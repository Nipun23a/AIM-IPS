"""
pipeline/network_level/flow_acuumulator.py
──────────────────────────────────────────
Packet → Flow accumulation and feature extraction.

UNIT CONTRACT — must match CICFlowMeter / CICIDS2017 exactly:
  flow duration  → microseconds
  flow bytes/s   → bytes per second   (total_bytes / duration_seconds)
  flow packets/s → packets per second (total_pkts  / duration_seconds)
  idle mean/std  → microseconds
"""

import time
import threading
import logging
import numpy as np
from typing import Callable, Dict, List, Optional

from pipeline.network_level.feature import (
    THREAT_FEATURES, FLOW_TIMEOUT_SECONDS, MIN_PACKETS_PER_FLOW, MAX_FLOW_DURATION
)

logger = logging.getLogger(__name__)


class PacketRecord:
    __slots__ = ("timestamp", "length", "direction", "flags")

    def __init__(self, timestamp: float, length: int, direction: str, flags: int):
        self.timestamp = timestamp
        self.length    = length
        self.direction = direction
        self.flags     = flags


class Flow:
    def __init__(self, src_ip: str, start_time: float):
        self.src_ip     = src_ip
        self.start_time = start_time
        self.last_seen  = start_time
        self.packets: List[PacketRecord] = []
        self._idle_times: List[float] = []   # stored in SECONDS internally

    def add_packet(self, pkt: PacketRecord) -> None:
        self.packets.append(pkt)
        gap = pkt.timestamp - self.last_seen
        if gap > 1.0:
            self._idle_times.append(gap)
        self.last_seen = pkt.timestamp

    def is_expired(self, now: float) -> bool:
        return (
            (now - self.last_seen)  > FLOW_TIMEOUT_SECONDS or
            (now - self.start_time) > MAX_FLOW_DURATION
        )

    def has_fin_rst(self) -> bool:
        FIN, RST = 0x01, 0x04
        return any(p.flags & (FIN | RST) for p in self.packets if p.flags)

    def packet_count(self) -> int:
        return len(self.packets)


def extract_flow_features(flow: Flow) -> Optional[dict]:
    pkts = flow.packets
    if len(pkts) < MIN_PACKETS_PER_FLOW:
        return None

    # ── Duration ──────────────────────────────────────────────
    duration_sec = max(flow.last_seen - flow.start_time, 1e-9)
    duration_us  = duration_sec * 1_000_000          # µs — matches CICIDS2017

    # ── Packet splits ─────────────────────────────────────────
    fwd_pkts = [p for p in pkts if p.direction == "fwd"]
    bwd_pkts = [p for p in pkts if p.direction == "bwd"]

    fwd_lengths = [p.length for p in fwd_pkts] or [0]
    bwd_lengths = [p.length for p in bwd_pkts] or [0]
    all_lengths = [p.length for p in pkts]
    total_bytes = sum(all_lengths)

    # ── TCP flags ─────────────────────────────────────────────
    SYN, ACK, PSH = 0x02, 0x10, 0x08
    syn_count = sum(1 for p in pkts if p.flags and p.flags & SYN)
    ack_count = sum(1 for p in pkts if p.flags and p.flags & ACK)
    psh_count = sum(1 for p in pkts if p.flags and p.flags & PSH)

    # ── Idle times → microseconds ─────────────────────────────
    # _idle_times stored in seconds; convert to µs for the model
    idle_us = [t * 1_000_000 for t in flow._idle_times] if flow._idle_times else [0.0]

    features = {
        "flow duration":               duration_us,
        "total fwd packets":           len(fwd_pkts),
        "total backward packets":      len(bwd_pkts),
        "total length of fwd packets": sum(fwd_lengths),
        "total length of bwd packets": sum(bwd_lengths),
        "fwd packet length mean":      float(np.mean(fwd_lengths)),
        "bwd packet length mean":      float(np.mean(bwd_lengths)),
        "flow bytes/s":                total_bytes / duration_sec,
        "flow packets/s":              len(pkts)   / duration_sec,
        "syn flag count":              syn_count,
        "ack flag count":              ack_count,
        "psh flag count":              psh_count,
        "packet length mean":          float(np.mean(all_lengths)),
        "packet length std":           float(np.std(all_lengths)) if len(all_lengths) > 1 else 0.0,
        "idle mean":                   float(np.mean(idle_us)),
        "idle std":                    float(np.std(idle_us)) if len(idle_us) > 1 else 0.0,
    }

    missing = set(THREAT_FEATURES) - set(features.keys())
    if missing:
        logger.error(f"[Flow] Missing features: {missing}")
        return None

    return features


class FlowAccumulator:
    def __init__(self, on_flow_complete: Callable[[str, dict], None]):
        self._flows: Dict[str, Flow] = {}
        self._lock            = threading.Lock()
        self._on_complete     = on_flow_complete
        self._completed_count = 0
        self._dropped_count   = 0

    def add_packet(
        self,
        src_ip:    str,
        dst_ip:    str,
        length:    int,
        flags:     int,
        timestamp: float,
        local_ip:  str,
    ) -> None:
        direction = "bwd" if src_ip == local_ip else "fwd"
        pkt = PacketRecord(timestamp=timestamp, length=length,
                           direction=direction, flags=flags)

        with self._lock:
            flow_key = src_ip if direction == "fwd" else dst_ip
            if flow_key not in self._flows:
                self._flows[flow_key] = Flow(src_ip=flow_key, start_time=timestamp)
            flow = self._flows[flow_key]
            flow.add_packet(pkt)
            if flow.has_fin_rst():
                self._complete_flow(flow_key, flow)

    def sweep_expired(self) -> int:
        now     = time.time()
        expired = []
        with self._lock:
            for key, flow in self._flows.items():
                if flow.is_expired(now):
                    expired.append(key)
            for key in expired:
                flow = self._flows.pop(key)
                self._complete_flow_unlocked(key, flow)
        return len(expired)

    def _complete_flow(self, key: str, flow: Flow) -> None:
        del self._flows[key]
        self._complete_flow_unlocked(key, flow)

    def _complete_flow_unlocked(self, key: str, flow: Flow) -> None:
        features = extract_flow_features(flow)
        if features is None:
            self._dropped_count += 1
            logger.debug(
                f"[Flow] Dropped {key} — too few packets ({flow.packet_count()})"
            )
            return

        self._completed_count += 1
        logger.debug(
            f"[Flow] Complete {key} — "
            f"{flow.packet_count()} pkts | "
            f"duration={features['flow duration']:.0f}µs | "
            f"bytes/s={features['flow bytes/s']:.0f}"
        )
        try:
            self._on_complete(key, features)
        except Exception as e:
            logger.error(f"[Flow] on_complete callback failed for {key}: {e}")

    def stats(self) -> dict:
        with self._lock:
            active = len(self._flows)
        return {
            "active_flows":    active,
            "completed_flows": self._completed_count,
            "dropped_flows":   self._dropped_count,
        }