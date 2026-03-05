"""
network_ips/network_ips.py
───────────────────────────
Network Layer IPS — Async Background Process

Architecture role (from diagram):
  Runs independently from the Application Layer IPS.
  Captures raw packets via Scapy, builds flows, classifies them,
  and writes threat scores to Redis (TTL=60s).

  The Application Layer IPS reads these scores via 0.10ms Redis lookup
  on every HTTP request — completely decoupled.

Run as a standalone background process:
    sudo python -m network_ips.network_ips --interface eth0
    sudo python -m network_ips.network_ips --interface eth0 --debug

Must be run with root/sudo (Scapy requires raw socket access).

For local testing without root:
    python -m network_ips.network_ips --simulate
    (injects synthetic flows directly without packet capture)
"""

import argparse
import logging
import signal
import sys
import time
import socket
import threading
from pathlib import Path

from pipeline.network_level.flow_acuumulator import FlowAccumulator
from pipeline.network_level.network_classifier import NetworkClassifier
from pipeline.network_level.feature import (
    LGBM_MODEL_PATH, LGBM_FEATURES_PATH,
    TCN_TFLITE_PATH, TCN_SCALER_PATH, TCN_FEATURES_PATH,
)

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    """Get the primary local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class NetworkLayerIPS:
    """
    Network Layer IPS — full background process.

    Lifecycle:
      1. Load models (LightGBM + TCN)
      2. Start flow sweep thread (expires timed-out flows every 5s)
      3. Start Scapy packet capture (blocking — runs until stopped)
      4. On each completed flow → NetworkClassifier → Redis write
    """

    def __init__(
        self,
        interface:  str  = "eth0",
        bpf_filter: str  = "ip",
        sweep_interval: int = 5,
        stats_interval: int = 30,
    ):
        self.interface      = interface
        self.bpf_filter     = bpf_filter
        self.sweep_interval = sweep_interval
        self.stats_interval = stats_interval
        self.local_ip       = get_local_ip()

        self._running    = False
        self._classifier = NetworkClassifier()  # uses paths from features.py
        self._accumulator   = FlowAccumulator(
            on_flow_complete=self._on_flow_complete
        )

        # Stats
        self._packets_seen    = 0
        self._start_time      = None

        logger.info(f"[NetworkIPS] Local IP: {self.local_ip}")
        logger.info(f"[NetworkIPS] Interface: {self.interface}")

    # ─────────────────────────────────────────────────────────
    # STARTUP
    # ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Load models and start capture. Blocking."""
        logger.info("=" * 60)
        logger.info("Network Layer IPS Starting")
        logger.info("=" * 60)

        # Load models
        logger.info("[NetworkIPS] Loading models...")
        self._classifier.load()
        status = self._classifier.status()
        logger.info(f"[NetworkIPS] LightGBM ready: {status['lgbm_ready']}")
        logger.info(f"[NetworkIPS] TCN ready:      {status['tcn_ready']}")
        logger.info(f"[NetworkIPS] Redis ready:    {status['redis_ok']}")

        if not status['lgbm_ready'] and not status['tcn_ready']:
            logger.error("[NetworkIPS] No models loaded — exiting")
            return

        self._running    = True
        self._start_time = time.time()

        # Start background threads
        sweep_thread = threading.Thread(
            target=self._sweep_loop,
            daemon=True,
            name="flow-sweep"
        )
        sweep_thread.start()

        stats_thread = threading.Thread(
            target=self._stats_loop,
            daemon=True,
            name="stats"
        )
        stats_thread.start()

        logger.info("[NetworkIPS] Ready — starting packet capture")
        logger.info("=" * 60)

        # Start capture (blocking)
        self._start_capture()

    def stop(self) -> None:
        self._running = False
        logger.info("[NetworkIPS] Stopping...")

    # ─────────────────────────────────────────────────────────
    # PACKET CAPTURE
    # ─────────────────────────────────────────────────────────

    def _start_capture(self) -> None:
        """
        Start Scapy packet capture.
        Requires root/sudo.
        """
        try:
            from scapy.all import sniff, IP, TCP, UDP
        except ImportError:
            logger.error("[NetworkIPS] Scapy not installed: pip install scapy")
            return

        logger.info(
            f"[NetworkIPS] Capturing on {self.interface} "
            f"filter='{self.bpf_filter}'"
        )

        try:
            from scapy.all import sniff
            sniff(
                iface   = self.interface,
                filter  = self.bpf_filter,
                prn     = self._process_packet,
                store   = False,          # don't store in memory
                stop_filter = lambda _: not self._running,
            )
        except PermissionError:
            logger.error(
                "[NetworkIPS] Permission denied — run with sudo:\n"
                "    sudo python -m network_ips.network_ips"
            )
        except Exception as e:
            logger.error(f"[NetworkIPS] Capture error: {e}", exc_info=True)

    def _process_packet(self, packet) -> None:
        """
        Scapy packet callback — called for every captured packet.
        Extracts minimal info and passes to FlowAccumulator.
        """
        try:
            from scapy.all import IP, TCP, UDP

            if not packet.haslayer(IP):
                return

            ip_layer  = packet[IP]
            src_ip    = ip_layer.src
            dst_ip    = ip_layer.dst
            length    = len(packet)
            timestamp = float(packet.time)
            flags     = 0

            # Extract TCP flags
            if packet.haslayer(TCP):
                flags = int(packet[TCP].flags)

            self._packets_seen += 1
            self._accumulator.add_packet(
                src_ip    = src_ip,
                dst_ip    = dst_ip,
                length    = length,
                flags     = flags,
                timestamp = timestamp,
                local_ip  = self.local_ip,
            )

        except Exception as e:
            logger.debug(f"[NetworkIPS] Packet processing error: {e}")

    # ─────────────────────────────────────────────────────────
    # FLOW COMPLETION CALLBACK
    # ─────────────────────────────────────────────────────────

    def _on_flow_complete(self, src_ip: str, features: dict) -> None:
        """
        Called by FlowAccumulator when a flow completes.
        Runs classifier and writes to Redis.
        """
        self._classifier.classify_flow(src_ip, features)

    # ─────────────────────────────────────────────────────────
    # BACKGROUND THREADS
    # ─────────────────────────────────────────────────────────

    def _sweep_loop(self) -> None:
        """Periodically expire timed-out flows."""
        while self._running:
            try:
                swept = self._accumulator.sweep_expired()
                if swept > 0:
                    logger.debug(f"[NetworkIPS] Swept {swept} expired flows")
            except Exception as e:
                logger.error(f"[NetworkIPS] Sweep error: {e}")
            time.sleep(self.sweep_interval)

    def _stats_loop(self) -> None:
        """Periodically log pipeline statistics."""
        while self._running:
            time.sleep(self.stats_interval)
            try:
                uptime   = time.time() - self._start_time
                flow_stats = self._accumulator.stats()
                cls_stats  = self._classifier.status()

                logger.info(
                    "[NetworkIPS] Stats | "
                    "uptime=%.0fs | packets=%d | "
                    "flows(active=%d completed=%d dropped=%d) | "
                    "threats=%d | redis_writes=%d",
                    uptime,
                    self._packets_seen,
                    flow_stats["active_flows"],
                    flow_stats["completed_flows"],
                    flow_stats["dropped_flows"],
                    cls_stats["threat_flows"],
                    cls_stats["redis_writes"],
                )
            except Exception as e:
                logger.error(f"[NetworkIPS] Stats error: {e}")

    # ─────────────────────────────────────────────────────────
    # SIMULATION MODE  (no root needed — for testing)
    # ─────────────────────────────────────────────────────────

    def simulate(self, flows: list = None) -> list:
        """
        Inject synthetic flows directly — no Scapy, no root needed.
        Used by test_network.py.

        Args:
            flows: list of (src_ip, feature_dict) tuples
                   if None, uses built-in test flows

        Returns:
            list of classification result dicts
        """
        self._classifier.load()

        if flows is None:
            flows = _default_test_flows()

        results = []
        for src_ip, features in flows:
            result = self._classifier.classify_flow(src_ip, features)
            results.append(result)

        return results


# ─────────────────────────────────────────────────────────────
# DEFAULT TEST FLOWS  (built-in synthetic data)
# ─────────────────────────────────────────────────────────────

def _default_test_flows() -> list:
    """
    Synthetic flow feature vectors for testing.
    Based on typical CICIDS2017 feature distributions.
    """
    return [
        # Normal web browsing
        ("192.168.1.100", {
            "flow duration": 2.5,
            "total fwd packets": 8,
            "total backward packets": 6,
            "total length of fwd packets": 1200,
            "total length of bwd packets": 8000,
            "fwd packet length mean": 150.0,
            "bwd packet length mean": 1333.0,
            "flow bytes/s": 3680.0,
            "flow packets/s": 5.6,
            "syn flag count": 1,
            "ack flag count": 12,
            "psh flag count": 4,
            "packet length mean": 654.0,
            "packet length std": 580.0,
            "idle mean": 0.1,
            "idle std": 0.05,
        }),

        # DDoS — high packet rate, tiny packets, many SYN flags
        ("10.0.0.50", {
            "flow duration": 0.5,
            "total fwd packets": 5000,
            "total backward packets": 0,
            "total length of fwd packets": 300000,
            "total length of bwd packets": 0,
            "fwd packet length mean": 60.0,
            "bwd packet length mean": 0.0,
            "flow bytes/s": 600000.0,
            "flow packets/s": 10000.0,
            "syn flag count": 5000,
            "ack flag count": 0,
            "psh flag count": 0,
            "packet length mean": 60.0,
            "packet length std": 5.0,
            "idle mean": 0.0,
            "idle std": 0.0,
        }),

        # Port Scan — many short flows, sequential ports, SYN only
        ("10.0.0.51", {
            "flow duration": 0.01,
            "total fwd packets": 1,
            "total backward packets": 0,
            "total length of fwd packets": 60,
            "total length of bwd packets": 0,
            "fwd packet length mean": 60.0,
            "bwd packet length mean": 0.0,
            "flow bytes/s": 6000.0,
            "flow packets/s": 100.0,
            "syn flag count": 1,
            "ack flag count": 0,
            "psh flag count": 0,
            "packet length mean": 60.0,
            "packet length std": 0.0,
            "idle mean": 0.0,
            "idle std": 0.0,
        }),

        # Botnet — periodic beaconing, regular intervals, small payloads
        ("10.0.0.52", {
            "flow duration": 60.0,
            "total fwd packets": 12,
            "total backward packets": 12,
            "total length of fwd packets": 720,
            "total length of bwd packets": 720,
            "fwd packet length mean": 60.0,
            "bwd packet length mean": 60.0,
            "flow bytes/s": 24.0,
            "flow packets/s": 0.4,
            "syn flag count": 1,
            "ack flag count": 22,
            "psh flag count": 10,
            "packet length mean": 60.0,
            "packet length std": 2.0,
            "idle mean": 5.0,
            "idle std": 0.1,
        }),
    ]


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AIM-IPS Network Layer IPS")
    parser.add_argument("--interface",  default="eth0",                          help="Network interface to capture on")
    parser.add_argument("--filter",     default="ip",                            help="BPF packet filter (default: 'ip')")
    # Model paths are configured in network_ips/src/features.py
    parser.add_argument("--simulate",   action="store_true",                     help="Run in simulation mode (no root needed)")
    parser.add_argument("--debug",      action="store_true",                     help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level   = logging.DEBUG if args.debug else logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    ips = NetworkLayerIPS(
        interface  = args.interface,
        bpf_filter = args.filter,
    )

    # Graceful shutdown on Ctrl+C
    def handle_signal(sig, frame):
        logger.info("[NetworkIPS] Caught signal — shutting down")
        ips.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if args.simulate:
        logger.info("[NetworkIPS] Running in SIMULATION mode")
        results = ips.simulate()
        print("\nSimulation Results:")
        print("-" * 60)
        for r in results:
            threat_str = "⚠ THREAT" if r["is_threat"] else "✓ CLEAN"
            print(
                f"{threat_str} | {r['src_ip']:15s} | "
                f"{r['attack_type']:12s} | "
                f"fused={r['fused_score']:.3f} | "
                f"lgbm={r['lgbm_score']:.3f} | "
                f"tcn={r['tcn_score']:.3f} | "
                f"redis={'✓' if r['redis_written'] else '✗'}"
            )
    else:
        ips.start()


if __name__ == "__main__":
    main()