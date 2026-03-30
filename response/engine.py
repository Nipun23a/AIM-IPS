import logging
from typing import Tuple, Optional

from firewall.decisions import FirewallDecision
from shared.schemas import RequestContext,LayerScore
from shared.constants import (
    SCORE_ALLOW_MAX,SCORE_DELAY_MAX,SCORE_THROTTLE_MAX,SCORE_CAPTCHA_MAX,
    WEIGHT_LAYER1_REGEX,WEIGHT_LAYER2_LGBM,WEIGHT_LAYER2_CNN,WEIGHT_NETWORK,
    LAYER_1,LAYER_2_LGBM,LAYER_2_CNN,LAYER_NETWORK,
    ACTION_ALLOW,ACTION_DELAY,ACTION_THROTTLE,ACTION_CAPTCHA,ACTION_BLOCK
    
)
from utils.redis_client import RedisClient

logger = logging.getLogger(__name__)
class ResponseEngine:
    """
       Layer 3 - Dynamic response engine for AIM-IPS.
       Replaces the old dict-based ResponseEngine with full RequestContext support.
       Backward-compatible: also acceps old dict format via decide_lagancy().

    """

    def __init__(self,auto_blacklist : bool = True):
        """
           auto_blacklist: if True, automatically blacklist IPs on BLOCK decisions.
        """
        self.auto_blacklist = auto_blacklist
        self._redis = None 


    @property
    def redis(self):
        if self._redis is None:
            try:
                self._redis = RedisClient.get_redis()
            except Exception as e:
                logger.warning(f"[ResponseEngine] Failed to connect to Redis: {e}")
        return self._redis
    
    def decide(self, ctx:RequestContext) -> Tuple[FirewallDecision, dict]:
        if ctx.was_hard_blocked() or ctx.short_circuited:
            triggered = next((s for s in ctx.layer_scores if s.triggered), None)
            label = triggered.layer if triggered else "pattern_match"
            reason = triggered.metadata.get("reason", "Hard block ") if triggered else "Layer 0/1 match"

            ctx.final_score = 1.0
            ctx.action = ACTION_BLOCK
            ctx.block_reason = reason

            self._handle_block(ctx.ip, reason)

            return FirewallDecision.BLOCK, {
                "action": ACTION_BLOCK,
                "risk": ctx.final_score,
                "reason": reason,
                "label" : label,
                "layer" : triggered.layer if triggered else "Layer0"
            }
        
        scores = ctx.scores_dict()

        fused_score = (
            WEIGHT_LAYER1_REGEX * scores.get("regex_conf",     0.0) +
            WEIGHT_LAYER2_LGBM  * scores.get("app_lgbm_score", 0.0) +
            WEIGHT_LAYER2_CNN   * scores.get("deep_anomaly",   0.0) +
            WEIGHT_NETWORK      * scores.get("net_lgbm_score", 0.0)
        )
        fused_score = round(max(0.0, min(1.0, fused_score)), 4)

        ctx.final_score = fused_score

        decision, action_str = self._score_to_action(fused_score)
        ctx.action = action_str

        if decision == FirewallDecision.BLOCK:
            reason = self._build_reason(ctx)
            ctx.block_reason = reason
            self._handle_block(ctx.ip, reason)
        elif decision == FirewallDecision.CAPTCHA and self.redis:
            self.redis.set_captcha_session(ctx.ip)

        logger.info(
            f"[ResponseEngine] {ctx.ip} → {action_str} "
            f"(score={fused_score} | "
            f"regex={scores['regex_conf']:.2f} "
            f"lgbm={scores['app_lgbm_score']:.2f} "
            f"cnn={scores['deep_anomaly']:.2f} "
            f"net={scores['net_lgbm_score']:.2f})"
        )

        return decision, {
            "action": action_str,
            "risk": fused_score,
            "reason": ctx.block_reason or self._build_reason(ctx),
            "scores": scores,
            "request_id": ctx.request_id,
        }
    
    def decide_legacy(self, req: dict) -> Tuple[FirewallDecision, dict]:
        """
        Backward-compatible interface for old dict-based callers.
        Maps to new decide() internally.

        Old usage:
            engine.decide({"scores": {...}, "flags": {...}})
        New usage:
            engine.decide(request_context)
        """
        from shared.schemas import RequestContext, LayerScore
        from shared.constants import LAYER_1, LAYER_2_LGBM, LAYER_2_CNN, LAYER_NETWORK

        scores = req.get("scores", {})
        flags  = req.get("flags", {})
        ip     = req.get("ip", "0.0.0.0")

        ctx = RequestContext(ip=ip, method="GET", path="/")
        ctx.network_score = scores.get("net_lgbm_score", 0.0)

        if scores.get("regex_conf", 0.0) > 0:
            ctx.add_score(LayerScore(
                score=scores["regex_conf"], label="regex_match",
                confidence=scores["regex_conf"], layer=LAYER_1, triggered=False,
            ))
        if scores.get("app_lgbm_score", 0.0) > 0:
            ctx.add_score(LayerScore(
                score=scores["app_lgbm_score"], label=flags.get("app_attack_type", "unknown"),
                confidence=scores["app_lgbm_score"], layer=LAYER_2_LGBM, triggered=False,
            ))
        if scores.get("deep_anomaly", 0.0) > 0:
            ctx.add_score(LayerScore(
                score=scores["deep_anomaly"], label="anomaly",
                confidence=scores["deep_anomaly"], layer=LAYER_2_CNN, triggered=False,
            ))

        return self.decide(ctx)
    
    def _score_to_action(self, score: float) -> Tuple[FirewallDecision, str]:
        """Map fused score to graduated action. Exact thresholds from diagram."""
        if score >= SCORE_CAPTCHA_MAX:          # ≥ 0.85
            return FirewallDecision.BLOCK,    ACTION_BLOCK
        elif score >= SCORE_THROTTLE_MAX:       # 0.65–0.85
            return FirewallDecision.CAPTCHA,  ACTION_CAPTCHA
        elif score >= SCORE_DELAY_MAX:          # 0.50–0.65
            return FirewallDecision.THROTTLE, ACTION_THROTTLE
        elif score >= SCORE_ALLOW_MAX:          # 0.35–0.50
            return FirewallDecision.DELAY,    ACTION_DELAY
        else:                                   # < 0.35
            return FirewallDecision.ALLOW,    ACTION_ALLOW

    def _build_reason(self, ctx: RequestContext) -> str:
        """Build human-readable reason string from layer scores."""
        triggered = [s for s in ctx.layer_scores if s.score > 0.5]
        if not triggered:
            return "Clean traffic"
        top = max(triggered, key=lambda s: s.score)
        return f"Detected {top.label} (layer={top.layer}, score={top.score:.2f})"

    def _handle_block(self, ip: str, reason: str) -> None:
        """Write IP to Redis blacklist only — fast, stays in critical path.
        PostgreSQL audit trail is handled separately in middleware._log_async()."""
        if self.auto_blacklist and self.redis:
            self.redis.blacklist_ip(ip, reason=reason, permanent=False)
    
