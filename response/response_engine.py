from firewall.decisions import FirewallDecision


class ResponseEngine:
    """
    Dynamic response engine for AIM-IPS.
    Aggregates scores from different layers and
    produces a final security decision.
    """

    def __init__(
        self,
        block_threshold=0.85,
        throttle_threshold=0.6,
        monitor_threshold=0.4
    ):
        self.block_threshold = block_threshold
        self.throttle_threshold = throttle_threshold
        self.monitor_threshold = monitor_threshold

    def decide(self, req):
        """
        Decide final action based on accumulated scores.
        """

        scores = req.get("scores", {})
        flags = req.get("flags", {})

        # Extract scores safely
        regex_conf = scores.get("regex_conf", 0.0)
        app_lgbm = scores.get("app_lgbm_score", 0.0)
        net_lgbm = scores.get("net_lgbm_score", 0.0)   # future
        deep_score = scores.get("deep_anomaly", 0.0)   # future

        # ---------- Risk aggregation ----------
        risk_score = (
            0.25 * regex_conf +
            0.45 * app_lgbm +
            0.20 * net_lgbm +
            0.10 * deep_score
        )

        # Store for logging / explainability
        req["scores"]["final_risk"] = round(risk_score, 4)

        # ---------- Decision logic ----------
        if risk_score >= self.block_threshold:
            return FirewallDecision.BLOCK, {
                "action": "BLOCK",
                "risk": risk_score,
                "reason": self._reason(flags, "High confidence attack")
            }

        if risk_score >= self.throttle_threshold:
            return FirewallDecision.THROTTLE, {
                "action": "THROTTLE",
                "risk": risk_score,
                "reason": self._reason(flags, "Suspicious behavior")
            }

        if risk_score >= self.monitor_threshold:
            return FirewallDecision.ALLOW, {
                "action": "ALLOW_MONITOR",
                "risk": risk_score,
                "reason": "Allowed but monitored"
            }

        return FirewallDecision.ALLOW, {
            "action": "ALLOW",
            "risk": risk_score,
            "reason": "Clean traffic"
        }

    def _reason(self, flags, default):
        """
        Generate explainable reason for decision.
        """
        if "app_attack_type" in flags:
            return f"Detected {flags['app_attack_type']} pattern"
        return default
