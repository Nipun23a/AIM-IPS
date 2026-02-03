class AppDeepAnomalyLayer:
    """
    Pipeline adapter for application-level deep anomaly detection
    """

    def __init__(self, detector):
        self.detector = detector

    def inspect(self, req):
        payload = req.get("body", "") or req.get("path", "")

        score, is_anomaly = self.detector.predict(payload)

        req["scores"]["app_deep_anomaly"] = score
        req["flags"]["app_zero_day"] = bool(is_anomaly)

        return score
