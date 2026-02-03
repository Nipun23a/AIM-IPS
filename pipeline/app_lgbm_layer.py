class AppLGBMLayer:
    def __init__(self,classifier):
        self.classifier = classifier

    def inspect(self, req):
        payload = req.get("body", "") or req.get("path", "")

        score, attack_type, probs = self.classifier.predict(payload)

        req["scores"]["app_lgbm_score"] = score
        req["flags"]["app_attack_type"] = attack_type
        req["flags"]["app_probs"] = probs

        return score
