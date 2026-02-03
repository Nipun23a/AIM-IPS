from firewall.decisions import FirewallDecision
from utils.request_normalizer import normalize_request
class AIM_IPS_Pipeline:
    def __init__(self,firewall,lgbm,deep,responder):
        self.firewall = firewall
        self.lgbm = lgbm
        self.deep = deep
        self.responder = responder

    def process(self,raw_request):
        req = normalize_request(raw_request)

        # Layer 1: Firewall Inspection
        decision,reason = self.firewall.inspect(req)

        if decision == FirewallDecision.MITIGATE:
            return decision, reason
        if decision == FirewallDecision.ALLOW:
            return FirewallDecision.ALLOW, "Clean Traffic"
        
        if decision == FirewallDecision.FORWARD_TO_ML:
            # Layer 2: LightGBM Model
            lgbm_score = self.lgbm.inspect(req)

            if lgbm_score >= 0.8:
                return FirewallDecision.BLOCK, {
                    "reason" : "Known Attack (ML)",
                    "score" : lgbm_score
                }
            
            # Layer 3: Deep Learning Model
            if lgbm_score < 0.4:
                self.deep.inspect(req)

            final_decision,info = self.responder.decide(req)
            return final_decision, info
        