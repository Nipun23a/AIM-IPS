from firewall.decisions import FirewallDecision
from utils.request_normalizer import normalize_request
class AIM_IPS_Pipeline:
    def _s_init__(self,firewall,lgbm,deep,responder):
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
        
        # Layer 2: LightGBM Model
        