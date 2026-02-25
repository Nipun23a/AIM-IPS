from enum import Enum

class FirewallDecision(str,Enum):
    ALLOW = "ALLOW"
    DELAY = "DELAY"
    THROTTLE = "THROTTLE"
    CAPTCHA = "CAPTCHA"
    BLOCK = "BLOCK"
    MITIGATE = "MITIGATE"
    FORWARD_TO_ML = "FORWARD_TO_ML"

    def is_blocking(self) -> bool:
        return self in {FirewallDecision.BLOCK, FirewallDecision.MITIGATE}
    
    def is_graduated(self) -> bool:
        return self in (
            FirewallDecision.ALLOW,
            FirewallDecision.DELAY,
            FirewallDecision.THROTTLE,
            FirewallDecision.CAPTCHA,
            FirewallDecision.BLOCK,
        )

    