from enum import Enum

class FirewallDecision(Enum):
    ALLOW = "allow"
    MITIGATE = "mitigate"
    FORWARD_TO_ML = "forward_to_ml"
    BLOCK = "block"

    