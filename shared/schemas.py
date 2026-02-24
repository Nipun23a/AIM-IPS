from dataclasses import dataclass,field
from typing import Optional,List

import time
import uuid

@dataclass
class LayerScore:
    """
    Standardized score output from any pipeline layer.
    score -> always 0.0 - 1.0
    triggered -> True = hard-block, skip fusion immediately
    layer -> use LAYER_* constants from shared.constants
    label -> use LABEL_* constants from shared.constants
    """
    score: float
    label: str
    confidence: float
    layer: str
    triggered: bool 
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.score = max(0.0, min(1.0, float(self.score)))  
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @classmethod
    def hard_block(cls, layer:str, label:str, reason:str = "") -> "LayerScore":
        """ Layer 0/1 pattern match - score = 1.0, triggered = True, skip fusion."""
        return cls(
            score=1.0,
            label = label,
            confidence = 1.0,
            layer=layer,
            triggered=True,
            metadata={"reason": reason}
        )
    
    @classmethod
    def clean(cls, layer:str) -> "LayerScore":
        """ Clean traffic - score = 0.0, triggered = False."""
        return cls(
            score=0.0,
            label="clean",
            confidence=1.0,
            layer=layer,
            triggered=False
        )

# -----------------------------------------
# Request Context
# One instance per HTTP request, passed through the full pipeline.
# -----------------------------------------

@dataclass
class RequestContext:
    """
    Full context for a single HTTP request flowing through AIM-IPS.
    Populated at FastAPI middleware entry. enriched layer by layer
    """
    # --------- HTTP request fields -----------------------
    ip : str
    method: str
    path: str
    headers : dict = field(default_factory=dict)
    query_params: dict = field(default_factory=dict)
    body: str = ""
    timestamp: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    #
    layer_scores: List[LayerScore] = field(default_factory=list)

    network_threat: Optional[dict] = None
    network_score: float = 0.0

    # Final Output (set by Layer 3)
    final_score: float = 0.0
    action: str = "PENDING"       # ALLOW| DELAY | THROTTLE | CAPTCHA | BLOCK
    block_reason: str = ""

    # Pipeline control
    short_circuited: bool = False  # True = Layer 0/1 hard block, skip ML

    def add_score(self, layer_score: LayerScore):
        self.layer_scores.append(layer_score)

    def get_score_by_layer(self, layer:str) -> Optional[LayerScore]:
        for ls in self.layer_scores:
            if ls.layer == layer:
                return ls
        return None
    
    def was_hard_blocked(self) -> bool:
        return any(ls.triggered for ls in self.layer_scores)
    
    def scores_dict(self) -> dict:
        """ Flat dict for ResponseEngine and logging - matches existing code keys."""
        d = {
            "regex_conf" : 0.0,
            "app_lgbm_score": 0.0,
            "net_lgbm_score": self.network_score,
            "deep_anomaly": 0.0,
            "final_risk": self.final_score
        }

        for ls in self.layer_scores:
            if ls.layer == "layer1_regex":
                d["regex_conf"] = ls.score
            elif ls.layer == "layer2_lgbm":
                d["app_lgbm_score"] = ls.score
            elif ls.layer == "layer2_cnn":
                d["deep_anomaly"] = ls.score
            elif ls.layer == "network_ips":
                d["net_lgbm_score"] = ls.score
        return d
    
    def to_log_dict(self) -> dict:
        """Serializable dict for DB logging and dashboard."""
        return {
            "request_id":      self.request_id,
            "ip":              self.ip,
            "method":          self.method,
            "path":            self.path,
            "timestamp":       self.timestamp,
            "final_score":     round(self.final_score, 4),
            "action":          self.action,
            "block_reason":    self.block_reason,
            "short_circuited": self.short_circuited,
            "network_score":   round(self.network_score, 4),
            "scores":          self.scores_dict(),
            "layer_scores": [
                {
                    "layer":      ls.layer,
                    "score":      round(ls.score, 4),
                    "label":      ls.label,
                    "confidence": round(ls.confidence, 4),
                    "triggered":  ls.triggered,
                    "metadata":   ls.metadata,
                }
                for ls in self.layer_scores
            ],
        }



