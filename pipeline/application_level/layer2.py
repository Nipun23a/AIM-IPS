import logging
from pathlib import Path

from shared.schemas import RequestContext,LayerScore
from shared.constants import LAYER_2_LGBM,LAYER_2_CNN
from threat_classifier.lgbm_classifier import LGBMAppClassifier
from anomly_detector.cnn_detector import CNNAnomalyDetector

logger = logging.getLogger(__name__)

class Layer2MLOrchestrator:
    def __init__(
        self,
        lgbm_model_dir: Path = Path("models/application_layer"),
        cnn_model_dir: Path = Path("models/anomaly_detector/application_level_attacks"),
        run_lgbm : bool = True,
        run_cnn: bool = True,
    ) : 
        self.run_lgbm = run_lgbm
        self.cnn = run_cnn
        self.lgbm: LGBMAppClassifier = None
        self.cnn : CNNAnomalyDetector = None

        if run_lgbm:
            self.lgbm = LGBMAppClassifier(model_dir=lgbm_model_dir)
        if run_cnn:
            self.cnn = CNNAnomalyDetector(model_dir=cnn_model_dir)

    def load(self) -> "Layer2MLOrchestrator":
        if self.lgbm:
            try:
                self.lgbm.load()
                logger.info("[Layer2] LightGBM app classifier loaded")
            except Exception as e:
                logger.error(f"[Layer2] Failed to load LightGBM : {e}")
                self.lgbm = None
        
        if self.cnn:
            try:
                self.cnn.load()
                logger.info("[Layer2] CNN Autoencoder loaded")
            except Exception as e:
                logger.error(f"[Layer2] Failed to load CNN: {e}")
                self.cnn = None

        return self
    
    def run(self, ctx: RequestContext) -> RequestContext:
        # ── Step 1: CNN Autoencoder as anomaly gate ───────────────────
        # Run CNN first. Only anomalous requests proceed to LightGBM
        # classification. Clean traffic is short-circuited here,
        # avoiding LightGBM false-positives on benign payloads.
        if self.cnn and self.cnn.is_ready():
            cnn_score = self.cnn.predict(ctx)
            ctx.add_score(cnn_score)
            logger.debug(
                f"[Layer2] CNN -> {cnn_score.label} "
                f"score={cnn_score.score:.3f}"
            )
        else:
            cnn_score = LayerScore.clean(LAYER_2_CNN)
            ctx.add_score(cnn_score)
            logger.warning("[Layer2] CNN unavailable - using zero score")

        # ── Step 2: LightGBM — only if CNN flagged an anomaly ────────
        cnn_flagged = cnn_score.label in {"anomaly", "zeroday"} or cnn_score.score > 0.5

        if cnn_flagged:
            if self.lgbm and self.lgbm.is_ready():
                lgbm_score = self.lgbm.predict(ctx)
                ctx.add_score(lgbm_score)
                logger.debug(
                    f"[Layer2] LGBM -> {lgbm_score.label} "
                    f"score={lgbm_score.score:.3f}"
                )
            else:
                ctx.add_score(LayerScore.clean(LAYER_2_LGBM))
                logger.warning("[Layer2] LightGBM unavailable - using zero score")
        else:
            # CNN passed as clean — skip LightGBM, treat as normal
            ctx.add_score(LayerScore(
                score=0.0,
                label="clean",
                confidence=1.0,
                layer=LAYER_2_LGBM,
                triggered=False,
                metadata={"skipped_reason": "cnn_gate_clean"},
            ))
            logger.debug("[Layer2] CNN gate clean — LightGBM skipped")

        return ctx
    
    def status(self) -> dict:
        return{
            "lgbm_ready": self.lgbm is not None and self.lgbm.is_ready(),
            "cnn_ready":  self.cnn  is not None and self.cnn.is_ready(),
        }


