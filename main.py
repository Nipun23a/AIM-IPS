from pathlib import Path
from firewall.engine import StaticFirewall
from pipeline.aim_ips_pipeline import AIM_IPS_Pipeline
from response.response_engine import ResponseEngine

from threat_classifier.src.application_level_threat_classifier.inference import ApplicationThreatClassifier
from pipeline.app_lgbm_layer import AppLGBMLayer
from pipeline.app_deep_anomly_layer import AppDeepAnomalyLayer
from anomly_detector.src.application_level_attacks_anomality.inference import ApplicationDeepAnomalyDetector
APP_MODEL_DIR = Path("models/application_layer")
APP_DEEP_MODEL_DIR = Path("models/anomaly_detector/application_level_attacks")

app_classifier = ApplicationThreatClassifier(APP_MODEL_DIR)
app_deep_detector = ApplicationDeepAnomalyDetector(model_dir=APP_DEEP_MODEL_DIR,use_fusion=True)
app_lgbm_layer = AppLGBMLayer(app_classifier)

firewall = StaticFirewall()
responder = ResponseEngine()
app_deep_layer = AppDeepAnomalyLayer(app_deep_detector)

pipeline = AIM_IPS_Pipeline(
    firewall = firewall,
    lgbm = app_lgbm_layer,
    deep = app_deep_layer,
    responder = responder
)