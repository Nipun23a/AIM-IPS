import pandas as pd
import numpy as np
import joblib
import json
import time
from pathlib import Path
from typing import Dict, List

from threat_classifier.src.network_level_threat_classifier.features import THREAT_FEATURES
from threat_classifier.src.network_level_threat_classifier.labels import LABEL_MAP


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "threat_classifier"


class NetworkThreatClassifier:
    """
    Fast network-level inference for CICIDS flow-based detection.
    Optimized for real-time IDS deployment.
    """

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = MODEL_DIR

        self.model_dir = Path(model_dir)
        self.model = None
        self.inverse_label_map = {v: k for k, v in LABEL_MAP.items()}

        self._load_model()

    # ======================================================
    # Load Model
    # ======================================================
    def _load_model(self):
        print("[INFO] Loading network threat classifier...")

        model_path = self.model_dir / "lgb_model.pkl"
        features_path = self.model_dir / "features.pkl"

        self.model = joblib.load(model_path)
        self.features = joblib.load(features_path)

        print("[INFO] Model loaded successfully")
        print(f"[INFO] Supported attack types: {list(LABEL_MAP.keys())}")

    # ======================================================
    # Single Flow Prediction
    # ======================================================
    def predict(self, flow_features: Dict, return_probabilities=False) -> Dict:
        """
        Predict threat type for a single network flow.

        Args:
            flow_features: Dictionary containing the 16 required flow features
            return_probabilities: Return class probabilities if True
        """

        start_time = time.perf_counter()

        # Ensure feature order consistency
        input_vector = [flow_features.get(f, 0.0) for f in self.features]
        df = pd.DataFrame([input_vector], columns=self.features)

        # Inference
        inference_start = time.perf_counter()
        probabilities = self.model.predict_proba(df)[0]
        inference_time = (time.perf_counter() - inference_start) * 1000

        predicted_idx = np.argmax(probabilities)
        predicted_label = self.inverse_label_map[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        total_time = (time.perf_counter() - start_time) * 1000

        result = {
            "attack_type": predicted_label,
            "confidence": confidence,
            "is_malicious": predicted_label != "benign",
            "timing": {
                "model_inference_ms": inference_time,
                "total_ms": total_time
            }
        }

        if return_probabilities:
            result["probabilities"] = {
                self.inverse_label_map[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }

        return result

    # ======================================================
    # Batch Prediction
    # ======================================================
    def predict_batch(self, flows: List[Dict]) -> List[Dict]:

        start_time = time.perf_counter()

        rows = []
        for flow in flows:
            row = [flow.get(f, 0.0) for f in self.features]
            rows.append(row)

        df = pd.DataFrame(rows, columns=self.features)

        probabilities = self.model.predict_proba(df)

        results = []

        for i, proba in enumerate(probabilities):
            predicted_idx = np.argmax(proba)
            predicted_label = self.inverse_label_map[predicted_idx]
            confidence = float(proba[predicted_idx])

            results.append({
                "attack_type": predicted_label,
                "confidence": confidence,
                "is_malicious": predicted_label != "benign"
            })

        total_time = (time.perf_counter() - start_time) * 1000
        avg_time = total_time / len(flows)

        print(f"[INFO] Batch prediction completed: {len(flows)} flows")
        print(f"[INFO] Average time per flow: {avg_time:.3f} ms")

        return results


# ==========================================================
# BENCHMARK
# ==========================================================
def benchmark_latency(classifier, num_iterations=5000):

    print("\n" + "="*80)
    print(f"NETWORK INFERENCE BENCHMARK - {num_iterations} iterations")
    print("="*80)

    # Example synthetic flow
    sample_flow = {
        "flow_duration": 500000,
        "total_fwd_packets": 10,
        "total_backward_packets": 5,
        "total_length_of_fwd_packets": 2000,
        "total_length_of_bwd_packets": 1500,
        "fwd_packet_length_mean": 200,
        "bwd_packet_length_mean": 180,
        "flow_bytes/s": 50000,
        "flow_packets/s": 20,
        "syn_flag_count": 1,
        "ack_flag_count": 1,
        "psh_flag_count": 0,
        "packet_length_mean": 190,
        "packet_length_std": 30,
        "idle_mean": 1000,
        "idle_std": 50,
    }

    timings = []

    for _ in range(num_iterations):
        result = classifier.predict(sample_flow)
        timings.append(result["timing"]["total_ms"])

    data = np.array(timings)

    print(f"\nMean latency: {np.mean(data):.3f} ms")
    print(f"P95 latency: {np.percentile(data, 95):.3f} ms")
    print(f"P99 latency: {np.percentile(data, 99):.3f} ms")
    print(f"Throughput: ~{1000/np.mean(data):.0f} flows/sec (single-threaded)")


# ==========================================================
# DEMO
# ==========================================================
def demo_prediction(classifier):

    print("\n" + "="*80)
    print("NETWORK FLOW DEMO PREDICTION")
    print("="*80)

    sample_flow = {
        "flow_duration": 1000000,
        "total_fwd_packets": 15,
        "total_backward_packets": 3,
        "total_length_of_fwd_packets": 5000,
        "total_length_of_bwd_packets": 800,
        "fwd_packet_length_mean": 300,
        "bwd_packet_length_mean": 120,
        "flow_bytes/s": 80000,
        "flow_packets/s": 30,
        "syn_flag_count": 2,
        "ack_flag_count": 1,
        "psh_flag_count": 1,
        "packet_length_mean": 250,
        "packet_length_std": 40,
        "idle_mean": 500,
        "idle_std": 20,
    }

    result = classifier.predict(sample_flow, return_probabilities=True)

    print(f"\nDetected: {result['attack_type']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Malicious: {result['is_malicious']}")
    print(f"Latency: {result['timing']['total_ms']:.3f} ms")


# ==========================================================
# MAIN
# ==========================================================
def main():

    print("\n" + "="*80)
    print("NETWORK LEVEL THREAT CLASSIFIER - INFERENCE")
    print("="*80)

    classifier = NetworkThreatClassifier()

    demo_prediction(classifier)
    benchmark_latency(classifier, num_iterations=10000)

    print("\nINFERENCE COMPLETED!")


if __name__ == "__main__":
    main()
