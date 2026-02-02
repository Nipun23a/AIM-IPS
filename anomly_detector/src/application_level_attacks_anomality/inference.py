import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from pathlib import Path
import sys
import math
import re
import time
from collections import Counter
from statistics import mean, median, stdev

# Feature extraction functions (same as training)
SQL_KW = re.compile(r"(select|union|insert|drop|or\s+1=1)", re.I)
XSS_KW = re.compile(r"(<script|onerror=|alert\()", re.I)
PATH_KW = re.compile(r"(\.\./|\.\.\\|/etc/passwd)", re.I)
CMD_KW = re.compile(r"(;|\|\||&&|\b(cat|ls|whoami)\b)", re.I)

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [c/len(s) for c in counts.values()]
    return -sum(p*math.log(p) for p in probs)

def extract_payload_features(payload: str) -> dict:
    payload = str(payload)
    return {
        "payload_len": len(payload),
        "digit_count": sum(c.isdigit() for c in payload),
        "alpha_count": sum(c.isalpha() for c in payload),
        "special_char_count": sum(not c.isalnum() for c in payload),
        "slash_count": payload.count("/"),
        "dot_count": payload.count("."),
        "percent_count": payload.count("%"),
        "equals_count": payload.count("="),
        "question_count": payload.count("?"),
        "amp_count": payload.count("&"),
        "has_sql_kw": int(bool(SQL_KW.search(payload))),
        "has_xss_kw": int(bool(XSS_KW.search(payload))),
        "has_path_kw": int(bool(PATH_KW.search(payload))),
        "has_cmd_kw": int(bool(CMD_KW.search(payload))),
        "entropy": shannon_entropy(payload),
    }


class FastWebPayloadDetector:
    """
    OPTIMIZED detector using ONLY Autoencoder (no PCA/Mahalanobis)
    Much faster for real-time IPS scenarios
    """
    
    def __init__(self, models_dir, model_prefix="fusion"):
        """
        Load only essential models for fast inference
        """
        self.models_dir = Path(models_dir)
        print(f"[INFO] Loading OPTIMIZED models from: {self.models_dir}")
        
        try:
            # Load only scaler and autoencoder
            self.scaler = joblib.load(self.models_dir / "scaler.pkl")
            print("✓ Loaded scaler")
            
            self.ae = load_model(self.models_dir / f"{model_prefix}_ae.keras")
            print("✓ Loaded autoencoder")
            
            self.ae_threshold = joblib.load(self.models_dir / f"{model_prefix}_ae_threshold.pkl")
            print(f"✓ Loaded AE threshold: {self.ae_threshold:.6f}")
            
            self.features = joblib.load(self.models_dir / f"{model_prefix}_features.pkl")
            print(f"✓ Loaded features list ({len(self.features)} features)")
            
            print("\n⚡ OPTIMIZATION: Skipped PCA + Mahalanobis for maximum speed")
            print("   Expected latency improvement: 3-5x faster\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            raise
    
    def predict_single_payload(self, payload: str, threshold_multiplier=1.0):
        """
        Fast prediction for a single payload
        
        Args:
            payload: URL/request payload string
            threshold_multiplier: Adjust sensitivity (default=1.0)
                - <1.0 = more sensitive (more detections)
                - >1.0 = less sensitive (fewer false positives)
        """
        # Extract features
        features_dict = extract_payload_features(payload)
        df = pd.DataFrame([features_dict])
        
        # Predict
        result = self.predict(df, threshold_multiplier)
        
        return {
            'payload': payload,
            'is_anomaly': bool(result['is_anomaly'][0]),
            'anomaly_score': float(result['anomaly_scores'][0]),
            'recon_error': float(result['recon_errors'][0]),
            'threshold_used': float(result['threshold_used'])
        }
    
    def predict(self, payload_df, threshold_multiplier=1.0):
        """
        FAST prediction using only autoencoder
        
        Args:
            payload_df: DataFrame with feature columns
            threshold_multiplier: Adjust threshold (1.0 = default, >1.0 = stricter)
            
        Returns:
            dict with predictions and scores
        """
        # Ensure correct feature order and fill missing
        for feat in self.features:
            if feat not in payload_df.columns:
                payload_df[feat] = 0.0
        
        X = payload_df[self.features].values.astype(np.float64)
        X_scaled = self.scaler.transform(X)
        
        # Only autoencoder reconstruction
        reconstructed = self.ae.predict(X_scaled, verbose=0)
        recon_err = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        
        # Adjusted threshold
        threshold = self.ae_threshold * threshold_multiplier
        
        # Normalize scores to 0-1 range for easier interpretation
        anomaly_scores = np.clip(recon_err / (threshold + 1e-12), 0, 1)
        
        return {
            'is_anomaly': (recon_err > threshold).astype(int),
            'anomaly_scores': anomaly_scores,
            'recon_errors': recon_err,
            'threshold_used': threshold
        }


def test_sample_payloads(detector):
    """
    Test the detector with sample payloads
    """
    test_payloads = [
        # Normal payloads
        ("/index.html", "Normal - Homepage"),
        ("/api/users?id=123", "Normal - API call"),
        ("/products/search?query=laptop", "Normal - Search"),
        ("/static/css/style.css", "Normal - Static file"),
        
        # SQL Injection
        ("' OR 1=1--", "SQLi - Classic"),
        ("admin' UNION SELECT * FROM users--", "SQLi - Union"),
        ("1' AND 1=1--", "SQLi - Boolean"),
        
        # XSS
        ("<script>alert('XSS')</script>", "XSS - Script tag"),
        ("<img src=x onerror=alert(1)>", "XSS - Event handler"),
        ("javascript:alert(1)", "XSS - JavaScript protocol"),
        
        # Path Traversal
        ("../../etc/passwd", "Path Traversal - Unix"),
        ("..\\..\\windows\\system32\\config\\sam", "Path Traversal - Windows"),
        
        # Command Injection
        ("; cat /etc/passwd", "Command Injection - Semicolon"),
        ("| whoami", "Command Injection - Pipe"),
        ("$(cat /etc/passwd)", "Command Injection - Subshell"),
    ]
    
    print("\n" + "="*80)
    print("TESTING SAMPLE PAYLOADS (OPTIMIZED AE-ONLY)")
    print("="*80 + "\n")
    
    for payload, description in test_payloads:
        result = detector.predict_single_payload(payload)
        
        status = "🚨 ANOMALY" if result['is_anomaly'] else "✓ NORMAL"
        print(f"{status} | Score: {result['anomaly_score']:.4f} | {description}")
        print(f"         | Payload: {payload[:60]}")
        print(f"         | Recon Error: {result['recon_error']:.6f}")
        print()


def benchmark_performance(detector, n_requests=1000, batch_sizes=[1, 10, 50, 100]):
    """
    Benchmark optimized detector performance
    """
    print("\n" + "="*80)
    print(f"PERFORMANCE BENCHMARK - {n_requests} REQUESTS (AE-ONLY OPTIMIZED)")
    print("="*80 + "\n")
    
    # Generate test payloads
    test_payloads = [
        "/index.html",
        "/api/users?id=123",
        "/products/search?query=laptop",
        "' OR 1=1--",
        "admin' UNION SELECT * FROM users--",
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert(1)>",
        "../../etc/passwd",
        "..\\..\\windows\\system32\\config\\sam",
        "; cat /etc/passwd",
    ]
    
    payloads = (test_payloads * (n_requests // len(test_payloads) + 1))[:n_requests]
    
    print(f"Generated {len(payloads)} test payloads\n")
    
    # Extract features
    print("Extracting features...")
    start_feature_extraction = time.time()
    features_list = [extract_payload_features(p) for p in payloads]
    df_all = pd.DataFrame(features_list)
    feature_extraction_time = time.time() - start_feature_extraction
    print(f"Feature extraction: {feature_extraction_time:.4f}s total")
    print(f"  Per request: {(feature_extraction_time/len(payloads))*1000:.4f}ms\n")
    
    # Test different batch sizes
    print("="*60)
    print("BATCH PROCESSING PERFORMANCE")
    print("="*60 + "\n")
    
    results_table = []
    
    for batch_size in batch_sizes:
        times = []
        n_batches = (len(payloads) + batch_size - 1) // batch_size
        
        print(f"Testing batch size: {batch_size} ({n_batches} batches)...")
        
        start_total = time.time()
        
        for i in range(0, len(payloads), batch_size):
            batch_df = df_all.iloc[i:i+batch_size]
            
            start_batch = time.time()
            _ = detector.predict(batch_df)
            batch_time = time.time() - start_batch
            
            times.append(batch_time)
        
        total_time = time.time() - start_total
        
        # Calculate statistics
        time_per_request = (total_time / len(payloads)) * 1000  # ms
        throughput = len(payloads) / total_time  # requests/sec
        
        results_table.append({
            'batch_size': batch_size,
            'total_time': total_time,
            'avg_time_per_request_ms': time_per_request,
            'median_batch_time_ms': median(times) * 1000,
            'std_batch_time_ms': stdev(times) * 1000 if len(times) > 1 else 0,
            'throughput_req_per_sec': throughput,
            'min_batch_time_ms': min(times) * 1000,
            'max_batch_time_ms': max(times) * 1000,
        })
        
        print(f"  ✓ Completed in {total_time:.4f}s")
        print(f"    Avg time per request: {time_per_request:.4f}ms")
        print(f"    Throughput: {throughput:.2f} req/sec\n")
    
    # Print summary table
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80 + "\n")
    
    print(f"{'Batch Size':<12} {'Avg Time/Req':<16} {'Throughput':<18} {'Total Time':<12}")
    print(f"{'':12} {'(ms)':<16} {'(req/sec)':<18} {'(sec)':<12}")
    print("-" * 80)
    
    for r in results_table:
        print(f"{r['batch_size']:<12} {r['avg_time_per_request_ms']:<16.4f} "
              f"{r['throughput_req_per_sec']:<18.2f} {r['total_time']:<12.4f}")
    
    # Single request performance
    print("\n" + "="*80)
    print("SINGLE REQUEST PERFORMANCE (CRITICAL FOR IPS)")
    print("="*80 + "\n")
    
    single_req_result = [r for r in results_table if r['batch_size'] == 1][0]
    
    print(f"Average latency per request: {single_req_result['avg_time_per_request_ms']:.4f} ms")
    print(f"Median batch time:          {single_req_result['median_batch_time_ms']:.4f} ms")
    print(f"Std deviation:              {single_req_result['std_batch_time_ms']:.4f} ms")
    print(f"Min time:                   {single_req_result['min_batch_time_ms']:.4f} ms")
    print(f"Max time:                   {single_req_result['max_batch_time_ms']:.4f} ms")
    print(f"Max throughput:             {single_req_result['throughput_req_per_sec']:.2f} requests/second")
    
    # Real-world context
    print("\n" + "="*60)
    print("REAL-WORLD CONTEXT")
    print("="*60 + "\n")
    
    avg_ms = single_req_result['avg_time_per_request_ms']
    
    print(f"Per 1,000 requests:   {avg_ms * 1000 / 1000:.2f} sec")
    print(f"Per 10,000 requests:  {avg_ms * 10000 / 1000:.2f} sec")
    print(f"Per 100,000 requests: {avg_ms * 100000 / 1000:.2f} sec ({avg_ms * 100000 / 60000:.2f} min)")
    
    # Latency classification
    print("\nLatency Assessment:")
    if avg_ms < 1:
        print("  ⚡ EXCELLENT - Sub-millisecond latency (real-time capable)")
        print("     Perfect for inline IPS deployment!")
    elif avg_ms < 5:
        print("  ✅ VERY GOOD - Low latency (suitable for inline IPS)")
        print("     Ready for production real-time blocking!")
    elif avg_ms < 10:
        print("  ✅ GOOD - Acceptable latency (suitable for most IPS scenarios)")
        print("     Acceptable for real-time IPS with moderate traffic")
    elif avg_ms < 20:
        print("  ⚠️  MODERATE - Borderline for real-time IPS")
        print("     Consider further optimization or hybrid approach")
    else:
        print("  ❌ HIGH - Too slow for real-time IPS")
        print("     Use for batch/log analysis only")
    
    # Speed improvement vs original
    print("\n" + "="*60)
    print("OPTIMIZATION IMPACT")
    print("="*60 + "\n")
    print("Previous (AE + PCA + Mahalanobis): ~55.94 ms/request")
    print(f"Current  (AE only):                 ~{avg_ms:.2f} ms/request")
    
    if avg_ms < 55.94:
        speedup = 55.94 / avg_ms
        print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster!")
        print(f"   Latency reduced by {((55.94 - avg_ms) / 55.94 * 100):.1f}%")
    
    return results_table


def test_csv_file(detector, csv_path):
    """
    Test the detector with a CSV file
    """
    print("\n" + "="*80)
    print(f"TESTING CSV FILE: {csv_path}")
    print("="*80 + "\n")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from CSV")
        
        # Predict
        start = time.time()
        results = detector.predict(df)
        elapsed = time.time() - start
        
        # Summary
        n_anomalies = results['is_anomaly'].sum()
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {len(df)}")
        print(f"Processing time: {elapsed:.4f}s ({elapsed/len(df)*1000:.4f}ms per request)")
        print(f"Anomalies detected: {n_anomalies} ({n_anomalies/len(df)*100:.2f}%)")
        print(f"Normal samples: {len(df) - n_anomalies} ({(len(df)-n_anomalies)/len(df)*100:.2f}%)")
        print(f"\nAnomaly score stats:")
        print(f"  Mean: {results['anomaly_scores'].mean():.4f}")
        print(f"  Std:  {results['anomaly_scores'].std():.4f}")
        print(f"  Min:  {results['anomaly_scores'].min():.4f}")
        print(f"  Max:  {results['anomaly_scores'].max():.4f}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Failed to process CSV: {e}")
        return None


def main():
    """
    Main execution function
    """
    print("\n" + "="*80)
    print("OPTIMIZED WEB PAYLOAD ANOMALY DETECTOR (AE-ONLY)")
    print("="*80 + "\n")
    
    # Set your models directory path
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    MODELS_DIR = PROJECT_ROOT / "models" / "anomaly_detector" / "application_level_attacks"
    
    # Alternative: Use absolute path
    # MODELS_DIR = Path(r"D:\Final-Year-Research\AIM-IPS\models\anomaly_detector\application_level_attacks")
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Models directory: {MODELS_DIR}")
    
    if not MODELS_DIR.exists():
        print(f"\n[ERROR] Models directory not found: {MODELS_DIR}")
        print("Please update the MODELS_DIR path in the script.")
        return
    
    # Initialize optimized detector
    try:
        detector = FastWebPayloadDetector(MODELS_DIR)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize detector: {e}")
        return
    
    # Test with sample payloads
    test_sample_payloads(detector)
    
    # Performance benchmark
    benchmark_performance(detector, n_requests=1000, batch_sizes=[1, 10, 50, 100])
    
    # Test with CSV if available
    csv_path = PROJECT_ROOT / "data_collector" / "data_sets" / "web_features_test.csv"
    if csv_path.exists():
        test_csv_file(detector, csv_path)
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()