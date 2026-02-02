"""
Network-Level Anomaly Detection Inference
For CICIDS2017 dataset with Autoencoder + PCA/Mahalanobis + Fusion Gate

Usage:
    python -m anomly_detector.src.network_level_attacks_anomality.inference

Expected latency: 
- Autoencoder: ~50ms (needs TFLite optimization)
- Feature extraction: minimal (features already in network flow)
"""

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from pathlib import Path
import sys
import time
from statistics import mean, median, stdev


def mahalanobis_scores(X_scaled, pca, mean_vec, inv_cov):
    """Calculate Mahalanobis distance scores"""
    X_pca = pca.transform(X_scaled)
    diffs = X_pca - mean_vec
    scores = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)
    return scores


class NetworkAnomalyDetector:
    """
    Network-level anomaly detector using Fusion Gate
    
    Architecture:
    - Autoencoder for reconstruction-based anomaly detection
    - PCA + Mahalanobis for statistical analysis
    - Fusion gate combining both approaches
    
    For: CICIDS2017 network flow features
    """
    
    def __init__(self, models_dir, model_prefix="fusion"):
        """
        Load all trained models
        
        Args:
            models_dir: Path to models directory
            model_prefix: Model filename prefix (default: "fusion")
        """
        self.models_dir = Path(models_dir)
        print(f"[INFO] Loading network anomaly models from: {self.models_dir}")
        
        try:
            # Load scaler
            self.scaler = joblib.load(self.models_dir / "scaler.pkl")
            print("✓ Loaded scaler")
            
            # Load autoencoder
            ae_path = self.models_dir / f"{model_prefix}_ae.keras"
            self.ae = load_model(ae_path)
            print("✓ Loaded autoencoder")
            
            # Load AE threshold
            self.ae_threshold = joblib.load(self.models_dir / f"{model_prefix}_ae_threshold.pkl")
            print(f"✓ Loaded AE threshold: {self.ae_threshold:.6f}")
            
            # Load PCA + Mahalanobis
            self.pca = joblib.load(self.models_dir / f"{model_prefix}_pca.pkl")
            self.maha_mean = joblib.load(self.models_dir / f"{model_prefix}_maha_mean.pkl")
            self.maha_inv_cov = joblib.load(self.models_dir / f"{model_prefix}_maha_inv_cov.pkl")
            print("✓ Loaded PCA + Mahalanobis models")
            
            # Load features
            self.features = joblib.load(self.models_dir / f"{model_prefix}_features.pkl")
            print(f"✓ Loaded features list ({len(self.features)} features)")
            
            print("\n⚡ NETWORK ANOMALY DETECTOR READY")
            print(f"   Feature count: {len(self.features)}")
            print(f"   Detection method: Fusion Gate (AE + Statistical)\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            raise
    
    def predict_single_flow(self, flow_features: dict, w_ae=0.7, w_stat=0.3):
        """
        Predict if a single network flow is anomalous
        
        Args:
            flow_features: Dictionary with network flow features
            w_ae: Weight for autoencoder score (default: 0.7)
            w_stat: Weight for statistical score (default: 0.3)
        
        Returns:
            dict with prediction results
        """
        # Create DataFrame
        df = pd.DataFrame([flow_features])
        
        # Predict
        result = self.predict(df, w_ae=w_ae, w_stat=w_stat)
        
        return {
            'is_anomaly': bool(result['is_anomaly'][0]),
            'fusion_score': float(result['fusion_scores'][0]),
            'ae_score': float(result['ae_scores'][0]),
            'maha_score': float(result['maha_scores'][0]),
            'detection_method': 'fusion'
        }
    
    def predict(self, flows_df, w_ae=0.7, w_stat=0.3):
        """
        Predict anomalies for a DataFrame of network flows
        
        Args:
            flows_df: DataFrame with network flow features
            w_ae: Weight for AE in fusion
            w_stat: Weight for statistical in fusion
            
        Returns:
            dict with predictions and scores
        """
        # Ensure correct feature order and fill missing
        for feat in self.features:
            if feat not in flows_df.columns:
                flows_df[feat] = 0.0
        
        X = flows_df[self.features].copy()
        
        # CRITICAL: Clean data before scaling
        # Replace infinity with NaN, then fill with 0
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)
        
        # Clip extreme values to prevent overflow
        CLIP_MAG = 1e12
        X = X.clip(lower=-CLIP_MAG, upper=CLIP_MAG)
        
        X = X.values.astype(np.float64)
        X_scaled = self.scaler.transform(X)
        
        # Autoencoder reconstruction error
        reconstructed = self.ae.predict(X_scaled, verbose=0)
        recon_err = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        
        # Mahalanobis distance
        maha_scores = mahalanobis_scores(X_scaled, self.pca, self.maha_mean, self.maha_inv_cov)
        
        # Normalize scores
        ae_norm = np.clip(recon_err / (self.ae_threshold + 1e-12), 0.0, 1.0)
        maha_norm = (maha_scores - maha_scores.min()) / (np.ptp(maha_scores) + 1e-12)
        maha_norm = np.clip(maha_norm, 0.0, 1.0)
        
        # Fusion
        fusion_scores = (w_ae * ae_norm) + (w_stat * maha_norm)
        fusion_scores = np.clip(fusion_scores, 0.0, 1.0)
        
        # Predictions
        predictions = (fusion_scores > 0.5).astype(int)
        
        return {
            'is_anomaly': predictions,
            'fusion_scores': fusion_scores,
            'ae_scores': recon_err,
            'maha_scores': maha_scores
        }


def test_sample_flows(detector):
    """
    Test the detector with sample network flows
    Note: These are synthetic examples - real CICIDS2017 flows have many more features
    """
    print("\n" + "="*80)
    print("TESTING SAMPLE NETWORK FLOWS")
    print("="*80 + "\n")
    
    print("⚠️  Note: Using zero-filled features for demonstration")
    print("   In production, use actual network flow features from CICIDS2017\n")
    
    # Create sample flows with zero features (will be filled with actual values in production)
    sample_flows = []
    
    for i in range(5):
        flow = {feat: 0.0 for feat in detector.features}
        # Simulate some variation
        flow[detector.features[0]] = float(i * 100)  # Vary first feature
        sample_flows.append(flow)
    
    df_samples = pd.DataFrame(sample_flows)
    
    # Predict
    results = detector.predict(df_samples)
    
    for i, (is_anom, fusion, ae, maha) in enumerate(zip(
        results['is_anomaly'],
        results['fusion_scores'],
        results['ae_scores'],
        results['maha_scores']
    )):
        status = "🚨 ANOMALY" if is_anom else "✓ NORMAL"
        print(f"{status} | Flow {i+1}")
        print(f"         | Fusion: {fusion:.4f} | AE: {ae:.6f} | Maha: {maha:.4f}")
        print()


def benchmark_performance(detector, n_flows=1000, batch_sizes=[1, 10, 50, 100]):
    """
    Benchmark detector performance on network flows
    """
    print("\n" + "="*80)
    print(f"PERFORMANCE BENCHMARK - {n_flows} NETWORK FLOWS")
    print("="*80 + "\n")
    
    # Generate test flows (zero-filled for benchmark)
    print(f"Generating {n_flows} test flows...")
    flows = []
    for i in range(n_flows):
        flow = {feat: float(np.random.randn()) for feat in detector.features}
        flows.append(flow)
    
    df_all = pd.DataFrame(flows)
    print(f"Generated flows with {len(detector.features)} features\n")
    
    # Test different batch sizes
    print("="*60)
    print("BATCH PROCESSING PERFORMANCE")
    print("="*60 + "\n")
    
    results_table = []
    
    for batch_size in batch_sizes:
        times = []
        n_batches = (len(flows) + batch_size - 1) // batch_size
        
        print(f"Testing batch size: {batch_size} ({n_batches} batches)...")
        
        start_total = time.time()
        
        for i in range(0, len(flows), batch_size):
            batch_df = df_all.iloc[i:i+batch_size]
            
            start_batch = time.time()
            _ = detector.predict(batch_df)
            batch_time = time.time() - start_batch
            
            times.append(batch_time)
        
        total_time = time.time() - start_total
        
        # Calculate statistics
        time_per_flow = (total_time / len(flows)) * 1000  # ms
        throughput = len(flows) / total_time  # flows/sec
        
        results_table.append({
            'batch_size': batch_size,
            'total_time': total_time,
            'avg_time_per_flow_ms': time_per_flow,
            'median_batch_time_ms': median(times) * 1000,
            'std_batch_time_ms': stdev(times) * 1000 if len(times) > 1 else 0,
            'throughput_flows_per_sec': throughput,
            'min_batch_time_ms': min(times) * 1000,
            'max_batch_time_ms': max(times) * 1000,
        })
        
        print(f"  ✓ Completed in {total_time:.4f}s")
        print(f"    Avg time per flow: {time_per_flow:.4f}ms")
        print(f"    Throughput: {throughput:.2f} flows/sec\n")
    
    # Print summary table
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80 + "\n")
    
    print(f"{'Batch Size':<12} {'Avg Time/Flow':<16} {'Throughput':<18} {'Total Time':<12}")
    print(f"{'':12} {'(ms)':<16} {'(flows/sec)':<18} {'(sec)':<12}")
    print("-" * 80)
    
    for r in results_table:
        print(f"{r['batch_size']:<12} {r['avg_time_per_flow_ms']:<16.4f} "
              f"{r['throughput_flows_per_sec']:<18.2f} {r['total_time']:<12.4f}")
    
    # Single flow performance
    print("\n" + "="*80)
    print("SINGLE FLOW PERFORMANCE (CRITICAL FOR REAL-TIME IPS)")
    print("="*80 + "\n")
    
    single_flow_result = [r for r in results_table if r['batch_size'] == 1][0]
    
    print(f"Average latency per flow: {single_flow_result['avg_time_per_flow_ms']:.4f} ms")
    print(f"Median batch time:        {single_flow_result['median_batch_time_ms']:.4f} ms")
    print(f"Std deviation:            {single_flow_result['std_batch_time_ms']:.4f} ms")
    print(f"Min time:                 {single_flow_result['min_batch_time_ms']:.4f} ms")
    print(f"Max time:                 {single_flow_result['max_batch_time_ms']:.4f} ms")
    print(f"Max throughput:           {single_flow_result['throughput_flows_per_sec']:.2f} flows/second")
    
    # Real-world context
    print("\n" + "="*60)
    print("REAL-WORLD CONTEXT")
    print("="*60 + "\n")
    
    avg_ms = single_flow_result['avg_time_per_flow_ms']
    
    print(f"Per 1,000 flows:    {avg_ms * 1000 / 1000:.2f} sec")
    print(f"Per 10,000 flows:   {avg_ms * 10000 / 1000:.2f} sec")
    print(f"Per 100,000 flows:  {avg_ms * 100000 / 1000:.2f} sec ({avg_ms * 100000 / 60000:.2f} min)")
    
    # Latency classification
    print("\nLatency Assessment:")
    if avg_ms < 1:
        print("  ⚡ EXCELLENT - Sub-millisecond latency")
        print("     Perfect for real-time network monitoring!")
    elif avg_ms < 5:
        print("  ✅ VERY GOOD - Low latency")
        print("     Suitable for inline network IPS!")
    elif avg_ms < 10:
        print("  ✅ GOOD - Acceptable latency")
        print("     Acceptable for most IPS scenarios")
    elif avg_ms < 20:
        print("  ⚠️  MODERATE - Borderline for real-time")
        print("     Consider TFLite optimization")
    else:
        print("  ❌ HIGH - Too slow for real-time")
        print("     STRONGLY RECOMMEND: Convert to TFLite (10-20x speedup)")
        print("     Run: python tflite_optimization_network.py")
    
    return results_table


def test_csv_file(detector, csv_path):
    """
    Test the detector with a CSV file of network flows
    """
    print("\n" + "="*80)
    print(f"TESTING CSV FILE: {csv_path}")
    print("="*80 + "\n")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} flows from CSV")
        
        # Check if label column exists
        label_col = None
        for cand in ['Label', 'label', 'Attack', 'attack', 'class']:
            if cand in df.columns:
                label_col = cand
                break
        
        # Predict
        start = time.time()
        results = detector.predict(df)
        elapsed = time.time() - start
        
        # Summary
        n_anomalies = results['is_anomaly'].sum()
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total flows: {len(df)}")
        print(f"Processing time: {elapsed:.4f}s ({elapsed/len(df)*1000:.4f}ms per flow)")
        print(f"Anomalies detected: {n_anomalies} ({n_anomalies/len(df)*100:.2f}%)")
        print(f"Normal flows: {len(df) - n_anomalies} ({(len(df)-n_anomalies)/len(df)*100:.2f}%)")
        
        print(f"\nFusion score stats:")
        print(f"  Mean: {results['fusion_scores'].mean():.4f}")
        print(f"  Std:  {results['fusion_scores'].std():.4f}")
        print(f"  Min:  {results['fusion_scores'].min():.4f}")
        print(f"  Max:  {results['fusion_scores'].max():.4f}")
        
        # If we have labels, calculate accuracy
        if label_col:
            # Map labels to 0/1
            labels = df[label_col].astype(str).str.lower()
            y_true = labels.apply(lambda x: 0 if x in ['benign', 'normal', '0'] else 1).values
            
            y_pred = results['is_anomaly']
            
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
            
            acc = accuracy_score(y_true, y_pred)
            prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            
            print(f"\n{'='*60}")
            print("MODEL PERFORMANCE (on labeled flows)")
            print(f"{'='*60}")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"                Normal  Attack")
            print(f"Actual Normal   {cm[0][0]:<7} {cm[0][1]:<7}")
            print(f"       Attack   {cm[1][0]:<7} {cm[1][1]:<7}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Failed to process CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main execution function
    """
    print("\n" + "="*80)
    print("NETWORK-LEVEL ANOMALY DETECTOR - INFERENCE")
    print("="*80 + "\n")
    
    # Set your models directory path
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    MODELS_DIR = PROJECT_ROOT / "models" / "anomaly_detector" / "network_level_anomality"
    
    # Alternative: Use absolute path
    # MODELS_DIR = Path(r"D:\Final-Year-Research\AIM-IPS\models\anomaly_detector\network_level_anomality")
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Models directory: {MODELS_DIR}")
    
    if not MODELS_DIR.exists():
        print(f"\n[ERROR] Models directory not found: {MODELS_DIR}")
        print("Please update the MODELS_DIR path in the script.")
        return
    
    # Initialize detector
    try:
        detector = NetworkAnomalyDetector(MODELS_DIR)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize detector: {e}")
        return
    
    # Test with sample flows
    test_sample_flows(detector)
    
    # Performance benchmark
    benchmark_performance(detector, n_flows=1000, batch_sizes=[1, 10, 50, 100])
    
    # Test with CSV if available
    DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets"
    test_csv = DATA_DIR / "cicids_test.csv"
    
    if test_csv.exists():
        test_csv_file(detector, test_csv)
    else:
        print(f"\n[INFO] Test CSV not found: {test_csv}")
        print("       Skipping CSV testing")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80 + "\n")
    
    # Recommendations
    print("💡 OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 80)
    print("1. If latency > 10ms: Convert to TFLite for 10-20x speedup")
    print("   → Run: python tflite_optimization_network.py")
    print("\n2. For production deployment: Use batched inference")
    print("   → Collect 50-100 flows and predict in batch")
    print("\n3. For accuracy improvement: Retrain with more diverse data")
    print("   → Include more attack types from CICIDS2017")
    print("\n4. Monitor false positives in production")
    print("   → Adjust fusion weights if needed (w_ae, w_stat)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()