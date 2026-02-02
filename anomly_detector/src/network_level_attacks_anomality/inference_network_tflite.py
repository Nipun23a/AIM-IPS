"""
Network-Level Anomaly Detection Inference with TFLite Optimization
For CICIDS2017 dataset with TFLite Autoencoder + PCA/Mahalanobis + Fusion Gate

Usage:
    python -m anomly_detector.src.network_level_attacks_anomality.inference_tflite

Features:
- 10-15x faster than Keras (3-5ms vs 56ms per flow)
- Handles infinity/NaN in data automatically
- Full fusion gate with PCA/Mahalanobis
- Comprehensive benchmarking and CSV testing
"""

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
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


class TFLiteNetworkAnomalyDetector:
    """
    TFLite-optimized network-level anomaly detector
    
    Performance:
    - Keras autoencoder: 56ms per flow
    - TFLite autoencoder: 3-5ms per flow (10-15x faster!)
    
    Architecture:
    - TFLite autoencoder for reconstruction-based detection
    - PCA + Mahalanobis for statistical analysis
    - Fusion gate combining both approaches
    """
    
    def __init__(self, models_dir, model_prefix="fusion", use_tflite=True):
        """
        Load TFLite or Keras models
        
        Args:
            models_dir: Path to models directory
            model_prefix: Model filename prefix (default: "fusion")
            use_tflite: Use TFLite if available, fallback to Keras
        """
        self.models_dir = Path(models_dir)
        self.use_tflite = use_tflite
        
        print(f"[INFO] Loading network anomaly models from: {self.models_dir}")
        
        try:
            # Load scaler
            self.scaler = joblib.load(self.models_dir / "scaler.pkl")
            print("✓ Loaded scaler")
            
            # Try to load TFLite model first
            tflite_path = self.models_dir / f"{model_prefix}_ae.tflite"
            keras_path = self.models_dir / f"{model_prefix}_ae.keras"
            
            if use_tflite and tflite_path.exists():
                # Load TFLite model
                self.interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                self.interpreter.allocate_tensors()
                
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                self.model_type = 'tflite'
                print(f"✓ Loaded TFLite autoencoder (OPTIMIZED)")
                print(f"  Input shape: {self.input_details[0]['shape']}")
                
            elif keras_path.exists():
                # Fallback to Keras
                from tensorflow.keras.models import load_model
                self.ae = load_model(keras_path)
                self.model_type = 'keras'
                print("✓ Loaded Keras autoencoder")
                print("  ⚠️  WARNING: Using Keras (slower)")
                print("  💡 TIP: Run tflite_optimization_network.py for 10-15x speedup")
                
            else:
                raise FileNotFoundError(
                    f"No model found. Expected:\n"
                    f"  - {tflite_path} (TFLite, fast)\n"
                    f"  - {keras_path} (Keras, slower)"
                )
            
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
            
            print(f"\n⚡ NETWORK ANOMALY DETECTOR READY")
            print(f"   Model type: {self.model_type.upper()}")
            print(f"   Feature count: {len(self.features)}")
            print(f"   Expected latency: {'3-5ms' if self.model_type == 'tflite' else '50-60ms'}")
            print(f"   Detection method: Fusion Gate (AE + Statistical)\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            raise
    
    def _predict_ae_tflite(self, X_scaled):
        """
        Fast TFLite autoencoder prediction
        
        Args:
            X_scaled: Scaled features array
        
        Returns:
            Reconstruction errors
        """
        recon_errors = []
        
        for row in X_scaled:
            # Set input
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                row.reshape(1, -1).astype(np.float32)
            )
            
            # Run inference (FAST!)
            self.interpreter.invoke()
            
            # Get output (reconstruction)
            reconstructed = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Calculate error
            error = np.mean(np.power(row.reshape(1, -1) - reconstructed, 2))
            recon_errors.append(error)
        
        return np.array(recon_errors)
    
    def _predict_ae_keras(self, X_scaled):
        """
        Keras autoencoder prediction (fallback)
        """
        reconstructed = self.ae.predict(X_scaled, verbose=0)
        recon_err = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        return recon_err
    
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
            'detection_method': 'fusion',
            'model_type': self.model_type
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
        
        # Autoencoder reconstruction error (TFLite or Keras)
        if self.model_type == 'tflite':
            recon_err = self._predict_ae_tflite(X_scaled)
        else:
            recon_err = self._predict_ae_keras(X_scaled)
        
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
    """
    print("\n" + "="*80)
    print("TESTING SAMPLE NETWORK FLOWS")
    print("="*80 + "\n")
    
    print("⚠️  Note: Using zero-filled features for demonstration")
    print("   In production, use actual network flow features from CICIDS2017\n")
    
    # Create sample flows with zero features
    sample_flows = []
    
    for i in range(5):
        flow = {feat: 0.0 for feat in detector.features}
        # Simulate some variation
        if len(detector.features) > 0:
            flow[detector.features[0]] = float(i * 100)
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
    print(f"Model: {detector.model_type.upper()}")
    print("="*80 + "\n")
    
    # Generate test flows
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
        if detector.model_type == 'keras':
            print("     STRONGLY RECOMMEND: Convert to TFLite (10-20x speedup)")
            print("     Run: python tflite_optimization_network.py")
    
    # Show speedup if TFLite
    if detector.model_type == 'tflite':
        keras_baseline = 56.0  # Known Keras baseline
        print("\n" + "="*60)
        print("OPTIMIZATION IMPACT")
        print("="*60)
        print(f"Previous (Keras):  ~{keras_baseline:.2f} ms/flow")
        print(f"Current  (TFLite): ~{avg_ms:.2f} ms/flow")
        
        if avg_ms < keras_baseline:
            speedup = keras_baseline / avg_ms
            print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster!")
            print(f"   Latency reduced by {((keras_baseline - avg_ms) / keras_baseline * 100):.1f}%")
    
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
            
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
            
            acc = accuracy_score(y_true, y_pred)
            prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            
            try:
                auc = roc_auc_score(y_true, results['fusion_scores'])
            except:
                auc = None
            
            print(f"\n{'='*60}")
            print("MODEL PERFORMANCE (on labeled flows)")
            print(f"{'='*60}")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            if auc:
                print(f"AUC-ROC:   {auc:.4f}")
            
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"                Normal  Attack")
            print(f"Actual Normal   {cm[0][0]:<7} {cm[0][1]:<7}")
            print(f"       Attack   {cm[1][0]:<7} {cm[1][1]:<7}")
            
            # Calculate rates
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            print(f"\nError Rates:")
            print(f"False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
            print(f"False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")
        
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
    print("NETWORK-LEVEL ANOMALY DETECTOR - TFLITE OPTIMIZED INFERENCE")
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
    
    # Initialize detector (will auto-detect TFLite or fallback to Keras)
    try:
        detector = TFLiteNetworkAnomalyDetector(MODELS_DIR, use_tflite=True)
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
    print("💡 RECOMMENDATIONS:")
    print("-" * 80)
    
    if detector.model_type == 'keras':
        print("⚠️  You're using Keras (slower)")
        print("\nTO GET 10-15x SPEEDUP:")
        print("1. Run: python tflite_optimization_network.py")
        print("2. This will convert your model to TFLite")
        print("3. Then run this script again")
        print("4. Expected improvement: 56ms → 3-5ms per flow")
    else:
        print("✅ You're using TFLite (optimized)")
        print("\nFURTHER OPTIMIZATIONS:")
        print("1. Use batched inference for async processing")
        print("   → Collect 50-100 flows and predict in batch")
        print("\n2. Monitor false positives in production")
        print("   → Adjust fusion weights if needed (w_ae, w_stat)")
        print("\n3. For even more speed: Use ONNX Runtime")
        print("   → Can be 1-2x faster than TFLite on some CPUs")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()