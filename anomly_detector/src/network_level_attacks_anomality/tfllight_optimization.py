"""
TensorFlow Lite Optimization for Network-Level Anomaly Detection
Converts Keras autoencoder to TFLite for 10-20x faster inference

Usage:
    python tflite_optimization_network.py

Expected improvement: 56ms -> 3-5ms per flow
"""

import tensorflow as tf
import numpy as np
import joblib
from pathlib import Path
import time

def convert_to_tflite(keras_model_path, output_path, optimize=True):
    """
    Convert Keras model to TensorFlow Lite
    
    Args:
        keras_model_path: Path to .keras model file
        output_path: Path to save .tflite model
        optimize: Apply optimizations (quantization)
    
    Returns:
        Path to TFLite model
    """
    print(f"\n[TFLite] Converting {keras_model_path} to TFLite...")
    
    # Load Keras model
    model = tf.keras.models.load_model(keras_model_path)
    print(f"[TFLite] Loaded model with input shape: {model.input_shape}")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if optimize:
        # Apply optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("[TFLite] Applying optimizations (quantization)...")
    
    tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"[TFLite] ✓ Saved TFLite model -> {output_path}")
    print(f"[TFLite]   Size: {len(tflite_model) / 1024:.2f} KB")
    
    return output_path


class TFLiteFastNetworkDetector:
    """
    Ultra-fast network detector using TensorFlow Lite
    
    Expected performance:
    - Keras model: 56ms per flow
    - TFLite model: 3-5ms per flow (10-15x faster!)
    """
    
    def __init__(self, tflite_model_path, scaler_path, features_path, 
                 pca_path, maha_mean_path, maha_inv_cov_path):
        """
        Load TFLite model and preprocessing components
        """
        print("\n[TFLite] Loading optimized TFLite network detector...")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"✓ Loaded TFLite model")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        
        # Load preprocessing
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)
        self.pca = joblib.load(pca_path)
        self.maha_mean = joblib.load(maha_mean_path)
        self.maha_inv_cov = joblib.load(maha_inv_cov_path)
        
        print(f"✓ Loaded preprocessing components ({len(self.features)} features)")
        
        print("\n⚡ TFLITE NETWORK DETECTOR READY")
        print("   Expected latency: 3-5ms per flow")
        print("   Speedup vs Keras: 10-15x faster\n")
    
    def predict_single(self, flow_features):
        """
        Fast single flow prediction using TFLite
        
        Args:
            flow_features: Dictionary of network flow features
        
        Returns:
            Reconstruction error
        """
        # Prepare input
        X = np.array([[flow_features.get(f, 0.0) for f in self.features]], dtype=np.float32)
        
        # Clean data
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -1e12, 1e12)
        
        X_scaled = self.scaler.transform(X)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], X_scaled.astype(np.float32))
        
        # Run inference (FAST!)
        self.interpreter.invoke()
        
        # Get output (reconstruction)
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Calculate reconstruction error
        recon_err = np.mean(np.power(X_scaled - output, 2))
        
        return float(recon_err)


def benchmark_tflite_vs_keras(keras_model_path, tflite_model_path, models_dir, n_samples=100):
    """
    Compare TFLite vs Keras inference speed for network flows
    """
    print("\n" + "="*70)
    print("TFLITE VS KERAS PERFORMANCE COMPARISON (NETWORK FLOWS)")
    print("="*70)
    
    # Load Keras model
    print("\n[1] Loading Keras model...")
    keras_model = tf.keras.models.load_model(keras_model_path)
    
    # Load TFLite model
    print("[2] Loading TFLite model...")
    models_dir = Path(models_dir)
    tflite_detector = TFLiteFastNetworkDetector(
        tflite_model_path,
        models_dir / "scaler.pkl",
        models_dir / "fusion_features.pkl",
        models_dir / "fusion_pca.pkl",
        models_dir / "fusion_maha_mean.pkl",
        models_dir / "fusion_maha_inv_cov.pkl"
    )
    
    # Load preprocessing
    scaler = joblib.load(models_dir / "scaler.pkl")
    features = joblib.load(models_dir / "fusion_features.pkl")
    
    # Generate test data
    print(f"\n[3] Generating {n_samples} test flows...")
    np.random.seed(42)
    X_test = np.random.randn(n_samples, len(features)).astype(np.float32)
    X_test = np.clip(X_test, -10, 10)  # Keep reasonable range
    X_test_scaled = scaler.transform(X_test)
    
    # Benchmark Keras
    print("\n" + "="*70)
    print("KERAS MODEL (Original)")
    print("="*70)
    
    # Warmup
    _ = keras_model.predict(X_test_scaled[:10], verbose=0)
    
    keras_times = []
    print("Running Keras predictions...")
    for i in range(100):
        start = time.time()
        _ = keras_model.predict(X_test_scaled[i:i+1], verbose=0)
        keras_times.append(time.time() - start)
    
    keras_avg = np.mean(keras_times) * 1000
    print(f"✓ Keras average latency: {keras_avg:.4f}ms per flow")
    
    # Benchmark TFLite
    print("\n" + "="*70)
    print("TFLITE MODEL (Optimized)")
    print("="*70)
    
    tflite_times = []
    print("Running TFLite predictions...")
    for i in range(100):
        flow_features = {features[j]: float(X_test[i, j]) for j in range(len(features))}
        
        start = time.time()
        _ = tflite_detector.predict_single(flow_features)
        tflite_times.append(time.time() - start)
    
    tflite_avg = np.mean(tflite_times) * 1000
    print(f"✓ TFLite average latency: {tflite_avg:.4f}ms per flow")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"Keras latency:  {keras_avg:.4f}ms")
    print(f"TFLite latency: {tflite_avg:.4f}ms")
    print(f"\n🚀 SPEEDUP: {keras_avg / tflite_avg:.2f}x faster!")
    print(f"   Latency reduced by {((keras_avg - tflite_avg) / keras_avg * 100):.1f}%")
    
    if tflite_avg < 5:
        print("\n✅ READY FOR REAL-TIME NETWORK IPS!")
    elif tflite_avg < 10:
        print("\n✅ GOOD for real-time IPS with moderate traffic")
    else:
        print("\n⚠️  Still needs more optimization")
    
    return {
        'keras_avg_ms': keras_avg,
        'tflite_avg_ms': tflite_avg,
        'speedup': keras_avg / tflite_avg
    }


if __name__ == "__main__":
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    MODELS_DIR = PROJECT_ROOT / "models" / "anomaly_detector" / "network_level_anomality"
    
    # Alternative: Use absolute path
    # MODELS_DIR = Path(r"D:\Final-Year-Research\AIM-IPS\models\anomaly_detector\network_level_anomality")
    
    keras_model_path = MODELS_DIR / "fusion_ae.keras"
    tflite_model_path = MODELS_DIR / "fusion_ae.tflite"
    
    print("="*70)
    print("NETWORK-LEVEL ANOMALY DETECTION - TFLITE OPTIMIZATION")
    print("="*70)
    
    # Check if Keras model exists
    if not keras_model_path.exists():
        print(f"\n[ERROR] Keras model not found: {keras_model_path}")
        print("Please train the model first:")
        print("  python -m anomly_detector.src.network_level_attacks_anomality.train")
        exit(1)
    
    # Step 1: Convert Keras to TFLite
    print("\n" + "="*70)
    print("STEP 1: CONVERT KERAS MODEL TO TFLITE")
    print("="*70)
    
    if not tflite_model_path.exists():
        convert_to_tflite(
            keras_model_path,
            tflite_model_path,
            optimize=True
        )
    else:
        print(f"\n[TFLite] Model already exists: {tflite_model_path}")
        print("[TFLite] Skipping conversion (delete file to reconvert)")
    
    # Step 2: Benchmark
    print("\n" + "="*70)
    print("STEP 2: BENCHMARK PERFORMANCE")
    print("="*70)
    
    results = benchmark_tflite_vs_keras(
        keras_model_path,
        tflite_model_path,
        MODELS_DIR,
        n_samples=100
    )
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    print(f"\nTo use TFLite model in production:")
    print(f"1. Load: TFLiteFastNetworkDetector('{tflite_model_path}', ...)")
    print(f"2. Predict: detector.predict_single(flow_features)")
    print(f"3. Expected latency: {results['tflite_avg_ms']:.2f}ms per flow")
    print(f"4. Speedup achieved: {results['speedup']:.2f}x faster")
    print("\n" + "="*70)