"""
TensorFlow Lite Optimization for Real-Time IPS
Converts Keras models to TFLite for 10-20x faster inference

Expected improvement: 52ms -> 2-5ms per request
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
        
        # Optional: Use float16 quantization for even more speed
        # converter.target_spec.supported_types = [tf.float16]
        
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


class TFLiteFastDetector:
    """
    Ultra-fast detector using TensorFlow Lite
    
    Expected performance:
    - Keras model: 52ms per request
    - TFLite model: 2-5ms per request (10-20x faster!)
    """
    
    def __init__(self, tflite_model_path, scaler_path, features_path, 
                 pca_path=None, maha_mean_path=None, maha_inv_cov_path=None):
        """
        Load TFLite model and preprocessing components
        """
        print("\n[TFLite] Loading optimized TFLite model...")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"✓ Loaded TFLite model")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Input dtype: {self.input_details[0]['dtype']}")
        
        # Load preprocessing
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)
        print("✓ Loaded scaler and features")
        
        # Load PCA/Mahalanobis (optional for fusion)
        self.use_fusion = False
        if pca_path and maha_mean_path and maha_inv_cov_path:
            self.pca = joblib.load(pca_path)
            self.maha_mean = joblib.load(maha_mean_path)
            self.maha_inv_cov = joblib.load(maha_inv_cov_path)
            self.use_fusion = True
            print("✓ Loaded PCA + Mahalanobis (fusion enabled)")
        
        print("\n⚡ TFLITE DETECTOR READY")
        print("   Expected latency: 2-5ms per request")
        print("   Speedup vs Keras: 10-20x faster\n")
    
    def predict_single(self, features_dict):
        """
        Fast single prediction using TFLite
        
        Args:
            features_dict: Dictionary of extracted features
        
        Returns:
            CNN score (0-1)
        """
        # Prepare input
        X = np.array([[features_dict[f] for f in self.features]], dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], X_scaled)
        
        # Run inference (THIS IS FAST!)
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return float(output[0][0])
    
    def predict_batch(self, features_df):
        """
        Batch prediction (still uses TFLite for speed)
        """
        # Prepare data
        X = features_df[self.features].values.astype(np.float32)
        X_scaled = self.scaler.transform(X)
        
        # Predict in batches
        scores = []
        for row in X_scaled:
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                row.reshape(1, -1)
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            scores.append(output[0][0])
        
        return np.array(scores)


def benchmark_tflite_vs_keras(keras_model_path, tflite_model_path, scaler_path, 
                               features_path, n_samples=1000):
    """
    Compare TFLite vs Keras inference speed
    """
    print("\n" + "="*70)
    print("TFLITE VS KERAS PERFORMANCE COMPARISON")
    print("="*70)
    
    # Load both models
    print("\n[1] Loading Keras model...")
    keras_model = tf.keras.models.load_model(keras_model_path)
    
    print("[2] Loading TFLite model...")
    tflite_detector = TFLiteFastDetector(
        tflite_model_path, scaler_path, features_path
    )
    
    # Load preprocessing
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    
    # Generate test data
    print(f"\n[3] Generating {n_samples} test samples...")
    np.random.seed(42)
    X_test = np.random.randn(n_samples, len(features)).astype(np.float32)
    X_test_scaled = scaler.transform(X_test)
    
    # Benchmark Keras
    print("\n" + "="*70)
    print("KERAS MODEL (Original)")
    print("="*70)
    
    # Warmup
    _ = keras_model.predict(X_test_scaled[:10], verbose=0)
    
    keras_times = []
    print("Running Keras predictions...")
    for i in range(100):  # Test 100 samples
        start = time.time()
        _ = keras_model.predict(X_test_scaled[i:i+1], verbose=0)
        keras_times.append(time.time() - start)
    
    keras_avg = np.mean(keras_times) * 1000  # Convert to ms
    print(f"✓ Keras average latency: {keras_avg:.4f}ms per request")
    
    # Benchmark TFLite
    print("\n" + "="*70)
    print("TFLITE MODEL (Optimized)")
    print("="*70)
    
    tflite_times = []
    print("Running TFLite predictions...")
    for i in range(100):
        features_dict = {features[j]: X_test[i, j] for j in range(len(features))}
        
        start = time.time()
        _ = tflite_detector.predict_single(features_dict)
        tflite_times.append(time.time() - start)
    
    tflite_avg = np.mean(tflite_times) * 1000
    print(f"✓ TFLite average latency: {tflite_avg:.4f}ms per request")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"Keras latency:  {keras_avg:.4f}ms")
    print(f"TFLite latency: {tflite_avg:.4f}ms")
    print(f"\n🚀 SPEEDUP: {keras_avg / tflite_avg:.2f}x faster!")
    print(f"   Latency reduced by {((keras_avg - tflite_avg) / keras_avg * 100):.1f}%")
    
    if tflite_avg < 5:
        print("\n✅ READY FOR REAL-TIME IPS!")
    elif tflite_avg < 10:
        print("\n✅ GOOD for real-time IPS with moderate traffic")
    else:
        print("\n⚠️  Still need more optimization")
    
    return {
        'keras_avg_ms': keras_avg,
        'tflite_avg_ms': tflite_avg,
        'speedup': keras_avg / tflite_avg
    }


if __name__ == "__main__":
    # Paths (adjust to your project)
    MODELS_DIR = Path("models/anomaly_detector/application_level_attacks")
    
    keras_model_path = MODELS_DIR / "fusion_cnn.keras"
    tflite_model_path = MODELS_DIR / "fusion_cnn.tflite"
    scaler_path = MODELS_DIR / "scaler.pkl"
    features_path = MODELS_DIR / "fusion_features.pkl"
    
    # Step 1: Convert Keras to TFLite
    print("="*70)
    print("STEP 1: CONVERT KERAS MODEL TO TFLITE")
    print("="*70)
    
    if not tflite_model_path.exists():
        convert_to_tflite(
            keras_model_path,
            tflite_model_path,
            optimize=True
        )
    else:
        print(f"[TFLite] Model already exists: {tflite_model_path}")
    
    # Step 2: Benchmark
    print("\n" + "="*70)
    print("STEP 2: BENCHMARK PERFORMANCE")
    print("="*70)
    
    results = benchmark_tflite_vs_keras(
        keras_model_path,
        tflite_model_path,
        scaler_path,
        features_path,
        n_samples=1000
    )
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    print(f"\nTo use TFLite model in production:")
    print(f"1. Load: TFLiteFastDetector('{tflite_model_path}', ...)")
    print(f"2. Predict: detector.predict_single(features_dict)")
    print(f"3. Expected latency: {results['tflite_avg_ms']:.2f}ms")