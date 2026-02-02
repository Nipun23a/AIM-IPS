import tensorflow as tf
import tf2onnx
from pathlib import Path

keras_model_path = r"models\anomaly_detector\application_level_attacks\fusion_ae.keras"
onnx_model_path  = r"models\anomaly_detector\application_level_attacks\fusion_ae.onnx"

# Load Keras model
model = tf.keras.models.load_model(keras_model_path)

# Build input signature
input_dim = model.input_shape[1]
spec = (tf.TensorSpec((None, input_dim), tf.float32, name="input"),)

# Convert to ONNX
tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    output_path=onnx_model_path,
    opset=13
)

print("✅ ONNX model saved to:", onnx_model_path)
