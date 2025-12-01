import joblib
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def save_scaler(scaler,path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def save_isof(iso,path):
    joblib.dump(iso, path)

def load_isof(path):
    return joblib.load(path)

def save_ae(model, path):
    # path should include .keras or .h5 or be a directory for TF SavedModel
    model.save(path, include_optimizer=False)


def load_ae(path):
    return tf.keras.models.load_model(path)

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


