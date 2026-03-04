import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

def predict_next_day(input_sequence, model_path, scaler_path):
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    scaled_seq = scaler.transform(input_sequence)
    X = np.expand_dims(scaled_seq, axis=0)
    pred = model.predict(X)
    pred_price = scaler.inverse_transform(np.concatenate([np.zeros((1,4)), pred], axis=1))[0, -1]
    return pred_price
