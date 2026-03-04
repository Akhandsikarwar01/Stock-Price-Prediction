import numpy as np
import pandas as pd

def create_sequences(data: pd.DataFrame, seq_length: int = 60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data.iloc[i-seq_length:i].values)
        y.append(data.iloc[i]['Close'])
    return np.array(X), np.array(y)
