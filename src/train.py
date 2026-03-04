import pandas as pd
from data_fetcher import download_stock_data
from data_preprocessing import preprocess_data
from feature_engineering import create_sequences
from model import build_lstm_model
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

TICKER = 'AAPL'
SEQ_LEN = 60
EPOCHS = 20
BATCH_SIZE = 32

# 1. Download data
df = download_stock_data(TICKER)

# 2. Preprocess
data, scaler = preprocess_data(df)

# 3. Feature engineering
X, y = create_sequences(data, SEQ_LEN)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. Build model
model = build_lstm_model((SEQ_LEN, X.shape[2]))

# 6. Train
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# 7. Save model and scaler
model.save('../models/lstm_model.h5')
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
