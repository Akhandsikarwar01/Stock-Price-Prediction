import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df: pd.DataFrame) -> (pd.DataFrame, MinMaxScaler):
    """Handle missing values, select features, and scale data."""
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.ffill(inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns), scaler
