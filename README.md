# 📈 Stock Price Prediction System

A complete end-to-end Machine Learning project that predicts future stock prices using **LSTM (Long Short-Term Memory) Neural Networks**. The system fetches real-time historical stock data from Yahoo Finance, trains a deep learning model, and provides an interactive **Streamlit web application** for visualization and prediction.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Features

- **Real-time Data Fetching** — Downloads historical stock data using Yahoo Finance API
- **Data Preprocessing** — Handles missing values, feature scaling with MinMaxScaler
- **LSTM Deep Learning Model** — Multi-layer LSTM architecture for time-series forecasting
- **Model Evaluation** — Performance metrics including MSE, RMSE, and MAE
- **Interactive Visualization** — Actual vs Predicted price charts using Matplotlib
- **Streamlit Web App** — User-friendly interface to select tickers and view predictions
- **Model Persistence** — Trained models saved for reuse without retraining

---

## 📁 Project Structure
Stock-Price-Prediction-System/ │ ├── README.md # Project documentation ├── requirements.txt # Python dependencies ├── LICENSE # MIT License │ ├── src/ │ ├── init.py # Package initializer │ ├── data_fetcher.py # Download stock data from Yahoo Finance │ ├── data_preprocessing.py # Clean, scale, and prepare data │ ├── feature_engineering.py # Create time-series sequences │ ├── model.py # LSTM model architecture │ ├── train.py # Model training pipeline │ ├── evaluate.py # Evaluation metrics and reporting │ ├── predict.py # Make predictions on new data │ └── visualize.py # Plotting and visualization utilities │ ├── app/ │ └── streamlit_app.py # Streamlit web application │ ├── notebooks/ │ └── exploration.ipynb # Jupyter notebook for EDA │ ├── models/ # Saved trained models │ └── .gitkeep │ ├── data/ # Cached datasets │ └── .gitkeep │ └── images/ # Screenshots and plots └── .gitkeep


---

## 🛠️ Tech Stack

| Category        | Technology                          |
|-----------------|-------------------------------------|
| Language        | Python 3.9+                         |
| Deep Learning   | TensorFlow / Keras                  |
| ML Utilities    | scikit-learn, NumPy, Pandas         |
| Data Source     | Yahoo Finance (yfinance)            |
| Visualization   | Matplotlib, Plotly                  |
| Web App         | Streamlit                           |
| Model Saving    | Keras `.h5` + Pickle (scaler)       |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Akhandsikarwar01/Stock-Price-Prediction-System.git
cd Stock-Price-Prediction-System
