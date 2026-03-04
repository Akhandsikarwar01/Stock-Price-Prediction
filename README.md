# 📈 Stock Price Prediction System

A complete end-to-end Machine Learning project that predicts future stock prices using **LSTM (Long Short-Term Memory) Neural Networks**. The system fetches real-time historical stock data from Yahoo Finance, trains a deep learning model, and provides an interactive **Streamlit web application** for visualization and prediction.

## 🚀 Features
- Real-time Data Fetching (Yahoo Finance)
- Data Preprocessing (missing values, MinMaxScaler, feature selection)
- LSTM Deep Learning Model for time-series forecasting
- Model Evaluation (MSE, RMSE, MAE)
- Interactive Visualization (Actual vs Predicted)
- Streamlit Web App for user interaction
- Model Persistence (Keras + Pickle)

## 📁 Project Structure
```
Stock-Price-Prediction/
├── README.md
├── requirements.txt
├── src/
│   ├── data_fetcher.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── visualize.py
├── app/
│   └── streamlit_app.py
├── data/
├── models/
├── notebooks/
```

## 🛠️ Tech Stack
- Python 3.9+
- pandas, numpy, matplotlib, scikit-learn, tensorflow, yfinance
- Streamlit

## ⚡ Quick Start
1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
2. Train the model:
	```bash
	python src/train.py
	```
3. Run the Streamlit app:
	```bash
	streamlit run app/streamlit_app.py
	```

## 📚 License
MIT

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Akhandsikarwar01/Stock-Price-Prediction-System.git
cd Stock-Price-Prediction-System
