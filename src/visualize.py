import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(10,6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.title('Actual vs Predicted Stock Prices')
    plt.show()
