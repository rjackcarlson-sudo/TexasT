import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Compute standard regression + directional metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'R2': r2_score(y_true, y_pred),
        'Directional_Accuracy_%': np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
    }
    return pd.Series(metrics)

def walk_forward_validation(model, X, y, n_test=30):
    """Walk-forward validation for realistic time-series evaluation."""
    predictions = []
    actuals = []
    n_records = len(X)
    start_idx = n_records - (5 * n_test)  
    
    for i in range(start_idx, n_records - n_test + 1):
        X_train = X[:i]
        y_train = y[:i]
        X_test = X[i:i + n_test]
        y_test = y[i:i + n_test]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        predictions.extend(y_pred)
        actuals.extend(y_test)
    
    metrics = calculate_metrics(actuals, predictions)
    results_df = pd.DataFrame({'actual': actuals, 'predicted': predictions})
    
    return metrics, results_df

def plot_actual_vs_pred(results_df, title="WTI Next-Day Price Prediction"):
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['actual'], label='Actual', alpha=0.8)
    plt.plot(results_df['predicted'], label='Predicted', alpha=0.8)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("WTI Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Residuals plot
    plt.figure(figsize=(12, 4))
    residuals = results_df['actual'] - results_df['predicted']
    plt.plot(residuals, label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Prediction Residuals")
    plt.legend()
    plt.grid(True)
    plt.show()
