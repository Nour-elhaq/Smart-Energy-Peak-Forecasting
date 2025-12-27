
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_metrics(y_true, y_pred, model_name="Model"):
    """
    Computes MAE, RMSE, and Peak MAE (top 10% highest actuals).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Peak MAE (Top 10% of actuals)
    threshold = np.percentile(y_true, 90)
    mask_top = y_true >= threshold
    if np.sum(mask_top) > 0:
        peak_mae = mean_absolute_error(y_true[mask_top], y_pred[mask_top])
    else:
        peak_mae = np.nan
        
    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "Peak_MAE": peak_mae
    }

def plot_actual_vs_pred(df, title, out_path):
    """
    Plot actual vs predicted for a subset or aggregated.
    Expects df to have 'Date', 'y', 'y_pred'.
    Aggregates by Date for clearer visualization if many feeders.
    """
    daily_agg = df.groupby('Date')[['y', 'y_pred']].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_agg['Date'], daily_agg['y'], label='Actual (Sum)', alpha=0.7)
    plt.plot(daily_agg['Date'], daily_agg['y_pred'], label='Predicted (Sum)', alpha=0.7, linestyle='--')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_error_dist(y_true, y_pred, title, out_path):
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=50)
    plt.title(f"{title} - Error Distribution")
    plt.xlabel("Error (Actual - Pred)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_scatter(y_true, y_pred, title, out_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.1, s=10)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f"{title} - Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
