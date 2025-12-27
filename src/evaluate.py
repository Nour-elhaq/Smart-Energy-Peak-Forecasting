
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import joblib
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Model Evaluation")
    parser.add_argument("--data", type=str, default="data/processed_daily.csv", help="Processed daily CSV")
    parser.add_argument("--model", type=str, default="models/xgb_pipeline.joblib", help="Path to joblib model")
    parser.add_argument("--outdir", type=str, default="plots/evaluation", help="Output directory")
    return parser.parse_args()

def plot_actual_vs_pred_scatter(y_true, y_pred, out_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.1, s=10, color='blue')
    
    # Identity line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Peak (kW)")
    plt.ylabel("Predicted Peak (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_residuals(y_true, y_pred, out_path):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.1, s=10, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted Peak (kW)")
    plt.ylabel("Residuals (Actual - Pred)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_residual_hist_qq(y_true, y_pred, out_dir):
    residuals = y_true - y_pred
    
    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title("Residual Distribution")
    plt.xlabel("Residual Error")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/residual_histogram.png")
    plt.close()
    
    # QQ Plot
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/qq_plot.png")
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    print("Loading model and data...")
    pipeline = joblib.load(args.model)
    df = pd.read_csv(args.data)
    
    # Prepare features (same as train.py logic)
    # Ideally we should use a shared function, but for now we duplicate logic for safety
    cat_features = ['SerialNo', 'Substation', 'is_weekend']
    num_features = [c for c in df.columns if c not in ['y', 'Date', 'datetime'] + cat_features]
    
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype(str)
            
    X = df[num_features + cat_features]
    y_true = df['y']
    
    print("Generating predictions...")
    y_pred = pipeline.predict(X)
    
    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"Evaluation Metrics:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.4f}")
    
    # Plots
    print("Generating plots...")
    plot_actual_vs_pred_scatter(y_true, y_pred, f"{args.outdir}/actual_vs_pred.png")
    plot_residuals(y_true, y_pred, f"{args.outdir}/residuals_vs_pred.png")
    plot_residual_hist_qq(y_true, y_pred, args.outdir)
    
    print(f"Evaluation complete. Plots saved to {args.outdir}")

if __name__ == "__main__":
    main()
