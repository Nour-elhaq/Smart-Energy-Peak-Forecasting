
import pandas as pd
import numpy as np
import argparse
import yaml
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import evaluate_metrics, plot_actual_vs_pred, plot_error_dist, plot_scatter
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def parse_args():
    parser = argparse.ArgumentParser(description="Run Walk-Forward Backtest")
    parser.add_argument("--data", type=str, default="data/processed_daily.csv", help="Processed daily CSV")
    parser.add_argument("--config", type=str, default="configs/backtest.yaml", help="Path to config yaml")
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb"], help="Model type")
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_xgb_pipeline(X_train, y_train, cat_features, num_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    
    model = xgb.XGBRegressor(
        n_estimators=100, # Reduced for faster backtest
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def main():
    args = parse_args()
    config = load_config(args.config)
    
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Dates
    dates = df['Date'].sort_values().unique()
    start_date = pd.Timestamp(config['backtest']['start_date'])
    end_date = pd.Timestamp(config.get('backtest', {}).get('end_date', dates[-1]))
    
    step_size_days = config['backtest']['step_size_days']
    initial_train_days = config['backtest']['min_train_days']
    
    # Walk forward
    current_date = dates[0] + pd.Timedelta(days=initial_train_days)
    if current_date < start_date:
        current_date = start_date
        
    metrics_list = []
    
    # Output dirs
    out_dir = "outputs/backtest"
    plot_dir = "plots/backtest"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Define features
    target = 'y'
    ignore_cols = [target, 'Date', 'datetime']
    cat_features = ['SerialNo', 'Substation', 'is_weekend']
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype(str)
            
    num_features = [c for c in df.columns if c not in ignore_cols + cat_features]
    
    print(f"Starting Walk-Forward Validation from {current_date} to {end_date}...")
    
    all_preds = []
    
    fold = 0
    while current_date < end_date:
        fold += 1
        test_end = current_date + pd.Timedelta(days=step_size_days)
        if test_end > end_date:
            test_end = end_date
            
        train_mask = df['Date'] < current_date
        test_mask = (df['Date'] >= current_date) & (df['Date'] < test_end)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) > 0 and len(test_df) > 0:
            train_start_real = train_df['Date'].min().date()
            train_end_real = train_df['Date'].max().date()
            test_start_real = test_df['Date'].min().date()
            test_end_real = test_df['Date'].max().date()
            
            print(f"Fold {fold}: Train [{train_start_real} .. {train_end_real}] | Test [{test_start_real} .. {test_end_real}]")
            
            # LEAKAGE CHECK:
            if train_end_real >= test_start_real:
                 raise ValueError(f"CRITICAL LEAKAGE: Train max {train_end_real} >= Test min {test_start_real}")
        else:
             print(f"Fold {fold}: Skipping due to empty train/test split.")
             continue
        
        if len(test_df) == 0:
            break
            
        X_train = train_df[num_features + cat_features]
        y_train = train_df[target]
        X_test = test_df[num_features + cat_features]
        y_test = test_df[target]
        
        # Train
        # Note: In a real expanded project, we'd cache the model or update it incrementally if supported.
        # Here we retrain for full correctness (no leakage).
        pipeline = train_xgb_pipeline(X_train, y_train, cat_features, num_features)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        fold_metrics = evaluate_metrics(y_test, y_pred, model_name=f"XGB_Fold_{fold}")
        fold_metrics['Fold'] = fold
        fold_metrics['StartDate'] = current_date
        fold_metrics['EndDate'] = test_end
        metrics_list.append(fold_metrics)
        
        # Store preds
        test_df = test_df.copy()
        test_df['y_pred'] = y_pred
        all_preds.append(test_df)
        
        current_date = test_end
        
    # Aggregate results
    results_df = pd.DataFrame(metrics_list)
    avg_metrics = results_df[['MAE', 'RMSE', 'Peak_MAE']].mean()
    print("\nAverage Metrics:")
    print(avg_metrics)
    
    results_df.to_csv(f"{out_dir}/metrics.csv", index=False)
    
    if all_preds:
        full_pred_df = pd.concat(all_preds)
        full_pred_df.to_csv(f"{out_dir}/predictions.csv", index=False)
        
        # Plots
        plot_actual_vs_pred(full_pred_df, "Backtest: Actual vs Predicted (Daily Sum)", f"{plot_dir}/actual_vs_pred_agg.png")
        plot_error_dist(full_pred_df['y'], full_pred_df['y_pred'], "Backtest Errors", f"{plot_dir}/error_dist.png")
        plot_scatter(full_pred_df['y'], full_pred_df['y_pred'], "Backtest Scatter", f"{plot_dir}/scatter.png")
        print(f"Plots saved to {plot_dir}")
        
if __name__ == "__main__":
    main()
