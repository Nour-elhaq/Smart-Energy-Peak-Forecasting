
import pandas as pd
import numpy as np
import argparse
import joblib
import os
import features as fe
# Better usage: reuse the pipeline's expectations.
# The model expects specific features. 
# We assume 'data' passed here is processed daily data up to today.
# We need to construct the feature vector for tomorrow.

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Forecasts")
    parser.add_argument("--model", type=str, default="models/xgb_pipeline.joblib", help="Path to joblib model")
    parser.add_argument("--data", type=str, default="data/processed_daily.csv", help="Processed daily CSV (historical)")
    parser.add_argument("--date", type=str, default="2014-11-18", help="Forecast date YYYY-MM-DD")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading model from {args.model}...")
    pipeline = joblib.load(args.model)
    
    print(f"Loading history from {args.data}...")
    df = pd.read_csv(args.data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    target_date = pd.Timestamp(args.date)
    prev_date = target_date - pd.Timedelta(days=1)
    
    # We need to create a row for each feeder for target_date
    # Features required: Lags (derived from history), Rolling (derived from history), Calendar
    
    feeders = df['SerialNo'].unique()
    print(f"Forecasting for {len(feeders)} feeders on {target_date}...")
    
    # Construct input dataframe for target_date
    # For lags/rolling, we need to look up values from the dataframe relative to target_date
    
    # Optimally, we'd just append the new date row to the DF and re-run feature engineering, 
    # but that might be slow if we re-calc everything.
    # Given the features are just Lags and Rolling:
    # Lag 1 = value at prev_date
    # Lag 7 = value at target_date - 7 days
    # Rolling 7 = mean(value) [target_date-7, target_date-1]
    
    # Let's filter df to relevant history window (last 30 days is plenty for our features)
    history_start = target_date - pd.Timedelta(days=30)
    history = df[df['Date'] >= history_start].copy()
    
    forecast_rows = []
    
    for feeder in feeders:
        feeder_hist = history[history['SerialNo'] == feeder].set_index('Date').sort_index()
        
        # Substation info
        substation = feeder_hist['Substation'].iloc[0] if not feeder_hist.empty else 'Unknown'
        
        row = {
            'SerialNo': feeder,
            'Date': target_date,
            'Substation': substation,
            'is_weekend': 1 if target_date.weekday() >= 5 else 0,
            'day_of_week': target_date.weekday(),
            'month': target_date.month
        }
        
        # Helper to safely get value
        def get_val(dt):
            if dt in feeder_hist.index:
                return feeder_hist.loc[dt, 'y']
            return np.nan # Or imputed mean
        
        # Lags
        row['lag_1'] = get_val(target_date - pd.Timedelta(days=1))
        row['lag_7'] = get_val(target_date - pd.Timedelta(days=7))
        
        # Rolling
        # Get last 7 days range
        r7_start = target_date - pd.Timedelta(days=7)
        r7_end = target_date - pd.Timedelta(days=1)
        r7_vals = feeder_hist.loc[r7_start:r7_end, 'y']
        
        row['rolling_mean_7'] = r7_vals.mean() if len(r7_vals) > 0 else np.nan
        row['rolling_max_7'] = r7_vals.max() if len(r7_vals) > 0 else np.nan
        
        # Rolling 14
        r14_start = target_date - pd.Timedelta(days=14)
        r14_vals = feeder_hist.loc[r14_start:r7_end, 'y'] # Note: using same end
        row['rolling_mean_14'] = r14_vals.mean() if len(r14_vals) > 0 else np.nan
        
        forecast_rows.append(row)
        
    forecast_df = pd.DataFrame(forecast_rows)
    
    # Handle NaNs (if history missing) - pipeline imputer will handle, but ideally we warn
    if forecast_df['lag_1'].isna().any():
        print("Warning: Some feeders missing historical data for features.")
        
    # Convert types to match training
    cat_features = ['SerialNo', 'Substation', 'is_weekend']
    for c in cat_features:
        forecast_df[c] = forecast_df[c].astype(str)
        
    # Predict
    # Make sure we select columns that match training (pipeline ignores extras usually, but good to be clean)
    X_new = forecast_df 
    preds = pipeline.predict(X_new)
    
    forecast_df['Forecast_Peak'] = preds
    
    # Save
    out_file = "outputs/predictions.csv"
    os.makedirs("outputs", exist_ok=True)
    forecast_df[['SerialNo', 'Date', 'Forecast_Peak']].to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")

if __name__ == "__main__":
    main()
