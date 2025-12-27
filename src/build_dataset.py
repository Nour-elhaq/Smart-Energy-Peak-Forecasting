
import pandas as pd
import numpy as np
import argparse
import os
from features import add_calendar_features, add_lag_features, add_rolling_features

def parse_args():
    parser = argparse.ArgumentParser(description="Build processed daily dataset")
    parser.add_argument("--data", type=str, default="data/EXPORT HourlyData - Feeders.csv", help="Path to raw CSV")
    parser.add_argument("--out", type=str, default="data/processed_daily.csv", help="Path to save processed CSV")
    return parser.parse_args()

def load_and_clean(path):
    print(f"Loading data from {path}...")
    # Fix 1: Dtype Warning - Load SerialNo as string
    df = pd.read_csv(path, dtype={'SerialNo': str})
    
    # Parse datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        raise ValueError("Column 'datetime' not found in dataset")

    # Fix 2: Robust Ignore Flag Filtering
    # Coerce to numeric (errors='coerce' turns non-numeric to NaN)
    # Then fill NaN with 0 (assume valid if missing flag) or 1 (safe)?
    # Usually flags are 0/1. Let's assume 0 if missing.
    ignore_cols = ['Ignore_CurrentNotConnected', 'Ignore_VoltageNotConnected']
    
    mask = pd.Series(False, index=df.index)
    
    for col in ignore_cols:
        if col in df.columns:
            # Force numeric, invalid becomes NaN
            s_num = pd.to_numeric(df[col], errors='coerce').fillna(0)
            # Flag is active if == 1
            mask = mask | (s_num == 1)
            
            # Log specific counts
            count = (s_num == 1).sum()
            print(f"  - Flag '{col}': found {count} rows with flag=1.")
    
    initial_len = len(df)
    df = df[~mask].copy()
    dropped_count = initial_len - len(df)
    print(f"Dropped {dropped_count} rows total due to ignore flags.")
    
    return df

def aggregate_daily(df):
    print("Aggregating to daily peaks...")
    
    # Interpolate for small gaps per feeder
    phase_cols = ['PA_MAX', 'PB_MAX', 'PC_MAX']
    df = df.sort_values(by=['SerialNo', 'datetime'])
    
    # Limit interpolation to prevent large gap filling (which should be dropped)
    # Use forward/backward limit
    df[phase_cols] = df.groupby('SerialNo')[phase_cols].transform(lambda x: x.interpolate(method='linear', limit=4))
    
    # Drop remaining NaNs
    df = df.dropna(subset=phase_cols)
    
    df['P_total_max'] = df['PA_MAX'] + df['PB_MAX'] + df['PC_MAX']
    
    # Fix: Clamp negative to 0
    df['P_total_max'] = df['P_total_max'].clip(lower=0)
    
    # Group by Feeder and Date
    df['Date'] = df['datetime'].dt.floor('D')
    
    # Daily max per feeder
    daily = df.groupby(['SerialNo', 'Date', 'Substation'])['P_total_max'].max().reset_index()
    daily.rename(columns={'P_total_max': 'y'}, inplace=True)
    
    # Fix 5: Anomaly Detection (System Peak Collapse)
    # Calculate total system load per day
    system_daily = daily.groupby('Date')['y'].sum()
    
    # Define threshold: < 10% of median system peak (relaxed from 5%)
    median_sys = system_daily.median()
    min_sys = system_daily.min()
    mean_sys = system_daily.mean()
    
    threshold = 0.10 * median_sys
    
    print(f"System Peak Stats: Median={median_sys:.2f}, Mean={mean_sys:.2f}, Min={min_sys:.2f}")
    print(f"Anomaly Threshold (< 10% median): {threshold:.2f}")
    
    anomalies = system_daily[system_daily < threshold]
    anomaly_dates = anomalies.index
    
    if len(anomaly_dates) > 0:
        print(f"Detected {len(anomaly_dates)} anomaly days:")
        for d in anomaly_dates:
            print(f"  - {d.date()} (Load: {anomalies[d]:.2f} kW)")
            
        # Drop these dates
        daily = daily[~daily['Date'].isin(anomaly_dates)]
        print(f"Dropped anomaly days. New shape: {daily.shape}")
    else:
        print("No system peak anomalies detected.")
            
    return daily

def main():
    args = parse_args()
    
    df_raw = load_and_clean(args.data)
    df_daily = aggregate_daily(df_raw)
    
    print("Adding features...")
    # Add features
    df_daily = add_calendar_features(df_daily, date_col='Date')
    
    # Lags: 1 day, 7 days
    df_daily = add_lag_features(df_daily, group_col='SerialNo', target_col='y', lags=[1, 7])
    
    # Rolling: 7d mean/max, 14d mean
    df_daily = add_rolling_features(df_daily, group_col='SerialNo', target_col='y', windows=[7])
    
    # Drop rows with NaNs created by lags/rolling (first 14 days will be lost)
    before_drop = len(df_daily)
    df_daily = df_daily.dropna()
    print(f"Dropped {before_drop - len(df_daily)} rows due to lag/rolling NaN (warmup).")
    
    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_daily.to_csv(args.out, index=False)
    print(f"Saved processed data to {args.out} (Shape: {df_daily.shape})")

if __name__ == "__main__":
    main()
