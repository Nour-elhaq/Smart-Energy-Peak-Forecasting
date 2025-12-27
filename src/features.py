
import pandas as pd
import numpy as np

def add_calendar_features(df, date_col='Date'):
    """
    Adds calendar features: day_of_week, month, is_weekend
    Expects df to have a date column (datetime64).
    """
    # Ensure date_col is datetime
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])
        
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

def add_lag_features(df, group_col, target_col, lags):
    """
    Adds lag features for a specific target column, grouped by group_col (Feeder).
    """
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
    return df

def add_rolling_features(df, group_col, target_col, windows):
    """
    Adds rolling mean and max features, grouped by group_col (Feeder).
    Assumes data is sorted by date.
    
    LEAKAGE PREVENTION:
    We explicitly shift the target by 1 BEFORE computing rolling stats.
    This ensures that for a target y_t (at row t), the rolling feature is computed 
    using {y_{t-1}, y_{t-2}, ...} and NOT y_t itself.
    This aligns with the goal of forecasting 'tomorrow' using 'today's' known data.
    """
    for window in windows:
        # Shift by 1 first to avoid leakage (rolling stats of *past* days)
        # We want features known for 'tomorrow' at time T. 
        # If we are predicting T+1, we know data up to T.
        # So for row T+1 (target), we use rolling stats of T, T-1, ...
        # The standard shift(1) aligns T with T+1 row.
        
        grouped = df.groupby(group_col)[target_col].shift(1)
        
        df[f'rolling_mean_{window}'] = grouped.rolling(window=window).mean().reset_index(0, drop=True)
        df[f'rolling_max_{window}'] = grouped.rolling(window=window).max().reset_index(0, drop=True)
        
    # Additional specific request: 14 day mean
    grouped_14 = df.groupby(group_col)[target_col].shift(1)
    df['rolling_mean_14'] = grouped_14.rolling(window=14).mean().reset_index(0, drop=True)

    return df
