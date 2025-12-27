
import pandas as pd
import argparse
import joblib
import os
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost Model")
    parser.add_argument("--data", type=str, default="data/processed_daily.csv", help="Processed daily CSV")
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb"], help="Model type")
    parser.add_argument("--outdir", type=str, default="models", help="Output directory")
    return parser.parse_args()

def train_model(data_path, out_dir):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature selection
    # Exclude targets and non-numeric cols not needed
    target = 'y'
    drop_cols = [target, 'Date', 'datetime'] 
    
    # Identify features
    # Numerical: lags, rolling, calendar (if numeric)
    # Categorical: SerialNo, Substation
    
    cat_features = ['SerialNo', 'Substation', 'is_weekend']
    # Ensure they are strings/categorical
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype(str)
            
    num_features = [c for c in df.columns if c not in drop_cols + cat_features]
    
    feature_names = num_features + cat_features
    X = df[feature_names]
    y = df[target]
    
    print(f"Features: {feature_names}")
    print(f"Training on {len(X)} samples...")
    
    # Preprocessing
    # One-hot encode categoricals, pass through numericals
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), num_features), # Handle remaining NaNs if any
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    
    # XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=500,
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
    
    pipeline.fit(X, y)
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "xgb_pipeline.joblib")
    joblib.dump(pipeline, out_path)
    print(f"Model saved to {out_path}")

def train_with_tuning(X, y, preprocessor, out_dir):
    from sklearn.model_selection import RandomizedSearchCV
    
    print("Starting Hyperparameter Tuning...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    param_dist = {
        'model__n_estimators': [100, 300, 500, 700],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7, 9],
        'model__min_child_weight': [1, 3, 5],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        verbose=1, 
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X, y)
    print(f"Best Params: {search.best_params_}")
    print(f"Best CV MAE: {-search.best_score_:.4f}")
    
    best_model = search.best_estimator_
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "xgb_tuned_pipeline.joblib")
    joblib.dump(best_model, out_path)
    print(f"Tuned model saved to {out_path}")

def main():
    args = parse_args()
    
    # Check for tuning flag (hacky parsing since we didn't add it to argparse yet)
    # Better: Update parse_args
    import sys
    do_tune = "--tune" in sys.argv
    
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    target = 'y'
    drop_cols = [target, 'Date', 'datetime'] 
    cat_features = ['SerialNo', 'Substation', 'is_weekend']
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype(str)
            
    num_features = [c for c in df.columns if c not in drop_cols + cat_features]
    
    X = df[num_features + cat_features]
    y = df[target]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    
    if do_tune:
        train_with_tuning(X, y, preprocessor, args.outdir)
    else:
        # Standard training
        print(f"Training standard model (use --tune for optimization)...")
        # Re-instantiate standard pipeline logic
        model = xgb.XGBRegressor(
            n_estimators=500,
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
        pipeline.fit(X, y)
        os.makedirs(args.outdir, exist_ok=True)
        out_path = os.path.join(args.outdir, "xgb_pipeline.joblib")
        joblib.dump(pipeline, out_path)
        print(f"Model saved to {out_path}")

if __name__ == "__main__":
    main()
