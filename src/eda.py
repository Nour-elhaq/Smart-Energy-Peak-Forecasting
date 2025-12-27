
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Perform Exploratory Data Analysis")
    parser.add_argument("--data", type=str, default="data/processed_daily.csv", help="Processed daily CSV")
    parser.add_argument("--outdir", type=str, default="plots/eda", help="Output directory")
    return parser.parse_args()

def plot_target_dist(df, out_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['y'], kde=True, bins=50)
    plt.title("Distribution of Daily Peak Load (y)")
    plt.xlabel("Peak Load (kW)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_total_load_ts(df, out_path):
    # Aggregated load
    daily_total = df.groupby('Date')['y'].sum().reset_index()
    daily_total['Date'] = pd.to_datetime(daily_total['Date'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_total['Date'], daily_total['y'])
    plt.title("Total System Daily Peak Load (Sum of all Feeders)")
    plt.xlabel("Date")
    plt.ylabel("Total Peak Load (kW)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_correlation(df, out_path):
    # Select numeric feats
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_feeder_boxplots(df, out_path):
    # Boxplot of load per feeder (top 30 by volume to avoid clutter)
    # Sort by median load
    medians = df.groupby('SerialNo')['y'].median().sort_values(ascending=False)
    top_feeders = medians.head(30).index
    
    subset = df[df['SerialNo'].isin(top_feeders)]
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='SerialNo', y='y', data=subset, order=top_feeders)
    plt.xticks(rotation=90)
    plt.title("Daily Peak Distribution per Feeder (Top 30)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    print("Generating EDA plots...")
    plot_target_dist(df, f"{args.outdir}/target_distribution.png")
    plot_total_load_ts(df, f"{args.outdir}/total_load_timeseries.png")
    plot_correlation(df, f"{args.outdir}/correlation_matrix.png")
    plot_feeder_boxplots(df, f"{args.outdir}/feeder_boxplots.png")
    
    print(f"EDA complete. Plots saved to {args.outdir}")

if __name__ == "__main__":
    main()
