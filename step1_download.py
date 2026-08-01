"""
================================================================
step1_download.py — Dataset Downloader
================================================================
Downloads CIC-DDoS2019 from Kaggle, combines all parquet files,
extracts 77 features, creates binary labels, and saves a clean
CSV ready for model training.

Run this first:
    python step1_download.py
================================================================
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ── Configuration ──────────────────────────────────────────────
DATASET_NAME  = "dhoogla/cicddos2019"
OUTPUT_PATH   = "data/dataset.csv"
PARQUET_PATH  = None  # will be set after download


def download_dataset():
    """
    Downloads CIC-DDoS2019 from Kaggle using kagglehub.
    Returns the local path to the downloaded files.
    """
    print("[1/4] Downloading CIC-DDoS2019 dataset from Kaggle...")
    print("      This may take a few minutes on first run.")
    print("      Subsequent runs use the cached version.")

    import kagglehub
    path = kagglehub.dataset_download(DATASET_NAME)

    print(f"      Saved to: {path}")
    return path


def load_parquet_files(path):
    """
    Loads all parquet files from the dataset directory,
    combines them into one DataFrame, and adds metadata columns.
    Each file is named like 'NTP-testing.parquet' or 'Syn-training.parquet'.
    """
    print("\n[2/4] Loading parquet files...")

    all_dfs = []

    for filename in sorted(os.listdir(path)):
        if not filename.endswith('.parquet'):
            continue

        filepath = os.path.join(path, filename)
        df_temp  = pd.read_parquet(filepath)

        # Track how many rows came from each file
        print(f"      {filename:<40} {len(df_temp):>8,} rows")
        all_dfs.append(df_temp)

    # Combine all files into one dataframe
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n      Total rows combined: {len(df):,}")
    print(f"      Total columns:       {df.shape[1]}")

    return df


def create_binary_labels(df):
    """
    Creates a binary label column:
        0 = Benign (normal traffic)
        1 = Attack (any DDoS type)

    The Label column contains values like:
        'Benign', 'DrDoS_NTP', 'TFTP', 'Syn', etc.
    """
    print("\n[3/4] Creating binary labels...")

    # Check label distribution before encoding
    print("\n      Label distribution (original):")
    for label, count in df['Label'].value_counts().items():
        pct = count / len(df) * 100
        print(f"        {label:<25} {count:>8,}  ({pct:.1f}%)")

    # Binary encoding: Benign=0, everything else=1
    df['binary_label'] = (df['Label'] != 'Benign').astype(int)

    normal_count = (df['binary_label'] == 0).sum()
    attack_count = (df['binary_label'] == 1).sum()

    print(f"\n      Binary encoding:")
    print(f"        Normal (0): {normal_count:>8,}  ({normal_count/len(df)*100:.1f}%)")
    print(f"        Attack (1): {attack_count:>8,}  ({attack_count/len(df)*100:.1f}%)")

    return df


def extract_features(df):
    """
    Drops non-feature columns, keeping only the 77 numeric
    CICFlowMeter feature columns used for ML classification.

    Dropped columns:
        Label        — original text label (replaced by binary_label)
        attack_type  — added by some dataset versions
        split        — added by some dataset versions
    """
    print("\n[4/4] Extracting feature matrix...")

    # Identify columns to drop
    non_feature_cols = ['Label', 'binary_label']
    optional_drops   = ['attack_type', 'split']

    for col in optional_drops:
        if col in df.columns:
            non_feature_cols.append(col)

    # Feature columns = everything except non-feature columns
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # Build feature matrix
    X = df[feature_cols].copy()
    y = df['binary_label'].values

    # Clean: replace infinity values and NaN with 0
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Clip extreme outliers at 99.9th percentile per feature
    # This prevents a single extreme value from distorting scaling
    for col in X.columns:
        cap = X[col].quantile(0.999)
        X[col] = X[col].clip(upper=cap)

    print(f"      Feature columns: {len(feature_cols)}")
    print(f"      Feature names (first 5): {feature_cols[:5]}")
    print(f"      Feature names (last 5):  {feature_cols[-5:]}")
    print(f"      Null values remaining:   {X.isnull().sum().sum()}")

    return X, y, feature_cols


def save_dataset(X, y, feature_cols):
    """
    Saves the cleaned feature matrix and labels to a CSV file.
    Also saves the feature column names for reference.
    """
    os.makedirs('data', exist_ok=True)

    # Combine features and labels into one dataframe for saving
    df_save = X.copy()
    df_save['label'] = y

    df_save.to_csv(OUTPUT_PATH, index=False)

    print(f"\n      Saved: {OUTPUT_PATH}")
    print(f"      Size:  {os.path.getsize(OUTPUT_PATH)/1024/1024:.1f} MB")

    # Save feature column names separately
    import json
    with open('data/feature_cols.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"      Saved: data/feature_cols.json")


def main():
    print("=" * 60)
    print("DDoS Defense System — Step 1: Dataset Download")
    print("=" * 60)

    # Skip download if dataset CSV already exists
    if os.path.exists(OUTPUT_PATH):
        print(f"\nDataset already exists at {OUTPUT_PATH}")
        print("Delete it and re-run to force a fresh download.")
        return

    # Download from Kaggle
    path = download_dataset()

    # Load and combine parquet files
    df = load_parquet_files(path)

    # Create binary labels
    df = create_binary_labels(df)

    # Extract feature matrix
    X, y, feature_cols = extract_features(df)

    # Save to CSV
    save_dataset(X, y, feature_cols)

    print("\n" + "=" * 60)
    print("Step 1 complete. Run next: python step2_train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
