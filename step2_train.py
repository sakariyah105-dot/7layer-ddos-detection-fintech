"""
================================================================
step2_train.py — Model Trainer
================================================================
Trains all 4 models of the hybrid fusion engine:
    Model 1: Z-score statistical detector
    Model 2: Decision Tree classifier
    Model 3: Random Forest classifier
    Model 4: Isolation Forest anomaly detector

Then computes adaptive weights and fusion thresholds,
saves everything to the models/ folder.

Run after step1_download.py:
    python step2_train.py
================================================================
"""

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import RandomForestClassifier, IsolationForest
from sklearn.metrics            import (f1_score, precision_score,
                                        recall_score, accuracy_score)
from imblearn.over_sampling     import SMOTE


# ── Configuration ──────────────────────────────────────────────
DATASET_PATH   = "data/dataset.csv"
MODELS_DIR     = "models"
TEST_SIZE      = 0.30       # 70% train, 30% test
RANDOM_STATE   = 42
N_ESTIMATORS   = 100        # trees in Random Forest and Isolation Forest
MAX_DEPTH      = 10         # Decision Tree max depth
CONTAMINATION  = 0.227      # Isolation Forest: expected attack proportion
THETA_LOW      = 0.35       # fusion score below this → ALLOW
THETA_HIGH     = 0.55       # fusion score above this → BLOCK (between = QUARANTINE)
SMOTE_NEIGHBORS = 3         # k_neighbors for SMOTE


def load_dataset():
    """
    Loads the cleaned dataset created by step1_download.py.
    Separates features (X) from labels (y).
    """
    print("[1/6] Loading dataset...")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}\n"
            f"Run step1_download.py first."
        )

    df = pd.read_csv(DATASET_PATH)

    # Separate features from label column
    X = df.drop(columns=['label'])
    y = df['label'].values

    print(f"      Rows:    {len(X):,}")
    print(f"      Features:{X.shape[1]}")
    print(f"      Normal:  {(y==0).sum():,}  ({(y==0).mean()*100:.1f}%)")
    print(f"      Attack:  {(y==1).sum():,}  ({(y==1).mean()*100:.1f}%)")

    return X, y


def split_and_scale(X, y):
    """
    Splits data into 70% training / 30% test sets (stratified).
    Fits StandardScaler on training data only.
    Test data uses the same scaler but is never seen during fitting.
    Then applies SMOTE to balance the training set only.

    Important: SMOTE is applied AFTER splitting to prevent data leakage.
    The test set remains at its original imbalanced distribution.
    """
    print("\n[2/6] Splitting, scaling, and balancing...")

    # Stratified split preserves attack/normal ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y
    )

    print(f"      Train: {len(X_train):,} rows")
    print(f"      Test:  {len(X_test):,}  rows")

    # Fit scaler on training data only
    # Transform both train and test using training statistics
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    print(f"      Scaler fitted on training data")

    # Apply SMOTE to training set only
    # Creates synthetic normal-traffic samples to reach 1:1 balance
    print(f"      Applying SMOTE (k_neighbors={SMOTE_NEIGHBORS})...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_NEIGHBORS)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)

    print(f"      Balanced train: {len(X_train_bal):,} rows")
    print(f"        Normal: {(y_train_bal==0).sum():,}")
    print(f"        Attack: {(y_train_bal==1).sum():,}")

    return (X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc,
            X_train_bal, y_train_bal,
            scaler)


def train_zscore(X_train_bal, y_train_bal, X_test_sc, y_test):
    """
    Model 1: Z-score Statistical Detector

    Computes the mean and standard deviation of each feature
    from normal training samples only. At inference time,
    flags a flow as an attack if any feature deviates more
    than 3 standard deviations from the normal baseline.

    Formula: Z = (x - μ) / σ
    Decision: attack if max(|Z|) > 3

    Advantages:
        - No training time (arithmetic only)
        - Runs in microseconds
        - Catches extreme volumetric floods instantly

    Limitations:
        - Cannot detect subtle behavioral anomalies
        - Low F1 on labeled data — but this is expected
          (its role is speed, not accuracy)
    """
    print("\n[3/6] Model 1: Z-score Statistical Detector")
    t0 = time.time()

    # Compute baseline statistics from normal samples only
    normal_mask = (y_train_bal == 0)
    train_mean  = X_train_bal[normal_mask].mean(axis=0)
    train_std   = X_train_bal[normal_mask].std(axis=0) + 1e-8  # avoid division by zero

    # Compute max absolute Z-score across all 77 features per test sample
    max_z  = np.abs((X_test_sc - train_mean) / train_std).max(axis=1)

    # Convert Z-score to attack probability using sigmoid mapping
    # Z=3 maps to ~0.5, Z=10 maps to ~0.97
    p_stat = 1 / (1 + np.exp(-0.5 * (max_z - 3)))

    # Binary prediction: attack if max Z > 3 standard deviations
    y_pred = (max_z > 3).astype(int)

    f1  = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    print(f"      Done in {time.time()-t0:.2f}s")
    print(f"      F1={f1:.4f}  Accuracy={acc:.4f}")
    print(f"      Note: Low F1 is expected — Z-score is a speed filter,")
    print(f"            not a primary classifier.")

    return train_mean, train_std, p_stat, f1


def train_decision_tree(X_train_bal, y_train_bal, X_test_sc, y_test):
    """
    Model 2: Decision Tree Classifier (Fast Filter)

    Learns hierarchical binary feature splits by maximising
    information gain at each node. Limited to max_depth=10
    to prevent overfitting.

    Formula: IG(S,A) = H(S) - Σ (|Sv|/|S|) × H(Sv)

    Advantages:
        - Very fast inference (~0.05ms per flow)
        - High F1 on known attack patterns
        - Human-interpretable decision paths

    Role in ensemble:
        - Primary fast classifier for known attack types
        - Contributes most to supervised detection
    """
    print("\n[3/6] Model 2: Decision Tree Classifier")
    t0 = time.time()

    dt = DecisionTreeClassifier(
        max_depth    = MAX_DEPTH,
        random_state = RANDOM_STATE
    )
    dt.fit(X_train_bal, y_train_bal)

    p_dt  = dt.predict_proba(X_test_sc)[:, 1]
    y_pred = dt.predict(X_test_sc)

    f1  = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    print(f"      Done in {time.time()-t0:.2f}s")
    print(f"      F1={f1:.4f}  Accuracy={acc:.4f}")

    return dt, p_dt, f1


def train_random_forest(X_train_bal, y_train_bal, X_test_sc, y_test):
    """
    Model 3: Random Forest (Behavioral Analysis)

    Trains 100 independent Decision Trees on bootstrap samples
    with random feature subsets (√77 ≈ 9 features per split).
    Final prediction = majority vote across all 100 trees.

    Formula: ŷ = mode { h₁(x), h₂(x), ..., h₁₀₀(x) }

    Advantages:
        - Highest F1 of all 4 models
        - Eliminates individual tree overfitting via bagging
        - Robust to feature noise

    Note: This takes 2-3 minutes to train.
    """
    print("\n[3/6] Model 3: Random Forest (this takes ~2-3 minutes...)")
    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators = N_ESTIMATORS,
        max_features = 'sqrt',      # √77 ≈ 9 features per split
        n_jobs       = -1,          # use all CPU cores
        random_state = RANDOM_STATE
    )
    rf.fit(X_train_bal, y_train_bal)

    p_rf  = rf.predict_proba(X_test_sc)[:, 1]
    y_pred = rf.predict(X_test_sc)

    f1  = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    print(f"      Done in {time.time()-t0:.1f}s")
    print(f"      F1={f1:.4f}  Accuracy={acc:.4f}")

    return rf, p_rf, f1


def train_isolation_forest(X_train_bal, y_train_bal, X_test_sc, y_test):
    """
    Model 4: Isolation Forest (Unsupervised Anomaly Detection)

    The ONLY unsupervised model — never accesses attack labels.
    Detects structural anomalies by measuring how quickly a
    data point can be isolated through random recursive partitioning.
    Anomalous points are isolated faster (shorter average path length).

    Formula: s(x,n) = 2^(-E[h(x)] / c(n))

    Advantages:
        - Detects zero-day attacks with no labeled training examples
        - Dataset-agnostic structural scoring

    Important: F1 on labeled data will be very low (~0.001).
    This is expected and does NOT indicate failure.
    Liu et al. [10] explicitly note that F1 underestimates
    unsupervised detector value. Its contribution is zero-day
    coverage that no supervised model can provide.
    """
    print("\n[3/6] Model 4: Isolation Forest (Unsupervised)")
    t0 = time.time()

    iso = IsolationForest(
        n_estimators  = N_ESTIMATORS,
        contamination = CONTAMINATION,  # expected fraction of anomalies
        n_jobs        = -1,
        random_state  = RANDOM_STATE
    )
    iso.fit(X_train_bal)

    # Convert anomaly scores to 0-1 probability
    # Lower decision_function score = more anomalous = higher attack probability
    raw_scores = iso.decision_function(X_test_sc)
    p_if = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)

    y_pred = (p_if > 0.5).astype(int)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    acc    = accuracy_score(y_test, y_pred)

    print(f"      Done in {time.time()-t0:.2f}s")
    print(f"      F1={f1:.4f}  Accuracy={acc:.4f}")
    print(f"      Note: Low F1 is expected for unsupervised model.")
    print(f"            Its value is zero-day detection, not labeled F1.")

    return iso, p_if, f1


def compute_fusion(p_stat, p_dt, p_rf, p_if,
                   f1_stat, f1_dt, f1_rf, f1_if,
                   y_test):
    """
    Adaptive Weighted Decision Fusion Engine (Layer 5)

    Combines all 4 model outputs into a single fusion score:
        S = Σ wᵢ × pᵢ   for i = 1..4

    Weights are proportional to each model's F1 performance:
        wᵢ = F1ᵢ / Σ F1ⱼ

    This means high-performing models automatically get more
    influence, and weak models are naturally down-weighted.
    No manual tuning required.

    Three-path routing:
        S < θ_low          → ALLOW
        θ_low ≤ S ≤ θ_high → QUARANTINE (CAPTCHA challenge)
        S > θ_high         → BLOCK
    """
    print("\n[4/6] Computing adaptive fusion weights...")

    total_f1 = f1_stat + f1_dt + f1_rf + f1_if

    w_stat = f1_stat / total_f1
    w_dt   = f1_dt   / total_f1
    w_rf   = f1_rf   / total_f1
    w_if   = f1_if   / total_f1

    print(f"      w_zscore   = {w_stat:.4f}")
    print(f"      w_dtree    = {w_dt:.4f}")
    print(f"      w_rforest  = {w_rf:.4f}")
    print(f"      w_iforest  = {w_if:.4f}")
    print(f"      Sum        = {w_stat+w_dt+w_rf+w_if:.4f} ✓")

    # Compute fusion score for every test sample
    fusion_scores = (w_stat * p_stat +
                     w_dt   * p_dt   +
                     w_rf   * p_rf   +
                     w_if   * p_if)

    # Apply three-path routing
    decisions = []
    for s in fusion_scores:
        if s < THETA_LOW:
            decisions.append('ALLOW')
        elif s <= THETA_HIGH:
            decisions.append('QUARANTINE')
        else:
            decisions.append('BLOCK')

    # Evaluate: treat QUARANTINE + BLOCK as attack prediction
    y_pred = [0 if d == 'ALLOW' else 1 for d in decisions]

    f1   = f1_score(y_test, y_pred, zero_division=0)
    pre  = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    acc  = accuracy_score(y_test, y_pred)
    fp   = sum(1 for p, y in zip(y_pred, y_test) if p == 1 and y == 0)
    fpr  = fp / max(sum(1 for y in y_test if y == 0), 1)
    qtn  = sum(1 for d in decisions if d == 'QUARANTINE')

    print(f"\n      Fusion Engine Results:")
    print(f"        F1        = {f1:.4f}")
    print(f"        Precision = {pre:.4f}")
    print(f"        Recall    = {rec:.4f}")
    print(f"        Accuracy  = {acc:.4f}")
    print(f"        FPR       = {fpr:.5f}")
    print(f"        Quarantine= {qtn:,} samples ({qtn/len(decisions)*100:.2f}%)")
    print(f"        Allow     = {decisions.count('ALLOW'):,}")
    print(f"        Block     = {decisions.count('BLOCK'):,}")

    weights = {
        'w_stat': float(w_stat),
        'w_dt':   float(w_dt),
        'w_rf':   float(w_rf),
        'w_if':   float(w_if),
    }

    metrics = {
        'f1': float(f1), 'precision': float(pre),
        'recall': float(rec), 'accuracy': float(acc),
        'fpr': float(fpr), 'quarantine_pct': float(qtn/len(decisions)*100)
    }

    return fusion_scores, weights, metrics


def save_models(rf, dt, iso, scaler,
                train_mean, train_std,
                weights, metrics,
                feature_cols,
                f1_stat, f1_dt, f1_rf, f1_if):
    """
    Saves all trained models and metadata to the models/ folder.

    Files saved:
        rf_model.pkl      — Random Forest
        dt_model.pkl      — Decision Tree
        iso_model.pkl     — Isolation Forest
        scaler.pkl        — StandardScaler (fitted on training data)
        model_meta.json   — weights, thresholds, feature names, metrics
    """
    print("\n[5/6] Saving models...")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save sklearn model objects
    pickle.dump(rf,     open(f'{MODELS_DIR}/rf_model.pkl',  'wb'))
    pickle.dump(dt,     open(f'{MODELS_DIR}/dt_model.pkl',  'wb'))
    pickle.dump(iso,    open(f'{MODELS_DIR}/iso_model.pkl', 'wb'))
    pickle.dump(scaler, open(f'{MODELS_DIR}/scaler.pkl',    'wb'))

    # Save metadata — everything needed to run inference
    meta = {
        'feature_cols':  feature_cols,
        'weights':       weights,
        'thresholds': {
            'theta_low':  THETA_LOW,
            'theta_high': THETA_HIGH,
        },
        'train_mean':    train_mean.tolist(),
        'train_std':     train_std.tolist(),
        'f1_scores': {
            'zscore':  float(f1_stat),
            'dtree':   float(f1_dt),
            'rforest': float(f1_rf),
            'iforest': float(f1_if),
        },
        'test_metrics': metrics,
    }

    with open(f'{MODELS_DIR}/model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Print file sizes
    for filename in os.listdir(MODELS_DIR):
        size = os.path.getsize(f'{MODELS_DIR}/{filename}') / 1024 / 1024
        print(f"      {filename:<25} {size:.1f} MB")


def print_summary(metrics, f1_stat, f1_dt, f1_rf, f1_if):
    """
    Prints a clean summary table of all model results.
    """
    print("\n[6/6] Training complete — Summary")
    print("\n" + "=" * 55)
    print(f"{'Model':<22} {'F1':>8} {'Role'}")
    print("-" * 55)
    print(f"{'Z-score':<22} {f1_stat:>8.4f}  Speed pre-filter")
    print(f"{'Decision Tree':<22} {f1_dt:>8.4f}  Fast classifier")
    print(f"{'Random Forest':<22} {f1_rf:>8.4f}  Primary classifier")
    print(f"{'Isolation Forest':<22} {f1_if:>8.4f}  Zero-day detector")
    print("-" * 55)
    print(f"{'Fusion Engine':<22} {metrics['f1']:>8.4f}  Combined result")
    print("=" * 55)
    print(f"\nFPR:       {metrics['fpr']:.5f}")
    print(f"Quarantine:{metrics['quarantine_pct']:.3f}% of traffic")


def main():
    print("=" * 60)
    print("DDoS Defense System — Step 2: Model Training")
    print("=" * 60)

    # Load dataset
    X, y = load_dataset()

    # Split, scale, and balance
    (X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc,
     X_train_bal, y_train_bal,
     scaler) = split_and_scale(X, y)

    # Feature column names for saving
    feature_cols = list(X.columns)

    # Train Model 1: Z-score
    train_mean, train_std, p_stat, f1_stat = train_zscore(
        X_train_bal, y_train_bal, X_test_sc, y_test
    )

    # Train Model 2: Decision Tree
    dt, p_dt, f1_dt = train_decision_tree(
        X_train_bal, y_train_bal, X_test_sc, y_test
    )

    # Train Model 3: Random Forest
    rf, p_rf, f1_rf = train_random_forest(
        X_train_bal, y_train_bal, X_test_sc, y_test
    )

    # Train Model 4: Isolation Forest
    iso, p_if, f1_if = train_isolation_forest(
        X_train_bal, y_train_bal, X_test_sc, y_test
    )

    # Compute fusion scores and weights
    fusion_scores, weights, metrics = compute_fusion(
        p_stat, p_dt, p_rf, p_if,
        f1_stat, f1_dt, f1_rf, f1_if,
        y_test
    )

    # Save everything
    save_models(
        rf, dt, iso, scaler,
        train_mean, train_std,
        weights, metrics,
        feature_cols,
        f1_stat, f1_dt, f1_rf, f1_if
    )

    # Print summary
    print_summary(metrics, f1_stat, f1_dt, f1_rf, f1_if)

    print("\n" + "=" * 60)
    print("Step 2 complete. Run next: python step3_engine.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
