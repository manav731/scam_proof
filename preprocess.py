"""
preprocess.py
-------------
Data loading, feature engineering, scaling and splitting
for the Sparkov Credit Card Fraud Detection dataset.

Raw columns used:
  trans_date_trans_time, amt, category, gender,
  lat, long, merch_lat, merch_long,
  city_pop, dob, is_fraud

Engineered features (all human-readable):
  amt, category_enc, gender_enc, hour, day_of_week,
  age, city_pop, distance, amt_log
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# ── Paths ──
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH   = os.path.join(BASE_DIR, "data", "fraudTrain.csv")
TEST_PATH    = os.path.join(BASE_DIR, "data", "fraudTest.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoders.pkl")

TARGET_COL   = "is_fraud"
RANDOM_STATE = 42

FEATURE_COLS = [
    "amt",           # raw transaction amount
    "amt_log",       # log(1+amt) — handles skew
    "category_enc",  # merchant category label-encoded
    "gender_enc",    # M/F encoded
    "hour",          # 0–23
    "day_of_week",   # 0=Mon, 6=Sun
    "age",           # derived from dob
    "city_pop",      # city population
    "distance",      # haversine(customer, merchant) in km
]

# All valid Sparkov categories
ALL_CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets",
    "misc_net", "misc_pos", "personal_care", "shopping_net",
    "shopping_pos", "travel",
]


# ──────────────────────────────────────────────
# Haversine distance
# ──────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ──────────────────────────────────────────────
# Load CSVs
# ──────────────────────────────────────────────
def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    print("[INFO] Loading Sparkov dataset...")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f"[INFO] Train : {train.shape}  |  Fraud: {train[TARGET_COL].sum():,} ({train[TARGET_COL].mean()*100:.2f}%)")
    print(f"[INFO] Test  : {test.shape}   |  Fraud: {test[TARGET_COL].sum():,}  ({test[TARGET_COL].mean()*100:.2f}%)")
    return train, test


# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()

    # Datetime
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]        = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

    # Age
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = ((df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25).astype(int)

    # Log amount
    df["amt_log"] = np.log1p(df["amt"])

    # Distance customer ↔ merchant
    df["distance"] = haversine(
        df["lat"].values, df["long"].values,
        df["merch_lat"].values, df["merch_long"].values,
    )

    # Encode categorical
    if fit:
        encoders = {}
        for col in ["category", "gender"]:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in ["category", "gender"]:
            le = encoders[col]
            df[f"{col}_enc"] = df[col].astype(str).apply(
                lambda x: int(le.transform([x])[0]) if x in le.classes_ else 0
            )

    return df, encoders


# ──────────────────────────────────────────────
# Scale numeric columns
# ──────────────────────────────────────────────
SCALE_COLS = ["amt", "amt_log", "age", "city_pop", "distance"]

def scale_features(X_train, X_test=None, fit=True):
    if fit:
        scaler = StandardScaler()
        X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
        if X_test is not None:
            X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[INFO] Scaler saved → {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        X_train[SCALE_COLS] = scaler.transform(X_train[SCALE_COLS])
    return X_train, X_test, scaler


# ──────────────────────────────────────────────
# SMOTE
# ──────────────────────────────────────────────
def apply_smote(X_train, y_train):
    print(f"[INFO] Before SMOTE — Legit: {(y_train==0).sum():,}  Fraud: {(y_train==1).sum():,}")
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.15)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After  SMOTE — Legit: {(y_res==0).sum():,}  Fraud: {(y_res==1).sum():,}")
    return X_res, y_res


# ──────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────
def run_preprocessing(use_smote=True):
    train_df, test_df = load_data()

    print("[INFO] Engineering features...")
    train_df, encoders = engineer_features(train_df, fit=True)
    test_df,  _        = engineer_features(test_df, encoders=encoders, fit=False)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(encoders, ENCODER_PATH)
    print(f"[INFO] Encoders saved → {ENCODER_PATH}")

    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET_COL].copy()
    X_test  = test_df[FEATURE_COLS].copy()
    y_test  = test_df[TARGET_COL].copy()

    X_train, X_test, _ = scale_features(X_train, X_test, fit=True)

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    print(f"\n[DONE] Preprocessing complete")
    print(f"  X_train : {X_train.shape}  |  Fraud: {int(y_train.sum()):,}")
    print(f"  X_test  : {X_test.shape}   |  Fraud: {int(y_test.sum()):,}")
    return X_train, X_test, y_train, y_test, encoders


# ──────────────────────────────────────────────
# EDA Plots
# ──────────────────────────────────────────────
def plot_eda(df, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "static", "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET_COL].value_counts()
    ax.bar(["Legitimate", "Fraud"], counts.values, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Class Distribution", fontweight="bold")
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.max() * 0.01, f"{v:,}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=120)
    plt.close()

    # Fraud rate by category
    fig, ax = plt.subplots(figsize=(13, 5))
    cat_fraud = df.groupby("category")[TARGET_COL].mean().sort_values(ascending=False)
    cat_fraud.plot(kind="bar", ax=ax, color="#e74c3c", alpha=0.85)
    ax.set_title("Fraud Rate by Merchant Category", fontweight="bold")
    ax.set_ylabel("Fraud Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fraud_by_category.png"), dpi=120)
    plt.close()

    # Fraud rate by hour
    df = df.copy()
    df["_hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
    fig, ax = plt.subplots(figsize=(10, 4))
    hour_fraud = df.groupby("_hour")[TARGET_COL].mean()
    ax.plot(hour_fraud.index, hour_fraud.values, color="#e74c3c", lw=2, marker="o", ms=4)
    ax.fill_between(hour_fraud.index, hour_fraud.values, alpha=0.15, color="#e74c3c")
    ax.set_title("Fraud Rate by Hour of Day", fontweight="bold")
    ax.set_xlabel("Hour"); ax.set_ylabel("Fraud Rate"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fraud_by_hour.png"), dpi=120)
    plt.close()

    # Amount distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for cls, color, label, ax in zip([0,1],["#2ecc71","#e74c3c"],["Legitimate","Fraud"],axes):
        df[df[TARGET_COL] == cls]["amt"].clip(0, 1000).hist(bins=60, ax=ax, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(f"Amount — {label}", fontweight="bold")
        ax.set_xlabel("Amount ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "amount_distribution.png"), dpi=120)
    plt.close()

    print(f"[INFO] EDA plots saved → {output_dir}/")


if __name__ == "__main__":
    run_preprocessing()
