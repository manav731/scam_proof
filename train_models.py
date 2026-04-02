"""
train_models.py
---------------
Train Logistic Regression, Random Forest, and XGBoost on
the Sparkov fraud dataset. Saves .pkl models + evaluation plots.

Usage:
  python train_models.py
"""

import os, json, time, warnings, argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from xgboost                import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)

from preprocess import run_preprocessing, FEATURE_COLS

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
PLOTS_DIR    = os.path.join(BASE_DIR, "static", "plots")
RESULTS_PATH = os.path.join(MODELS_DIR, "model_results.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

MODEL_PATHS = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.pkl"),
    "Random Forest"      : os.path.join(MODELS_DIR, "random_forest.pkl"),
    "XGBoost"            : os.path.join(MODELS_DIR, "xgboost_model.pkl"),
}


# ──────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────
def get_models(scale_pos_weight=1.0):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42, n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=7,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=42, n_jobs=-1,
        ),
    }


# ──────────────────────────────────────────────
# Evaluate one model
# ──────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model"     : name,
        "precision" : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"    : round(recall_score(y_test, y_pred), 4),
        "f1_score"  : round(f1_score(y_test, y_pred), 4),
        "roc_auc"   : round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc"    : round(average_precision_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_prob"    : y_prob.tolist(),
    }

    cm = metrics["confusion_matrix"]
    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1-Score  : {metrics['f1_score']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print(f"  PR-AUC    : {metrics['pr_auc']}")
    print(f"  TN={cm[0][0]:>7,}  FP={cm[0][1]:>5,}")
    print(f"  FN={cm[1][0]:>7,}  TP={cm[1][1]:>5,}")
    return metrics


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────
def plot_confusion_matrices(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    for ax, res in zip(axes, results):
        cm = np.array(res["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                    xticklabels=["Legit","Fraud"],
                    yticklabels=["Legit","Fraud"], ax=ax)
        ax.set_title(res["model"], fontweight="bold")
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), dpi=120, bbox_inches="tight")
    plt.close()

def plot_roc_curves(results, y_test):
    plt.figure(figsize=(8, 6))
    for res, color in zip(results, ["#3498db","#2ecc71","#e74c3c"]):
        fpr, tpr, _ = roc_curve(y_test, np.array(res["y_prob"]))
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{res['model']}  (AUC={res['roc_auc']:.4f})")
    plt.plot([0,1],[0,1],"k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=120)
    plt.close()

def plot_pr_curves(results, y_test):
    plt.figure(figsize=(8, 6))
    for res, color in zip(results, ["#3498db","#2ecc71","#e74c3c"]):
        prec, rec, _ = precision_recall_curve(y_test, np.array(res["y_prob"]))
        plt.plot(rec, prec, color=color, lw=2, label=f"{res['model']}  (PR-AUC={res['pr_auc']:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pr_curves.png"), dpi=120)
    plt.close()

def plot_feature_importance(xgb_model):
    imp = xgb_model.feature_importances_
    feat_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": imp})
    feat_df = feat_df.sort_values("importance", ascending=False)
    plt.figure(figsize=(8, 5))
    colors = ["#e74c3c" if i < 3 else "#3498db" for i in range(len(feat_df))]
    plt.barh(feat_df["feature"][::-1], feat_df["importance"][::-1], color=colors[::-1])
    plt.xlabel("Importance")
    plt.title("XGBoost — Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=120)
    plt.close()

def plot_model_comparison(results):
    metrics = ["precision","recall","f1_score","roc_auc","pr_auc"]
    labels  = [r["model"].replace(" ","\n") for r in results]
    x = np.arange(len(labels)); width = 0.15
    colors = ["#3498db","#2ecc71","#e74c3c","#9b59b6","#f39c12"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [r[m] for r in results]
        bars = ax.bar(x + i*width, vals, width, label=m.replace("_"," ").title(), color=c, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x + width*2); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=120)
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def train_all():
    print("\n" + "="*52)
    print("  SPARKOV FRAUD DETECTION — MODEL TRAINING")
    print("="*52)

    # Step 1: Preprocess
    print("\n[STEP 1] Preprocessing...")
    X_train, X_test, y_train, y_test, _ = run_preprocessing(use_smote=True)

    # Step 2: scale_pos_weight for XGBoost
    neg = int((y_test == 0).sum())
    pos = int((y_test == 1).sum())
    spw = round(neg / pos, 2) if pos > 0 else 1.0
    print(f"\n[INFO] XGBoost scale_pos_weight = {spw}")

    # Step 3: Train
    models_def = get_models(spw)
    trained = {}
    print("\n[STEP 2] Training models...")
    for name, model in models_def.items():
        print(f"\n  → {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        print(f"    Done in {round(time.time()-t0, 1)}s")
        joblib.dump(model, MODEL_PATHS[name])
        print(f"    Saved → {MODEL_PATHS[name]}")
        trained[name] = model

    # Step 4: Evaluate
    print("\n[STEP 3] Evaluating...")
    results = []
    for name, model in trained.items():
        results.append(evaluate_model(model, X_test, y_test, name))

    # Step 5: Best model
    best = max(results, key=lambda r: r["f1_score"])
    print(f"\n{'='*52}")
    print(f"  ★  BEST MODEL : {best['model']}")
    print(f"     F1-Score   : {best['f1_score']}")
    print(f"     ROC-AUC    : {best['roc_auc']}")
    print(f"{'='*52}")

    # Step 6: Save results JSON
    clean = [{k:v for k,v in r.items() if k != "y_prob"} for r in results]
    clean.append({"best_model": best["model"]})
    with open(RESULTS_PATH, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n[INFO] Results → {RESULTS_PATH}")

    # Step 7: Plots
    print("\n[STEP 4] Generating plots...")
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_pr_curves(results, y_test)
    plot_model_comparison(results)
    plot_feature_importance(trained["XGBoost"])

    print("\n[DONE] All models trained and saved!")
    return trained, results


if __name__ == "__main__":
    train_all()
