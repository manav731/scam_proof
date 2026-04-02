"""
app.py — FraudShield Flask Application (Sparkov Dataset)
"""

import os
from flask import Flask, request, jsonify, render_template
from predict            import predict_transaction, predict_all_models, get_model_metrics
from transaction_lookup import lookup_transaction, is_dataset_available, get_dataset_size
from preprocess         import ALL_CATEGORIES

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")


def _plot(filename, title):
    path   = os.path.join(PLOTS_DIR, filename)
    exists = os.path.exists(path)
    return {"title": title, "url": f"/static/plots/{filename}", "exists": exists}


@app.route("/")
def index():
    return render_template(
        "index.html",
        dataset_available = is_dataset_available(),
        dataset_size      = get_dataset_size(),
        categories        = ALL_CATEGORIES,
    )


@app.route("/analytics")
def analytics():
    metrics = get_model_metrics()
    eval_plots = [
        _plot("roc_curves.png",         "ROC Curves — All Models"),
        _plot("pr_curves.png",          "Precision-Recall Curves"),
        _plot("model_comparison.png",   "Model Comparison — All Metrics"),
        _plot("feature_importance.png", "XGBoost Feature Importances"),
    ]
    eda_plots = [
        _plot("class_distribution.png",  "Class Distribution (Fraud vs Legit)"),
        _plot("fraud_by_category.png",   "Fraud Rate by Merchant Category"),
        _plot("fraud_by_hour.png",       "Fraud Rate by Hour of Day"),
        _plot("amount_distribution.png", "Transaction Amount Distribution"),
    ]
    return render_template(
        "analytics.html",
        metrics    = metrics,
        eval_plots = eval_plots,
        eda_plots  = eda_plots,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data provided"}), 400
    model_key = data.pop("model", "xgboost")
    valid = {"xgboost", "random_forest", "logistic_regression"}
    if model_key not in valid:
        return jsonify({"error": f"Invalid model. Choose from: {valid}"}), 400
    try:
        result = predict_transaction(data, model_key=model_key)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Model not found. Run: python train_models.py"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/all", methods=["POST"])
def predict_all():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data provided"}), 400
    try:
        return jsonify(predict_all_models(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/lookup", methods=["POST"])
def lookup():
    body   = request.get_json(force=True)
    txn_id = str(body.get("transaction_id", "")).strip()
    if not txn_id:
        return jsonify({"error": "transaction_id is required"}), 400
    result = lookup_transaction(txn_id)
    if result is None:
        return jsonify({
            "error"   : "Transaction not found",
            "message" : f"ID {txn_id} not found. Valid range: 0 – {(get_dataset_size() or 1) - 1}",
            "fallback": True,
        }), 404
    return jsonify(result)


@app.route("/metrics")
def metrics():
    data = get_model_metrics()
    if not data:
        return jsonify({"error": "No metrics found. Run train_models.py first."}), 404
    return jsonify(data)


@app.route("/health")
def health():
    files = {
        "xgboost"            : "xgboost_model.pkl",
        "random_forest"      : "random_forest.pkl",
        "logistic_regression": "logistic_regression.pkl",
        "scaler"             : "scaler.pkl",
        "encoders"           : "encoders.pkl",
    }
    models_dir = os.path.join(BASE_DIR, "models")
    status = {k: os.path.exists(os.path.join(models_dir, v)) for k, v in files.items()}
    ready  = all(status.values())
    return jsonify({
        "status"         : "ready" if ready else "models_missing",
        "models"         : status,
        "dataset_present": is_dataset_available(),
        "dataset_size"   : get_dataset_size(),
    }), 200 if ready else 503


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  FraudShield — Sparkov Dataset")
    print("  http://localhost:5000")
    print("  http://localhost:5000/analytics")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
