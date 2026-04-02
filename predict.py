"""
predict.py
----------
Prediction logic for the Sparkov fraud detection system.
Accepts human-readable inputs, engineers features, and returns prediction.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
RESULTS_PATH = os.path.join(MODELS_DIR, "model_results.json")

MODEL_FILES = {
    "xgboost"            : "xgboost_model.pkl",
    "random_forest"      : "random_forest.pkl",
    "logistic_regression": "logistic_regression.pkl",
}

FEATURE_COLS = [
    "amt", "amt_log", "category_enc", "gender_enc",
    "hour", "day_of_week", "age", "city_pop", "distance",
]
SCALE_COLS = ["amt", "amt_log", "age", "city_pop", "distance"]

HIGH_RISK   = 0.75
MEDIUM_RISK = 0.40

_cache = {}


def _load(key):
    if key not in _cache:
        if key == "scaler":
            _cache[key] = joblib.load(SCALER_PATH)
        elif key == "encoders":
            _cache[key] = joblib.load(ENCODER_PATH)
        elif key in MODEL_FILES:
            path = os.path.join(MODELS_DIR, MODEL_FILES[key])
            _cache[key] = joblib.load(path)
        else:
            raise ValueError(f"Unknown key: {key}")
    return _cache[key]


def _risk_level(prob):
    if prob >= HIGH_RISK:   return "HIGH"
    if prob >= MEDIUM_RISK: return "MEDIUM"
    return "LOW"


def haversine(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(min(1, a**0.5))


def build_feature_vector(data: dict) -> pd.DataFrame:
    """
    Convert human-readable transaction dict → scaled feature DataFrame.

    Expected keys:
      amt          : float  — transaction amount
      category     : str    — merchant category (e.g. 'grocery_pos')
      gender       : str    — 'M' or 'F'
      hour         : int    — 0–23
      day_of_week  : int    — 0=Mon, 6=Sun
      age          : int    — customer age
      city_pop     : int    — city population
      distance     : float  — km between customer and merchant
                              (or computed from lat/long if provided)
    """
    # Compute distance from lat/lon if provided instead of direct distance
    if "distance" not in data and all(k in data for k in ["lat","long","merch_lat","merch_long"]):
        data["distance"] = haversine(
            float(data["lat"]), float(data["long"]),
            float(data["merch_lat"]), float(data["merch_long"])
        )

    encoders = _load("encoders")
    scaler   = _load("scaler")

    amt          = float(data.get("amt", 0))
    category_raw = str(data.get("category", "misc_pos"))
    gender_raw   = str(data.get("gender", "M"))

    # Encode category
    le_cat = encoders["category"]
    cat_enc = int(le_cat.transform([category_raw])[0]) if category_raw in le_cat.classes_ else 0

    # Encode gender
    le_gen = encoders["gender"]
    gen_enc = int(le_gen.transform([gender_raw])[0]) if gender_raw in le_gen.classes_ else 0

    row = {
        "amt"         : amt,
        "amt_log"     : float(np.log1p(amt)),
        "category_enc": cat_enc,
        "gender_enc"  : gen_enc,
        "hour"        : int(data.get("hour", 12)),
        "day_of_week" : int(data.get("day_of_week", 0)),
        "age"         : int(data.get("age", 35)),
        "city_pop"    : int(data.get("city_pop", 50000)),
        "distance"    : float(data.get("distance", 10.0)),
    }

    df = pd.DataFrame([row], columns=FEATURE_COLS)

    # Scale numeric columns
    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
    return df


def predict_transaction(data: dict, model_key: str = "xgboost") -> dict:
    """
    Predict fraud for a transaction.

    Returns dict with is_fraud, verdict, confidence, risk_level, model_used.
    """
    model = _load(model_key)
    X     = build_feature_vector(data)

    prediction  = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    confidence  = round(probability * 100, 2)

    return {
        "is_fraud"  : bool(prediction),
        "verdict"   : "FRAUDULENT" if prediction else "LEGITIMATE",
        "confidence": confidence,
        "risk_level": _risk_level(probability),
        "model_used": model_key,
    }


def predict_all_models(data: dict) -> dict:
    results = {}
    for key in MODEL_FILES:
        try:
            results[key] = predict_transaction(data, model_key=key)
        except FileNotFoundError:
            results[key] = {"error": "Model not found. Run train_models.py first."}
    return results


def get_model_metrics() -> list:
    if not os.path.exists(RESULTS_PATH):
        return []
    with open(RESULTS_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    sample = {
        "amt": 952.00, "category": "shopping_net",
        "gender": "F", "hour": 2, "day_of_week": 5,
        "age": 45, "city_pop": 30000, "distance": 2000.0,
    }
    print(predict_transaction(sample))
