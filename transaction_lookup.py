"""
transaction_lookup.py
---------------------
Looks up a transaction by TransactionID from the Sparkov dataset
(fraudTrain.csv or fraudTest.csv).

Returns human-readable fields directly usable by predict.py.
"""

import os
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "fraudTrain.csv")
TEST_PATH  = os.path.join(BASE_DIR, "data", "fraudTest.csv")

_df_cache = None


def _load_df():
    global _df_cache
    if _df_cache is None:
        dfs = []
        for path in [TRAIN_PATH, TEST_PATH]:
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
        if dfs:
            _df_cache = pd.concat(dfs, ignore_index=True)
            _df_cache["_row_id"] = _df_cache.index
            print(f"[INFO] Loaded {len(_df_cache):,} transactions for lookup.")
    return _df_cache


def is_dataset_available():
    return os.path.exists(TRAIN_PATH) or os.path.exists(TEST_PATH)


def get_dataset_size():
    df = _load_df()
    return len(df) if df is not None else None


def lookup_transaction(transaction_id: str) -> dict | None:
    """
    Look up by row index. Returns human-readable transaction dict or None.
    """
    df = _load_df()
    if df is None:
        return None

    try:
        idx = int(transaction_id)
        if idx < 0 or idx >= len(df):
            return None

        row = df.iloc[idx]

        # Compute hour and day_of_week from timestamp
        ts  = pd.to_datetime(row["trans_date_trans_time"])
        dob = pd.to_datetime(row["dob"])
        age = int((ts - dob).days / 365.25)

        # Distance
        import math
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            lat1,lon1,lat2,lon2 = map(math.radians,[lat1,lon1,lat2,lon2])
            dlat=lat2-lat1; dlon=lon2-lon1
            a=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            return round(R*2*math.asin(min(1,a**0.5)), 2)

        distance = haversine(
            float(row["lat"]), float(row["long"]),
            float(row["merch_lat"]), float(row["merch_long"])
        )

        return {
            "transaction_id"  : idx,
            "amt"             : round(float(row["amt"]), 2),
            "category"        : str(row["category"]),
            "merchant"        : str(row["merchant"]),
            "gender"          : str(row["gender"]),
            "hour"            : int(ts.hour),
            "day_of_week"     : int(ts.dayofweek),
            "age"             : age,
            "city_pop"        : int(row["city_pop"]),
            "distance"        : distance,
            "timestamp"       : str(ts),
            "city"            : str(row["city"]),
            "state"           : str(row["state"]),
            "true_label"      : int(row["is_fraud"]),
        }

    except (ValueError, KeyError, IndexError):
        return None
