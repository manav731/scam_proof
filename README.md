# ScamProof — FraudShield

> A full-stack machine learning web application for real-time credit card fraud detection, built as an academic team project integrating Machine Learning and Web Technologies coursework.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-REST%20API-lightgrey?style=flat-square&logo=flask" alt="Flask"/>
  <img src="https://img.shields.io/badge/XGBoost-Primary%20Model-orange?style=flat-square" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?style=flat-square&logo=scikit-learn" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" alt="Status"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Team](#team)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Academic Context](#academic-context)

---

## Overview

ScamProof is a full-stack machine learning platform that detects fraudulent credit card transactions in real time. It was developed as a college team project applying concepts from our Machine Learning and Web Technologies courses in a single, end-to-end deployable application.

The system covers the complete data science lifecycle — from raw CSV ingestion through feature engineering, class balancing, model training, and serialisation — and exposes a Flask REST API that returns a fraud verdict, confidence score, and three-tier risk level (**HIGH / MEDIUM / LOW**) within milliseconds. An interactive analytics dashboard lets users explore model performance through ROC curves, precision-recall curves, confusion matrices, and feature importance charts.

**The core challenge:** Credit card fraud accounts for only ~0.58% of all transactions. A naïve model predicting "legitimate" for everything achieves 99.4% accuracy while being completely useless. ScamProof addresses this with SMOTE oversampling, cost-sensitive learning, and PR-AUC as the primary evaluation metric.

---

## Team

| Member | Primary Ownership | Files |
|---|---|---|
| **Annanya** | ML Model 1 — Logistic Regression · Preprocessing Lead | `preprocess.py` |
| **Khushi** | ML Model 2 — Random Forest · Training & Evaluation Lead | `train_models.py` |
| **Manav** | ML Model 3 — XGBoost · Inference Engine Lead | `predict.py` |
| **Siddhi** | Flask Backend · Frontend · Integration Lead | `app.py`, `index.html`, `analytics.html`, `transaction_lookup.py` |

---

## Features

- **Real-time fraud detection** — Submit any transaction and receive a verdict in milliseconds
- **Three-model comparison** — Run Logistic Regression, Random Forest, and XGBoost side by side on the same input via a single API call
- **Risk tiering** — Fraud probability mapped to three actionable levels: HIGH (≥ 0.75), MEDIUM (≥ 0.40), LOW (< 0.40)
- **Interactive analytics dashboard** — ROC curves, PR curves, confusion matrices, feature importance, and EDA plots all rendered in-browser
- **Dataset lookup mode** — Retrieve real historical transactions by row index to populate the form automatically for demo and testing
- **Deployment readiness check** — `/health` endpoint returns HTTP 200 when all model files are present, HTTP 503 if any are missing
- **No data leakage** — Scaler and encoders are fitted on training data only, serialised, and reloaded identically at inference time

---

## System Architecture

The platform is organised into six layers:

```
┌─────────────────────────────────────────────────┐
│  LAYER 1 — Data Sources                         │
│  fraudTrain.csv (1.3M rows) · fraudTest.csv     │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  LAYER 2 — Preprocessing  (preprocess.py)       │
│  Feature Eng → Encoding → Scaling → SMOTE       │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  LAYER 3 — Model Training  (train_models.py)    │
│  Logistic Regression · Random Forest · XGBoost  │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  LAYER 4 — Evaluation                           │
│  PR-AUC · F1 · Recall · Precision · ROC-AUC    │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  LAYER 5 — Flask Backend  (app.py + predict.py) │
│  6 REST endpoints · Inference engine            │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  LAYER 6 — Frontend Output                      │
│  Fraud Verdict Panel · Analytics Dashboard      │
└─────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Framework | scikit-learn, XGBoost |
| Class Balancing | imbalanced-learn (SMOTE) |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Frontend | HTML, CSS, JavaScript |
| Serialisation | pickle (.pkl) |
| Version Control | Git |

---

## Dataset

- **Source:** [Sparkov Credit Card Fraud Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — `kartik2112/fraud-detection` on Kaggle
- **Generator:** Sparkov Data Generation (Brandon Harris) — synthetically simulated to preserve real fraud data statistics without containing any personal information
- **Period:** January 2019 – December 2020
- **Cardholders:** 1,000 simulated customers across 693 merchants

| Split | File | Rows |
|---|---|---|
| Training | `fraudTrain.csv` | 1,296,675 |
| Test | `fraudTest.csv` | 555,719 |
| **Total** | — | **1,852,394** |

**Class imbalance:** ~0.58% fraud rate (approximately 1 in 172 transactions is fraudulent). The test set retains this natural rate to produce honest evaluation metrics.

---

## Methodology

ScamProof follows the **CRISP-DM** framework extended with a software deployment phase:

### Phase 1 — Business Understanding
Defined the problem as binary classification (fraud = 1, legitimate = 0). Rejected accuracy as a metric given the 0.58% fraud rate. Designated **PR-AUC** and **F1-Score** as primary success metrics.

### Phase 2 — Data Understanding
Exploratory analysis revealing:
- **Fraud by category:** `travel` and `shopping_net` exhibit the highest fraud rates
- **Fraud by hour:** Peaks between midnight and 3 AM
- **Amount distribution:** Fraud concentrates in the mid-range band, not at high values (card-testing behaviour)

### Phase 3 — Data Preparation (`preprocess.py`)

**Feature Engineering** — 6 features derived from raw columns:

| Feature | Source | Method |
|---|---|---|
| `hour` | `trans_date_trans_time` | Datetime parsing |
| `day_of_week` | `trans_date_trans_time` | Datetime parsing |
| `month` | `trans_date_trans_time` | Datetime parsing |
| `age` | `dob` + transaction timestamp | Integer years difference |
| `distance` | `lat/long` + `merch_lat/long` | Haversine formula (km) |
| `amt_log` | `amt` | `log(1 + amt)` |

**Encoding** — `LabelEncoder` applied to `category` (14 classes) and `gender`. Fitted on training data only, serialised to `encoders.pkl`.

**Scaling** — `StandardScaler` applied to `amt`, `amt_log`, `age`, `city_pop`, `distance`. Fitted on training data only, serialised to `scaler.pkl`.

**SMOTE** — `sampling_strategy=0.15` raises the fraud ratio from 0.58% to ~13% in the training set only. The test set is never resampled.

### Phase 4 — Modelling (`train_models.py`)

| Model | Key Configuration | Role |
|---|---|---|
| Logistic Regression | `class_weight=balanced`, L2, lbfgs | Linear baseline |
| Random Forest | `n_estimators=200`, `max_depth=12`, balanced | Ensemble benchmark |
| XGBoost ★ | `n_estimators=300`, `lr=0.05`, `eval_metric=aucpr` | Primary production model |

### Phase 5 — Evaluation

All models evaluated on the untouched `fraudTest.csv`. Metrics computed: **PR-AUC** (primary), **F1-Score**, **Precision**, **Recall**, **ROC-AUC**.

### Phase 6 — Deployment (`app.py` + `predict.py`)

Trained models serialised as `.pkl` files and loaded lazily (cached in memory). The inference engine in `predict.py` mirrors the training-time feature engineering pipeline exactly — same feature order, same saved scaler and encoders — to prevent distribution mismatch.

---

## Project Structure

```
scamproof/
│
├── preprocess.py            # Feature engineering, encoding, scaling, SMOTE, EDA plots
├── train_models.py          # Model training, evaluation, serialisation, result plots
├── predict.py               # Inference engine — loads models, builds feature vectors
├── transaction_lookup.py    # Retrieves dataset rows by index for UI demo mode
├── app.py                   # Flask application — 6 REST endpoints
├── missingvalue.py          # Linked list data structure component (academic exercise)
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── model_results.json
│
├── static/
│   └── plots/
│       ├── roc_curves.png
│       ├── pr_curves.png
│       ├── confusion_matrices.png
│       ├── model_comparison.png
│       ├── feature_importance.png
│       ├── class_distribution.png
│       ├── fraud_by_category.png
│       ├── fraud_by_hour.png
│       └── amount_distribution.png
│
├── templates/
│   ├── index.html           # Transaction submission UI
│   └── analytics.html       # Model performance dashboard
│
├── data/
│   ├── fraudTrain.csv       # Download from Kaggle (not tracked in Git)
│   └── fraudTest.csv        # Download from Kaggle (not tracked in Git)
│
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/manav731/scam_proof
cd scamproof
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `fraudTrain.csv` and `fraudTest.csv` from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place them in the `data/` directory.

### 4. Run preprocessing and training

```bash
# Run feature engineering, encoding, scaling, SMOTE, and EDA plots
python preprocess.py

# Train all three models and generate evaluation plots
python train_models.py
```

This will populate the `models/` and `static/plots/` directories.

### 5. Start the Flask server

```bash
python app.py
```

The application will be available at `http://localhost:5000`.

### 6. Verify deployment readiness

```bash
curl http://localhost:5000/health
# Expected: HTTP 200 {"status": "ok"}
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Submit a transaction to a single model and receive a fraud verdict |
| `POST` | `/predict/all` | Run all three models on the same transaction and compare outputs |
| `POST` | `/lookup` | Retrieve a real dataset row by index to auto-populate the UI form |
| `GET` | `/metrics` | Returns F1, Precision, Recall, ROC-AUC, PR-AUC for all three models as JSON |
| `GET` | `/analytics` | Serves the analytics dashboard HTML page |
| `GET` | `/health` | Returns HTTP 200 if all model files are present, HTTP 503 if any are missing |

### Example — POST /predict

**Request:**
```json
{
  "amt": 952.00,
  "category": "shopping_net",
  "gender": "M",
  "hour": 2,
  "age": 34,
  "city_pop": 45000,
  "distance": 2000.0,
  "model": "xgboost"
}
```

**Response:**
```json
{
  "is_fraud": true,
  "verdict": "FRAUDULENT",
  "confidence": 0.91,
  "risk_level": "HIGH"
}
```

**Risk level thresholds:**

| Risk Level | Fraud Probability |
|---|---|
| HIGH | ≥ 0.75 |
| MEDIUM | ≥ 0.40 |
| LOW | < 0.40 |

---

## Model Performance

All metrics computed on the held-out `fraudTest.csv` (natural 0.58% fraud rate, never resampled).

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.0561 | 0.7417 | 0.1043 | 0.8774 | 0.1442 |
| Random Forest | 0.2216 | 0.9441 | 0.3589 | 0.9961 | 0.84 |
| **XGBoost ★** | 0.08 | 0.9832 | 0.148 | 0.9966 | 0.8676 |



**Why PR-AUC over accuracy?**
With a 0.58% fraud rate, a model predicting "legitimate" for every transaction achieves 99.42% accuracy. PR-AUC directly measures the precision-recall trade-off that matters operationally — catching fraud (recall) while minimising false alarms (precision).

---

## Screenshots


<img width="2812" height="1515" alt="image" src="https://github.com/user-attachments/assets/1eee7a12-a1ba-4aec-b224-e27e14a9981f" />

<img width="1563" height="1290" alt="image" src="https://github.com/user-attachments/assets/fea4d090-aee8-4f02-b180-6eb9e6276fff" />

<img width="1554" height="1064" alt="image" src="https://github.com/user-attachments/assets/d1c7992a-3f3f-4c17-a4ef-324b5e7e1865" />

<img width="2826" height="1275" alt="image" src="https://github.com/user-attachments/assets/a74f1c23-cc06-4097-8553-547ad2050af4" />

<img width="1526" height="1252" alt="image" src="https://github.com/user-attachments/assets/ddb9e06a-4e38-4329-80ab-dcc3c597d4cb" />


---

## Academic Context

This project was developed as part of our **second year college curriculum**, applying concepts from:

- **Machine Learning** — Supervised classification, ensemble methods, gradient boosting, class imbalance handling (SMOTE), model evaluation metrics
- **Web Technologies** — REST API design, Flask routing, frontend development with HTML/CSS/JavaScript, client-server architecture

The project demonstrates how academic ML work transitions into a production-grade application architecture — covering the full data science lifecycle from raw data ingestion to a live, browser-accessible web service.

**Framework used:** CRISP-DM (Cross-Industry Standard Process for Data Mining), extended with a software deployment phase.

---

## Requirements

```
flask
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
pickle-mixin
```

> See `requirements.txt` for pinned versions.

---

## License

This project was created for academic purposes. All dataset usage is subject to the [Kaggle dataset license](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

---

<p align="center">Made with Python · Flask · XGBoost &nbsp;|&nbsp; Academic Year 2025–26 &nbsp;|&nbsp; Annanya · Khushi · Manav · Siddhi</p>
