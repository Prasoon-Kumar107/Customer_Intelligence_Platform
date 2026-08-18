from pathlib import Path

# ============================================================
# Project Path
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# Dataset
# ============================================================

DATA_DIR = BASE_DIR/"data"
DATA_FILE = DATA_DIR/"Dataset.csv"

# ============================================================
# Train-Test Split
# ============================================================

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TEST_FILE = DATA_DIR/"X_test.csv"
TEST_PRED_FILE = DATA_DIR/"y_test.csv"

# ============================================================
# Model
# ============================================================

MODEL_DIR = BASE_DIR/"models"
MODEL_FILE = MODEL_DIR/"model.pkl"

# ============================================================
# Best XGBoost Parameters
# Obtained from Optuna
# ============================================================

BEST_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.010135741504846071,
    'max_depth': 5,
    'min_child_weight': 10,
    'gamma': 2.1767206092334814,
    'subsample': 0.8639688151739384,
    'colsample_bytree': 0.6184420428439027,
    'reg_alpha': 8.517451253173012e-08,
    'reg_lambda': 2.922504089966854e-08,
    'scale_pos_weight': 9.111395831318026,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1
}

# ============================================================
# Model Evaluation
# ============================================================

BEST_THRESHOLD = 0.72


