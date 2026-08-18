import pandas as pd
import pickle as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from src.data_preprocessing import data_preprocess
from src.feature_engineering import feature_engineering
from src.config import DATA_FILE,TARGET_COLUMN,RANDOM_STATE,TEST_SIZE,MODEL_FILE,BEST_PARAMS,BEST_THRESHOLD,TEST_FILE,TEST_PRED_FILE
from src.mlflow_utils import set_experiment,start_run,log_params_and_metrics,log_model,set_tags

data = pd.read_csv(DATA_FILE)
data = data_preprocess(data)
data = feature_engineering(data)

X = data.drop(columns = [TARGET_COLUMN])
y = data[TARGET_COLUMN].map({"Yes":1,"No":0})

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = TEST_SIZE,random_state = RANDOM_STATE,stratify=y)

X_test.to_csv(TEST_FILE,index=False)
y_test.to_csv(TEST_PRED_FILE,index=False)

num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object","string"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("Imputation",SimpleImputer(strategy="median"),num_cols),
    ("Encoding",OneHotEncoder(sparse_output=True,handle_unknown="ignore"),cat_cols)
],remainder="drop")

pipeline = Pipeline(
    [
        ("preprocessing",preprocessor),
        ("model",XGBClassifier(**BEST_PARAMS,random_state=RANDOM_STATE))
    ]
)

param = {
    **BEST_PARAMS,
    "random_state_value":RANDOM_STATE,
    "decision_threshold": BEST_THRESHOLD
}

pipeline.fit(X_train,y_train)

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= BEST_THRESHOLD).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_prob),
}

set_experiment("Customer Churn Model")
with start_run(run_name="XGBoost Churn"):
    set_tags({
        "model_type": "XGBoost",
        "optimization": "Optuna",
        "task": "Binary Classification"})
    log_params_and_metrics(params=param,metrics=metrics)
    log_model(pipeline,artifact_path="model")

with open(MODEL_FILE, "wb") as f:
    p.dump(pipeline, f)

