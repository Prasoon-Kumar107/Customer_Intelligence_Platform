import pickle as p
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from src.config import TEST_FILE,TEST_PRED_FILE,BEST_THRESHOLD,MODEL_FILE
with open(MODEL_FILE,"rb") as f:
    model = p.load(f)

X_test = pd.read_csv(TEST_FILE)
y_test = pd.read_csv(TEST_PRED_FILE).squeeze()

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= BEST_THRESHOLD).astype(int)

print(f"Accuracy: "f"{accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: "f"{precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall: "f"{recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1: "f"{f1_score(y_test, y_pred)*100:.2f}%")
print(f"ROC-AUC: "f"{roc_auc_score(y_test, y_prob)*100:.2f}%")




