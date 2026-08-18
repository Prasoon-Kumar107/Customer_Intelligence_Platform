import pandas as pd
import matplotlib.pyplot as plt
import shap
from src.config import MODEL_FILE,DATA_FILE
from src.utils import load_model
from src.data_preprocessing import data_preprocess
from src.feature_engineering import feature_engineering
model = load_model(MODEL_FILE)

def summary_plot():
    data = pd.read_csv(DATA_FILE)
    data = data_preprocess(data)
    data = feature_engineering(data)
    preprocessing = model.named_steps["preprocessing"]
    data_trans = preprocessing.transform(data)
    feature_names = preprocessing.get_feature_names_out()
    # Convert sparse matrix to dense
    if hasattr(data_trans, "toarray"):
        data_trans = data_trans.toarray()
    data_transformed = pd.DataFrame(data_trans,columns=feature_names)
    xgb_model = model.named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(data_transformed)
    plt.figure()
    shap.plots.beeswarm(shap_values,max_display=15,show=False)
    fig = plt.gcf()
    return fig

def waterfall_plot(df):
    df = feature_engineering(df)
    preprocessing = model.named_steps["preprocessing"]
    trans_df = preprocessing.transform(df)
    if hasattr(trans_df, "toarray"):
        trans_df = trans_df.toarray()
    trans_df = pd.DataFrame(trans_df,columns=preprocessing.get_feature_names_out())
    xgb_model = model.named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(trans_df)
    plt.figure()
    shap.plots.waterfall(shap_values[0],max_display=15,show=False)
    fig = plt.gcf()
    return fig

def customer_segmentation():
    data = pd.read_csv(DATA_FILE)
    data = data_preprocess(data)
    data = feature_engineering(data)
    data["Churn_binary"] = data["Churn"].map({"Yes": 1,"No": 0})
    risk_summary = (data.groupby("RiskType")["Churn_binary"].agg(Customer_Count="count",Churn_Rate="mean").assign(Churn_Rate=lambda x: (x["Churn_Rate"] * 100).round(2)).reset_index())
    return risk_summary