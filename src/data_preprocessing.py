import pandas as pd
def data_preprocess(data):
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"],errors="coerce")
    return data