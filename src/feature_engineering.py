def feature_engineering(data):
    data_processed = data.copy()
    def familysize(row):
        if row["Partner"] == "Yes" and row["Dependents"] == "No":
            return "has spouse"
        elif row["Partner"] == "No" and row["Dependents"] == "No":
            return "Alone"
        else:
            return "Has family"
    data_processed["familytype"] = data_processed.apply(familysize,axis=1)
    data_processed["SecurityScore"] = (
    (data_processed["OnlineSecurity"] == "Yes").astype(int)
    +(data_processed["OnlineBackup"] == "Yes").astype(int)
    +(data_processed["DeviceProtection"] == "Yes").astype(int)
    +(data_processed["TechSupport"] == "Yes").astype(int)
    )
    data_processed["EntertainmentScore"] = (
    (data_processed["StreamingTV"] == "Yes").astype(int)
    +(data_processed["StreamingMovies"] == "Yes").astype(int)
    )
    def duration(row):
        if row["tenure"] > 25:
            return "Old"
        elif row["tenure"] < 10:
            return "New"
        else:
            return "Moderate"
    data_processed["Customertype"] = data_processed.apply(duration,axis=1)
    def risk(row):
        if (row["Contract"] == "Month-to-month" and row["PaymentMethod"] == "Electronic check") and (row["InternetService"] == "Fiber optic" and row["TechSupport"] == "No"):
            return "High risk"
        elif (row["Contract"] == "Month-to-month" and row["PaymentMethod"] == "Mailed check") and (row["InternetService"] == "DSL" and row["SeniorCitizen"] == 0):
            return "Medium risk"
        else:
            return "Low risk"
    data_processed["RiskType"] = data_processed.apply(risk,axis=1)
    data_processed = data_processed.drop(["customerID","StreamingTV","StreamingMovies","gender","MultipleLines"],axis=1,errors="ignore")
    return data_processed