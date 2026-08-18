import streamlit as st
import pandas as pd
import os
import requests
from src.config import BEST_PARAMS,BEST_THRESHOLD
from src.explain import summary_plot,customer_segmentation
st.header("🔮 CUSTOMER INTELLIGENCE PLATFORM")
st.markdown("### AI-powered customer churn prediction, risk prioritization, and explainable retention insights.")
st.divider()
st.sidebar.title("📌 Project Information")
st.sidebar.markdown("""
### Technologies Used
- 🤖 Machine Learning
- 📊 Scikit-learn
- 🌳 XGBOOST
- ⚡ Streamlit
- 🚀 FastAPI
- 🔍 SHAP
- 🐳 Docker
- ☁️ Render
- ⚙️ MLOps
""")
st.sidebar.markdown(f"""
### 🤖 Model Details
- **Algorithm:** XGBoost Classifier
- **Threshold:** `{BEST_THRESHOLD}`""")
with st.sidebar.expander("⚙️ Model Parameters"):
    st.json(BEST_PARAMS)
st.subheader("Customer Prediction")
tab1,tab2,tab3,tab4 = st.tabs(["Customer Information","Result Breakdown","Global Feature Importance","Customer-Specific Explanation"])
with tab1:
    col1,col2 = st.columns(2)
    with col1:
        gender = st.selectbox("What is gender of Customer?",["Male","Female"])
        SeniorCitizen = st.radio("Is Customer a Senior Citizen?",[0,1],horizontal=True)
        Partner = st.radio("Does Customer have a Partner?",["Yes","No"],horizontal=True)
        Dependents = st.radio("Does Customer have Dependents?",["Yes","No"],horizontal=True)
        tenure = st.slider("For how many months Customer have been a member for?",0,100,12)
        PhoneService = st.radio("Does Customer have Phone Service?",["Yes","No"],horizontal=True)
        MultipleLines = st.selectbox("Does Customer have Multiple Lines?",["Yes","No","No phone service"])
        InternetService = st.selectbox("Does Customer have Internet Service?",['DSL', 'Fiber optic', 'No'])
        OnlineSecurity = st.selectbox("Does Customer have Online Security?",['No', 'Yes', 'No internet service'])
        OnlineBackup = st.selectbox("Does Customer have Online Backup?",['Yes', 'No', 'No internet service'])
    with col2:
        DeviceProtection = st.selectbox("Does Customer have Device Protection?",['No', 'Yes', 'No internet service'])
        TechSupport = st.selectbox("Does Customer have Tech Support?",['No', 'Yes', 'No internet service'])
        StreamingTV = st.selectbox("Does Customer Streams on TV?",['No', 'Yes', 'No internet service'])
        StreamingMovies = st.selectbox("Does Customer Streams Movies?",['No', 'Yes', 'No internet service'])
        Contract = st.selectbox("What type of Contract Customer have?",['Month-to-month', 'One year', 'Two year'])
        PaperlessBilling = st.radio("Do Customer do Paperless Billing?",["Yes","No"],horizontal=True)
        PaymentMethod = st.selectbox("What type of Payment Method is used by Customer?",['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'])
        MonthlyCharges = st.slider("What are monthly charges paid by Customer?",0,200,100)
        TotalCharges = st.slider("What are the Total Charges paid by Customer till now?",0,10000,4500)
input_data = {
        "gender" : gender,
        "SeniorCitizen" : SeniorCitizen,
        "Partner" : Partner,
        "Dependents" : Dependents,
        "tenure" : tenure,
        "PhoneService" : PhoneService,
        "MultipleLines" : MultipleLines,
        "InternetService" : InternetService,
        "OnlineSecurity" : OnlineSecurity,
        "OnlineBackup" : OnlineBackup,
        "DeviceProtection" : DeviceProtection,
        "TechSupport" : TechSupport,
        "StreamingTV" : StreamingTV,
        "StreamingMovies" : StreamingMovies,
        "Contract" : Contract,
        "PaperlessBilling" : PaperlessBilling,
        "PaymentMethod" : PaymentMethod,
        "MonthlyCharges" : MonthlyCharges,
        "TotalCharges" : TotalCharges
}
API_URL = os.getenv("API_URL","http://127.0.0.1:8000/predict")
EXPLAIN_API_URL = os.getenv("EXPLAIN_API_URL","http://127.0.0.1:8000/explain")
if st.button("Predict",width="stretch"):
        try:
            response = requests.post(API_URL,json=input_data,timeout=30)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']}")
                st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
                st.metric("Threshold", f"{result['threshold']:.2f}")
                with tab2:
                    churn_probability = result["churn_probability"]
                    threshold = result["threshold"]
                    if churn_probability >= threshold:
                        risk_level = "HIGH"
                        risk_icon = "⚠️"
                        recommendation = "Contact customer within 7 days"
                    elif churn_probability >= 0.40:
                        risk_level = "MEDIUM"
                        risk_icon = "🟡"
                        recommendation = "Engage customer with a retention offer"
                    else:
                        risk_level = "LOW"
                        risk_icon = "🟢"
                        recommendation = "Continue regular customer engagement"
                    # Display risk information
                    st.subheader(f"{risk_icon} {risk_level} RISK")
                    st.info(
                        f"**Recommended Action:** {recommendation}")
                    st.subheader("👥 CUSTOMER SEGMENTS")
                    st.dataframe(customer_segmentation(),width="stretch")
                with tab3:
                    st.subheader("🔍 Global Feature Importance")
                    st.markdown("This shows which features have the greatest overall"
                        "impact on the model's churn predictions.")
                    try:
                        fig = summary_plot()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Unable to generate SHAP summary plot: {e}")
                with tab4:
                    st.subheader("🔎 Explanation for Current Customer")
                    try:
                        explain_response = requests.post(EXPLAIN_API_URL,json=input_data,timeout=30)
                        if explain_response.status_code == 200:
                            st.image(explain_response.content,caption="SHAP Waterfall Explanation",width="stretch")
                        else:
                            st.error(f"Explanation API Error: "f"{explain_response.status_code} - "f"{explain_response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to the FastAPI Server.")
                    except requests.exceptions.Timeout:
                        st.error("Explanation request timed out.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Unable to generate SHAP explanation: {e}")
            else:
                st.error(f"API Error:{response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI Server.")
