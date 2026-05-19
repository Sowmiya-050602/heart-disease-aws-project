import streamlit as st
import requests
import json
import boto3
import pandas as pd
import plotly.express as px
from decimal import Decimal


API_URL = "https://p3bvm69zp1.execute-api.us-east-1.amazonaws.com/prod/predict"

dyn = boto3.resource('dynamodb', region_name='us-east-1')
table = dyn.Table('HeartDiseasePredictions')

# UI
st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️")
st.title("Heart Disease Risk Predictor")

tab1, tab2 = st.tabs(["Predict", "History"])

# Predict
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 20, 90, 55)
        sex = st.selectbox("Sex", [0,1], format_func=lambda x: ["Female","Male"][x])
        cp = st.selectbox("Chest Pain Type", [0,1,2,3])
        trestbps = st.number_input("Resting BP", 80, 200, 130)
        chol = st.number_input("Cholesterol", 100, 600, 220)
        fbs = st.selectbox("FBS > 120", [0,1])

    with col2:
        restecg = st.selectbox("Rest ECG", [0,1,2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Angina", [0,1])
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [0,1,2])
        ca = st.selectbox("CA", [0,1,2,3])
        thal = st.selectbox("Thal", [3,6,7])

    if st.button("Predict Risk", type="primary"):

        payload = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

        
            if "body" in result:
                try:
                    out = json.loads(result["body"])
                except:
                    out = result
            else:
                out = result

            score = float(out.get("score", 0))
            risk = out.get("risk", "Unknown")
            pred_id = out.get("id", "N/A")

            st.metric("Risk Score", f"{score:.1%}")
            st.metric("Risk Level", risk)
            st.info(f"Prediction ID: {pred_id}")

        else:
            st.error("API call failed")

with tab2:

    if st.button("Load History"):

        try:
            resp = table.scan(Limit=200)
            items = resp.get("Items", [])

            if items:

                def clean(v):
                    return float(v) if isinstance(v, Decimal) else v

                rows = [{k: clean(v) for k, v in item.items()} for item in items]
                df = pd.DataFrame(rows)

                if "timestamp" in df.columns:
                    df = df.sort_values("timestamp", ascending=False)

                st.subheader("Summary")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Predictions", len(df))
                col2.metric("High Risk", (df["risk_label"] == "High").sum())
                col3.metric("Avg Score", f"{df['prediction'].astype(float).mean():.2f}")

                st.plotly_chart(
                    px.histogram(df, x="prediction", nbins=20, title="Risk Score Distribution")
                )

                st.dataframe(df[["timestamp", "risk_label", "prediction", "age", "cp"]])

            else:
                st.info("No predictions yet.")

        except Exception as e:
            st.warning(f"DynamoDB not accessible: {e}")
