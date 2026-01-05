import streamlit as st
import pandas as pd
import joblib

from tabs.single_customer import render_single_customer_tab
from tabs.churn_ranking import render_churn_ranking_tab

st.set_page_config(page_title="Telco Churn", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("Model/churn_model.joblib")

@st.cache_data
def load_data():
    # Single source of truth for the dataset path
    df = pd.read_csv("Dataset/TelcoCustomerChurn.csv")

    # Clean TotalCharges (common Telco issue: blanks like " ")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df

model = load_model()
df = load_data()

st.title("Telco Customer Churn Prediction")
st.write(
    "Use this app to predict churn for a single customer, "
    "or rank all customers by churn risk.\n\n"
    "➡️ Use the sidebar to open the **Dashboard** page."
)

tab_single, tab_all = st.tabs(["Single Prediction", "All Customers Ranking"])

with tab_single:
    render_single_customer_tab(model)

with tab_all:
    render_churn_ranking_tab(model, df)
