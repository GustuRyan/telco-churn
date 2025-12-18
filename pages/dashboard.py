import streamlit as st
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="Telco Churn Overview", page_icon="📊", layout="centered")

# 2. Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Dataset/TelcoCustomerChurn.csv")
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'Dataset/TelcoCustomerChurn.csv' exists.")
        return pd.DataFrame()

    # Numeric conversion
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Create a binary column for calculation ease
    if "Churn" in df.columns:
        df["is_churn"] = (df["Churn"] == "Yes").astype(int)
        
    return df

df = load_data()

if df.empty:
    st.stop()

# 3. Header
st.title("📊 Telco Customer Churn")
st.write("A high-level overview of customer retention and risk factors.")
st.divider()

# 4. Top-Level KPIs
churn_rate = df["is_churn"].mean() * 100
total_customers = len(df)
avg_revenue = df["MonthlyCharges"].mean()

k1, k2, k3 = st.columns(3)
k1.metric("Total Customers", f"{total_customers:,}")
k2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
k3.metric("Avg Monthly Bill", f"${avg_revenue:.2f}")

st.divider()

# --- EXISTING SECTIONS ---

# 5. Contracts
st.subheader("1. Which contracts drive churn?")
st.caption("Month-to-month contracts usually have the highest turnover.")
if "Contract" in df.columns:
    contract_churn = df.groupby("Contract")["is_churn"].mean() * 100
    st.bar_chart(contract_churn, horizontal=True,x_label="Contract Type", y_label="Churn Rate (%)", color="#FF4B4B")

# 6. Tenure
st.subheader("2. When do customers leave?")
st.caption("Churn is highest in the first year. Long-term customers are very stable.")
if "tenure" in df.columns:
    def tenure_group(t):
        if t <= 12: return "0–1 Year"
        if t <= 24: return "1–2 Years"
        if t <= 36: return "2–3 Years"
        if t <= 48: return "3–4 Years"
        if t <= 60: return "4–5 Years"
        return "5+ Years"

    df["tenure_group"] = df["tenure"].apply(tenure_group)
    order = ["0–1 Year", "1–2 Years", "2–3 Years", "3–4 Years", "4–5 Years", "5+ Years"]
    tenure_churn = df.groupby("tenure_group")["is_churn"].mean() * 100
    tenure_churn = tenure_churn.reindex(order)
    st.line_chart(tenure_churn, x_label="Tenure Group", y_label="Churn Rate (%)")

# --- NEW SECTIONS BELOW ---

# 7. Payment Methods (Crucial Insight)
st.subheader("3. Payment Method Risks")
st.caption("Electronic checks often indicate higher churn risk compared to automatic payments.")

if "PaymentMethod" in df.columns:
    # We use a horizontal bar chart because the labels (e.g., "Bank transfer (automatic)") are long
    pay_churn = df.groupby("PaymentMethod")["is_churn"].mean() * 100
    st.bar_chart(pay_churn, horizontal=True, x_label="Churn Rate (%)", color="#FFA500")

# 8. "Sticky" Services (Tech Support & Security)
st.subheader("4. The 'Stickiness' of Support Services")
st.caption("Customers who subscribe to support or security services are significantly less likely to leave.")

col_a, col_b = st.columns(2)

with col_a:
    if "TechSupport" in df.columns:
        st.markdown("**Tech Support**")
        tech_churn = df.groupby("TechSupport")["is_churn"].mean() * 100
        st.bar_chart(tech_churn, y_label="Churn Rate (%)", color="#1F77B4")

with col_b:
    if "OnlineSecurity" in df.columns:
        st.markdown("**Online Security**")
        sec_churn = df.groupby("OnlineSecurity")["is_churn"].mean() * 100
        st.bar_chart(sec_churn, y_label="Churn Rate (%)", color="#1F77B4")

# 9. Paperless Billing
st.subheader("5. Paperless Billing Impact")
st.caption("Paperless billing is efficient, but often correlates with higher churn.")

if "PaperlessBilling" in df.columns:
    paper_churn = df.groupby("PaperlessBilling")["is_churn"].mean() * 100
    st.bar_chart(paper_churn, horizontal=True, x_label="Churn Rate (%)", color="#2ECC71")
