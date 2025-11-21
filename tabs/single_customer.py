import streamlit as st
import pandas as pd

def yn(val: bool) -> str:
    return "Yes" if val else "No"

def render_single_customer_tab(model):
    st.subheader("Single Customer Prediction")

    with st.form("churn_form"):
        st.markdown("### Customer Profile")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)

            senior_toggle = st.toggle("Senior Citizen", value=False)
            senior = 1 if senior_toggle else 0

            partner_toggle = st.toggle("Has Partner", value=False)
            partner = yn(partner_toggle)

            dependents_toggle = st.toggle("Has Dependents", value=False)
            dependents = yn(dependents_toggle)

        with col2:
            tenure = st.slider(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12,
                help="How many months the customer has stayed.",
            )

            monthly_charges = st.slider(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0,
            )

            total_charges = st.slider(
                "Total Charges",
                min_value=0.0,
                max_value=10000.0,
                value=1000.0,
                step=10.0,
            )

        st.markdown("### Services")

        col3, col4 = st.columns(2)

        with col3:
            phone_service_toggle = st.toggle("Phone Service", value=True)
            phone_service = yn(phone_service_toggle)

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No phone service", "No", "Yes"],
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"],
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"],
            )

        with col4:
            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"],
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"],
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"],
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"],
            )

        st.markdown("### Contract & Billing")

        col5, col6 = st.columns(2)

        with col5:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )

            paperless_billing_toggle = st.toggle("Paperless Billing", value=True)
            paperless_billing = yn(paperless_billing_toggle)

        with col6:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        input_df = pd.DataFrame([input_dict])

        submitted = st.form_submit_button("Predict churn")

        if submitted:
            proba = model.predict_proba(input_df)[0, 1]
            pred = model.predict(input_df)[0]

            churn_label = "Churn" if pred == 1 else "Not churn"
            st.subheader(f"Prediction: **{churn_label}**")
            st.write(f"Churn probability: **{proba:.2%}**")
