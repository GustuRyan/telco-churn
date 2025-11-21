import streamlit as st
import pandas as pd

def render_churn_ranking_tab(model, df: pd.DataFrame):
    st.subheader("Churn Ranking for All Customers")

    # Work on a copy so we don't mutate the original df
    df = df.copy()

    # 💡 Clean TotalCharges just like in your training notebook
    if "TotalCharges" in df.columns:
        n_before = len(df)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])

    feature_cols = list(model.feature_names_in_)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"The dataset is missing columns the model expects: {missing}")
        return

    if "customerID" not in df.columns:
        st.error("Dataset does not contain 'customerID' column.")
        return

    # Use the same features as during training
    X_all = df[feature_cols]

    # Predict churn probability for every row
    proba_all = model.predict_proba(X_all)[:, 1]

    results = pd.DataFrame(
        {
            "customerID": df["customerID"],
            "churn_probability": proba_all,
        }
    )

    # Sort by highest churn risk
    results_sorted = results.sort_values(
        by="churn_probability",
        ascending=False,
    ).reset_index(drop=True)

    top_n = st.slider(
        "Show top N highest-risk customers",
        min_value=5,
        max_value=min(200, len(results_sorted)),
        value=20,
    )

    display_df = results_sorted.head(top_n).copy()
    display_df["churn_probability"] = (
        display_df["churn_probability"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(display_df, use_container_width=True)

    # Optional download
    csv = results_sorted.to_csv(index=False)
    st.download_button(
        label="Download full churn ranking as CSV",
        data=csv,
        file_name="churn_ranking.csv",
        mime="text/csv",
    )
