import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Risk Scoring", page_icon="🏦", layout="wide")

COLS = ["utilization", "age", "late_30_59", "debt_ratio", "income",
        "open_lines", "late_90", "real_estate", "late_60_89", "dependents"]

LOW_CUT, HIGH_CUT = 0.30, 0.60  # same cut-offs used in the notebook


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("credit_model.json")
    return model


model = load_model()

st.title("Credit risk scoring")
st.caption(
    "Demo model trained on the public Give Me Some Credit dataset (150,000 borrowers). "
    "It is a portfolio project, not a real lending tool."
)

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Borrower details")
    age = st.number_input("Age", 18, 100, 40)
    income = st.number_input("Monthly income", 0, 100000, 5000, step=500)
    dependents = st.number_input("Number of dependents", 0, 10, 0)
    st.subheader("Credit profile")
    utilization = st.slider(
        "Credit utilisation (balance as a share of limit)", 0.0, 1.5, 0.30, 0.01
    )
    debt_ratio = st.slider("Debt ratio (monthly debt as a share of income)", 0.0, 5.0, 0.35, 0.01)
    open_lines = st.number_input("Open credit lines and loans", 0, 40, 8)
    real_estate = st.number_input("Real estate loans", 0, 10, 1)
    st.subheader("Payment history")
    late_30_59 = st.number_input("Times 30-59 days late", 0, 10, 0)
    late_60_89 = st.number_input("Times 60-89 days late", 0, 10, 0)
    late_90 = st.number_input("Times 90+ days late", 0, 10, 0)

row = pd.DataFrame(
    [[utilization, age, late_30_59, debt_ratio, income,
      open_lines, late_90, real_estate, late_60_89, dependents]],
    columns=COLS,
)
score = float(model.predict_proba(row)[0, 1])

if score < LOW_CUT:
    band = "Low"
elif score < HIGH_CUT:
    band = "Medium"
else:
    band = "High"

# ---------------- Output ----------------
tab_score, tab_model = st.tabs(["Score a borrower", "About the model"])

with tab_score:
    c1, c2 = st.columns([1, 2])
    c1.metric("Risk score", f"{score:.0%}")
    c1.metric("Risk band", band)
    with c2:
        if band == "Low":
            st.success("Low risk. The profile looks similar to borrowers who repay on time.")
        elif band == "Medium":
            st.warning("Medium risk. Review the payment history and utilisation before deciding.")
        else:
            st.error("High risk. The profile looks similar to borrowers who later defaulted.")
        st.progress(min(score, 1.0))
        st.caption(
            "The score is a ranking signal, not a calibrated probability of default. "
            "The model was trained with extra weight on defaulters, which pushes scores up."
        )

    st.subheader("What drives scores in this model")
    importance = pd.Series(model.feature_importances_, index=COLS).sort_values()
    st.bar_chart(importance, horizontal=True)

with tab_model:
    st.subheader("Model and data")
    st.markdown(
        """
- **Data:** Give Me Some Credit, with default defined as serious delinquency within two years (about 7% of borrowers).
- **Cleaning:** impossible ages and placeholder late-payment codes treated as missing; extreme values capped at the 99th percentile.
- **Models compared:** logistic regression (AUC 0.859) and XGBoost (AUC 0.869, 77.8% recall on defaulters).
- **Risk bands:** Low below 30%, Medium 30-60%, High above 60% of the model score.
- **Limits:** one historical dataset, no check for fairness across groups, and no calibration of the score to real default rates.
        """
    )
