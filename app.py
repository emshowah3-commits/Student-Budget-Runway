import os

import joblib
import pandas as pd
import streamlit as st

from src.runway import assess_all, describe, fix_text, prepare_ledger

st.set_page_config(page_title="Student Budget Runway", page_icon=" ", layout="wide")


# APP CODE
def show_df(frame):
    try:
        st.dataframe(frame, width="stretch")
    except Exception:
        st.dataframe(frame, use_container_width=True)


def show_image(path, caption):
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.image(path, caption=caption, **kwargs)
            return
        except Exception:
            continue


def banner(kind, text):
    {"error": st.error, "warning": st.warning, "success": st.success}.get(kind, st.info)(text)


st.title("Student Budget Runway Predictor")
st.markdown("**Veritas AI Hackathon MVP:** AI-powered transaction categorization and a pre-payday runway warning, "
            "tested on synthetic data.")

# 1. Load model pipeline
model_path = "models/categorizer.joblib"
if os.path.exists(model_path):
    clf = joblib.load(model_path)
else:
    st.error("Model file not found at `models/categorizer.joblib`. Run notebook 01 to create it, "
             "or place the saved model file there.")
    st.stop()

# 2. Sidebar controls
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Student Transactions (CSV)", type=["csv"])

st.sidebar.header("Forecast settings")
start_balance = st.sidebar.number_input("Starting balance ($), used only if the CSV has no running_balance column",
                                        value=500.0, step=50.0)
buffer = st.sidebar.number_input("Safety buffer ($)", value=0.0, step=25.0)
fallback_days = st.sidebar.number_input("Days until next payday (used if it cannot be worked out)",
                                        min_value=1, max_value=31, value=7)
threshold_pct = st.sidebar.slider("Warn when the chance of running out is at least (%)", 5, 80, 20, 5)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("The uploaded CSV file is empty.")
            st.stop()
        required_cols = ["date", "description", "amount"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Uploaded CSV is missing required columns: `{missing_cols}`. Please check your file headers!")
            st.stop()
    except pd.errors.EmptyDataError:
        st.error("The uploaded file has no columns or data to parse.")
        st.stop()
else:
    default_path = "sample_transactions.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.sidebar.info("Loaded default demo data. Upload your own CSV above to test a custom ledger.")
    else:
        st.warning("Please upload a transaction CSV file in the sidebar to begin.")
        st.stop()

if pd.to_datetime(df["date"], errors="coerce").isna().all():
    st.error("None of the values in the `date` column could be read as dates (try YYYY-MM-DD).")
    st.stop()

# 3. Categorize every transaction, then run the forecast for every student
descriptions = df["description"].astype(str)
df["pred_label"] = clf.predict(descriptions)
try:
    df["confidence"] = (clf.predict_proba(descriptions).max(axis=1) * 100).round(2)
except Exception:
    pass

ledger = prepare_ledger(df, start_balance)
if ledger.empty:
    st.error("No usable rows: check the `date` and `amount` columns.")
    st.stop()
threshold = threshold_pct / 100
assessments = assess_all(ledger, buffer=buffer, threshold=threshold, fallback_days=int(fallback_days))
flagged = sum(a.status == "warning" for a in assessments)

# 4. Main dashboard layout
tab_app, tab_eval, tab_info = st.tabs(["Live Dashboard", "Evaluation & Metrics", "How it works"])

with tab_app:
    st.subheader("Transaction Categorization & Runway Analysis")

    low_conf = int((ledger["confidence"] < 0.5).sum()) if "confidence" in ledger.columns else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", len(ledger))
    c2.metric("Unique Students", ledger["user_id"].nunique())
    c3.metric("Students with a runway warning", flagged)
    c4.metric("Low-confidence labels", "n/a" if low_conf is None else low_conf)

    st.write("### Classified Transactions Preview")
    preview = ledger.copy()
    preview["date"] = preview["date"].dt.strftime("%Y-%m-%d")
    cols = ["date", "description", "amount", "pred_label"] + (["confidence"] if "confidence" in preview.columns else [])
    show_df(preview[cols].head(10))
    if low_conf:
        st.caption(f"{low_conf} rows were labelled with under 50% confidence. Check those before trusting the forecast.")

    # pick a student
    ids = [a.user_id for a in assessments]
    student = st.selectbox("Student", ids) if len(ids) > 1 else ids[0]
    a = next(x for x in assessments if x.user_id == student)
    g = ledger[ledger["user_id"] == student]

    st.write("### Where the money goes")
    spend = g[(g["amount"] < 0) & (g["pred_label"] != "Income")]
    if spend.empty:
        st.info("No spending rows for this student.")
    else:
        st.bar_chart(spend.groupby("pred_label")["amount"].sum().abs().sort_values(ascending=False))

    st.write("### Pre-Payday Runway Warning")
    kind, message = describe(a, horizon_days=3)
    banner(kind, message)
    advice = fix_text(a)
    if advice:
        st.info(advice)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Balance now", f"${a.balance_now:,.2f}")
    m2.metric("Days to payday", "?" if a.days_to_payday is None else a.days_to_payday)
    m3.metric("Safe to spend per day", "n/a" if a.safe_per_day is None else f"${a.safe_per_day:,.2f}")
    m4.metric("Chance of running out", "n/a" if a.risk is None else f"{a.risk:.0%}")
    if a.payday_source:
        st.caption(f"Payday timing: {a.payday_source}.")

    if a.projection is not None:
        st.write("#### Projected balance until payday")
        st.line_chart(a.projection.set_index("date"))
        b1, b2 = st.columns(2)
        b1.markdown("**Spend-as-usual baseline**")
        b1.write("Would warn." if a.baseline_warns else "Would stay quiet.")
        if a.baseline_runout is not None:
            b1.caption(f"Expects to run out around {a.baseline_runout.strftime('%b %d')}.")
        b2.markdown("**Category-aware forecast**")
        b2.write("Warns." if a.status == "warning" else "Stays quiet.")
        if a.expected_runout is not None:
            b2.caption(f"Expects to run out around {a.expected_runout.strftime('%b %d')}.")
        if a.bills:
            st.write("Fixed bills expected before payday:")
            bills = pd.DataFrame(a.bills)
            bills["due"] = bills["due"].dt.strftime("%Y-%m-%d")
            bills["amount"] = bills["amount"].round(2)
            show_df(bills)

    st.write("### All students")
    summary = pd.DataFrame([x.summary_row() for x in assessments])
    show_df(summary.sort_values("chance of running out", ascending=False, na_position="last"))

    if "runs_out_within_3d_pre_payday" in df.columns:
        n_flag = int((df["runs_out_within_3d_pre_payday"] == 1).sum())
        st.caption(f"The uploaded file also contains {n_flag} rows pre-labelled as at risk. "
                    "They are not used by this forecast.")

with tab_eval:
    st.subheader("Model Performance vs. Naive Baseline")
    st.markdown("Results below were computed offline in the notebooks on **synthetic** Canadian student ledgers. "
                "The number of real warning cases is small, so read them as indicative, not conclusive.")

    sorting_path = "results/tables/sorting_results.csv"
    if os.path.exists(sorting_path):
        st.write("#### Sorting bank lines")
        show_df(pd.read_csv(sorting_path))
    errors_path = "results/tables/messy_errors.csv"
    if os.path.exists(errors_path):
        with st.expander("Where the sorting model gets it wrong (own test lines)"):
            show_df(pd.read_csv(errors_path).head(30))

    st.write("#### Early-warning forecast")
    table_path = "results/tables/forecast_comparison.csv"
    if os.path.exists(table_path):
        show_df(pd.read_csv(table_path))
    else:
        st.info("Evaluation metrics table not found in `results/tables/`.")

    fig_path = "results/figures/forecast_comparison.png"
    if os.path.exists(fig_path):
        show_image(fig_path, "3-Day Pre-Payday Cash-Out Warning Performance Comparison")
    else:
        st.info("Evaluation chart image not found in `results/figures/`.")

with tab_info:
    st.subheader("How it works")
    st.markdown(
        """
**1. Sort the bank lines.** A TF-IDF + logistic regression model labels each transaction (Groceries, Housing, ...).

**2. Split spending into fixed bills and flexible spending.** Housing, utilities, insurance, subscriptions,
education and fees are treated as fixed bills. Everything else going out is flexible.

**3. Look ahead to payday (plain code, no AI).**
- *Expected path:* today's balance minus the average flexible spending of the last 14 days, minus fixed bills as they come due.
- *Chance of running out:* 1,000 simulated futures, drawing each day's flexible spending from the student's own recent days.
- *Safe to spend:* (balance minus fixed bills before payday minus your safety buffer) divided by the days left.
- *Smallest fix:* the smallest cut to flexible spending that brings the chance of running out down to your chosen level.

**Definitions.** *Run out* means the balance falls below the safety buffer before the next payday. *Early* means at least
3 days before it happens. The *baseline* assumes the average daily spending of the last 30 days simply continues.

**Limits.** Synthetic data only. Fixed bills are guessed from repeated payments. A mislabelled transaction can change the
forecast. Educational prototype, **not financial advice**.
"""
    )

st.caption("Prototype built with synthetic data. Educational, not financial advice.")