"""Streamlit frontend for the Customer Retention Intelligence Agent."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from backend.predictor import predict_dataframe, predict_customer
    from backend.explainability import generate_customer_explanation
    from backend.advisor import generate_recommendations
    BACKEND_ERROR = None
except ImportError as exc:
    BACKEND_ERROR = str(exc)

st.set_page_config("Retention Intelligence", "🎯", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp{background:#f6f8fc}.block-container{max-width:1440px;padding-top:2rem;padding-bottom:3rem}
[data-testid='stSidebar']{background:linear-gradient(180deg,#102a43 0%,#0b4f6c 100%)}
[data-testid='stSidebar'] *{color:#f7fbff!important}[data-testid='stSidebar'] [data-baseweb='radio']{padding:4px 0}
[data-testid='stMetric']{background:#fff;border:1px solid #e8eef7;border-radius:16px;padding:18px;box-shadow:0 8px 20px rgba(28,52,84,.06)}
[data-testid='stMetricLabel']{font-size:.84rem;color:#61738a!important;font-weight:600}[data-testid='stMetricValue']{color:#102a43}
[data-testid='stDataFrame']{border:1px solid #e8eef7;border-radius:14px;overflow:hidden}.stButton>button{border-radius:10px;font-weight:650;padding:.55rem 1rem}
div[data-testid='stVerticalBlockBorderWrapper']{border-radius:16px;border-color:#e4ebf4;background:#fff;box-shadow:0 6px 18px rgba(28,52,84,.035)}
</style>""", unsafe_allow_html=True)
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#1261a0", "#1a8e99", "#f7b267", "#e76f51"]


def backend_ready() -> bool:
    if BACKEND_ERROR:
        st.error("Backend is not available yet. Add the supplied `backend/` package to enable this action.")
        st.caption(BACKEND_ERROR)
        return False
    return True


def frame(result: Any, source: pd.DataFrame | None = None) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame): out = result.copy()
    elif isinstance(result, dict): out = pd.DataFrame(result) if any(isinstance(v, (list, tuple)) for v in result.values()) else pd.DataFrame([result])
    else: out = pd.DataFrame(result)
    aliases = {"Customer": ["Customer", "customer", "customer_id", "CustomerID", "Customer ID", "name"], "Probability": ["Probability", "probability", "churn_probability", "risk_probability", "score"], "Risk Segment": ["Risk Segment", "risk_segment", "segment", "risk", "Risk"]}
    out = out.rename(columns={candidate: target for target, names in aliases.items() for candidate in names if candidate in out.columns})
    if "Customer" not in out: out.insert(0, "Customer", source.index.astype(str) if source is not None else out.index.astype(str))
    if "Probability" in out: out["Probability"] = pd.to_numeric(out["Probability"], errors="coerce")
    if "Risk Segment" not in out and "Probability" in out: out["Risk Segment"] = pd.cut(out.Probability, [-.01, .35, .65, 1], labels=["Low", "Medium", "High"]).astype(str)
    return out


def data() -> pd.DataFrame | None: return st.session_state.get("predictions")
def heading(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
    st.divider()


def dashboard() -> None:
    heading("Customer Retention Dashboard", "Monitor churn exposure and act on customers who need attention.")
    df = data()
    if df is None or df.empty: st.info("Upload a customer CSV on Predictions to populate this dashboard."); return
    risk = pd.to_numeric(df.get("Probability"), errors="coerce").fillna(0); segment = df.get("Risk Segment", pd.Series("Unknown", index=df.index)).astype(str); high = segment.str.lower().eq("high") | risk.ge(.65)
    monthly = pd.to_numeric(df.get("MonthlyCharges", df.get("Monthly Charges", 0)), errors="coerce").fillna(0)
    a,b,c,d = st.columns(4); a.metric("Total Customers", f"{len(df):,}"); b.metric("Average Risk", f"{risk.mean():.1%}"); c.metric("High Risk Customers", f"{high.sum():,}", f"{high.mean():.1%} of base", delta_color="inverse"); d.metric("Revenue At Risk", f"${monthly[high].sum():,.0f}")
    left,right = st.columns(2)
    with left.container(border=True):
        st.subheader("Risk distribution", divider="blue")
        st.plotly_chart(px.histogram(df, x="Probability", nbins=12, color_discrete_sequence=["#1261a0"]).update_layout(margin=dict(l=0,r=0,t=10,b=0), yaxis_title="Customers", xaxis_title="Churn probability"), use_container_width=True)
    segments = segment.value_counts().rename_axis("Risk Segment").reset_index(name="Customers")
    with right.container(border=True):
        st.subheader("Risk segmentation", divider="blue")
        st.plotly_chart(px.pie(segments, names="Risk Segment", values="Customers", hole=.6).update_traces(textinfo="percent+label").update_layout(margin=dict(l=0,r=0,t=10,b=0)), use_container_width=True)


def predictions() -> None:
    heading("Predictions", "Upload a customer file to identify risk across your customer base.")
    with st.container(border=True):
        st.subheader("Upload customer data")
        st.caption("Use a CSV with the same fields expected by your prediction backend.")
        upload = st.file_uploader("Customer CSV", type="csv", label_visibility="collapsed")
    if upload:
        try:
            source = pd.read_csv(upload); st.caption(f"Loaded {len(source):,} customers and {len(source.columns)} fields.")
            if st.button("Run risk prediction", type="primary") and backend_ready():
                with st.spinner("Scoring customers..."): st.session_state.predictions = frame(predict_dataframe(source), source)
                st.success("Predictions generated.")
        except Exception as exc: st.error(f"Could not process this CSV: {exc}")
    if data() is not None:
        df = data(); cols = [c for c in ["Customer", "Probability", "Risk Segment"] if c in df]
        st.subheader("Scored customers", divider="blue")
        st.dataframe(df[cols] if cols else df, hide_index=True, use_container_width=True, height=420)
        st.download_button("Download predictions CSV", df.to_csv(index=False).encode(), "customer_risk_predictions.csv", "text/csv", type="primary")


def risk_analysis() -> None:
    heading("Risk Analysis", "Understand priority customers and the drivers behind their risk.")
    df = data()
    if df is None or df.empty: st.info("Run predictions first to see customer-level analysis."); return
    ranked = df.sort_values("Probability", ascending=False) if "Probability" in df else df
    st.subheader("Top 10 High Risk Customers"); st.dataframe(ranked.head(10), hide_index=True, use_container_width=True)
    st.subheader("Feature Importance"); image = next((p for p in [Path(__file__).parent / "assets" / "feature_importance.png", ROOT / "feature_importance.png"] if p.exists()), None)
    st.image(str(image), use_container_width=True) if image else st.warning("Add `feature_importance.png` to `frontend/assets/` to display the chart.")
    customer = st.selectbox("Customer for explanation", ranked.Customer.astype(str).tolist())
    if st.button("Generate explanation") and backend_ready():
        try:
            record = ranked[ranked.Customer.astype(str).eq(customer)].iloc[0].to_dict()
            with st.spinner("Generating explanation..."): answer = generate_customer_explanation(record)
            st.info(str(answer))
        except Exception as exc: st.error(f"Explanation could not be generated: {exc}")


def advisor() -> None:
    heading("Retention Advisor", "Get focused, practical actions for each at-risk customer.")
    df = data()
    if df is None or df.empty: st.info("Run predictions first to use the advisor."); return
    customer = st.selectbox("Select customer", df.Customer.astype(str).tolist()); record = df[df.Customer.astype(str).eq(customer)].iloc[0]
    risk = float(record.get("Probability", 0)); st.progress(min(max(risk, 0), 1), text=f"Current predicted risk: {risk:.1%}")
    if risk >= .65: st.warning("High-risk customer: prioritise timely outreach and a tailored offer.")
    if st.button("Generate retention actions", type="primary") and backend_ready():
        try:
            with st.spinner("Creating recommendations..."): actions = generate_recommendations(record.to_dict())
            if isinstance(actions, (list, tuple)):
                for action in actions: st.markdown(f"- {action}")
            else: st.write(actions)
        except Exception as exc: st.error(f"Recommendations could not be generated: {exc}")


def simulator() -> None:
    heading("What-if Simulator", "Estimate how a proposed plan or payment change could improve retention risk.")
    df = data(); options = df.Customer.astype(str).tolist() if df is not None and "Customer" in df else ["New customer"]
    customer = st.selectbox("Customer", options); base = df[df.Customer.astype(str).eq(customer)].iloc[0].to_dict() if df is not None and customer != "New customer" else {}
    a,b = st.columns(2)
    with a: contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]); tenure = st.number_input("Tenure (months)", min_value=0, value=int(base.get("tenure", 12)))
    with b: charge = st.number_input("Monthly Charges", min_value=0., value=float(base.get("MonthlyCharges", 70.))); payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    if st.button("Simulate risk", type="primary") and backend_ready():
        try:
            with st.spinner("Simulating..."): raw = predict_customer({**base, "Contract":contract, "tenure":tenure, "MonthlyCharges":charge, "PaymentMethod":payment})
            modified = float(raw.get("Probability", raw.get("probability", raw.get("risk", 0))) if isinstance(raw, dict) else raw); current = float(base.get("Probability", base.get("probability", 0)))
            x,y,z = st.columns(3); x.metric("Current Risk", f"{current:.1%}"); y.metric("Modified Risk", f"{modified:.1%}"); z.metric("Risk Reduction", f"{current-modified:.1%}"); st.progress(min(max(modified, 0), 1), text="Modified customer risk")
            if modified >= .65: st.warning("The modified scenario is still high risk. Consider a longer contract or proactive offer.")
        except Exception as exc: st.error(f"Simulation could not be completed: {exc}")


with st.sidebar:
    st.title("🎯 Retention IQ"); st.caption("CUSTOMER RETENTION INTELLIGENCE")
    st.divider()
    page = st.radio("Navigation", ["Dashboard", "Predictions", "Risk Analysis", "Retention Advisor", "What-if Simulator"])
    st.divider()
    if data() is not None: st.success("Prediction data loaded")
    else: st.caption("No prediction data loaded")
{"Dashboard":dashboard, "Predictions":predictions, "Risk Analysis":risk_analysis, "Retention Advisor":advisor, "What-if Simulator":simulator}[page]()
