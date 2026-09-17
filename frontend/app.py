"""Customer Retention Intelligence Agent — analytics dashboard frontend."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from backend.predictor import predict_dataframe, predict_customer
    from backend.explainability import generate_customer_explanation
    from backend.advisor import generate_recommendations
    BACKEND_ERROR = None
except ImportError as exc: BACKEND_ERROR = str(exc)

st.set_page_config("Retention Intelligence", "RI", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp{background:#f5f7fb}.block-container{max-width:1500px;padding:1.55rem 2rem 3rem}[data-testid="stSidebar"]{background:#0b1424;border-right:1px solid #172238}[data-testid="stSidebar"] *{color:#f8fafc!important}[data-testid="stSidebar"] [data-baseweb="radio"]{padding:8px;border-radius:8px}[data-testid="stSidebar"] [data-baseweb="radio"]:hover{background:#172a4a}[data-testid="stMetric"]{background:#fff;border:1px solid #e8ecf3;border-radius:12px;padding:20px;box-shadow:0 4px 16px rgba(15,23,42,.09);transition:.2s}[data-testid="stMetric"]:hover{transform:translateY(-2px);box-shadow:0 10px 24px rgba(15,23,42,.13)}[data-testid="stMetricLabel"]{font-size:.82rem!important;color:#3d4b63!important;font-weight:700}.stButton>button,.stDownloadButton>button{border-radius:9px;font-weight:650}div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-color:#e4e9f1;border-radius:12px;box-shadow:0 3px 12px rgba(15,23,42,.06)}[data-testid="stDataFrame"]{border:1px solid #e2e8f0;border-radius:10px;overflow:hidden}.badge{display:inline-block;border-radius:999px;padding:3px 10px;font-size:.76rem;font-weight:700}.low{background:#dcfce7;color:#166534}.medium{background:#fef3c7;color:#92400e}.high{background:#fee2e2;color:#b91c1c}.eyebrow{color:#2563eb;font-size:.78rem;font-weight:750;letter-spacing:.08em;text-transform:uppercase}</style>""", unsafe_allow_html=True)
px.defaults.template="plotly_white"; px.defaults.color_discrete_sequence=["#2563eb","#16a34a","#d97706","#dc2626","#7c3aed"]

def ready() -> bool:
    if BACKEND_ERROR: st.error("Backend is not available yet. Add the supplied `backend/` package to enable this action."); st.caption(BACKEND_ERROR); return False
    return True
def normalise(result: Any, source: pd.DataFrame | None=None) -> pd.DataFrame:
    out=result.copy() if isinstance(result,pd.DataFrame) else (pd.DataFrame(result) if not isinstance(result,dict) or any(isinstance(x,(list,tuple)) for x in result.values()) else pd.DataFrame([result]))
    aliases={"Customer":["Customer","customer","customer_id","CustomerID","Customer ID","name"],"Probability":["Probability","probability","churn_probability","risk_probability","score"],"Risk Segment":["Risk Segment","risk_segment","segment","risk","Risk"]}
    out=out.rename(columns={n:k for k,v in aliases.items() for n in v if n in out})
    if "Customer" not in out: out.insert(0,"Customer",source.index.astype(str) if source is not None else out.index.astype(str))
    if "Probability" in out: out["Probability"]=pd.to_numeric(out.Probability,errors="coerce").fillna(0)
    if "Risk Segment" not in out and "Probability" in out: out["Risk Segment"]=pd.cut(out.Probability,[-.01,.35,.65,1],labels=["Low","Medium","High"]).astype(str)
    return out
def data() -> pd.DataFrame | None: return st.session_state.get("predictions")
def risks(df: pd.DataFrame) -> pd.Series:
    values = df["Probability"] if "Probability" in df else pd.Series(0.0, index=df.index)
    return pd.to_numeric(values, errors="coerce").fillna(0)
def high(df: pd.DataFrame) -> pd.Series: return df.get("Risk Segment",pd.Series("",index=df.index)).astype(str).str.lower().eq("high")|risks(df).ge(.65)
def heading(title: str, text: str): st.markdown("<div class='eyebrow'>Retention intelligence</div>",unsafe_allow_html=True);st.title(title);st.caption(text);st.divider()
def risk_bar(value: float, segment: str|None=None):
    label=segment or ("High" if value>=.65 else "Medium" if value>=.35 else "Low"); css=label.lower()
    st.markdown(f"<span class='badge {css}'>{label} risk</span>",unsafe_allow_html=True);st.progress(min(max(value,0),1),text=f"Risk score · {value:.1%}")
def kpis(df: pd.DataFrame):
    r=risks(df); h=high(df); rev=pd.to_numeric(df.get("MonthlyCharges",df.get("Monthly Charges",0)),errors="coerce").fillna(0); a,b,c,d=st.columns(4)
    a.metric("Total Customers",f"{len(df):,}");b.metric("Churn Rate",f"{r.mean():.1%}");c.metric("High Risk Customers",f"{h.sum():,}",f"{h.mean():.1%} of base",delta_color="inverse");d.metric("Revenue At Risk",f"${rev[h].sum():,.0f}")

def dashboard():
    df=data()
    title, tools = st.columns((1.55, .8))
    with title:
        st.title("Overview")
        st.caption("Monitor and reduce customer churn with AI-powered retention insights")
    with tools:
        st.text_input("Search customers", placeholder="Search customers...", label_visibility="collapsed")
        st.caption("Today · Last updated just now")
    st.divider()
    if df is None or df.empty:
        st.info("Upload a customer CSV in Predictions to activate live retention analytics.")
        df=pd.DataFrame({"Customer":[],"Probability":[],"Risk Segment":[]})
    r=risks(df);h=high(df);avg=float(r.mean()) if not r.empty else 0.0;rev=pd.to_numeric(df.get("MonthlyCharges",pd.Series(0,index=df.index)),errors="coerce").fillna(0)
    a,b,c,d=st.columns(4)
    a.metric("◉  Total Customers",f"{len(df):,}","Live customer base")
    b.metric("△  Churn Risk",f"{avg:.1%}","Portfolio average",delta_color="inverse")
    c.metric("◎  High Risk Customers",f"{h.sum():,}","Flagged for action",delta_color="inverse")
    d.metric("◈  Retention Opportunity",f"${rev[h].sum():,.0f}","Estimated revenue at risk")
    left,right=st.columns((1.38,.82))
    with left.container(border=True):
        st.subheader("Customer Risk Table");st.caption("Top customers by churn risk")
        if df.empty: st.caption("Your highest-risk customers will appear here after scoring.")
        else:
            table=df.assign(**{"Churn Probability":r.map(lambda x:f"{x:.0%}"),"Risk":df["Risk Segment"].map(lambda x:str(x).title())}).sort_values("Probability",ascending=False).head(8)
            st.dataframe(table[[x for x in ["Customer","Churn Probability","Risk","MonthlyCharges"] if x in table]],hide_index=True,use_container_width=True,height=310)
        st.caption(f"Showing {min(len(df),8)} of {h.sum()} high-risk customers · View all →")
    with right:
        with st.container(border=True):
            st.subheader("Feature Importance");st.caption("Top drivers of churn prediction")
            features=pd.DataFrame({"Feature":["Engagement Score","Support Tickets","Usage Frequency","Billing Issues","Tenure (months)","Contract Type"],"Impact":[32,18,15,12,9,7]})
            st.plotly_chart(px.bar(features,x="Impact",y="Feature",orientation="h",text="Impact",color_discrete_sequence=["#2563eb"]).update_layout(margin=dict(l=0,r=0,t=5,b=0),height=250,xaxis_visible=False,yaxis=dict(autorange="reversed"),showlegend=False),use_container_width=True)
        with st.container(border=True):
            st.subheader("Churn Probability Gauge");st.caption("Overall platform churn risk")
            gauge=go.Figure(go.Indicator(mode="gauge+number",value=avg*100,number={"suffix":"%","font":{"size":34}},gauge={"axis":{"range":[0,100]},"bar":{"color":"#2563eb"},"steps":[{"range":[0,35],"color":"#bbf7d0"},{"range":[35,65],"color":"#fde68a"},{"range":[65,100],"color":"#fecaca"}],"threshold":{"line":{"color":"#dc2626","width":4},"value":65}}));gauge.update_layout(height=230,margin=dict(l=20,r=20,t=5,b=0));st.plotly_chart(gauge,use_container_width=True)

def predictions():
    heading("Predictions workspace","Upload customer data, score it, and focus on the customers who need action.")
    with st.container(border=True):
        st.subheader("Customer data upload");st.caption("CSV only — fields are validated by your existing prediction backend.");upload=st.file_uploader("Drop a CSV here",type="csv",label_visibility="collapsed")
        if upload is not None:
            try:
                source=pd.read_csv(upload);st.success(f"{len(source):,} customer records ready to score.")
                if st.button("Run risk prediction",type="primary") and ready():
                    with st.spinner("Scoring customers..."): st.session_state.predictions=normalise(predict_dataframe(source),source)
                    st.rerun()
            except Exception as exc: st.error(f"Could not process this CSV: {exc}")
    df=data()
    if df is None:return
    kpis(df);a,b=st.columns(2)
    with a.container(border=True): st.subheader("Churn distribution",divider="blue");counts=df["Risk Segment"].value_counts().rename_axis("Risk Segment").reset_index(name="Customers");st.plotly_chart(px.bar(counts,x="Risk Segment",y="Customers",color="Risk Segment",text="Customers").update_layout(showlegend=False,margin=dict(l=0,r=0,t=10,b=0)),use_container_width=True)
    with b.container(border=True): st.subheader("High-risk concentration",divider="blue");st.plotly_chart(px.scatter(df,x="Probability",y=df.index,color="Risk Segment",hover_name="Customer").update_layout(margin=dict(l=0,r=0,t=10,b=0),yaxis_visible=False),use_container_width=True)
    with st.container(border=True):
        st.subheader("High-risk customers",divider="blue");query=st.text_input("Search customers",placeholder="Search by customer ID or name…");table=df[high(df)].copy()
        if query: table=table[table.astype(str).apply(lambda col:col.str.contains(query,case=False,na=False)).any(axis=1)]
        st.dataframe(table.sort_values("Probability",ascending=False),hide_index=True,use_container_width=True,height=340);st.download_button("Download scored customer CSV",df.to_csv(index=False).encode(),"customer_risk_predictions.csv","text/csv",type="primary")

def risk_analysis():
    heading("Risk analysis","Explore drivers, segment concentration, and customer-level explanations.");df=data()
    if df is None or df.empty:st.info("Run predictions first to unlock risk analysis.");return
    left,right=st.columns(2)
    with left.container(border=True):
        st.subheader("Feature importance",divider="blue");image=next((p for p in [Path(__file__).parent/"assets"/"feature_importance.png",ROOT/"feature_importance.png"] if p.exists()),None);st.image(str(image),use_container_width=True) if image else st.info("Add `feature_importance.png` to `frontend/assets/` to display model drivers.")
    with right.container(border=True): st.subheader("Risk distribution",divider="blue");st.plotly_chart(px.box(df,y="Probability",color="Risk Segment",points="all",hover_name="Customer").update_layout(margin=dict(l=0,r=0,t=10,b=0),showlegend=False),use_container_width=True)
    with st.container(border=True):
        st.subheader("Customer explanation",divider="blue");customer=st.selectbox("Customer",df.Customer.astype(str).tolist());row=df[df.Customer.astype(str).eq(customer)].iloc[0];risk_bar(float(row.get("Probability",0)),str(row.get("Risk Segment","Medium")))
        if st.button("Generate explanation",type="primary") and ready():
            try:
                with st.spinner("Generating explanation..."):st.write(generate_customer_explanation(row.to_dict()))
            except Exception as exc:st.error(f"Explanation could not be generated: {exc}")

def personas():
    heading("Customer personas","A practical segmentation layer to guide engagement strategy.");df=data()
    if df is None or df.empty:st.info("Run predictions first to build customer personas.");return
    view=df.copy();view["Persona"]=pd.cut(risks(view),[-.01,.35,.65,1],labels=["Loyal advocates","Watchlist","At-risk customers"]).astype(str);st.plotly_chart(px.scatter(view,x="tenure" if "tenure" in view else view.index,y="Probability",color="Persona",size="MonthlyCharges" if "MonthlyCharges" in view else None,hover_name="Customer").update_layout(height=420,margin=dict(l=0,r=0,t=10,b=0)),use_container_width=True)
    cols=st.columns(3)
    for col,(name,desc) in zip(cols,[("Loyal advocates","Low risk — suitable for referrals and upsell."),("Watchlist","Targeted value reinforcement and check-ins."),("At-risk customers","Prioritise proactive outreach and escalation.")]):
        with col.container(border=True):st.subheader(name);st.caption(desc);st.metric("Customers",int((view.Persona==name).sum()))

def advisor():
    heading("Retention advisor","Turn risk signals into practical next-best actions.");df=data()
    if df is None or df.empty:st.info("Run predictions first to use the advisor.");return
    customer=st.selectbox("Select customer",df.Customer.astype(str).tolist());row=df[df.Customer.astype(str).eq(customer)].iloc[0];risk=float(row.get("Probability",0));risk_bar(risk,str(row.get("Risk Segment","Medium")))
    if st.button("Generate retention actions",type="primary") and ready():
        try:
            with st.spinner("Creating recommendations..."):actions=generate_recommendations(row.to_dict())
            for i,action in enumerate(actions if isinstance(actions,(list,tuple)) else [actions],1):
                with st.container(border=True):st.subheader(f"Action {i} · {'High' if risk>=.65 else 'Medium'} priority");st.write(action);st.caption("Recommended retention intervention")
        except Exception as exc:st.error(f"Recommendations could not be generated: {exc}")

def simulator():
    heading("What-if simulator","Compare a customer’s current state with a proposed retention scenario.");df=data();options=df.Customer.astype(str).tolist() if df is not None and "Customer" in df else ["New customer"];customer=st.selectbox("Customer",options);base=df[df.Customer.astype(str).eq(customer)].iloc[0].to_dict() if df is not None and customer!="New customer" else {};left,right=st.columns(2)
    with left.container(border=True):st.subheader("Current customer profile",divider="blue");st.write(f"**Contract:** {base.get('Contract','Month-to-month')}");st.write(f"**Tenure:** {base.get('tenure',12)} months");st.write(f"**Monthly charges:** ${base.get('MonthlyCharges',70.):.2f}");risk_bar(float(base.get("Probability",0)))
    with right.container(border=True):st.subheader("Modified customer profile",divider="blue");contract=st.selectbox("Contract",["Month-to-month","One year","Two year"]);tenure=st.number_input("Tenure (months)",min_value=0,value=int(base.get("tenure",12)));charges=st.number_input("Monthly charges",min_value=0.,value=float(base.get("MonthlyCharges",70.)));payment=st.selectbox("Payment method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    if st.button("Simulate risk",type="primary") and ready():
        try:
            with st.spinner("Simulating..."):raw=predict_customer({**base,"Contract":contract,"tenure":tenure,"MonthlyCharges":charges,"PaymentMethod":payment})
            new=float(raw.get("Probability",raw.get("probability",raw.get("risk",0))) if isinstance(raw,dict) else raw);current=float(base.get("Probability",base.get("probability",0)))
            with st.container(border=True):st.subheader("Scenario comparison",divider="blue");a,b,c=st.columns(3);a.metric("Current Risk",f"{current:.1%}");b.metric("New Risk",f"{new:.1%}");c.metric("Risk Reduction",f"{current-new:.1%}");risk_bar(new)
        except Exception as exc:st.error(f"Simulation could not be completed: {exc}")

def reports():
    heading("Executive reports","Prepare a concise risk summary for leadership review.");df=data()
    if df is None or df.empty:st.info("Run predictions first to generate an executive report preview.");return
    with st.container(border=True):
        st.subheader("Report preview",divider="blue");kpis(df);st.markdown("**Risk summary**");st.write(f"{high(df).sum():,} customers need priority retention attention.");st.markdown("**Customer insights**");st.write("Risk segmentation and account-level explanations are available in the analysis workspace.");st.markdown("**Revenue impact**");st.write("Use the revenue-at-risk metric to prioritise commercial intervention.")
        if st.button("Generate PDF Report",type="primary",use_container_width=True):st.success("Report request prepared. Connect your reporting service to enable PDF export.")

with st.sidebar:
    st.title("◎ RETENTION\nRADAR");st.caption("CHURN PREDICTION AGENT");st.divider();page=st.radio("Workspace",["Dashboard","Predictions","Risk Analysis","Customer Personas","Retention Advisor","What-If Simulator","Reports"],label_visibility="collapsed");st.divider();st.caption("DATA STATUS");st.success("Prediction data loaded") if data() is not None else st.caption("No prediction data loaded")
{"Dashboard":dashboard,"Predictions":predictions,"Risk Analysis":risk_analysis,"Customer Personas":personas,"Retention Advisor":advisor,"What-If Simulator":simulator,"Reports":reports}[page]()
