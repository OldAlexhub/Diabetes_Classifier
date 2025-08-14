import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import plotly.express as px
import plotly.graph_objects as go

# ------------------------------- Page & Theme --------------------------------
st.set_page_config(
    page_title="Diabetes Risk Tool",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
:root{
  --bg: #f6f9fc;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #334155;
  --border: #e5e7eb;
  --accent: #1976d2;
  --accent-weak: #e3f2fd;
  --ok: #16a34a;
  --warn: #f59e0b;
  --risk: #dc2626;
}

/* Background */
[data-testid="stAppViewContainer"] { background: var(--bg); }

/* Container spacing */
.block-container { padding-top: 1.0rem; }

/* Card */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: 0 8px 20px rgba(2,6,23,0.06);
  padding: 1.0rem 1.1rem 0.8rem 1.1rem;
}

/* Typography */
h1, h2, h3, h4 { color: var(--text); letter-spacing: .2px; }
p, li, label, span, input, .stMarkdown { color: var(--muted) !important; }
.kpi { font-weight: 700; color: var(--text); font-size: 1.1rem; }

/* Inputs */
input, select, textarea, .stNumberInput input, .stTextInput input {
  background-color: #ffffff !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  height: 42px !important;
}
input:focus, select:focus, textarea:focus,
.stNumberInput input:focus, .stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(25,118,210,0.15);
  outline: none;
}

/* Buttons */
.stButton > button {
  background: var(--accent);
  color: #fff;
  border: 1px solid #1565c0;
  border-radius: 10px;
  height: 42px;
}
.stButton > button:hover { filter: brightness(0.95); }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--muted) !important; }
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
  background: #fff !important; color: var(--text) !important; border: 1px solid var(--border) !important; border-radius: 10px !important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div:nth-child(2) { background: var(--accent) !important; }
section[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] { background: var(--accent) !important; }

/* Pills */
.pill {
  display:inline-block; padding:6px 10px; border-radius:9999px;
  background: var(--accent-weak); color: var(--text); border: 1px solid #cfe3fa; font-size:.9rem;
}

/* Risk banner */
.banner {
  border-radius: 12px; padding: 12px 14px; border: 1px solid var(--border);
}
.banner.ok   { background: #ecfdf5; border-color: #bbf7d0; }
.banner.warn { background: #fffbeb; border-color: #fde68a; }
.banner.risk { background: #fef2f2; border-color: #fecaca; }
.banner h3 { margin: 0 0 4px 0; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------ Data Loading ---------------------------------
@st.cache_data
def load_data(csv_name="Multiclass Diabetes Dataset.csv"):
    if not os.path.exists(csv_name):
        raise FileNotFoundError(f'Could not find "{csv_name}" in the current folder.')
    df = pd.read_csv(csv_name)
    # Normalize "Class" name
    for c in df.columns:
        if c.strip().lower() == "class":
            if c != "Class":
                df = df.rename(columns={c: "Class"})
            break
    if "Class" not in df.columns:
        raise ValueError('Dataset must include a "Class" column with 0 and 1 labels. Rows with 2 will be dropped if present.')
    if 2 in df["Class"].unique():
        df = df[df["Class"] != 2].copy()
    df["Class"] = df["Class"].astype(int)
    return df

try:
    df = load_data("Multiclass Diabetes Dataset.csv")
except Exception as e:
    st.error(str(e))
    st.stop()

# Numeric features only for modeling
num_cols = [c for c in df.columns if c != "Class" and np.issubdtype(df[c].dtype, np.number)]
if not num_cols:
    st.error("No numeric feature columns found besides Class.")
    st.stop()

# ------------------------------- Sidebar -------------------------------------
with st.sidebar:
    st.header("Configuration")
    algo = st.selectbox("Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
    threshold = st.slider("Decision threshold", 0.20, 0.80, 0.50, 0.01)
    test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

    if algo == "Logistic Regression":
        C = st.slider("C (inverse regularization strength)", 0.01, 5.0, 1.0, 0.01)
    elif algo == "Random Forest":
        n_estimators = st.slider("Trees", 100, 1000, 300, 50)
        max_depth = st.selectbox("Max depth", [None, 3, 5, 8, 12, 16], index=0)
    else:
        gb_lr = st.slider("Learning rate", 0.01, 0.5, 0.05, 0.01)
        gb_estimators = st.slider("Estimators", 100, 1000, 300, 50)

# ------------------------------ Header & Summary ------------------------------
st.title("Diabetes Risk Tool")
st.caption("For education and decision support only. Not a diagnostic device.")

top = st.columns([1.2, 1.8])
with top[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dataset snapshot")
    cls_counts = df["Class"].value_counts().rename({0: "No diabetes", 1: "Diabetic"})
    pie = px.pie(values=cls_counts.values, names=cls_counts.index, hole=0.5)
    pie.update_traces(textinfo="percent+label")
    st.plotly_chart(pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with top[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feature summary")
    # Show correlations to outcome for first 12 numeric columns
    view_cols = num_cols[:12]
    corr = df[view_cols + ["Class"]].corr()
    heat = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    heat.update_layout(height=400)
    st.plotly_chart(heat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ Train & Evaluate ------------------------------
X = df[num_cols].copy()
y = df["Class"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_state
)

def build_pipeline(name):
    if name == "Logistic Regression":
        model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
        scaler = StandardScaler()
        pre = ColumnTransformer([("num", scaler, list(range(X_train.shape[1])))], remainder="drop")
        return Pipeline([("pre", pre), ("clf", model)])
    if name == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        pre = ColumnTransformer([("num", "passthrough", list(range(X_train.shape[1])))], remainder="drop")
        return Pipeline([("pre", pre), ("clf", model)])
    model = GradientBoostingClassifier(learning_rate=gb_lr, n_estimators=gb_estimators, random_state=42)
    pre = ColumnTransformer([("num", "passthrough", list(range(X_train.shape[1])))], remainder="drop")
    return Pipeline([("pre", pre), ("clf", model)])

pipe = build_pipeline(algo)
pipe.fit(X_train, y_train)

def proba_and_pred(model, X, thr):
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    if proba is None:
        # approximate probability using decision_function range
        dfc = model.decision_function(X)
        proba = (dfc - dfc.min()) / (dfc.max() - dfc.min() + 1e-9)
    pred = (proba >= thr).astype(int)
    return proba, pred

prob_tr, pred_tr = proba_and_pred(pipe, X_train, threshold)
prob_te, pred_te = proba_and_pred(pipe, X_test, threshold)

def metrics_block(y_true, pred, prob):
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan
    return acc, prec, rec, f1, auc

acc, prec, rec, f1, auc = metrics_block(y_test, pred_te, prob_te)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model performance")

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(f"<div class='kpi'>Accuracy<br>{acc:.3f}</div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi'>Precision<br>{prec:.3f}</div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi'>Recall<br>{rec:.3f}</div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi'>F1<br>{f1:.3f}</div>", unsafe_allow_html=True)
k5.markdown(f"<div class='kpi'>ROC AUC<br>{auc:.3f}</div>", unsafe_allow_html=True)

cm = confusion_matrix(y_test, pred_te, labels=[0,1])
cm_df = pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"])
cm_fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto", title="Confusion matrix")
st.plotly_chart(cm_fig, use_container_width=True)

# ROC
fpr, tpr, _ = roc_curve(y_test, prob_te)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
roc_fig.update_layout(title=f"ROC Curve (AUC={auc:.3f})", xaxis_title="False positive rate", yaxis_title="True positive rate", height=360)
st.plotly_chart(roc_fig, use_container_width=True)

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, prob_te, n_bins=10, strategy="uniform")
cal = go.Figure()
cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Calibration"))
cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Ideal", line=dict(dash="dash")))
cal.update_layout(title="Calibration curve", xaxis_title="Predicted probability", yaxis_title="Observed frequency", height=360)
st.plotly_chart(cal, use_container_width=True)

st.caption("Adjust the decision threshold in the sidebar to balance false negatives vs false positives.")
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ Patient Entry --------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Patient risk check")

with st.form("patient_form"):
    cols = st.columns(3)
    stats = df[num_cols].describe()
    patient = {}
    for i, col in enumerate(num_cols):
        c = cols[i % 3]
        vmin = float(np.floor(stats.loc["min", col]))
        vmax = float(np.ceil(stats.loc["max", col]))
        vmean = float(stats.loc["mean", col])
        step = max(0.1, round((vmax - vmin) / 200, 2))
        patient[col] = c.number_input(col, min_value=vmin, max_value=vmax, value=float(np.round(vmean, 2)), step=step)
    submit = st.form_submit_button("Run assessment")

if submit:
    row = pd.DataFrame([patient])[num_cols]
    p, yhat = proba_and_pred(pipe, row, threshold)
    prob = float(p[0])
    pred = int(yhat[0])

    # Risk band message
    if prob < 0.25:
        klass = "ok"; title = "Low risk estimate"
        lines = [
            "Maintain healthy diet patterns with adequate fiber",
            "Aim for regular physical activity if your clinician approves",
            "Continue regular checkups and monitoring"
        ]
    elif prob < 0.5:
        klass = "warn"; title = "Moderate risk estimate"
        lines = [
            "Consider a clinician visit for baseline labs such as A1C or fasting glucose",
            "Review nutrition and activity plan",
            "Track weight, waist size, and blood pressure"
        ]
    else:
        klass = "risk"; title = "High risk estimate"
        lines = [
            "Schedule a clinician visit soon for diagnostic testing, such as A1C",
            "Discuss treatment and lifestyle options",
            "Consider nutrition counseling and a structured activity plan"
        ]

    st.markdown(f"<div class='banner {klass}'><h3>{title}</h3>"
                f"<p>Predicted class: <b>{pred}</b> (0 no diabetes, 1 diabetic)</p>"
                f"<p>Estimated probability: <b>{prob:.3f}</b> at threshold <b>{threshold:.2f}</b></p></div>",
                unsafe_allow_html=True)
    st.markdown("**Suggested next steps**")
    st.markdown("\n".join([f"- {x}" for x in lines]))

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ Feature Effects ------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Feature signals")

# Simple coefficient or importance view
model = pipe.named_steps["clf"]
names = num_cols
imp_df = None
if hasattr(model, "coef_"):
    vals = model.coef_.ravel()
    imp_df = pd.DataFrame({"feature": names, "signal": vals}).sort_values("signal", ascending=False)
    fig = px.bar(imp_df, x="signal", y="feature", orientation="h", title="Logistic regression coefficients")
    st.plotly_chart(fig, use_container_width=True)
elif hasattr(model, "feature_importances_"):
    vals = model.feature_importances_
    imp_df = pd.DataFrame({"feature": names, "importance": vals}).sort_values("importance", ascending=False)
    fig = px.bar(imp_df, x="importance", y="feature", orientation="h", title=f"{algo} feature importance")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Feature importance not available for this model.")

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ Data Preview ---------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Data preview")
st.dataframe(df.head(30), use_container_width=True)
st.caption("Target column: Class. Values 0 and 1 are used. Any 2s are removed.")
st.markdown('</div>', unsafe_allow_html=True)
