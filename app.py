# app.py
# Streamlit Diabetes Classifier for your single dataset:
# - Loads "Multiclass Diabetes Dataset.csv" from the current folder
# - Binary target: 0 = no diabetes, 1 = diabetic
# - If Class == 2 exists, it will be dropped
# - Modern glassy UI, interactive training, metrics, visuals, and practical advice

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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Page config and style
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Diabetes Classifier", page_icon="🩺", layout="wide")

st.markdown("""
<style>
:root {
  --card-bg: rgba(255,255,255,0.06);
  --border: 1px solid rgba(255,255,255,0.15);
}

/* Main app background */
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 10% 10%, #2a2a72 0%, #009ffd 35%, #0b1b2b 100%);
}
.block-container { padding-top: 1.5rem; }

/* Glass card style */
.glass {
  background: var(--card-bg);
  border: var(--border);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  border-radius: 24px;
  padding: 1.1rem 1.2rem 0.8rem 1.2rem;
  backdrop-filter: blur(8px);
}

/* Text colors */
h1, h2, h3, h4 { color: #f7fafc; }
p, li, label, span, input, .stMarkdown { color: #f1f5f9 !important; }

/* Badges and metrics */
.metric-chip {
  display: inline-block;
  padding: 8px 12px;
  border-radius: 12px;
  margin: 4px 8px 8px 0;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.15);
}
.badge {
  display:inline-block; padding:6px 10px; border-radius:10px;
  border:1px solid rgba(255,255,255,0.2); background:rgba(255,255,255,0.06);
  margin-left:8px; font-size:0.9rem;
}
.small-note { font-size: 0.9rem; opacity: 0.95; }
hr { border: none; height: 1px; background: rgba(255,255,255,0.15); margin: 0.8rem 0; }

/* Inputs, selects, textareas */
input, select, textarea, .stNumberInput input, .stTextInput input {
  background-color: rgba(20, 20, 20, 0.85) !important;
  color: #f1f5f9 !important;
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  border-radius: 8px !important;
}

/* Closed dropdown */
.stSelectbox div[data-baseweb="select"] > div {
  background-color: rgba(20, 20, 20, 0.85) !important;
  color: #f1f5f9 !important;
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  border-radius: 8px !important;
}

/* Dropdown menu list */
.stSelectbox div[data-baseweb="select"] ul {
  background-color: rgba(20, 20, 20, 0.95) !important;
  color: #f1f5f9 !important;
  border-radius: 8px !important;
}

/* Dropdown menu options */
.stSelectbox div[data-baseweb="select"] li {
  background-color: transparent !important;
  color: #f1f5f9 !important;
}
.stSelectbox div[data-baseweb="select"] li:hover {
  background-color: rgba(255, 255, 255, 0.15) !important;
}

/* File uploader box */
.stFileUploader div[data-testid="stFileUploaderDropzone"] {
  background-color: rgba(20, 20, 20, 0.85) !important;
  color: #f1f5f9 !important;
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  border-radius: 8px !important;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
  background: rgba(12, 20, 28, 0.82) !important;
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(255,255,255,0.12);
}
section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }

/* Sidebar inputs */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stTextInput input {
  background-color: rgba(20, 20, 20, 0.9) !important;
  color: #f1f5f9 !important;
  border: 1px solid rgba(255, 255, 255, 0.18) !important;
  border-radius: 10px !important;
}

/* Sidebar dropdown closed */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
  background-color: rgba(20, 20, 20, 0.9) !important;
  color: #f1f5f9 !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 10px !important;
}

/* Sidebar dropdown menu */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] ul {
  background-color: rgba(15, 15, 15, 0.98) !important;
  color: #f1f5f9 !important;
  border-radius: 10px !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] li {
  background: transparent !important; color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] li:hover {
  background: rgba(255,255,255,0.12) !important;
}

/* Sidebar sliders */
section[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
  background: #00b3ff !important;
  border: 2px solid rgba(255,255,255,0.25) !important;
  box-shadow: 0 0 0 4px rgba(0,179,255,0.15);
}
section[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div:nth-child(2) {
  background: rgba(255,255,255,0.2) !important; /* track filled */
}
section[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div:nth-child(1) {
  background: rgba(255,255,255,0.12) !important; /* track background */
}

/* Sidebar file uploader */
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {
  background-color: rgba(20, 20, 20, 0.9) !important;
  color: #f1f5f9 !important;
  border: 1px solid rgba(255, 255, 255, 0.18) !important;
  border-radius: 12px !important;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] button[kind="secondary"] {
  background: rgba(255,255,255,0.1) !important;
  color: #f7fafc !important;
  border: 1px solid rgba(255,255,255,0.2) !important;
  border-radius: 12px !important;
}

/* Sidebar radios */
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 10px;
  padding: 6px 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(csv_name="Multiclass Diabetes Dataset.csv"):
    if not os.path.exists(csv_name):
        raise FileNotFoundError(f'Could not find "{csv_name}" in the current folder.')
    df = pd.read_csv(csv_name)
    # Normalize target column name to "Class" if needed
    rename_map = {c: "Class" for c in df.columns if c.strip().lower() == "class"}
    if rename_map:
        df = df.rename(columns=rename_map)
    if "Class" not in df.columns:
        raise ValueError('Dataset must have a "Class" column with values 0 and 1. Rows where Class == 2 will be dropped if present.')
    # Drop any multiclass value 2 if present
    if 2 in df["Class"].unique():
        df = df[df["Class"] != 2].copy()
    # Enforce int labels
    df["Class"] = df["Class"].astype(int)
    return df

try:
    df = load_data("Multiclass Diabetes Dataset.csv")
except Exception as e:
    st.error(str(e))
    st.stop()

# Keep only numeric features for modeling
num_cols = [c for c in df.columns if c != "Class" and np.issubdtype(df[c].dtype, np.number)]
if not num_cols:
    st.error("No numeric feature columns found. Add numeric predictors besides the Class column.")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🩺 Diabetes Classifier")
    st.caption("Binary: 0 = no diabetes, 1 = diabetic")

    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)

    model_name = st.selectbox("Algorithm", ["Logistic Regression", "Random Forest", "Gradient Boosting"], index=0)

    if model_name == "Logistic Regression":
        C = st.slider("C (inverse regularization)", 0.01, 5.0, 1.0, 0.01)
        penalty = st.selectbox("Penalty", ["l2"], index=0)
    elif model_name == "Random Forest":
        rf_estimators = st.slider("Trees", 100, 1000, 300, 50)
        rf_depth = st.selectbox("Max depth", [None, 3, 5, 8, 12, 16], index=0)
    else:
        gb_learn_rate = st.slider("Learning rate", 0.01, 0.5, 0.05, 0.01)
        gb_estimators = st.slider("Estimators", 100, 1000, 300, 50)

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.title("Diabetes Risk Classifier")
st.markdown(
    """
    <div class="small-note">
    Source: <b>Multiclass Diabetes Dataset.csv</b>
    <span class="badge">Binary classification</span>
    <span class="badge">0 = no diabetes</span>
    <span class="badge">1 = diabetic</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Quick class balance
cls_counts = df["Class"].value_counts().rename({0: "No Diabetes", 1: "Diabetic"})
col_a, col_b = st.columns([1.1, 1.9], gap="large")
with col_a:
    st.subheader("Class balance")
    pie = px.pie(values=cls_counts.values, names=cls_counts.index, hole=0.45)
    pie.update_traces(textinfo="percent+label")
    st.plotly_chart(pie, use_container_width=True)

with col_b:
    st.subheader("Feature overview")
    sample_df = df.sample(min(len(df), 600), random_state=random_state)
    # Correlation heatmap for up to 12 numeric cols
    view_cols = num_cols[:12]
    if len(view_cols) >= 2:
        corr = sample_df[view_cols + ["Class"]].corr()
        heat = px.imshow(corr, text_auto=False, aspect="auto", title="Correlation heatmap")
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Need at least two numeric columns to build a heatmap.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Train-test split and model
# -----------------------------------------------------------------------------
X = df[num_cols].copy()
y = df["Class"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_state
)

def build_model(model_name):
    if model_name == "Logistic Regression":
        model = LogisticRegression(penalty=penalty, C=C, solver="lbfgs", max_iter=1000)
        needs_scale = True
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_depth, random_state=42)
        needs_scale = False
    else:
        model = GradientBoostingClassifier(learning_rate=gb_learn_rate, n_estimators=gb_estimators, random_state=42)
        needs_scale = False

    pre = ("scale", StandardScaler()) if needs_scale else ("identity", "passthrough")
    pipeline = Pipeline([
        ("pre", ColumnTransformer([("num", pre[1], list(range(X_train.shape[1])))], remainder="drop")),
        ("clf", model)
    ])
    return pipeline

clf = build_model(model_name)
clf.fit(X_train, y_train)

def proba_and_pred(model, X, thr):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
    else:
        raw = model.decision_function(X)
        p = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    yhat = (p >= thr).astype(int)
    return p, yhat

prob_test, pred_test = proba_and_pred(clf, X_test, threshold)
prob_train, pred_train = proba_and_pred(clf, X_train, threshold)

# -----------------------------------------------------------------------------
# Metrics and visuals
# -----------------------------------------------------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("Model performance")

def metric_row(y_true, pred, prob):
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, prob)
    except Exception:
        auc = np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f'<div class="metric-chip">Accuracy<br><b>{acc:.3f}</b></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-chip">Precision<br><b>{prec:.3f}</b></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-chip">Recall<br><b>{rec:.3f}</b></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-chip">F1<br><b>{f1:.3f}</b></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="metric-chip">ROC AUC<br><b>{auc:.3f}</b></div>', unsafe_allow_html=True)
    return acc, prec, rec, f1, auc

st.markdown("**Test set**")
test_acc, test_prec, test_rec, test_f1, test_auc = metric_row(y_test, pred_test, prob_test)

st.markdown("**Train set**")
_ = metric_row(y_train, pred_train, prob_train)

# Confusion matrix
cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
cm_fig = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion matrix")
st.plotly_chart(cm_fig, use_container_width=True)

# ROC
try:
    fpr, tpr, thr = roc_curve(y_test, prob_test)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    roc_fig.update_layout(title=f"ROC Curve (AUC={test_auc:.3f})", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(roc_fig, use_container_width=True)
except Exception:
    st.info("ROC not available for this model.")
st.caption("Tip: adjust the decision threshold in the sidebar to trade recall vs precision.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Feature importance
# -----------------------------------------------------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("Top features")

def get_importance(pipeline, X_eval):
    model = pipeline.named_steps["clf"]
    names = list(X.columns)
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_).ravel()
        return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
    try:
        pi = permutation_importance(pipeline, X_eval, pipeline.predict(X_eval), n_repeats=5, random_state=42)
        return pd.DataFrame({"feature": names, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
    except Exception:
        return None

imp_df = get_importance(clf, X_test)
if imp_df is not None and not imp_df.empty:
    bar = px.bar(imp_df.head(15), x="importance", y="feature", orientation="h", title="Top feature importances")
    st.plotly_chart(bar, use_container_width=True)
else:
    st.info("Feature importance not available for this model.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Single patient simulator
# -----------------------------------------------------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("Single patient simulator")

with st.form("sim_form"):
    cols = st.columns(3)
    inputs = {}
    stats = df[num_cols].describe()
    for i, col in enumerate(num_cols):
        c = cols[i % 3]
        vmin = float(np.floor(stats.loc["min", col]))
        vmax = float(np.ceil(stats.loc["max", col]))
        vmean = float(stats.loc["mean", col])
        step = max(0.1, round((vmax - vmin) / 200, 2))
        inputs[col] = c.number_input(col, min_value=vmin, max_value=vmax, value=float(np.round(vmean, 2)), step=step)
    submitted = st.form_submit_button("Predict")

if submitted:
    user_row = pd.DataFrame([inputs])[num_cols]
    user_prob, user_pred = proba_and_pred(clf, user_row, threshold)
    p = float(user_prob[0])
    yhat = int(user_pred[0])

    st.markdown(f"**Predicted class:** `{yhat}`  (0 = no diabetes, 1 = diabetic)")
    st.markdown(f"**Risk probability:** `{p:.3f}` at threshold `{threshold:.2f}`")

    st.markdown("### 🧭 Guidance")
    st.info("This is a decision support tool. It is not a diagnosis. Always consult a licensed clinician.")
    tips_common = [
        "Aim for 150 minutes per week of moderate activity if your physician approves",
        "Prefer whole grains, legumes, vegetables, and lean proteins",
        "Limit sugary drinks and refined carbs",
        "Prioritize sleep quality and stress reduction",
    ]
    if yhat == 1:
        st.markdown("**Result suggests higher risk**")
        st.markdown("- Schedule a medical visit for tests like A1C or fasting plasma glucose")
        st.markdown("- Discuss lifestyle and medication options with a clinician")
        st.markdown("- Ask about nutrition counseling and a monitored activity plan")
    else:
        st.markdown("**Result suggests lower risk**")
        st.markdown("- Maintain healthy habits and routine checkups")
        st.markdown("- Track weight, waist size, and blood pressure")
        st.markdown("- Reassess if symptoms or risk factors change")
    st.markdown("**General tips**")
    st.markdown("\n".join([f"- {t}" for t in tips_common]))
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Dataset peek
# -----------------------------------------------------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("Dataset preview")
st.dataframe(df.head(25), use_container_width=True)
st.caption("Model uses only numeric columns as features. Target is the Class column.")
st.markdown('</div>', unsafe_allow_html=True)
