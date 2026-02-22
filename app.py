# FINAL PREMIUM AI DATA ANALYST APP

import streamlit as st
import pandas as pd
import plotly.express as px
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from modules.ai_engine import generate_ai_insights
from modules.storytelling import generate_story
from modules.ml_lab import run_auto_ml
from modules.pdf_engine import generate_pdf

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI Data Analyst", layout="wide")

# ================= PREMIUM DARK MODE =================
mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if mode:
    st.markdown("""
    <style>
    /* App background */
    .stApp {
        background-color: #0f172a;
    }

    /* Main content area */
    section.main > div {
        background-color: #0f172a;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 1px solid #1e293b;
    }

    /* Sidebar text FIX */
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }

    /* Paragraph text */
    p, span, label, div {
        color: #cbd5f5 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: white !important;
    }

    /* Cards */
    .card {
        background: #1e293b;
        color: #f8fafc;
        padding: 16px;
        border-radius: 16px;
        margin-bottom: 14px;
        border: 1px solid #334155;
        box-shadow: 0 8px 20px rgba(0,0,0,0.35);
    }

    /* Fix top padding gap */
    /* Fix top white gap */
    .block-container {
        padding-top: 1rem !important;
        background-color: #0f172a !important;
    }

    header[data-testid="stHeader"] {
        background: #0f172a !important;
    }

    section.main {
        background-color: #0f172a !important;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }

    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] * {
        color: #111827 !important;
    }

    .card {
        background: #f1f5f9;
        color: #0f172a;
        padding: 16px;
        border-radius: 16px;
        margin-bottom: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    </style>
    """, unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<h1 style='text-align:center'>🤖 AI Data Analyst</h1>
<p style='text-align:center;color:gray'>Turn raw CSV files into stories, insights, and decisions.</p>
""", unsafe_allow_html=True)
st.divider()

# ================= SIDEBAR =================
page = st.sidebar.radio(
    "Navigation",
    [
        "Landing",
        "Overview",
        "Story Mode",
        "AI Insights",
        "Visual Lab",
        "ML Studio",
        "Data Assistant",
        "Executive Report",
    ],
)

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ================= LANDING PAGE =================
if page == "Landing":
    st.markdown("## 🚀 Welcome")
    st.write("Upload a dataset to unlock insights, visuals, and data stories.")

    col1, col2, col3 = st.columns(3)
    col1.info("📖 Story Mode — Narrative insights")
    col2.info("📊 Visual Lab — Instant charts")
    col3.info("🤖 ML Studio — One-click models")

    st.stop()

# ================= LOAD DATA =================
if not file:
    st.warning("Upload a CSV file from sidebar.")
    st.stop()

df = pd.read_csv(file)
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# ================= OVERVIEW =================
if page == "Overview":
    st.subheader("📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", df.isnull().sum().sum())

    st.dataframe(df.head(), use_container_width=True)

# ================= STORY MODE =================
elif page == "Story Mode":
    st.subheader("📖 Story Mode")

    rows, cols = df.shape
    missing = df.isnull().sum().sum()

    id_cols = [c for c in df.columns if df[c].nunique() == len(df)]
    useful_numeric = [c for c in numeric_cols if c not in id_cols]

    # strongest correlation
    corr_text = "Not enough numeric data"
    if len(useful_numeric) > 1:
        corr = df[useful_numeric].corr().abs()
        for i in corr.index:
            corr.loc[i, i] = 0
        pair = corr.unstack().sort_values(ascending=False).idxmax()
        corr_text = f"{pair[0]} ↔ {pair[1]}"

    st.markdown("### 🧠 Data Story")
    st.write(
        f"This dataset contains **{rows:,} rows** and **{cols} columns**, making it suitable for analysis."
    )

    col1, col2 = st.columns(2)
    col1.success("Clean dataset" if missing == 0 else f"{missing} missing values")
    col2.info(f"Strongest pattern: {corr_text}")

    st.write(
        "The dataset appears structured and suitable for exploratory analysis and modeling."
    )

# ================= INSIGHTS =================
elif page == "AI Insights":
    st.subheader("🧠 Insight Cards")

    insights = generate_ai_insights(df)

    for ins in insights:
        st.markdown(f"""
        <div class="card">
            {ins}
        </div>
        """, unsafe_allow_html=True)

# ================= VISUAL LAB =================
elif page == "Visual Lab":
    st.subheader("📈 Visual Exploration")

    if numeric_cols:
        col = st.selectbox("Select column", numeric_cols)
        st.plotly_chart(px.histogram(df, x=col), use_container_width=True)

        if len(numeric_cols) > 1:
            st.markdown("### Correlation Heatmap")
            st.plotly_chart(px.imshow(df[numeric_cols].corr(), text_auto=True), use_container_width=True)
    else:
        st.warning("No numeric columns available.")

# ================= ML =================
elif page == "ML Studio":
    st.subheader("🤖 Auto ML Studio")

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
        st.stop()

    target = st.selectbox("Select target", numeric_cols)

    if st.button("Run Auto ML"):
        r2, importance = run_auto_ml(df, target)
        st.success(f"Model R² Score: {round(r2, 3)}")
        st.bar_chart(importance)

# ================= DATA ASSISTANT =================
elif page == "Data Assistant":

    st.markdown("### 🔍 Data Assistant")
    st.caption("Smart dataset exploration without external AI.")

    st.markdown(
        "**Try asking:**  \n"
        "- dataset summary  \n"
        "- missing values  \n"
        "- strongest correlation  \n"
        "- identifier columns  \n"
        "- column averages"
    )
    st.divider()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render chat history
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    user_input = st.chat_input("Ask about your dataset...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        q = user_input.lower()

        id_cols = [c for c in df.columns if df[c].nunique() == len(df)]
        useful_numeric = [c for c in numeric_cols if c not in id_cols]

        with st.chat_message("ai"):

            # SUMMARY
            if "summary" in q:
                st.markdown("### Dataset Summary")
                st.write(f"Rows: **{df.shape[0]:,}**")
                st.write(f"Columns: **{df.shape[1]}**")
                st.write(f"Missing values: **{df.isnull().sum().sum()}**")
                st.dataframe(df.describe(), use_container_width=True)

            # MISSING
            elif "missing" in q:
                total_missing = df.isnull().sum().sum()
                if total_missing == 0:
                    st.success("No missing values detected.")
                else:
                    st.write(df.isnull().sum().sort_values(ascending=False))

            # CORRELATION
            elif "correlation" in q:
                if len(useful_numeric) > 1:
                    corr = df[useful_numeric].corr()
                    corr_abs = corr.abs()
                    for i in corr_abs.index:
                        corr_abs.loc[i, i] = 0
                    pair = corr_abs.unstack().sort_values(ascending=False).idxmax()

                    st.markdown(f"**Strongest correlation:** {pair[0]} ↔ {pair[1]}")
                    st.plotly_chart(px.imshow(corr, text_auto=True), use_container_width=True)
                else:
                    st.warning("Not enough numeric columns.")

            # IDENTIFIERS
            elif "id" in q:
                if id_cols:
                    st.warning(f"Possible identifier columns: {', '.join(id_cols)}")
                else:
                    st.success("No clear identifier columns detected.")

            # AVERAGES
            elif "average" in q or "mean" in q:
                if useful_numeric:
                    avg = df[useful_numeric].mean().sort_values(ascending=False)
                    st.dataframe(avg.to_frame("Average"), use_container_width=True)
                else:
                    st.warning("No useful numeric columns.")

            else:
                st.info("Try asking about summary, missing values, correlations, or averages.")

# ================= Executive Report =================
elif page == "Executive Report":

    st.header("📄 Executive Report")

    rows, cols = df.shape
    missing = df.isnull().sum().sum()
    insights = generate_ai_insights(df)
    narrative = generate_story(df)

    id_cols = [c for c in df.columns if df[c].nunique() == len(df)]
    useful_numeric = [c for c in numeric_cols if c not in id_cols]

    # Strongest correlation
    corr_text = "Not enough numeric data"
    if len(useful_numeric) > 1:
        corr = df[useful_numeric].corr().abs()
        for i in corr.index:
            corr.loc[i, i] = 0
        pair = corr.unstack().sort_values(ascending=False).idxmax()
        corr_text = f"{pair[0]} ↔ {pair[1]}"

    # ===== SECTION 1: SUMMARY =====
    st.subheader("Dataset Summary")
    st.write(f"Rows: {rows:,}")
    st.write(f"Columns: {cols}")
    st.write(f"Missing Values: {missing}")

    # ===== SECTION 2: STORY =====
    st.subheader("Data Story")
    st.write(
        f"This dataset contains {rows:,} rows and {cols} columns. "
        f"Strongest relationship observed between {corr_text}. "
        f"Data quality appears {'clean' if missing == 0 else 'requires review'}."
    )

    # ===== SECTION 3: INSIGHTS =====
    st.subheader("AI Insights")
    for ins in insights:
        st.markdown(f"- {ins}")

    # ===== SECTION 4: STATS =====
    st.subheader("Statistical Snapshot")
    st.dataframe(df.describe(), use_container_width=True)

    # ===== PREMIUM PDF DOWNLOAD =====
    if st.button("Download Premium Executive Report"):

        pdf_path = generate_pdf(df, insights, narrative)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="AI_Executive_Report.pdf",
                mime="application/pdf"
            )

# ================= FOOTER =================
st.divider()
st.caption("Built by Meghana • AI Data Analyst")
