import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils.profiling import (
    get_basic_info,
    get_data_types,
    get_missing_values,
    get_summary_statistics,
)
from utils.visualization import (
    create_histogram,
    create_boxplot,
    create_correlation_heatmap,
    create_countplot,
)
from utils.profiling_report import generate_profile_report
from utils.llm_analysis import (
    generate_dataset_insights,
    chat_with_dataset,
    generate_ml_recommendations,
)
from utils.analytics import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_skewness,
    detect_duplicates,
    calculate_vif,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="EDA Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #080d1a; }

    /* ── Metric Cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0f1f3d 0%, #0a1628 100%);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(56,189,248,0.15);
    }
    [data-testid="metric-container"] label {
        color: #7dd3fc !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 28px !important;
        color: #f0f9ff !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #0a1628;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #1e3a5f;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-size: 13px;
        font-weight: 500;
        border-radius: 8px;
        color: #7dd3fc;
        padding: 6px 14px;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1d4ed8, #0ea5e9) !important;
        color: #fff !important;
        font-weight: 600;
    }

    /* ── Buttons ── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%);
        color: white;
        border: none;
        border-radius: 10px;
        height: 3em;
        font-size: 15px;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 0.02em;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(14,165,233,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(14,165,233,0.45);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border-radius: 12px;
        border: 1px solid #1e3a5f;
        overflow: hidden;
    }

    /* ── Alerts ── */
    .stAlert {
        border-radius: 10px;
        border-left-width: 4px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080d1a 0%, #0a1628 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #7dd3fc;
        font-family: 'Space Mono', monospace;
        font-size: 13px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* ── Health Score Badge ── */
    .health-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 64px; height: 64px;
        border-radius: 50%;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 18px;
        border: 3px solid;
    }

    /* ── Section Headers ── */
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #38bdf8;
        margin: 24px 0 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #1e3a5f;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid #1e3a5f;
        margin-bottom: 8px;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MAX_CHAT_HISTORY    = 20
VIF_SEVERE          = 10
VIF_MODERATE        = 5
MISSING_HIGH        = 20
MISSING_MODERATE    = 10
SKEW_THRESHOLD      = 1.0
HIGH_CORR_THRESHOLD = 0.8

# ─────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_data(show_spinner=False)
def compute_all_analytics(df: pd.DataFrame) -> dict:
    """Single cached call for all heavy analytics."""
    vif_result = _safe_vif(df)
    return {
        "iqr_outliers":    detect_outliers_iqr(df),
        "zscore_outliers": detect_outliers_zscore(df),
        "skewness":        detect_skewness(df),
        "duplicates":      detect_duplicates(df),
        "vif":             vif_result,
        "correlation":     df.corr(numeric_only=True),
    }


@st.cache_data(show_spinner=False)
def compute_profiling(df: pd.DataFrame) -> dict:
    return {
        "info":    get_basic_info(df),
        "dtypes":  get_data_types(df),
        "missing": get_missing_values(df),
        "summary": get_summary_statistics(df),
    }


def _safe_vif(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return calculate_vif(df)
    except Exception:
        return pd.DataFrame({"Feature": [], "VIF": []})


# ─────────────────────────────────────────────
# HEALTH SCORE
# ─────────────────────────────────────────────

def calculate_health_score(
    missing_percent: float,
    duplicates: int,
    avg_vif: float,
    skewed_columns: int,
) -> tuple[int, list[str]]:
    """Returns (score, list of reasons for deductions)."""
    score   = 100
    reasons = []

    if missing_percent > MISSING_HIGH:
        score -= 25
        reasons.append(f"-25 pts: High missing values ({missing_percent:.1f}%)")
    elif missing_percent > MISSING_MODERATE:
        score -= 15
        reasons.append(f"-15 pts: Moderate missing values ({missing_percent:.1f}%)")

    if duplicates > 0:
        score -= 10
        reasons.append(f"-10 pts: {duplicates} duplicate rows found")

    if avg_vif > VIF_SEVERE:
        score -= 20
        reasons.append(f"-20 pts: Severe multicollinearity (avg VIF={avg_vif:.1f})")
    elif avg_vif > VIF_MODERATE:
        score -= 10
        reasons.append(f"-10 pts: Moderate multicollinearity (avg VIF={avg_vif:.1f})")

    if skewed_columns > 3:
        score -= 15
        reasons.append(f"-15 pts: {skewed_columns} heavily skewed columns")

    return max(score, 0), reasons


def health_color(score: int) -> str:
    if score >= 80:
        return "#22c55e"
    if score >= 60:
        return "#f59e0b"
    return "#ef4444"


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "insights_cache" not in st.session_state:
        st.session_state.insights_cache = None
    if "recommendations_cache" not in st.session_state:
        st.session_state.recommendations_cache = {}


init_session_state()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px;">
            <div style="font-family:'Space Mono',monospace; font-size:26px; color:#38bdf8;">
                📊
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:16px;
                        font-weight:700; color:#f0f9ff; letter-spacing:0.05em;">
                EDA COPILOT
            </div>
            <div style="font-size:11px; color:#7dd3fc; margin-top:4px;">
                LLM-Augmented Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.info(
        "**Upload** a CSV to begin\n\n"
        "✅ Automated profiling\n"
        "✅ AI-powered insights\n"
        "✅ Anomaly detection\n"
        "✅ Conversational Q&A\n"
        "✅ ML recommendations"
    )
    st.markdown("---")
    st.markdown("### 🛠 Tech Stack")
    st.markdown(
        "- **Streamlit** — UI\n"
        "- **Pandas / NumPy** — Data\n"
        "- **Plotly** — Visuals\n"
        "- **Groq LLM** — AI layer\n"
        "- **ydata-profiling** — Reports"
    )
    st.markdown("---")
    st.caption("M.Tech Research Project")

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────

st.markdown(
    """
    <div style="padding: 10px 0 30px;">
        <h1 style="font-family:'Space Mono',monospace; font-size:28px;
                   color:#f0f9ff; margin:0; letter-spacing:-0.02em;">
            📊 LLM-Augmented EDA System
        </h1>
        <p style="color:#7dd3fc; margin:8px 0 0; font-size:15px;">
            Upload any CSV dataset for automated analysis, AI insights, and ML recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# FILE UPLOADER
# ─────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"],
    help="Accepts standard CSV files. Large files (>50 MB) may take longer to process.",
)

# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────

if uploaded_file is not None:

    # ── Load & cache ──────────────────────────

    with st.spinner("Loading dataset…"):
        raw_df = load_data(uploaded_file)

    # ── Sidebar column filter ─────────────────

    with st.sidebar:
        st.markdown("## 🎯 Dataset Filters")
        selected_columns = st.multiselect(
            "Select Columns",
            raw_df.columns.tolist(),
            default=raw_df.columns.tolist(),
        )
        if not selected_columns:
            st.warning("Select at least one column.")
            st.stop()

    # Use filtered view; never overwrite raw_df
    df = raw_df[selected_columns].copy()

    # ── Compute analytics (all cached together) ──

    with st.spinner("Running analytics…"):
        analytics = compute_all_analytics(df)
        profiling = compute_profiling(df)

    info           = profiling["info"]
    dtypes_df      = profiling["dtypes"]
    missing_values = profiling["missing"]
    summary_stats  = profiling["summary"]

    iqr_outliers      = analytics["iqr_outliers"]
    zscore_outliers   = analytics["zscore_outliers"]
    skewness_df       = analytics["skewness"]
    duplicates        = analytics["duplicates"]
    vif_df            = analytics["vif"]
    correlation_matrix = analytics["correlation"]

    rows    = info["rows"]
    columns = info["columns"]

    numerical_columns   = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # ── Health score ──────────────────────────

    total_cells     = rows * columns
    missing_percent = (df.isnull().sum().sum() / total_cells) * 100

    # FIX #4 — guard against missing "VIF" column if VIF computation failed
    avg_vif = (
        vif_df["VIF"].mean()
        if "VIF" in vif_df.columns and not vif_df.empty
        else 0
    )

    skewed_columns = int((skewness_df["Skewness"].abs() > SKEW_THRESHOLD).sum())

    health_score, health_reasons = calculate_health_score(
        missing_percent, duplicates, avg_vif, skewed_columns
    )
    h_color = health_color(health_score)

    # ── Dataset Overview ──────────────────────

    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows",         f"{rows:,}")
    col2.metric("Columns",      columns)
    col3.metric("Numerical",    len(numerical_columns))
    col4.metric("Categorical",  len(categorical_columns))
    col5.metric("Health Score", f"{health_score}/100")

    # Health score tooltip
    with st.expander("ℹ️ Health Score Breakdown"):
        if health_reasons:
            for r in health_reasons:
                st.markdown(f"- {r}")
        else:
            st.success("✅ No issues detected — dataset looks healthy!")

    # ── Smart Alerts ──────────────────────────

    if duplicates > 0:
        st.warning(f"⚠️ **{duplicates} duplicate rows** detected. Consider deduplication before modelling.")
    if avg_vif > VIF_SEVERE:
        st.error("🔥 **Severe multicollinearity** detected (avg VIF > 10). Drop or combine correlated features.")
    elif avg_vif > VIF_MODERATE:
        st.warning("⚠️ **Moderate multicollinearity** detected (avg VIF > 5). Review correlated features.")
    if missing_percent > MISSING_HIGH:
        st.warning(f"⚠️ **{missing_percent:.1f}% missing values** — consider imputation or column removal.")

    # ── Data Preview ──────────────────────────

    with st.expander("🔍 Dataset Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────

    (
        tab1, tab2, tab3, tab4,
        tab5, tab6, tab7, tab8, tab9,
    ) = st.tabs([
        "📊 Profiling",
        "📈 Visualizations",
        "🔥 Correlation",
        "📂 Categorical",
        "🧠 Full Report",
        "🤖 AI Insights",
        "💬 Chat",
        "🧠 Smart Analytics",
        "🚀 ML Recommendations",
    ])

    # ── TAB 1 — PROFILING ────────────────────

    with tab1:
        st.markdown('<div class="section-header">Column Names</div>', unsafe_allow_html=True)
        st.write(info["column_names"])

        st.markdown('<div class="section-header">Data Types</div>', unsafe_allow_html=True)
        st.dataframe(dtypes_df, use_container_width=True)

        st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
        st.dataframe(missing_values, use_container_width=True)

        st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
        st.dataframe(summary_stats, use_container_width=True)

        st.download_button(
            label="📥 Download Missing Values Report",
            data=missing_values.to_csv(index=False),
            file_name="missing_values.csv",
            mime="text/csv",
        )

    # ── TAB 2 — VISUALIZATIONS ───────────────

    with tab2:
        if numerical_columns:
            selected_numeric_col = st.selectbox(
                "Select Numerical Column", numerical_columns, key="viz_num"
            )
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Histogram")
                st.plotly_chart(
                    create_histogram(df, selected_numeric_col),
                    use_container_width=True,
                )
            with col2:
                st.subheader("Boxplot")
                st.plotly_chart(
                    create_boxplot(df, selected_numeric_col),
                    use_container_width=True,
                )
        else:
            st.info("No numerical columns available for visualisation.")

    # ── TAB 3 — CORRELATION ──────────────────

    with tab3:
        if len(numerical_columns) > 1:
            st.subheader("🔥 Correlation Heatmap")
            st.plotly_chart(
                create_correlation_heatmap(df),
                use_container_width=True,
            )

            # High-correlation pairs table
            corr_abs   = correlation_matrix.abs()
            corr_pairs = [
                {
                    "Feature 1":   corr_abs.columns[i],
                    "Feature 2":   corr_abs.columns[j],
                    "Correlation": round(corr_abs.iloc[i, j], 4),
                }
                for i in range(len(corr_abs.columns))
                for j in range(i)
                if corr_abs.iloc[i, j] > HIGH_CORR_THRESHOLD
            ]
            if corr_pairs:
                st.subheader("Highly Correlated Features (> 0.8)")
                st.dataframe(
                    pd.DataFrame(corr_pairs).sort_values("Correlation", ascending=False),
                    use_container_width=True,
                )
            else:
                st.success("✅ No highly correlated feature pairs found.")
        else:
            st.info("Need at least 2 numerical columns for correlation analysis.")

    # ── TAB 4 — CATEGORICAL ──────────────────

    with tab4:
        if categorical_columns:
            selected_cat_col = st.selectbox(
                "Select Categorical Column", categorical_columns, key="viz_cat"
            )
            st.plotly_chart(
                create_countplot(df, selected_cat_col),
                use_container_width=True,
            )
            # Show value counts table
            vc = df[selected_cat_col].value_counts().reset_index()
            vc.columns = [selected_cat_col, "Count"]
            vc["Proportion (%)"] = (vc["Count"] / len(df) * 100).round(2)
            st.dataframe(vc, use_container_width=True)
        else:
            st.info("No categorical columns found in the selected data.")

    # ── TAB 5 — FULL REPORT ──────────────────

    with tab5:
        st.subheader("🧠 Automated Full EDA Report")
        st.caption("Powered by ydata-profiling — may take 30–60 s for large datasets.")

        if st.button("Generate Full Report", key="gen_report"):
            with st.spinner("Generating comprehensive report…"):
                try:
                    os.makedirs("outputs", exist_ok=True)
                    profile     = generate_profile_report(df)
                    report_path = "outputs/eda_report.html"
                    profile.to_file(report_path)

                    with open(report_path, "r", encoding="utf-8") as f:
                        html_data = f.read()

                    components.html(html_data, height=1200, scrolling=True)

                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Full Report",
                            data=file,
                            file_name="eda_report.html",
                            mime="text/html",
                        )
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

    # ── TAB 6 — AI INSIGHTS ──────────────────
    # FIX #3 — simplified button logic, no duplicate keys, no dead code paths

    with tab6:
        st.subheader("🤖 AI Dataset Insights")

        if st.session_state.insights_cache:
            st.markdown(st.session_state.insights_cache)

        if st.button("🔄 Generate / Regenerate Insights", key="gen_insights"):
            st.session_state.insights_cache = None
            with st.spinner("Analysing dataset with AI…"):
                try:
                    insights = generate_dataset_insights(
                        rows=rows,
                        columns=columns,
                        missing_values=missing_values,
                        summary_stats=summary_stats,
                    )
                    st.session_state.insights_cache = insights
                    st.markdown(insights)
                    st.download_button(
                        label="📥 Download Insights",
                        data=insights,
                        file_name="ai_insights.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"AI insights generation failed: {e}")

    # ── TAB 7 — CHAT ─────────────────────────

    with tab7:
        st.subheader("💬 Chat with Your Dataset")

        # Trim history to avoid memory leak
        if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
            st.session_state.chat_history = (
                st.session_state.chat_history[-MAX_CHAT_HISTORY:]
            )

        # Render existing history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

        user_question = st.chat_input("Ask anything about your dataset…")

        if user_question:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        response = chat_with_dataset(
                            user_question=user_question,
                            rows=rows,
                            columns=columns,
                            column_names=info["column_names"],
                            missing_values=missing_values,
                            summary_stats=summary_stats,
                            correlation_matrix=correlation_matrix,
                            chat_history=st.session_state.chat_history,
                        )
                    except Exception as e:
                        response = f"⚠️ Error generating response: {e}"
                    st.markdown(response)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )

    # ── TAB 8 — SMART ANALYTICS ──────────────

    with tab8:
        st.subheader("🧠 Smart Dataset Analytics")

        # IQR Outliers
        st.markdown('<div class="section-header">IQR Outlier Detection</div>', unsafe_allow_html=True)
        if not iqr_outliers.empty:
            st.dataframe(iqr_outliers, use_container_width=True)
        else:
            st.success("✅ No IQR outliers detected.")

        # Z-Score Outliers
        st.markdown('<div class="section-header">Z-Score Outlier Detection</div>', unsafe_allow_html=True)
        if not zscore_outliers.empty:
            st.dataframe(zscore_outliers, use_container_width=True)
        else:
            st.success("✅ No Z-score outliers detected.")

        # Skewness
        st.markdown('<div class="section-header">Skewness Analysis</div>', unsafe_allow_html=True)
        if not skewness_df.empty:
            skewed = skewness_df[skewness_df["Skewness"].abs() > SKEW_THRESHOLD]
            if not skewed.empty:
                st.warning(f"{len(skewed)} column(s) are heavily skewed (|skew| > 1).")
            st.dataframe(skewness_df, use_container_width=True)
        else:
            st.info("Skewness data unavailable.")

        # Duplicates
        st.markdown('<div class="section-header">Duplicate Rows</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 3])
        col_a.metric("Duplicate Rows", duplicates)
        if duplicates == 0:
            col_b.success("✅ No duplicate rows found.")
        else:
            col_b.warning(f"⚠️ {duplicates} duplicate rows may inflate model performance.")

        # VIF
        st.markdown('<div class="section-header">Multicollinearity (VIF)</div>', unsafe_allow_html=True)
        if not vif_df.empty and "VIF" in vif_df.columns:
            st.dataframe(vif_df, use_container_width=True)
        else:
            st.info("VIF could not be calculated (requires ≥ 2 numerical columns).")

        st.info(
            "**VIF Guide:** VIF < 5 → Low | VIF 5–10 → Moderate | VIF > 10 → Severe multicollinearity"
        )

        # FIX #2 — align column names before concat to avoid NaN-filled mismatched columns
        if not iqr_outliers.empty or not zscore_outliers.empty:
            parts = []
            if not iqr_outliers.empty:
                iqr_export = (
                    iqr_outliers[["Column", "Outlier Count", "Outlier %", "Severity"]]
                    .copy()
                    .rename(columns={"Outlier Count": "Outliers"})
                    .assign(Source="IQR")
                )
                parts.append(iqr_export)

            if not zscore_outliers.empty:
                zscore_export = (
                    zscore_outliers[["Column", "Z-Score Outliers", "Outlier %", "Severity"]]
                    .copy()
                    .rename(columns={"Z-Score Outliers": "Outliers"})
                    .assign(Source="Z-Score")
                )
                parts.append(zscore_export)

            if parts:
                analytics_summary = pd.concat(parts, ignore_index=True)
                st.download_button(
                    label="📥 Download Outlier Report",
                    data=analytics_summary.to_csv(index=False),
                    file_name="outlier_report.csv",
                    mime="text/csv",
                )

    # ── TAB 9 — ML RECOMMENDATIONS ───────────

    with tab9:
        st.subheader("🚀 AI-Powered ML Recommendations")

        target_column = st.selectbox(
            "Select Target Column",
            df.columns.tolist(),
            key="ml_target",
        )

        cache_key = f"{target_column}_{rows}_{columns}"

        if cache_key in st.session_state.recommendations_cache:
            st.info("Showing cached recommendations. Click below to regenerate.")
            st.markdown(st.session_state.recommendations_cache[cache_key])
            regenerate = st.button("🔄 Regenerate Recommendations")
        else:
            regenerate = False

        if st.button("Generate Recommendations", key="gen_recs") or regenerate:
            with st.spinner("Generating ML recommendations…"):
                try:
                    # FIX #1 — pass target_column to generate_ml_recommendations
                    recommendations = generate_ml_recommendations(
                        rows=rows,
                        columns=columns,
                        column_names=info["column_names"],
                        dtypes=dtypes_df,
                        missing_values=missing_values,
                        summary_stats=summary_stats,
                        correlation_matrix=correlation_matrix,
                        outliers_iqr=iqr_outliers,
                        skewness=skewness_df,
                        duplicates=duplicates,
                        target_column=target_column,   # ✅ was missing
                    )
                    st.session_state.recommendations_cache[cache_key] = recommendations
                    st.markdown(recommendations)
                    st.download_button(
                        label="📥 Download Recommendations",
                        data=recommendations,
                        file_name="ml_recommendations.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Recommendation generation failed: {e}")

    # ─────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────

    st.markdown("---")
    st.caption(
        "Built with Streamlit · Plotly · Pandas · Groq · ydata-profiling "
        "| M.Tech Research Project"
    )

else:
    # ── Welcome state (no file uploaded) ─────
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px;
                    border: 1px dashed #1e3a5f; border-radius: 16px;
                    background: linear-gradient(135deg, #0a1628 0%, #080d1a 100%);">
            <div style="font-size: 52px; margin-bottom: 16px;">📂</div>
            <div style="font-family:'Space Mono',monospace; font-size:18px;
                        color:#7dd3fc; font-weight:700; margin-bottom:8px;">
                No dataset uploaded yet
            </div>
            <div style="color:#475569; font-size:14px; max-width:420px; margin:0 auto;">
                Upload a CSV file using the panel above to get started.
                The system will automatically profile, visualise, and analyse your data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )