import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Excel Column Analyser",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Upload box */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4f8ef7;
        border-radius: 12px;
        padding: 1.5rem;
        background: #1a1d27;
    }

    /* Metric cards */
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.5rem;
    }
    .metric-card h4 { margin: 0 0 0.3rem 0; font-size: 0.78rem;
                      color: #8892a4; letter-spacing: 0.06em; text-transform: uppercase; }
    .metric-card p  { margin: 0; font-size: 1.35rem; font-weight: 700; color: #e0e0e0; }

    /* Verdict badges */
    .badge-yes {
        display: inline-block;
        background: #12291a;
        color: #4ade80;
        border: 1px solid #4ade80;
        border-radius: 999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    .badge-no {
        display: inline-block;
        background: #2a1010;
        color: #f87171;
        border: 1px solid #f87171;
        border-radius: 999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }

    /* Section divider */
    hr { border-color: #2a2d3a; }

    /* Header gradient text */
    .title-gradient {
        background: linear-gradient(90deg, #4f8ef7 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="title-gradient">📊 Excel Column Analyser</p>', unsafe_allow_html=True)
st.markdown(
    "Upload any Excel file. For every **numerical column** you'll get a histogram, "
    "mean & median, and a verdict on whether the data is **suitable for forecasting**."
)
st.markdown("---")

# ── Threshold slider ──────────────────────────────────────────────────────────
with st.expander("⚙️ Forecasting threshold (% difference between mean & median)", expanded=False):
    threshold_pct = st.slider(
        "Max allowed % difference",
        min_value=1, max_value=30, value=10,
        help="If |(mean − median)| / mean  ≤  this %, the column is flagged as forecasting-suitable.",
    )

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Drop your Excel file here (.xlsx / .xls)", type=["xlsx", "xls"])

# ── Helpers ───────────────────────────────────────────────────────────────────
PLOT_BG   = "#1a1d27"
HIST_COL  = "#4f8ef7"
MEAN_COL  = "#f59e0b"
MED_COL   = "#a78bfa"

def pct_diff(mean: float, median: float) -> float:
    """Percentage difference of median from mean (using mean as base)."""
    if mean == 0:
        return 0.0 if median == 0 else float("inf")
    return abs(mean - median) / abs(mean) * 100


def make_histogram(series: pd.Series, col_name: str, mean: float, median: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)

    n_bins = min(40, max(10, int(np.sqrt(len(series)))))
    ax.hist(series.dropna(), bins=n_bins, color=HIST_COL, alpha=0.85, edgecolor="#0f1117", linewidth=0.4)

    ax.axvline(mean,   color=MEAN_COL, linewidth=1.8, linestyle="--", label=f"Mean {mean:.2f}")
    ax.axvline(median, color=MED_COL,  linewidth=1.8, linestyle=":",  label=f"Median {median:.2f}")

    ax.set_title(col_name, color="#e0e0e0", fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(colors="#8892a4", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3a")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.legend(fontsize=7, framealpha=0.15, labelcolor="#e0e0e0")
    fig.tight_layout(pad=0.8)
    return fig


# ── Main logic ────────────────────────────────────────────────────────────────
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not num_cols:
        st.warning("No numerical columns found in this file.")
        st.stop()

    st.markdown(f"### Found **{len(num_cols)}** numerical column(s)")
    st.markdown("---")

    # Summary table at top
    summary_rows = []
    for col in num_cols:
        s      = df[col].dropna()
        mean_  = s.mean()
        med_   = s.median()
        diff   = pct_diff(mean_, med_)
        ok     = diff <= threshold_pct
        summary_rows.append({
            "Column": col,
            "Count (non-null)": len(s),
            "Mean": round(mean_, 4),
            "Median": round(med_, 4),
            "% Difference": round(diff, 2),
            "Forecasting OK?": "✅ Yes" if ok else "❌ No",
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    # Per-column cards
    COLS_PER_ROW = 2
    for i in range(0, len(num_cols), COLS_PER_ROW):
        row_cols = num_cols[i : i + COLS_PER_ROW]
        grid = st.columns(len(row_cols))

        for j, col in enumerate(row_cols):
            s      = df[col].dropna()
            mean_  = s.mean()
            med_   = s.median()
            diff   = pct_diff(mean_, med_)
            ok     = diff <= threshold_pct

            with grid[j]:
                st.markdown(f"#### 🔹 `{col}`")

                # Histogram
                fig = make_histogram(s, col, mean_, med_)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=PLOT_BG)
                buf.seek(0)
                st.image(buf, use_container_width=True)
                plt.close(fig)

                # Metrics
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f'<div class="metric-card"><h4>Mean</h4><p>{mean_:,.3f}</p></div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f'<div class="metric-card"><h4>Median</h4><p>{med_:,.3f}</p></div>',
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f'<div class="metric-card"><h4>% Diff</h4><p>{diff:.1f}%</p></div>',
                        unsafe_allow_html=True,
                    )

                # Verdict
                if ok:
                    st.markdown(
                        '<span class="badge-yes">✅ Suitable for forecasting</span>'
                        f'<br><small style="color:#8892a4;">Mean ≈ Median (within {threshold_pct}%)</small>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="badge-no">❌ Not ideal for forecasting</span>'
                        f'<br><small style="color:#8892a4;">Skewed distribution — mean & median diverge by {diff:.1f}%</small>',
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

else:
    st.info("👆 Upload an Excel file to get started.")
