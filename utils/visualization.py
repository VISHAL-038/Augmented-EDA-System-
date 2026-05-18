import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# SHARED THEME
# ─────────────────────────────────────────────

_PALETTE = [
    "#378ADD", "#1D9E75", "#D85A30", "#7F77DD",
    "#D4537E", "#BA7517", "#E24B4A", "#5DCAA5",
]

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,13,26,0.6)",
    font=dict(family="DM Sans, sans-serif", color="#94a3b8"),
    title_font=dict(family="Space Mono, monospace", size=14, color="#f0f9ff"),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=_PALETTE,
    hoverlabel=dict(
        bgcolor="#0f1f3d",
        bordercolor="#1e3a5f",
        font_color="#f0f9ff",
        font_family="DM Sans, sans-serif",
    ),
    xaxis=dict(
        gridcolor="#1e3a5f",
        zerolinecolor="#1e3a5f",
        tickfont=dict(color="#64748b"),
    ),
    yaxis=dict(
        gridcolor="#1e3a5f",
        zerolinecolor="#1e3a5f",
        tickfont=dict(color="#64748b"),
    ),
)


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the shared dark theme and optional title to any figure."""
    fig.update_layout(title=dict(text=title, x=0.02), **_LAYOUT_DEFAULTS)
    return fig


# ─────────────────────────────────────────────
# HISTOGRAM
# ─────────────────────────────────────────────

def create_histogram(df: pd.DataFrame, column: str) -> go.Figure:
    """
    Enhanced histogram with:
    - KDE overlay curve
    - Mean & median reference lines
    - Skewness annotation
    - Descriptive hover template
    """
    series = df[column].dropna()

    # Compute stats
    mean_val   = series.mean()
    median_val = series.median()
    skewness   = series.skew()

    # KDE via numpy + scipy if available, else skip
    try:
        from scipy.stats import gaussian_kde
        kde_x = np.linspace(series.min(), series.max(), 300)
        kde_y = gaussian_kde(series)(kde_x)
        # Scale KDE to histogram counts
        bin_width = (series.max() - series.min()) / 30
        kde_y_scaled = kde_y * len(series) * bin_width
        has_kde = True
    except ImportError:
        has_kde = False

    fig = go.Figure()

    # Histogram bars
    fig.add_trace(go.Histogram(
        x=series,
        nbinsx=30,
        name="Count",
        marker=dict(
            color="#378ADD",
            opacity=0.75,
            line=dict(color="#185FA5", width=0.5),
        ),
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))

    # KDE overlay
    if has_kde:
        fig.add_trace(go.Scatter(
            x=kde_x,
            y=kde_y_scaled,
            mode="lines",
            name="KDE",
            line=dict(color="#1D9E75", width=2.5),
            hovertemplate="Value: %{x:.2f}<extra>KDE</extra>",
        ))

    # Mean line
    fig.add_vline(
        x=mean_val,
        line=dict(color="#EF9F27", width=1.5, dash="dash"),
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right",
        annotation_font=dict(color="#EF9F27", size=11),
    )

    # Median line
    fig.add_vline(
        x=median_val,
        line=dict(color="#D4537E", width=1.5, dash="dot"),
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="top left",
        annotation_font=dict(color="#D4537E", size=11),
    )

    # Skewness annotation
    skew_label = (
        "Highly right-skewed" if skewness > 1 else
        "Highly left-skewed"  if skewness < -1 else
        "Moderately skewed"   if abs(skewness) > 0.5 else
        "Approximately normal"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.96,
        text=f"Skewness: {skewness:.2f} ({skew_label})",
        showarrow=False,
        font=dict(size=11, color="#7dd3fc"),
        align="right",
        bgcolor="rgba(8,13,26,0.7)",
        bordercolor="#1e3a5f",
        borderwidth=1,
    )

    fig.update_layout(barmode="overlay", showlegend=has_kde)
    _apply_theme(fig, f"Distribution of {column}")
    return fig


# ─────────────────────────────────────────────
# BOXPLOT
# ─────────────────────────────────────────────

def create_boxplot(df: pd.DataFrame, column: str) -> go.Figure:
    """
    Enhanced boxplot with:
    - Overlaid strip (jitter) plot
    - Outlier count annotation
    - IQR band annotation
    - Mean marker
    """
    series = df[column].dropna()

    q1  = series.quantile(0.25)
    q3  = series.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_count = int(((series < lower_fence) | (series > upper_fence)).sum())

    fig = go.Figure()

    # Jitter strip (sample for large datasets)
    sample = series.sample(min(500, len(series)), random_state=42)
    fig.add_trace(go.Scatter(
        x=np.random.uniform(-0.2, 0.2, size=len(sample)),
        y=sample,
        mode="markers",
        marker=dict(
            color="#378ADD",
            opacity=0.35,
            size=4,
        ),
        name="Data points",
        hovertemplate="%{y:.3f}<extra>Point</extra>",
    ))

    # Box
    fig.add_trace(go.Box(
        y=series,
        name=column,
        boxmean=True,
        marker=dict(
            color="#D85A30",
            outliercolor="#E24B4A",
            size=5,
            line=dict(color="#993C1D"),
        ),
        line=dict(color="#D85A30", width=1.5),
        fillcolor="rgba(56,138,221,0.15)",
        whiskerwidth=0.5,
        hovertemplate=(
            "Max: %{upperfence:.3f}<br>"
            "Q3: %{q3:.3f}<br>"
            "Median: %{median:.3f}<br>"
            "Q1: %{q1:.3f}<br>"
            "Min: %{lowerfence:.3f}<extra></extra>"
        ),
    ))

    # IQR annotation
    fig.add_annotation(
        xref="paper", yref="y",
        x=1.02, y=(q1 + q3) / 2,
        text=f"IQR: {iqr:.2f}",
        showarrow=False,
        font=dict(size=11, color="#7dd3fc"),
    )

    # Outlier count annotation
    if outlier_count > 0:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=f"⚠ {outlier_count} outlier{'s' if outlier_count != 1 else ''}",
            showarrow=False,
            font=dict(size=11, color="#EF9F27"),
            bgcolor="rgba(8,13,26,0.7)",
            bordercolor="#1e3a5f",
            borderwidth=1,
        )

    _apply_theme(fig, f"Boxplot of {column}")
    fig.update_layout(showlegend=False)
    return fig


# ─────────────────────────────────────────────
# CORRELATION HEATMAP
# ─────────────────────────────────────────────

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Enhanced correlation heatmap with:
    - Diverging blue-white-red colorscale
    - Masked upper triangle (clean look)
    - Significance stars (*p<0.05, **p<0.01)
    - Sortable by hierarchical clustering
    """
    numeric_df = df.select_dtypes(include=["number"])

    # Optional hierarchical ordering
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        corr = numeric_df.corr()
        dist = 1 - corr.abs()
        dist_condensed = squareform(dist.values, checks=False)
        dist_condensed = np.clip(dist_condensed, 0, None)
        order = leaves_list(linkage(dist_condensed, method="average"))
        corr = corr.iloc[order, order]
    except Exception:
        corr = numeric_df.corr()

    n = len(corr.columns)
    z = corr.values.copy()

    # Mask upper triangle (set to NaN so it renders as transparent)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    z_masked = z.copy().astype(float)
    z_masked[mask] = np.nan

    # Build annotation text (values + significance stars)
    try:
        from scipy.stats import pearsonr
        ann_text = []
        for i in range(n):
            row = []
            for j in range(n):
                if mask[i, j] or i == j:
                    row.append("" if i != j else f"{z[i,j]:.2f}")
                else:
                    r_val = z[i, j]
                    _, p = pearsonr(
                        numeric_df.iloc[:, i].dropna(),
                        numeric_df.iloc[:, j].dropna(),
                    )
                    stars = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                    row.append(f"{r_val:.2f}{stars}")
            ann_text.append(row)
    except Exception:
        ann_text = corr.round(2).astype(str).values.tolist()

    fig = ff.create_annotated_heatmap(
        z=z_masked,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=ann_text,
        colorscale=[
            [0.0,  "#A32D2D"],
            [0.25, "#D85A30"],
            [0.5,  "#0f1f3d"],
            [0.75, "#185FA5"],
            [1.0,  "#378ADD"],
        ],
        showscale=True,
        zmin=-1,
        zmax=1,
        hovertemplate="<b>%{y} × %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
    )

    # Style annotation font
    for ann in fig.layout.annotations:
        ann.font = dict(size=10, color="#f0f9ff")

    fig.update_layout(
        height=max(500, n * 55),
        xaxis=dict(tickangle=-40, tickfont=dict(size=11, color="#94a3b8")),
        yaxis=dict(tickfont=dict(size=11, color="#94a3b8")),
    )

    # Colorbar styling
    fig.update_traces(
        colorbar=dict(
            thickness=12,
            len=0.8,
            tickfont=dict(color="#94a3b8"),
            title=dict(text="r", font=dict(color="#94a3b8")),
            tickvals=[-1, -0.5, 0, 0.5, 1],
        )
    )

    _apply_theme(fig, "Correlation Heatmap")
    # Add legend note for significance
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.0, y=-0.08,
        text="* p < 0.05   ** p < 0.01   (lower triangle only)",
        showarrow=False,
        font=dict(size=10, color="#64748b"),
    )
    return fig


# ─────────────────────────────────────────────
# COUNT PLOT
# ─────────────────────────────────────────────

def create_countplot(df: pd.DataFrame, column: str, top_n: int = 20) -> go.Figure:
    """
    Enhanced count/bar chart with:
    - Top-N cap with 'Others' bucket
    - Percentage labels on bars
    - Sorted by frequency (descending)
    - Color gradient encoding magnitude
    """
    vc = df[column].value_counts()
    total = len(df[column].dropna())

    if len(vc) > top_n:
        top    = vc.iloc[:top_n]
        others = pd.Series({"Others": vc.iloc[top_n:].sum()})
        vc     = pd.concat([top, others])

    labels      = vc.index.astype(str).tolist()
    counts      = vc.values.tolist()
    percentages = [c / total * 100 for c in counts]

    # Color gradient: deeper blue for higher counts
    norm    = [c / max(counts) for c in counts]
    colors  = [
        f"rgba({int(56 - 30*n)},{int(138 + 60*n)},{int(221 - 20*n)},0.85)"
        for n in norm
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        marker=dict(
            color=colors,
            line=dict(color="#185FA5", width=0.5),
        ),
        text=[f"{p:.1f}%" for p in percentages],
        textposition="outside",
        textfont=dict(size=11, color="#94a3b8"),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Count: %{y:,}<br>"
            "Proportion: %{text}<extra></extra>"
        ),
        name=column,
    ))

    fig.update_layout(
        xaxis=dict(
            tickangle=-35 if len(labels) > 8 else 0,
            categoryorder="total descending",
        ),
        yaxis_title="Count",
        bargap=0.25,
        showlegend=False,
    )

    if len(vc) > top_n:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            text=f"Top {top_n} shown · {len(df[column].value_counts()) - top_n} categories collapsed into 'Others'",
            showarrow=False,
            font=dict(size=10, color="#64748b"),
            align="right",
        )

    _apply_theme(fig, f"Value Distribution — {column}")
    return fig


# ─────────────────────────────────────────────
# MISSING VALUES HEATMAP  (bonus)
# ─────────────────────────────────────────────

def create_missing_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Row-level missingness heatmap.
    Shows which cells are missing across a sample of rows.
    """
    sample = df.sample(min(200, len(df)), random_state=42)
    z      = sample.isnull().astype(int).values

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(df.columns),
        y=[f"Row {i}" for i in sample.index],
        colorscale=[[0, "#0f1f3d"], [1, "#E24B4A"]],
        showscale=False,
        hovertemplate="Column: %{x}<br>Row: %{y}<br>Missing: %{z}<extra></extra>",
    ))

    fig.update_layout(
        height=max(300, min(len(sample) * 4, 600)),
        xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(showticklabels=False),
    )

    _apply_theme(fig, "Missing Values Map (sample of 200 rows)")
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        text="Red = missing   ·   Dark = present",
        showarrow=False,
        font=dict(size=11, color="#64748b"),
    )
    return fig


# ─────────────────────────────────────────────
# SCATTER WITH REGRESSION  (bonus)
# ─────────────────────────────────────────────

def create_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
) -> go.Figure:
    """
    Scatter plot with optional OLS trendline and R² annotation.
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        trendline="ols",
        trendline_color_override="#EF9F27",
        opacity=0.65,
        color_discrete_sequence=_PALETTE,
        hover_data=df.columns[:5].tolist(),
    )

    # Compute R²
    try:
        from scipy.stats import pearsonr
        r, _ = pearsonr(df[x_col].dropna(), df[y_col].dropna())
        r2   = r ** 2
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.04,
            text=f"R² = {r2:.3f}",
            showarrow=False,
            font=dict(size=12, color="#EF9F27"),
            bgcolor="rgba(8,13,26,0.7)",
            bordercolor="#1e3a5f",
            borderwidth=1,
        )
    except Exception:
        pass

    _apply_theme(fig, f"{y_col} vs {x_col}")
    return fig