import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import zscore, ks_2samp, shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

_NUMERIC_TYPES   = ["int8", "int16", "int32", "int64", "float32", "float64"]
_IQR_MULTIPLIER  = 1.5
_ZSCORE_DEFAULT  = 3.0
_SHAPIRO_LIMIT   = 5000     # shapiro-wilk is unreliable above this


# ─────────────────────────────────────────────
# SHARED HELPER
# ─────────────────────────────────────────────

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric column names using a broader dtype list."""
    return df.select_dtypes(include=_NUMERIC_TYPES).columns.tolist()


# ─────────────────────────────────────────────
# IQR OUTLIER DETECTION
# ─────────────────────────────────────────────

def detect_outliers_iqr(
    df: pd.DataFrame,
    multiplier: float = _IQR_MULTIPLIER,
) -> pd.DataFrame:
    """
    IQR-based outlier detection with:
    - Outlier count + percentage
    - Fence bounds
    - Severity classification
    - Sample outlier values
    """
    records = []

    for col in _numeric_cols(df):
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        # Zero IQR means constant column — skip
        if iqr == 0:
            continue

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        mask     = (series < lower) | (series > upper)
        n_out    = int(mask.sum())
        out_pct  = round(n_out / len(series) * 100, 2)

        severity = (
            "🔴 Severe"   if out_pct > 10 else
            "🟠 Moderate" if out_pct > 5  else
            "🟡 Mild"     if out_pct > 0  else
            "✅ None"
        )

        sample = series[mask].head(5).round(4).tolist()

        records.append({
            "Column":         col,
            "Outlier Count":  n_out,
            "Outlier %":      out_pct,
            "Lower Fence":    round(lower, 4),
            "Upper Fence":    round(upper, 4),
            "IQR":            round(iqr, 4),
            "Severity":       severity,
            "Sample Outliers": str(sample) if sample else "—",
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    return result.sort_values("Outlier Count", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# Z-SCORE OUTLIER DETECTION
# ─────────────────────────────────────────────

def detect_outliers_zscore(
    df: pd.DataFrame,
    threshold: float = _ZSCORE_DEFAULT,
) -> pd.DataFrame:
    """
    Z-score outlier detection with:
    - Outlier count + percentage
    - Max absolute z-score per column
    - Agreement flag with IQR method
    - Severity classification
    """
    records = []

    for col in _numeric_cols(df):
        series = df[col].dropna()
        if len(series) < 3:          # z-score meaningless on tiny series
            continue

        try:
            z = np.abs(zscore(series, ddof=1))
        except Exception:
            continue

        mask    = z > threshold
        n_out   = int(mask.sum())
        out_pct = round(n_out / len(series) * 100, 2)
        max_z   = round(float(z.max()), 4)

        severity = (
            "🔴 Severe"   if out_pct > 10 else
            "🟠 Moderate" if out_pct > 5  else
            "🟡 Mild"     if out_pct > 0  else
            "✅ None"
        )

        records.append({
            "Column":          col,
            "Z-Score Outliers": n_out,
            "Outlier %":       out_pct,
            "Max |Z-Score|":   max_z,
            "Threshold Used":  threshold,
            "Severity":        severity,
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    return result.sort_values("Z-Score Outliers", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# COMBINED OUTLIER REPORT
# ─────────────────────────────────────────────

def detect_outliers_combined(
    df: pd.DataFrame,
    iqr_multiplier: float = _IQR_MULTIPLIER,
    zscore_threshold: float = _ZSCORE_DEFAULT,
) -> pd.DataFrame:
    """
    Merges IQR and Z-score results into one table.
    Adds an 'Agreement' column flagging columns where both
    methods agree there are outliers — higher confidence signal.
    """
    iqr_df = detect_outliers_iqr(df, iqr_multiplier)[
        ["Column", "Outlier Count", "Outlier %", "Severity"]
    ].rename(columns={
        "Outlier Count": "IQR Outliers",
        "Outlier %":     "IQR %",
        "Severity":      "IQR Severity",
    })

    z_df = detect_outliers_zscore(df, zscore_threshold)[
        ["Column", "Z-Score Outliers", "Outlier %"]
    ].rename(columns={"Outlier %": "Z %"})

    merged = iqr_df.merge(z_df, on="Column", how="outer").fillna(0)

    merged["Both Agree"] = (
        (merged["IQR Outliers"] > 0) & (merged["Z-Score Outliers"] > 0)
    ).map({True: "✅ Yes", False: "—"})

    return merged.sort_values("IQR Outliers", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# SKEWNESS DETECTION
# ─────────────────────────────────────────────

def interpret_skewness(value: float) -> str:
    if value > 2:   return "⛔ Extreme positive skew"
    if value > 1:   return "🔴 High positive skew"
    if value > 0.5: return "🟡 Moderate positive skew"
    if value < -2:  return "⛔ Extreme negative skew"
    if value < -1:  return "🔴 High negative skew"
    if value < -0.5:return "🟡 Moderate negative skew"
    return "✅ Approximately normal"


def _suggest_skew_fix(value: float) -> str:
    if value > 1:
        return "Log / sqrt / Box-Cox transform"
    if value < -1:
        return "Square / reflect + log transform"
    if abs(value) > 0.5:
        return "Mild transform or leave as-is"
    return "No transform needed"


def detect_skewness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skewness analysis with:
    - Kurtosis (tail heaviness)
    - Interpretation labels (6-tier)
    - Suggested fix per column
    - Normality test (Shapiro-Wilk for n≤5000, else skips)
    """
    num_df  = df.select_dtypes(include=_NUMERIC_TYPES)
    records = []

    for col in num_df.columns:
        series = num_df[col].dropna()
        if len(series) < 3:
            continue

        skew_val = round(float(series.skew()), 4)
        kurt_val = round(float(series.kurt()), 4)     # excess kurtosis

        # Normality test
        if 3 <= len(series) <= _SHAPIRO_LIMIT:
            try:
                _, p_val = shapiro(series.sample(min(len(series), _SHAPIRO_LIMIT), random_state=42))
                normal   = "✅ Yes" if p_val > 0.05 else "❌ No"
                p_str    = f"{p_val:.4f}"
            except Exception:
                normal, p_str = "—", "—"
        else:
            normal, p_str = "N/A (n > 5000)", "—"

        records.append({
            "Column":          col,
            "Skewness":        skew_val,
            "Kurtosis":        kurt_val,
            "Interpretation":  interpret_skewness(skew_val),
            "Suggested Fix":   _suggest_skew_fix(skew_val),
            "Normal (p>0.05)": normal,
            "Shapiro p-value": p_str,
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    return result.sort_values("Skewness", key=abs, ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# DUPLICATE DETECTION
# ─────────────────────────────────────────────

def detect_duplicates(df: pd.DataFrame) -> int:
    """Returns count of fully duplicate rows."""
    return int(df.duplicated().sum())


def detect_duplicates_detailed(df: pd.DataFrame) -> dict:
    """
    Extended duplicate analysis:
    - Total duplicate rows
    - Per-column duplicate counts
    - Sample of duplicate rows
    - Estimated memory wasted
    """
    total      = int(df.duplicated().sum())
    total_pct  = round(total / len(df) * 100, 2) if len(df) else 0

    # Per-column: how many rows are duplicated by that column alone
    col_dups = {}
    for col in df.columns:
        n = int(df.duplicated(subset=[col]).sum())
        if n > 0:
            col_dups[col] = n

    col_dup_df = (
        pd.DataFrame
        .from_dict(col_dups, orient="index", columns=["Duplicates"])
        .sort_values("Duplicates", ascending=False)
    )

    memory_wasted_mb = round(
        df[df.duplicated()].memory_usage(deep=True).sum() / 1024 ** 2, 3
    )

    return {
        "total_duplicates":    total,
        "duplicate_percent":   total_pct,
        "clean_rows":          len(df) - total,
        "memory_wasted_mb":    memory_wasted_mb,
        "per_column":          col_dup_df,
        "sample_rows":         df[df.duplicated(keep=False)].head(10),
    }


# ─────────────────────────────────────────────
# VIF — MULTICOLLINEARITY
# ─────────────────────────────────────────────

def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    VIF calculation with:
    - Graceful handling of singular / near-singular matrices
    - Severity labels
    - Recommended action per feature
    - Pairwise Pearson r for context
    """
    num_df = df.select_dtypes(include=_NUMERIC_TYPES).dropna()

    # Need ≥ 2 columns for VIF to make sense
    if num_df.shape[1] < 2:
        return pd.DataFrame({"Info": ["Need ≥ 2 numeric columns for VIF."]})

    # Drop constant columns (cause singular matrix)
    num_df = num_df.loc[:, num_df.nunique() > 1]

    try:
        X = num_df.values.astype(float)
        vif_scores = [
            variance_inflation_factor(X, i)
            for i in range(X.shape[1])
        ]
    except Exception as e:
        logger.warning("VIF computation failed: %s", e)
        return pd.DataFrame({"Error": [f"VIF failed: {e}"]})

    def _severity(v: float) -> str:
        if v > 10:  return "🔴 Severe"
        if v > 5:   return "🟠 Moderate"
        if v > 2.5: return "🟡 Low"
        return "✅ None"

    def _action(v: float) -> str:
        if v > 10:  return "Drop or combine with correlated feature"
        if v > 5:   return "Investigate — consider PCA or feature selection"
        if v > 2.5: return "Monitor — acceptable for most models"
        return "Keep"

    result = pd.DataFrame({
        "Feature":    num_df.columns,
        "VIF":        [round(v, 3) for v in vif_scores],
        "Severity":   [_severity(v) for v in vif_scores],
        "Action":     [_action(v)   for v in vif_scores],
    })

    return result.sort_values("VIF", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# FEATURE CORRELATION PAIRS
# ─────────────────────────────────────────────

def get_high_correlation_pairs(
    df: pd.DataFrame,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Return all unique feature pairs with |correlation| above threshold.
    More useful than a full heatmap when the dataset has many columns.
    """
    num_df = df.select_dtypes(include=_NUMERIC_TYPES)
    corr   = num_df.corr().abs()

    pairs = []
    cols  = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if val >= threshold:
                pairs.append({
                    "Feature A":   cols[i],
                    "Feature B":   cols[j],
                    "|Correlation|": round(val, 4),
                    "Strength": (
                        "🔴 Very strong (≥0.95)" if val >= 0.95 else
                        "🟠 Strong (≥0.85)"      if val >= 0.85 else
                        "🟡 Moderate (≥0.80)"
                    ),
                    "Suggestion": "Consider dropping one or using PCA",
                })

    result = pd.DataFrame(pairs)
    if result.empty:
        return result

    return result.sort_values("|Correlation|", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# DISTRIBUTION COMPARISON (new)
# ─────────────────────────────────────────────

def compare_distributions(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> dict:
    """
    Kolmogorov-Smirnov test comparing two numeric columns.
    Useful for checking if two features come from the same distribution
    or for train/test split drift detection.
    """
    if col_a not in df.columns or col_b not in df.columns:
        return {"error": "One or both columns not found."}

    a = df[col_a].dropna()
    b = df[col_b].dropna()

    if a.empty or b.empty:
        return {"error": "One or both columns are empty after dropping nulls."}

    stat, p_val = ks_2samp(a, b)

    return {
        "column_a":         col_a,
        "column_b":         col_b,
        "ks_statistic":     round(stat, 6),
        "p_value":          round(p_val, 6),
        "same_distribution": p_val > 0.05,
        "interpretation": (
            "✅ Distributions are likely similar (p > 0.05)"
            if p_val > 0.05
            else "⚠️ Distributions are significantly different (p ≤ 0.05)"
        ),
    }


# ─────────────────────────────────────────────
# FULL ANALYTICS SNAPSHOT (single-call helper)
# ─────────────────────────────────────────────

def run_full_analytics(df: pd.DataFrame) -> dict:
    """
    Convenience wrapper — computes all analytics in one call.
    Returns a dict so the Streamlit app can unpack selectively.

    Usage:
        analytics = run_full_analytics(df)
        iqr       = analytics["iqr_outliers"]
        vif       = analytics["vif"]
    """
    return {
        "iqr_outliers":     detect_outliers_iqr(df),
        "zscore_outliers":  detect_outliers_zscore(df),
        "combined_outliers":detect_outliers_combined(df),
        "skewness":         detect_skewness(df),
        "duplicates":       detect_duplicates(df),
        "duplicates_detail":detect_duplicates_detailed(df),
        "vif":              calculate_vif(df),
        "high_corr_pairs":  get_high_correlation_pairs(df),
    }