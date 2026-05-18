import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


# ─────────────────────────────────────────────
# BASIC INFO
# ─────────────────────────────────────────────

def get_basic_info(df: pd.DataFrame) -> dict:
    """
    Returns a rich summary dict covering shape, memory,
    data types breakdown, and duplicate count.
    """
    total_cells  = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()

    return {
        "rows":               df.shape[0],
        "columns":            df.shape[1],
        "column_names":       list(df.columns),
        "total_cells":        total_cells,
        "missing_cells":      int(missing_cells),
        "missing_percent":    round(missing_cells / total_cells * 100, 2),
        "duplicate_rows":     int(df.duplicated().sum()),
        "memory_usage_mb":    round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3),
        "numeric_columns":    df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
        "datetime_columns":   df.select_dtypes(include="datetime").columns.tolist(),
        "bool_columns":       df.select_dtypes(include="bool").columns.tolist(),
        "constant_columns":   [c for c in df.columns if df[c].nunique() <= 1],
        "high_cardinality_columns": [
            c for c in df.select_dtypes(include="object").columns
            if df[c].nunique() > 50
        ],
    }


# ─────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────

def get_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with dtype, unique count, unique %,
    and a suggested cast for each column.
    """
    rows = df.shape[0]
    records = []

    for col in df.columns:
        dtype       = str(df[col].dtype)
        n_unique    = df[col].nunique(dropna=True)
        unique_pct  = round(n_unique / rows * 100, 2) if rows else 0
        n_missing   = int(df[col].isnull().sum())
        sample_vals = df[col].dropna().head(3).tolist()

        # Suggest a better dtype where applicable
        suggestion = _suggest_dtype(df[col], dtype, n_unique, rows)

        records.append({
            "Column":          col,
            "Current Dtype":   dtype,
            "Unique Values":   n_unique,
            "Unique %":        unique_pct,
            "Missing":         n_missing,
            "Sample Values":   str(sample_vals),
            "Suggested Dtype": suggestion,
        })

    return pd.DataFrame(records)


def _suggest_dtype(series: pd.Series, dtype: str, n_unique: int, n_rows: int) -> str:
    """Heuristic dtype suggestion to reduce memory or improve semantics."""
    if dtype == "object":
        if n_unique / max(n_rows, 1) < 0.05:
            return "category  ← low cardinality, saves memory"
        try:
            pd.to_datetime(series.dropna().head(50), infer_datetime_format=True)
            return "datetime64  ← looks like dates"
        except Exception:
            pass
        try:
            pd.to_numeric(series.dropna().head(50))
            return "numeric  ← looks like numbers stored as strings"
        except Exception:
            pass
        return "—"

    if dtype in ("int64", "int32"):
        mn, mx = series.min(), series.max()
        if mn >= 0 and mx <= 255:
            return "uint8  ← range fits, saves ~8x memory"
        if mn >= -128 and mx <= 127:
            return "int8"
        if mn >= -32768 and mx <= 32767:
            return "int16"
        return "—"

    if dtype == "float64":
        return "float32  ← halves memory, check precision needs"

    return "—"


# ─────────────────────────────────────────────
# MISSING VALUES
# ─────────────────────────────────────────────

def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-column missing value count, percentage,
    and a triage recommendation.
    """
    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df) * 100).round(2)

    recommendations = []
    for col in df.columns:
        pct = missing_pct[col]
        if pct == 0:
            rec = "✅ Complete"
        elif pct < 5:
            rec = "🟡 Impute (low missingness)"
        elif pct < 30:
            rec = "🟠 Impute carefully or flag"
        elif pct < 60:
            rec = "🔴 Consider dropping column"
        else:
            rec = "⛔ Drop column (>60% missing)"
        recommendations.append(rec)

    result = pd.DataFrame({
        "Column":         df.columns,
        "Missing Count":  missing_count.values,
        "Missing %":      missing_pct.values,
        "Recommendation": recommendations,
    })

    return result[result["Missing Count"] > 0].sort_values(
        "Missing %", ascending=False
    ).reset_index(drop=True)


# ─────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────

def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended numeric summary including skewness, kurtosis,
    coefficient of variation, IQR, and range.
    Falls back gracefully if scipy is unavailable.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame({"Info": ["No numerical columns found."]})

    base = numeric_df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T

    # Extra stats
    extra = pd.DataFrame(index=numeric_df.columns)

    try:
        extra["Skewness"]  = numeric_df.apply(lambda s: round(skew(s.dropna()), 4))
        extra["Kurtosis"]  = numeric_df.apply(lambda s: round(kurtosis(s.dropna()), 4))
    except Exception:
        extra["Skewness"]  = numeric_df.skew().round(4)
        extra["Kurtosis"]  = numeric_df.kurt().round(4)

    extra["IQR"]           = (numeric_df.quantile(0.75) - numeric_df.quantile(0.25)).round(4)
    extra["Range"]         = (numeric_df.max() - numeric_df.min()).round(4)
    extra["CV %"]          = (
        (numeric_df.std() / numeric_df.mean().replace(0, np.nan)) * 100
    ).round(2)
    extra["Zeros"]         = (numeric_df == 0).sum()
    extra["Negatives"]     = (numeric_df < 0).sum()

    result = base.join(extra)
    result.index.name = "Column"
    return result.round(4)


# ─────────────────────────────────────────────
# CATEGORICAL SUMMARY
# ─────────────────────────────────────────────

def get_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary for object/category columns:
    unique count, top value, top frequency, entropy, and missing %.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if cat_cols.empty:
        return pd.DataFrame({"Info": ["No categorical columns found."]})

    records = []
    for col in cat_cols:
        series      = df[col].dropna()
        n_unique    = series.nunique()
        top_val     = series.mode().iloc[0] if not series.empty else "—"
        top_freq    = int((series == top_val).sum())
        top_pct     = round(top_freq / len(df) * 100, 2)
        missing_pct = round(df[col].isnull().mean() * 100, 2)

        # Shannon entropy (diversity measure)
        try:
            freq   = series.value_counts(normalize=True)
            entropy = round(float(-(freq * np.log2(freq + 1e-10)).sum()), 4)
        except Exception:
            entropy = None

        records.append({
            "Column":       col,
            "Unique Values": n_unique,
            "Top Value":    str(top_val),
            "Top Freq":     top_freq,
            "Top %":        top_pct,
            "Entropy (bits)": entropy,
            "Missing %":    missing_pct,
        })

    return pd.DataFrame(records).sort_values("Unique Values", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# COLUMN QUALITY REPORT
# ─────────────────────────────────────────────

def get_column_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    A single comprehensive quality score per column (0–100).
    Penalizes missing values, low variance, high cardinality in categoricals,
    and constant columns.
    """
    records = []

    for col in df.columns:
        series      = df[col]
        score       = 100
        flags       = []

        # Missing
        miss_pct = series.isnull().mean() * 100
        if miss_pct > 0:
            penalty = min(miss_pct * 1.5, 50)
            score  -= penalty
            flags.append(f"{miss_pct:.1f}% missing")

        # Constant / near-constant
        n_unique = series.nunique(dropna=True)
        if n_unique <= 1:
            score -= 30
            flags.append("constant column")
        elif n_unique / len(df) < 0.01:
            score -= 10
            flags.append("near-constant")

        # Numeric-specific
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 1:
                cv = non_null.std() / abs(non_null.mean()) if non_null.mean() != 0 else 0
                if cv < 0.01:
                    score -= 10
                    flags.append("very low variance")
                try:
                    sk = abs(skew(non_null))
                    if sk > 3:
                        score -= 10
                        flags.append(f"extreme skew ({sk:.1f})")
                except Exception:
                    pass

        # Categorical-specific
        if pd.api.types.is_object_dtype(series):
            if n_unique > 100:
                score -= 15
                flags.append(f"very high cardinality ({n_unique})")

        score = max(round(score, 1), 0)
        grade = (
            "A" if score >= 90 else
            "B" if score >= 75 else
            "C" if score >= 60 else
            "D" if score >= 40 else
            "F"
        )

        records.append({
            "Column":        col,
            "Dtype":         str(series.dtype),
            "Quality Score": score,
            "Grade":         grade,
            "Flags":         ", ".join(flags) if flags else "—",
        })

    return (
        pd.DataFrame(records)
        .sort_values("Quality Score", ascending=True)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────
# DUPLICATE ANALYSIS
# ─────────────────────────────────────────────

def get_duplicate_analysis(df: pd.DataFrame) -> dict:
    """
    Returns full duplicate diagnostics including which columns
    drive the most duplications.
    """
    total_dups = int(df.duplicated().sum())
    dup_pct    = round(total_dups / len(df) * 100, 2)

    # Which column subsets produce duplicates
    col_dup_counts = {}
    for col in df.columns:
        n = int(df.duplicated(subset=[col]).sum())
        if n > 0:
            col_dup_counts[col] = n

    col_dup_df = (
        pd.DataFrame
        .from_dict(col_dup_counts, orient="index", columns=["Duplicates by Column"])
        .sort_values("Duplicates by Column", ascending=False)
    )

    return {
        "total_duplicates":    total_dups,
        "duplicate_percent":   dup_pct,
        "clean_rows":          len(df) - total_dups,
        "duplicate_by_column": col_dup_df,
        "sample_duplicates":   df[df.duplicated(keep=False)].head(10),
    }


# ─────────────────────────────────────────────
# FAST DATASET SNAPSHOT  (single-call summary)
# ─────────────────────────────────────────────

def get_dataset_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-stop per-column overview: dtype, missing, unique,
    min, max, mean (numeric), top value (categorical).
    Useful for a quick executive overview table.
    """
    records = []
    for col in df.columns:
        series   = df[col]
        dtype    = str(series.dtype)
        missing  = int(series.isnull().sum())
        miss_pct = round(missing / len(df) * 100, 2)
        n_unique = series.nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(series):
            col_min  = round(series.min(), 4)
            col_max  = round(series.max(), 4)
            col_mean = round(series.mean(), 4)
            top_val  = "—"
        else:
            col_min  = "—"
            col_max  = "—"
            col_mean = "—"
            top_val  = str(series.mode().iloc[0]) if not series.dropna().empty else "—"

        records.append({
            "Column":    col,
            "Dtype":     dtype,
            "Missing":   missing,
            "Missing %": miss_pct,
            "Unique":    n_unique,
            "Min":       col_min,
            "Max":       col_max,
            "Mean":      col_mean,
            "Top Value": top_val,
        })

    return pd.DataFrame(records)