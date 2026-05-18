import os
import time
import logging
from functools import lru_cache
from typing import Any

import pandas as pd
from groq import Groq, APIError, RateLimitError, APIConnectionError
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not _GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. "
        "Add it to your .env file: GROQ_API_KEY=your_key_here"
    )

_client = Groq(api_key=_GROQ_API_KEY)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

_DEFAULT_MODEL       = "llama-3.3-70b-versatile"
_FAST_MODEL          = "llama-3.1-8b-instant"   # fallback for quick calls
_DEFAULT_TEMPERATURE = 0.3
_MAX_RETRIES         = 3
_RETRY_DELAY         = 2        # seconds between retries
_MAX_TOKENS          = 4096
_STAT_ROWS           = 8        # max stat rows sent to LLM
_CORR_COLS           = 10       # max corr columns sent to LLM


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _safe_df_str(obj: Any, max_rows: int = _STAT_ROWS) -> str:
    """
    Safely convert a DataFrame / Series / dict to a
    compact string, truncating to max_rows to control token use.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.head(max_rows).to_string()
    if isinstance(obj, pd.Series):
        return obj.head(max_rows).to_string()
    if isinstance(obj, dict):
        items = list(obj.items())[:max_rows]
        return "\n".join(f"  {k}: {v}" for k, v in items)
    return str(obj)[:2000]          # hard cap for raw strings


def _trim_correlation(corr: Any, max_cols: int = _CORR_COLS) -> str:
    """Keep only the top-N columns of the correlation matrix."""
    if isinstance(corr, pd.DataFrame):
        cols = corr.columns[:max_cols]
        return corr.loc[cols, cols].round(3).to_string()
    return _safe_df_str(corr)


def _call_llm(
    messages: list[dict],
    model: str = _DEFAULT_MODEL,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _MAX_TOKENS,
    use_fast_model_on_retry: bool = True,
) -> str:
    """
    Robust LLM caller with:
    - Exponential back-off retry (up to _MAX_RETRIES)
    - Automatic fallback to fast model on rate-limit
    - Structured error messages instead of raw exceptions
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            current_model = (
                _FAST_MODEL
                if (attempt > 1 and use_fast_model_on_retry)
                else model
            )
            logger.info("LLM call attempt %d / %d — model: %s", attempt, _MAX_RETRIES, current_model)

            response = _client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except RateLimitError:
            wait = _RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning("Rate limit hit. Retrying in %ds…", wait)
            time.sleep(wait)

        except APIConnectionError as e:
            logger.error("Connection error: %s", e)
            if attempt == _MAX_RETRIES:
                return "⚠️ Could not connect to the AI service. Check your internet connection."
            time.sleep(_RETRY_DELAY)

        except APIError as e:
            logger.error("API error: %s", e)
            return f"⚠️ AI service returned an error: {e}"

        except Exception as e:
            logger.exception("Unexpected error during LLM call")
            return f"⚠️ Unexpected error: {e}"

    return "⚠️ All retry attempts failed. Please try again later."


def _build_system_message(role_description: str) -> dict:
    return {
        "role": "system",
        "content": (
            f"{role_description}\n\n"
            "Always respond in well-structured Markdown. "
            "Be concise, beginner-friendly, and avoid jargon. "
            "Never hallucinate — if data is missing, say so clearly."
        ),
    }


# ─────────────────────────────────────────────
# DATASET INSIGHTS
# ─────────────────────────────────────────────

def generate_dataset_insights(
    rows: int,
    columns: int,
    missing_values: Any,
    summary_stats: Any,
    extra_context: str = "",
) -> str:
    """
    Generate a structured AI analysis of a dataset.
    Includes trends, outlier hints, distribution notes,
    data quality issues, preprocessing steps, and business insights.

    Args:
        rows:           Number of rows in the dataset.
        columns:        Number of columns.
        missing_values: DataFrame / Series of missing value counts.
        summary_stats:  DataFrame from df.describe() or get_summary_statistics().
        extra_context:  Optional free-text extra info (domain, target column, etc.)
    """
    missing_str = _safe_df_str(missing_values)
    stats_str   = _safe_df_str(summary_stats)

    prompt = f"""
## Dataset Overview
- **Rows:** {rows:,}
- **Columns:** {columns}

## Missing Values
```
{missing_str}
```

## Summary Statistics
```
{stats_str}
```

{"## Additional Context\n" + extra_context if extra_context else ""}

---

Please provide a thorough analysis covering:

### 1. 📈 Key Trends
Identify the most important patterns or trends visible in the statistics.

### 2. 🚨 Outlier Signals
Based on mean/std/min/max ranges, flag columns likely to contain outliers.

### 3. 📊 Distribution Notes
Comment on skewed distributions and what they mean for modelling.

### 4. 🧹 Data Quality Issues
List any data quality concerns (missing values, suspicious ranges, etc.).

### 5. ⚙️ Preprocessing Recommendations
Concrete, step-by-step preprocessing suggestions before model building.

### 6. 💼 Business / Domain Insights
What would a domain expert notice first? What story does the data tell?

Keep each section to 3–5 bullet points. Use plain language suitable for beginners.
"""

    messages = [
        _build_system_message("You are a senior data scientist reviewing a dataset for a junior analyst."),
        {"role": "user", "content": prompt},
    ]
    return _call_llm(messages)


# ─────────────────────────────────────────────
# CONVERSATIONAL CHATBOT
# ─────────────────────────────────────────────

def chat_with_dataset(
    user_question: str,
    rows: int,
    columns: int,
    column_names: list[str],
    missing_values: Any,
    summary_stats: Any,
    correlation_matrix: Any,
    chat_history: list[dict],
    max_history_turns: int = 6,
) -> str:
    """
    Answer a free-form question about the dataset using conversation history.

    Args:
        user_question:      The latest user message.
        ...                 Dataset statistics (passed from app state).
        chat_history:       List of {"role": ..., "content": ...} dicts.
        max_history_turns:  How many past turns to include (controls token use).
    """
    # Compact context block — injected once as a system addendum
    context_block = f"""
<dataset_context>
Rows: {rows:,}
Columns: {columns}
Column names: {', '.join(str(c) for c in column_names)}

Missing values (top columns):
{_safe_df_str(missing_values, max_rows=6)}

Summary statistics:
{_safe_df_str(summary_stats, max_rows=6)}

Correlation matrix (first {_CORR_COLS} columns):
{_trim_correlation(correlation_matrix)}
</dataset_context>
"""

    system_msg = _build_system_message(
        "You are an expert AI data analyst embedded in an EDA tool.\n"
        + context_block
        + "\nAnswer the user's questions about THEIR dataset using only the context above. "
        "If you cannot determine something from the provided statistics, say so clearly."
    )

    # Trim old history to control token usage (keep last N turns = 2*N messages)
    trimmed_history = chat_history[-(max_history_turns * 2):]

    # Filter out non-standard roles (safety check)
    safe_history = [
        m for m in trimmed_history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    messages = [system_msg] + safe_history + [
        {"role": "user", "content": user_question}
    ]

    return _call_llm(messages, temperature=0.4)


# ─────────────────────────────────────────────
# ML RECOMMENDATIONS
# ─────────────────────────────────────────────

def generate_ml_recommendations(
    rows: int,
    columns: int,
    column_names: list[str],
    dtypes: Any,
    missing_values: Any,
    summary_stats: Any,
    correlation_matrix: Any,
    outliers_iqr: Any,
    skewness: Any,
    duplicates: int,
    target_column: str = "",
) -> str:
    """
    Generate a structured ML readiness report with preprocessing
    advice and model suggestions tailored to the dataset.

    Args:
        target_column: Optional. If provided, the LLM will tailor
                       problem-type detection around this column.
    """
    prompt = f"""
## Dataset Facts
| Property | Value |
|---|---|
| Rows | {rows:,} |
| Columns | {columns} |
| Duplicate rows | {duplicates} |
| Target column | {target_column or "Not specified"} |

## Column Names
{', '.join(str(c) for c in column_names)}

## Data Types
```
{_safe_df_str(dtypes)}
```

## Missing Values
```
{_safe_df_str(missing_values)}
```

## Summary Statistics
```
{_safe_df_str(summary_stats)}
```

## Correlation Matrix (top {_CORR_COLS} columns)
```
{_trim_correlation(correlation_matrix)}
```

## IQR Outlier Summary
```
{_safe_df_str(outliers_iqr)}
```

## Skewness
```
{_safe_df_str(skewness)}
```

---

Generate a structured ML readiness report with EXACTLY these sections:

# 📊 1. Dataset Quality Overview
- Overall quality rating: Excellent / Good / Fair / Poor
- Top 3 issues that must be fixed before modelling
- Is the dataset ready to model as-is? (Yes / No / With minor fixes)

# 🧹 2. Missing Value Action Plan
For each column with missing values, suggest ONE specific strategy:
- Median imputation (for skewed numeric)
- Mean imputation (for normal numeric)
- Mode imputation (for categorical)
- Drop column (>60% missing)
- Flag as separate binary column

# 🚨 3. Outlier Handling Plan
- Which columns have outliers?
- Severity: mild / moderate / severe
- Recommended action: cap (Winsorize) / log-transform / keep / investigate

# 🔥 4. Feature Engineering Hints
- Which columns are likely most predictive?
- Any obvious interaction features worth creating?
- Columns to drop (constant, duplicate, high-cardinality IDs)

# ⚙️ 5. Preprocessing Pipeline (in order)
List the exact steps in order:
1. Step
2. Step
...

# 🤖 6. Recommended ML Problem Type
Based on the target column (if specified) or general dataset structure:
- Problem type: Classification / Regression / Clustering / Other
- Reasoning in 2 sentences

# 🧠 7. Top 3 Suggested Models
For each model:
- **Model Name**
- Why it suits THIS dataset specifically (1–2 sentences)
- One key hyperparameter to tune first

# ✅ 8. Final Checklist Before Training
Provide a 5–8 point checklist the user should verify before fitting any model.

---

Rules:
- Use plain language — assume the reader is a first-year ML student
- No code
- All sections must be present even if data is insufficient (say "insufficient data to assess")
- Be specific to THIS dataset, not generic advice
"""

    messages = [
        _build_system_message(
            "You are a senior ML engineer writing a dataset readiness report "
            "for a junior data science team. Be specific, structured, and actionable."
        ),
        {"role": "user", "content": prompt},
    ]
    return _call_llm(messages, max_tokens=_MAX_TOKENS)


# ─────────────────────────────────────────────
# COLUMN EXPLAINER  (new)
# ─────────────────────────────────────────────

def explain_column(
    column_name: str,
    dtype: str,
    sample_values: list,
    missing_pct: float,
    stats: dict,
) -> str:
    """
    Ask the LLM to explain what a single column likely represents,
    its data quality, and how to handle it.
    Useful for the 'Chat' tab's one-click 'Explain this column' button.
    """
    prompt = f"""
A dataset contains a column with the following properties:

- **Column name:** `{column_name}`
- **Data type:** {dtype}
- **Missing values:** {missing_pct:.1f}%
- **Sample values:** {sample_values[:10]}
- **Statistics:** {stats}

Please explain:
1. What this column likely represents in plain English
2. Any data quality concerns
3. Whether it should be kept, transformed, or dropped
4. How to handle missing values if any
5. Whether it might be a useful feature for machine learning

Keep it under 200 words and beginner-friendly.
"""
    messages = [
        _build_system_message("You are a data analyst explaining a dataset column to a junior analyst."),
        {"role": "user", "content": prompt},
    ]
    return _call_llm(messages, model=_FAST_MODEL, max_tokens=512)


# ─────────────────────────────────────────────
# ANOMALY NARRATIVE  (new)
# ─────────────────────────────────────────────

def generate_anomaly_narrative(
    iqr_outliers: Any,
    zscore_outliers: Any,
    skewness: Any,
    duplicates: int,
) -> str:
    """
    Convert raw anomaly detection results into a human-readable
    narrative paragraph for the Smart Analytics tab.
    """
    prompt = f"""
The following anomaly detection results were found in a dataset:

**Duplicate rows:** {duplicates}

**IQR Outliers:**
{_safe_df_str(iqr_outliers, max_rows=10)}

**Z-Score Outliers:**
{_safe_df_str(zscore_outliers, max_rows=10)}

**Skewness per column:**
{_safe_df_str(skewness, max_rows=10)}

Write a short (3–5 paragraph) narrative report summarising:
1. The overall data health from an anomaly perspective
2. Which columns need the most attention and why
3. Whether the anomalies are likely data errors or genuine extreme values
4. What a data scientist should do first

Write in plain English. No bullet points — use flowing prose paragraphs.
"""
    messages = [
        _build_system_message("You are a data quality analyst writing an anomaly summary report."),
        {"role": "user", "content": prompt},
    ]
    return _call_llm(messages, model=_FAST_MODEL, max_tokens=800)