"""Rule-based natural-language question answering over a DataFrame.

The engine never hardcodes answers. It works in three generic steps:

1. Detect which dataset columns (and optionally which category value,
   e.g. 'Electronics') the question mentions.
2. Detect the intent from keywords (highest/average/most frequent/...).
3. Route to the matching reusable function in :mod:`analysis` and wrap
   the result - raw answer plus supporting groupby table - in a
   :class:`QAResult` that downstream chart/AI layers can consume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

import analysis
from analysis import DatasetError

# Keyword vocabularies used for intent detection.
MAX_WORDS = {"highest", "maximum", "max", "most", "top", "largest", "biggest", "best"}
MIN_WORDS = {"lowest", "minimum", "min", "least", "smallest", "fewest", "bottom", "worst"}
AVG_WORDS = {"average", "mean", "typical"}
TOTAL_WORDS = {"total", "sum", "overall"}
FREQ_WORDS = {"frequently", "frequent", "common", "often", "popular", "appears", "appear"}
COUNT_WORDS = {"count", "counts", "orders", "transactions", "records", "rows", "entries", "number", "times"}
SALES_WORDS = {"sales", "sale", "revenue", "earnings"}
TREND_WORDS = {"trend", "trends", "time", "monthly", "month", "daily"}


@dataclass
class QAResult:
    """Everything downstream layers need about one answered question.

    Attributes:
        question: The original free-text question.
        answer_text: Human-readable one-line answer.
        answer_value: The raw computed number or label.
        supporting_data: Groupby table / value counts / raw values used
            to produce the answer - this is what gets charted.
        chart_kind: Suggested chart type: 'bar', 'pie', 'line', 'hist' or 'none'.
        chart_title: Descriptive title for the generated chart.
        x_label: X-axis label for the chart.
        y_label: Y-axis label for the chart.
        stats: Extra context (shares, totals, runner-up) for the AI explainer.
        success: False when the question could not be understood.
    """

    question: str
    answer_text: str
    answer_value: Any = None
    supporting_data: Optional[pd.Series] = None
    chart_kind: str = "none"
    chart_title: str = ""
    x_label: str = ""
    y_label: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    success: bool = True


def _normalize(text: str) -> str:
    """Lowercase text and collapse punctuation to single spaces.

    Args:
        text: Raw question or column name.

    Returns:
        Normalized string safe for whole-word matching.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _column_aliases(col: str) -> list[str]:
    """Build matchable spellings of a column name (incl. simple plurals).

    Args:
        col: Original column name, e.g. 'customer_age'.

    Returns:
        Aliases like 'customer age' and 'customer ages', longest first so
        the most specific spelling wins during matching.
    """
    base = _normalize(col)
    aliases = {base}
    words = base.split()
    last = words[-1]
    # Naive pluralization of the final word: city -> cities, price -> prices.
    plural = last[:-1] + "ies" if last.endswith("y") else last + "s"
    aliases.add(" ".join(words[:-1] + [plural]))
    return sorted(aliases, key=len, reverse=True)


def detect_columns(question: str, df: pd.DataFrame) -> list[str]:
    """Find dataset columns mentioned in a question, in question order.

    Args:
        question: Free-text question.
        df: DataFrame whose column names are searched for.

    Returns:
        Matched column names ordered by their position in the question.
    """
    padded = f" {_normalize(question)} "
    found: list[tuple[int, str]] = []
    for col in df.columns:
        for alias in _column_aliases(col):
            pos = padded.find(f" {alias} ")
            if pos >= 0:
                found.append((pos, col))
                break
    return [col for _, col in sorted(found)]


def detect_entity(question: str, df: pd.DataFrame) -> Optional[tuple[str, str]]:
    """Find a category *value* (e.g. 'Electronics') mentioned in a question.

    Args:
        question: Free-text question.
        df: DataFrame whose categorical values are scanned.

    Returns:
        Tuple of (column name, matched value) or None if nothing matched.
    """
    padded = f" {_normalize(question)} "
    for col in analysis.categorical_columns(df):
        # Scanning unique values keeps this O(categories), not O(rows).
        for value in df[col].dropna().unique():
            if f" {_normalize(str(value))} " in padded:
                return col, str(value)
    return None


def _split_detected(cols: list[str], df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split detected columns into numeric and categorical groups.

    Args:
        cols: Column names detected in the question.
        df: The DataFrame the columns belong to.

    Returns:
        Tuple of (numeric column names, categorical column names).
    """
    numeric = [c for c in cols if c in analysis.numeric_columns(df)]
    categorical = [c for c in cols if c in analysis.categorical_columns(df)]
    return numeric, categorical


def _resolve_metric(df: pd.DataFrame, words: set[str], numeric: list[str]) -> tuple[pd.DataFrame, Optional[str]]:
    """Pick the numeric column to aggregate, deriving revenue if needed.

    If the question says 'sales'/'revenue' and the dataset has both a
    quantity and a price column, a derived revenue column (quantity *
    price) is added on a copy so the metric matches user intent.

    Args:
        df: The source DataFrame.
        words: Set of normalized words from the question.
        numeric: Numeric columns explicitly mentioned in the question.

    Returns:
        Tuple of (possibly augmented DataFrame, metric column name or None).
    """
    if numeric:
        return df, numeric[0]
    if words & SALES_WORDS:
        cols = {c.lower(): c for c in df.columns}
        if "quantity" in cols and "price" in cols:
            df = df.copy()
            df["revenue"] = df[cols["quantity"]] * df[cols["price"]]
            return df, "revenue"
    return df, None


def answer_question(df: pd.DataFrame, question: str) -> QAResult:
    """Answer a free-text question about a DataFrame.

    Args:
        df: The dataset to analyze.
        question: Free-text question, e.g. 'Which category generated the
            highest sales?'.

    Returns:
        A QAResult with the computed answer, supporting data and chart hints.
        On unrecognized questions, ``success`` is False and ``answer_text``
        explains what the engine can do.
    """
    try:
        words = set(_normalize(question).split())
        cols = detect_columns(question, df)
        numeric, categorical = _split_detected(cols, df)
        df, metric = _resolve_metric(df, words, numeric)

        if words & TREND_WORDS and metric:
            trend = _answer_trend(df, question, metric)
            if trend is not None:
                return trend
        if words & (MAX_WORDS | MIN_WORDS):
            return _answer_extreme(df, question, words, metric, categorical)
        if words & AVG_WORDS and metric:
            return _answer_average(df, question, metric, categorical)
        if words & TOTAL_WORDS and metric:
            return _answer_total(df, question, metric, categorical)
        if words & FREQ_WORDS and categorical:
            return _answer_frequency(df, question, categorical[0])
        # Count-all: "total students", "how many rows", "number of records", etc.
        if words & (TOTAL_WORDS | COUNT_WORDS | {"how", "many", "size", "students", "entries", "total"}):
            if not metric:
                return _answer_row_count(df, question)
        return _fallback(df, question)
    except DatasetError as exc:
        return QAResult(question=question, answer_text=str(exc), success=False)


def _answer_extreme(
    df: pd.DataFrame,
    question: str,
    words: set[str],
    metric: Optional[str],
    categorical: list[str],
) -> QAResult:
    """Handle highest/lowest/most style questions.

    Args:
        df: The dataset.
        question: Original question text.
        words: Normalized words of the question.
        metric: Numeric column to aggregate, if one was resolved.
        categorical: Categorical columns mentioned in the question.

    Returns:
        QAResult for a top/bottom-category or max-row question.
    """
    largest = not (words & MIN_WORDS)
    direction = "highest" if largest else "lowest"
    # 'Which product appears most frequently?' or 'maximum orders' both
    # mean counting rows, not summing a numeric column.
    wants_count = bool(words & (FREQ_WORDS | COUNT_WORDS)) and metric is None
    if categorical and wants_count:
        return _answer_frequency(df, question, categorical[0], largest)
    if categorical and metric:
        # 'highest average price' should rank by mean, not by sum.
        agg = "mean" if words & AVG_WORDS else "sum"
        label, value, grouped = analysis.get_top_category(df, categorical[0], metric, agg, largest)
        return _grouped_result(question, label, value, grouped, categorical[0], metric, agg, direction)
    if metric:
        return _answer_max_row(df, question, metric, largest)
    if categorical:
        return _answer_frequency(df, question, categorical[0], largest)
    return _fallback(df, question)


def _grouped_result(
    question: str,
    label: str,
    value: float,
    grouped: pd.Series,
    group_col: str,
    metric: str,
    agg: str,
    direction: str,
) -> QAResult:
    """Package a groupby-based answer with share stats and chart hints.

    Args:
        question: Original question text.
        label: Winning category label.
        value: Its aggregated value.
        grouped: Full aggregated Series (sorted descending) for charting.
        group_col: The category column that was grouped.
        metric: The numeric column that was aggregated.
        agg: Aggregation used ('sum' or 'mean').
        direction: 'highest' or 'lowest', for phrasing.

    Returns:
        A fully populated QAResult.
    """
    agg_word = "total" if agg == "sum" else "average"
    answer = f"{label} has the {direction} {agg_word} {metric}: {value:,.2f}"
    stats: dict[str, Any] = {"top": label, "group_count": int(len(grouped))}
    if agg == "sum" and float(grouped.sum()):
        # Share of the winner over all groups, useful for the AI explainer.
        stats["share_percent"] = round(float(grouped.iloc[0] / grouped.sum() * 100), 1)
        stats["grand_total"] = round(float(grouped.sum()), 2)
    if len(grouped) > 1:
        stats["runner_up"] = str(grouped.index[1])
        stats["runner_up_value"] = round(float(grouped.iloc[1]), 2)
    return QAResult(
        question=question,
        answer_text=answer,
        answer_value=value,
        supporting_data=grouped,
        chart_kind="bar",
        chart_title=f"{agg_word.title()} {metric} by {group_col}",
        x_label=f"{agg_word.title()} {metric}",
        y_label=group_col,
        stats=stats,
    )


def _answer_average(
    df: pd.DataFrame, question: str, metric: str, categorical: list[str]
) -> QAResult:
    """Handle 'What is the average X (of Y)?' questions.

    Args:
        df: The dataset.
        question: Original question text.
        metric: Numeric column to average.
        categorical: Categorical columns mentioned (used for grouping).

    Returns:
        QAResult with the mean, optionally filtered to a named entity
        (e.g. 'average price of Electronics') or grouped per category.
    """
    entity = detect_entity(question, df)
    if entity:
        col, value = entity
        subset = analysis.filter_by_value(df, col, value)
        if subset.empty:
            raise DatasetError(f"No rows found where {col} is '{value}'.")
        avg = analysis.get_average(subset, metric)
        overall = analysis.get_average(df, metric)
        grouped = analysis.get_group_totals(df, col, metric, "mean")
        return QAResult(
            question=question,
            answer_text=f"The average {metric} for {value} is {avg:,.2f}",
            answer_value=avg,
            supporting_data=grouped,
            chart_kind="bar",
            chart_title=f"Average {metric} by {col}",
            x_label=f"Average {metric}",
            y_label=col,
            stats={"overall_average": round(overall, 2), "entity": value, "rows_matched": int(len(subset))},
        )
    if categorical:
        _, value2, grouped = analysis.get_top_category(df, categorical[0], metric, "mean")
        return _grouped_result(question, str(grouped.index[0]), value2, grouped,
                               categorical[0], metric, "mean", "highest")
    return _answer_overall_average(df, question, metric)


def _answer_overall_average(df: pd.DataFrame, question: str, metric: str) -> QAResult:
    """Handle a plain 'What is the average X?' with no grouping.

    Args:
        df: The dataset.
        question: Original question text.
        metric: Numeric column to average.

    Returns:
        QAResult whose supporting data is the raw values, charted as a
        histogram so the mean has visual context.
    """
    avg = analysis.get_average(df, metric)
    lo, hi = analysis.get_min_max(df, metric)
    return QAResult(
        question=question,
        answer_text=f"The average {metric} is {avg:,.2f}",
        answer_value=avg,
        supporting_data=df[metric].dropna(),
        chart_kind="hist",
        chart_title=f"Distribution of {metric} (mean = {avg:,.2f})",
        x_label=metric,
        y_label="Number of rows",
        stats={"min": lo, "max": hi, "rows": int(df[metric].notna().sum())},
    )


def _answer_total(
    df: pd.DataFrame, question: str, metric: str, categorical: list[str]
) -> QAResult:
    """Handle 'What is the total X (by Y)?' questions.

    Args:
        df: The dataset.
        question: Original question text.
        metric: Numeric column to sum.
        categorical: Categorical columns mentioned (used for grouping).

    Returns:
        QAResult with the grand total; charts a per-category breakdown
        when a category column is available for context.
    """
    total = analysis.get_total(df, metric)
    group_col = categorical[0] if categorical else _default_group_col(df)
    grouped = analysis.get_group_totals(df, group_col, metric, "sum") if group_col else None
    return QAResult(
        question=question,
        answer_text=f"The total {metric} is {total:,.2f}",
        answer_value=total,
        supporting_data=grouped,
        chart_kind="bar" if grouped is not None else "none",
        chart_title=f"Total {metric} by {group_col}" if group_col else "",
        x_label=f"Total {metric}",
        y_label=group_col or "",
        stats={"rows": int(df[metric].notna().sum())},
    )


def _answer_frequency(
    df: pd.DataFrame, question: str, col: str, largest: bool = True
) -> QAResult:
    """Handle 'Which X appears most frequently / has the most orders?'.

    Args:
        df: The dataset.
        question: Original question text.
        col: Categorical column to count.
        largest: True for most frequent, False for least frequent.

    Returns:
        QAResult with the winning category and full value counts.
    """
    counts = analysis.get_value_counts(df, col)
    if counts.empty:
        raise DatasetError(f"Column '{col}' has no values to count.")
    label = str(counts.index[0] if largest else counts.index[-1])
    value = int(counts.iloc[0] if largest else counts.iloc[-1])
    share = round(value / int(counts.sum()) * 100, 1)
    word = "most" if largest else "least"
    return QAResult(
        question=question,
        answer_text=f"{label} appears {word} frequently in '{col}': {value} times ({share}% of rows)",
        answer_value=label,
        supporting_data=counts,
        chart_kind="bar",
        chart_title=f"Number of records by {col}",
        x_label="Number of records",
        y_label=col,
        stats={"top": label, "share_percent": share, "total_rows": int(counts.sum()), "group_count": int(len(counts))},
    )


def _answer_max_row(df: pd.DataFrame, question: str, metric: str, largest: bool) -> QAResult:
    """Handle 'Which row/order has the highest X?' with no grouping column.

    Args:
        df: The dataset.
        question: Original question text.
        metric: Numeric column to rank by.
        largest: True for maximum, False for minimum.

    Returns:
        QAResult describing the extreme row, charted as a histogram.
    """
    row = analysis.get_max_row(df, metric) if largest else df.loc[df[metric].idxmin()]
    value = float(row[metric])
    word = "highest" if largest else "lowest"
    # Show a compact identity for the row using its first few fields.
    ident = ", ".join(f"{k}={row[k]}" for k in list(df.columns)[:3])
    return QAResult(
        question=question,
        answer_text=f"The {word} {metric} is {value:,.2f} ({ident})",
        answer_value=value,
        supporting_data=df[metric].dropna(),
        chart_kind="hist",
        chart_title=f"Distribution of {metric} ({word} = {value:,.2f})",
        x_label=metric,
        y_label="Number of rows",
        stats={"row": {k: str(v) for k, v in row.items()}},
    )


def _answer_trend(df: pd.DataFrame, question: str, metric: str) -> Optional[QAResult]:
    """Handle 'How does X trend over time / by month?' questions.

    Args:
        df: The dataset.
        question: Original question text.
        metric: Numeric column to aggregate over time.

    Returns:
        QAResult with a monthly line chart, or None when the dataset has
        no datetime column (caller then falls through to other intents).
    """
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not date_cols:
        return None
    date_col = date_cols[0]
    # Resample by calendar month via a PeriodIndex for clean axis labels.
    monthly = df.dropna(subset=[date_col]).groupby(df[date_col].dt.to_period("M"))[metric].sum()
    monthly.index = monthly.index.astype(str)
    best = monthly.idxmax()
    return QAResult(
        question=question,
        answer_text=f"Total {metric} peaks in {best} at {monthly.max():,.2f}",
        answer_value=float(monthly.max()),
        supporting_data=monthly,
        chart_kind="line",
        chart_title=f"Total {metric} per month",
        x_label="Month",
        y_label=f"Total {metric}",
        stats={"months": int(len(monthly)), "peak_month": str(best)},
    )


def _default_group_col(df: pd.DataFrame) -> Optional[str]:
    """Pick a reasonable category column for context charts.

    Args:
        df: The dataset.

    Returns:
        The categorical column with the fewest distinct values (nicest to
        chart), or None if there are no categorical columns.
    """
    cats = analysis.categorical_columns(df)
    if not cats:
        return None
    return min(cats, key=lambda c: df[c].nunique())


def _answer_row_count(df: pd.DataFrame, question: str) -> QAResult:
    """Handle 'how many students/records/rows are there?' questions."""
    n = len(df)
    counts = pd.Series({"Total records": n})
    return QAResult(
        question=question,
        answer_text=f"There are {n:,} records in the dataset.",
        answer_value=n,
        supporting_data=counts,
        chart_kind="none",
        chart_title="",
        x_label="",
        y_label="",
        stats={"row_count": n, "column_count": int(df.shape[1])},
    )


def _fallback(df: pd.DataFrame, question: str) -> QAResult:
    """Try an AI answer first; fall back to a guidance message if unavailable.

    Args:
        df: The dataset (used to list its columns).
        question: The question that could not be routed by keyword matching.

    Returns:
        QAResult with an AI-generated answer (success=True) when a key is
        configured, or a guidance message (success=False) otherwise.
    """
    try:
        import ai_explainer  # Lazy import to avoid a load-time circular dependency.
        ai_text = ai_explainer.answer_free_question(df, question)
        if ai_text:
            return QAResult(question=question, answer_text=ai_text, success=True, chart_kind="none")
    except Exception:
        pass
    cols = ", ".join(df.columns)
    return QAResult(
        question=question,
        answer_text=(
            "Sorry, I couldn't understand that question. Try asking about "
            "the highest/lowest, average, total, or most frequent values of a column. "
            f"Available columns: {cols}."
        ),
        success=False,
    )
