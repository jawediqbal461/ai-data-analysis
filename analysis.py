"""Dataset loading and statistical analysis helpers.

Every function here is dataset-agnostic: pass any DataFrame plus column
names and it will compute totals, averages, extremes, value counts and
category distributions. Nothing is hardcoded to the bundled sample CSV.
"""

from __future__ import annotations

import os
from typing import IO, Any, Union

import numpy as np
import pandas as pd

# A CSV source can be a filesystem path or a file-like object
# (e.g. Streamlit's UploadedFile).
CsvSource = Union[str, IO[bytes], IO[str]]


class DatasetError(Exception):
    """Raised when a dataset cannot be loaded or a computation is invalid.

    The message is always written for end users, so callers can display
    ``str(exc)`` directly instead of a stack trace.
    """


def load_dataset(source: CsvSource) -> pd.DataFrame:
    """Load a CSV into a DataFrame with friendly error handling.

    Args:
        source: Path to a CSV file, or an open file-like object.

    Returns:
        The loaded DataFrame with date-like columns parsed to datetime.

    Raises:
        DatasetError: If the file is missing, empty, malformed, or uses
            an encoding that cannot be decoded.
    """
    if isinstance(source, str) and not os.path.exists(source):
        raise DatasetError(
            f"Could not find the file '{source}'. "
            "Check the path and try again."
        )
    try:
        df = pd.read_csv(source)
    except UnicodeDecodeError:
        df = _read_with_fallback_encoding(source)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("The CSV file is empty - there is nothing to analyze.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetError(
            "The file could not be parsed as CSV. "
            "Make sure it is a valid comma-separated file."
        ) from exc
    if df.empty:
        raise DatasetError("The CSV was read successfully but contains no data rows.")
    return _coerce_date_columns(df)


def _read_with_fallback_encoding(source: CsvSource) -> pd.DataFrame:
    """Retry reading a CSV with the permissive latin-1 encoding.

    Args:
        source: Path or file-like object that failed to decode as UTF-8.

    Returns:
        The DataFrame read with latin-1 encoding.

    Raises:
        DatasetError: If the fallback read also fails.
    """
    try:
        # File-like objects need their cursor reset after the failed read.
        if hasattr(source, "seek"):
            source.seek(0)  # type: ignore[union-attr]
        return pd.read_csv(source, encoding="latin-1")
    except Exception as exc:
        raise DatasetError(
            "The file's text encoding could not be understood. "
            "Try re-saving it as UTF-8 CSV."
        ) from exc


def _coerce_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns whose name mentions 'date' to datetime dtype.

    Args:
        df: The freshly loaded DataFrame.

    Returns:
        The same DataFrame with date-like text columns parsed in place.
    """
    for col in df.columns:
        if "date" in col.lower() and df[col].dtype == object:
            # errors="coerce" turns unparseable entries into NaT instead
            # of raising, so one bad cell never breaks the whole load.
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize a DataFrame's shape, schema and missing values.

    Args:
        df: The DataFrame to summarize.

    Returns:
        Dict with row_count, column_count, columns, dtypes (as strings)
        and missing_counts (per-column count of null cells).
    """
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col: int(n) for col, n in df.isna().sum().items()},
    }


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """List the numeric columns of a DataFrame.

    Args:
        df: The DataFrame to inspect.

    Returns:
        Column names with a numeric dtype.
    """
    return list(df.select_dtypes(include=[np.number]).columns)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """List the text/categorical columns of a DataFrame.

    Args:
        df: The DataFrame to inspect.

    Returns:
        Column names with object or categorical dtype.
    """
    return list(df.select_dtypes(include=["object", "category"]).columns)


def _require_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric column or raise a friendly error.

    Args:
        df: The DataFrame to read from.
        col: The column that must exist and be numeric.

    Returns:
        The column as a Series with nulls dropped.

    Raises:
        DatasetError: If the column is missing or not numeric.
    """
    if col not in df.columns:
        raise DatasetError(f"The dataset has no column named '{col}'.")
    if col not in numeric_columns(df):
        raise DatasetError(f"Column '{col}' is not numeric, so this statistic cannot be computed.")
    return df[col].dropna()


def get_numeric_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize every numeric column with count/mean/std/min/max.

    Args:
        df: The dataset to summarize.

    Returns:
        DataFrame indexed by column name with the summary statistics,
        or an empty DataFrame if the dataset has no numeric columns.
    """
    cols = numeric_columns(df)
    if not cols:
        return pd.DataFrame()
    return df[cols].describe().T[["count", "mean", "std", "min", "max"]].round(2)


def get_categorical_overview(
    df: pd.DataFrame, max_categories: int = 8, max_unique: int = 30
) -> dict[str, pd.DataFrame]:
    """Summarize each categorical column's top values with count and share.

    Columns that look like identifiers (almost as many unique values as
    rows) or that have only one distinct value are skipped since they
    carry no distributional information worth charting or tabulating.

    Args:
        df: The dataset to summarize.
        max_categories: Maximum number of top values to keep per column.
        max_unique: Skip columns with more distinct values than this.

    Returns:
        Dict mapping column name to its top-values distribution table.
    """
    overview: dict[str, pd.DataFrame] = {}
    for col in categorical_columns(df):
        nunique = df[col].nunique(dropna=True)
        if nunique < 2 or nunique > max_unique:
            continue
        overview[col] = get_category_distribution(df, col).head(max_categories)
    return overview


def get_average(df: pd.DataFrame, col: str) -> float:
    """Compute the mean of a numeric column, ignoring missing values.

    Args:
        df: The DataFrame to read from.
        col: Name of the numeric column.

    Returns:
        The column's mean as a float.
    """
    return float(_require_numeric(df, col).mean())


def get_total(df: pd.DataFrame, col: str) -> float:
    """Compute the sum of a numeric column, ignoring missing values.

    Args:
        df: The DataFrame to read from.
        col: Name of the numeric column.

    Returns:
        The column's total as a float.
    """
    return float(_require_numeric(df, col).sum())


def get_min_max(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Compute the minimum and maximum of a numeric column.

    Args:
        df: The DataFrame to read from.
        col: Name of the numeric column.

    Returns:
        Tuple of (min, max) as floats.
    """
    series = _require_numeric(df, col)
    return float(series.min()), float(series.max())


def get_max_row(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the full row where a numeric column reaches its maximum.

    Args:
        df: The DataFrame to read from.
        col: Name of the numeric column.

    Returns:
        The row (as a Series) containing the maximum value of ``col``.
    """
    _require_numeric(df, col)
    # idxmax skips NaN and returns the index label of the largest value.
    return df.loc[df[col].idxmax()]


def get_value_counts(df: pd.DataFrame, col: str) -> pd.Series:
    """Count occurrences of each value in a column, most frequent first.

    Args:
        df: The DataFrame to read from.
        col: Name of the column to count.

    Returns:
        Series indexed by value with occurrence counts, descending.
    """
    if col not in df.columns:
        raise DatasetError(f"The dataset has no column named '{col}'.")
    return df[col].value_counts()


def get_category_distribution(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Compute count and percentage share for each value of a column.

    Args:
        df: The DataFrame to read from.
        col: Name of the categorical column.

    Returns:
        DataFrame indexed by category with 'count' and 'percent' columns.
    """
    counts = get_value_counts(df, col)
    # Percentages are taken over non-null rows so they always sum to 100.
    percent = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"count": counts, "percent": percent})


def get_group_totals(
    df: pd.DataFrame, group_col: str, value_col: str, agg: str = "sum"
) -> pd.Series:
    """Aggregate a numeric column per category, sorted descending.

    Args:
        df: The DataFrame to read from.
        group_col: Categorical column to group by.
        value_col: Numeric column to aggregate.
        agg: Aggregation name understood by pandas ('sum', 'mean', ...).

    Returns:
        Series indexed by category with the aggregated values, descending.
    """
    if group_col not in df.columns:
        raise DatasetError(f"The dataset has no column named '{group_col}'.")
    _require_numeric(df, value_col)
    # groupby drops NaN group keys by default, which is what we want here.
    grouped = df.groupby(group_col)[value_col].agg(agg)
    return grouped.sort_values(ascending=False)


def get_top_category(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    agg: str = "sum",
    largest: bool = True,
) -> tuple[str, float, pd.Series]:
    """Find the category with the highest (or lowest) aggregated value.

    Args:
        df: The DataFrame to read from.
        group_col: Categorical column to group by.
        value_col: Numeric column to aggregate.
        agg: Aggregation name ('sum', 'mean', 'max', ...).
        largest: True for the top category, False for the bottom one.

    Returns:
        Tuple of (category label, aggregated value, full groupby Series
        sorted descending) - the Series is handy for charting.
    """
    grouped = get_group_totals(df, group_col, value_col, agg)
    if grouped.empty:
        raise DatasetError(f"No data available to group '{value_col}' by '{group_col}'.")
    label = grouped.index[0] if largest else grouped.index[-1]
    return str(label), float(grouped.loc[label]), grouped


def filter_by_value(df: pd.DataFrame, col: str, value: str) -> pd.DataFrame:
    """Filter rows where a column equals a value (case-insensitive).

    Args:
        df: The DataFrame to filter.
        col: Column to match against.
        value: Value to look for (compared as lowercase strings).

    Returns:
        The matching subset of the DataFrame.
    """
    if col not in df.columns:
        raise DatasetError(f"The dataset has no column named '{col}'.")
    # Compare as lowercase strings so 'electronics' matches 'Electronics'.
    mask = df[col].astype(str).str.lower() == str(value).lower()
    return df[mask]
