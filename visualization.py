"""Chart generation for QA results using matplotlib + seaborn.

Design rules applied to every chart:

* seaborn's default palette (no garish colors), one hue per measure -
  bars that compare a single measure share a single color,
* descriptive title, labeled axes, recessive grid, no chart junk,
* categories sorted by value with direct value labels on bars,
* ``plt.tight_layout()`` and a timestamped PNG saved to ``charts/``.
"""

from __future__ import annotations

import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # Render off-screen; works headless and in Streamlit.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import analysis
from qa_engine import QAResult

CHARTS_DIR = "charts"
MAX_BARS = 12  # Cap categories per chart so labels stay readable.

sns.set_theme(style="whitegrid", palette="deep")
# First color of seaborn's default 'deep' palette - used for single-measure marks.
ACCENT = sns.color_palette("deep")[0]


def _ensure_charts_dir() -> str:
    """Create the charts output folder if it does not exist yet.

    Returns:
        The path of the charts directory.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return CHARTS_DIR


def save_chart(fig: plt.Figure, prefix: str = "chart") -> str:
    """Save a figure to charts/ as a timestamped PNG.

    Args:
        fig: The matplotlib figure to save.
        prefix: Filename prefix describing the chart.

    Returns:
        The path of the saved PNG file.
    """
    _ensure_charts_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(CHARTS_DIR, f"{prefix}_{stamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def bar_chart(data: pd.Series, title: str, x_label: str, y_label: str) -> plt.Figure:
    """Draw a horizontal bar chart of one measure across categories.

    Args:
        data: Series indexed by category, sorted descending.
        title: Descriptive chart title.
        x_label: Label for the value axis.
        y_label: Label for the category axis.

    Returns:
        The finished matplotlib figure.
    """
    data = data.head(MAX_BARS)
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.5 * len(data) + 1)))
    # One color for one measure: differences are carried by length, not hue.
    ax.barh(data.index.astype(str), data.values, color=ACCENT, height=0.62)
    ax.invert_yaxis()  # Largest category on top.
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Direct value labels at bar ends replace squinting at the axis.
    for i, value in enumerate(data.values):
        ax.text(value, i, f" {value:,.0f}", va="center", fontsize=9, color="dimgray")
    ax.grid(axis="y", visible=False)  # Keep only the recessive value-axis grid.
    sns.despine(fig=fig, left=True, bottom=True)
    fig.tight_layout()
    return fig


def pie_chart(data: pd.Series, title: str) -> plt.Figure:
    """Draw a pie chart of category shares (falls back to bar if crowded).

    Args:
        data: Series indexed by category with positive values.
        title: Descriptive chart title.

    Returns:
        The finished matplotlib figure.
    """
    if len(data) > 6:
        # Pies with many slices are unreadable - a bar chart says it better.
        return bar_chart(data, title, "Value", str(data.index.name or "Category"))
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette("deep", len(data))
    ax.pie(
        data.values,
        labels=[str(i) for i in data.index],
        autopct="%1.1f%%",
        colors=colors,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},  # gaps between slices
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def line_chart(data: pd.Series, title: str, x_label: str, y_label: str) -> plt.Figure:
    """Draw a line chart of a measure over an ordered index (e.g. months).

    Args:
        data: Series indexed by ordered labels (time periods).
        title: Descriptive chart title.
        x_label: Label for the x axis.
        y_label: Label for the y axis.

    Returns:
        The finished matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(data.index.astype(str), data.values, color=ACCENT, linewidth=2, marker="o", markersize=5)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=45)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def histogram(data: pd.Series, title: str, x_label: str, y_label: str) -> plt.Figure:
    """Draw a histogram of raw numeric values with a mean marker line.

    Args:
        data: Raw numeric values (nulls already dropped).
        title: Descriptive chart title.
        x_label: Label for the value axis.
        y_label: Label for the count axis.

    Returns:
        The finished matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data, bins=20, color=ACCENT, edgecolor="white", ax=ax)
    mean = float(data.mean())
    ax.axvline(mean, color="dimgray", linestyle="--", linewidth=1.5)
    ax.text(mean, ax.get_ylim()[1] * 0.95, f" mean = {mean:,.2f}", fontsize=9, color="dimgray")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def box_chart(data: pd.Series, title: str, x_label: str, y_label: str) -> plt.Figure:
    """Draw a horizontal box plot summarizing a numeric distribution's spread.

    Args:
        data: Raw numeric values (nulls already dropped).
        title: Descriptive chart title.
        x_label: Label for the value axis.
        y_label: Label for the category axis (usually blank for one box).

    Returns:
        The finished matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(x=data, color=ACCENT, ax=ax, width=0.4)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    sns.despine(fig=fig, left=True)
    fig.tight_layout()
    return fig


def _render_one(data: pd.Series, kind: str, result: QAResult) -> plt.Figure:
    """Dispatch to the matplotlib builder for a single chart kind.

    Args:
        data: Supporting data to plot.
        kind: One of 'bar', 'pie', 'line', 'hist', 'box'.
        result: The QAResult providing title/axis labels.

    Returns:
        The finished matplotlib figure for that chart kind.
    """
    if kind == "pie":
        return pie_chart(data, result.chart_title)
    if kind == "line":
        return line_chart(data, result.chart_title, result.x_label, result.y_label)
    if kind == "hist":
        return histogram(data, result.chart_title, result.x_label, result.y_label)
    if kind == "box":
        return box_chart(data, f"Spread of {result.x_label}", result.x_label, "")
    return bar_chart(data, result.chart_title, result.x_label, result.y_label)  # bar default


def _companion_kinds(result: QAResult) -> list[str]:
    """Pick complementary chart kinds to accompany the QA engine's primary pick.

    Each question gets more than one automatically chosen visualization so
    the user sees the same answer from a couple of useful angles, without
    ever having to pick a chart type themselves.

    Args:
        result: The answered question with its suggested chart_kind and data.

    Returns:
        Ordered list of chart kinds to render (primary kind first).
    """
    kind = result.chart_kind
    data = result.supporting_data
    if kind == "bar":
        # A pie view works only when the category count stays readable.
        return ["bar", "pie"] if data is not None and len(data) <= 8 else ["bar"]
    if kind == "pie":
        return ["pie", "bar"]
    if kind == "line":
        return ["line", "bar"]
    if kind == "hist":
        return ["hist", "box"]
    return [kind]


def make_overview_charts(
    df: pd.DataFrame, max_categorical: int = 3, max_numeric: int = 2
) -> list[tuple[plt.Figure, str, str, str]]:
    """Build an automatic set of charts summarizing an uploaded dataset.

    Shown right after a file loads, before any question is asked: one bar
    chart per informative categorical column (fewest distinct values
    first, so the most readable charts come first) and one histogram per
    numeric column.

    Args:
        df: The freshly loaded dataset.
        max_categorical: Maximum number of categorical bar charts to build.
        max_numeric: Maximum number of numeric histograms to build.

    Returns:
        List of (figure, saved PNG path, chart kind, column name) tuples.
    """
    charts: list[tuple[plt.Figure, str, str, str]] = []

    cat_overview = analysis.get_categorical_overview(df)
    # Fewest-category columns first: they make the cleanest bar charts.
    cat_cols = sorted(cat_overview, key=lambda c: len(cat_overview[c]))[:max_categorical]
    for col in cat_cols:
        counts = analysis.get_value_counts(df, col)
        fig = bar_chart(counts, f"Number of records by {col}", "Number of records", col)
        path = save_chart(fig, prefix=f"overview_bar_{col}")
        charts.append((fig, path, "bar", col))

    num_cols = analysis.numeric_columns(df)[:max_numeric]
    for col in num_cols:
        values = df[col].dropna()
        if values.empty or values.nunique() < 2:
            continue
        fig = histogram(values, f"Distribution of {col}", col, "Number of rows")
        path = save_chart(fig, prefix=f"overview_hist_{col}")
        charts.append((fig, path, "hist", col))
    return charts


def make_charts(result: QAResult) -> list[tuple[plt.Figure, str, str]]:
    """Build and save every chart suggested for a QAResult.

    Args:
        result: The answered question with supporting data + chart hints.

    Returns:
        List of (figure, saved PNG path, chart kind) tuples, in a sensible
        order (the primary chart first). Empty when there is nothing to
        chart (no supporting data or chart_kind == 'none').
    """
    data = result.supporting_data
    if data is None or result.chart_kind == "none" or len(data) == 0:
        return []
    charts = []
    for kind in _companion_kinds(result):
        fig = _render_one(data, kind, result)
        path = save_chart(fig, prefix=kind)
        charts.append((fig, path, kind))
    return charts
