"""Chart generation using Plotly (interactive) + matplotlib (static/PDF only).

matplotlib is imported LAZILY inside PDF-only functions to avoid the
segmentation fault that occurs when matplotlib initialises at module
load time in Streamlit Cloud's headless Linux environment.
All interactive (Streamlit) charts use Plotly exclusively.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import analysis
from qa_engine import QAResult

CHARTS_DIR = "charts"
MAX_BARS = 12

PLOTLY_COLORS = px.colors.qualitative.Set2


def _mpl():
    """Lazy-load matplotlib + seaborn only when needed (PDF export)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="deep")
    return plt, sns


def _ensure_charts_dir() -> str:
    """Create the charts output folder if it does not exist yet.

    Returns:
        The path of the charts directory.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return CHARTS_DIR


def save_chart(fig, prefix: str = "chart") -> str:
    """Save a matplotlib figure to charts/ as a timestamped PNG."""
    _ensure_charts_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(CHARTS_DIR, f"{prefix}_{stamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def bar_chart(data: pd.Series, title: str, x_label: str, y_label: str):
    plt, sns = _mpl()
    accent = sns.color_palette("deep")[0]
    data = data.head(MAX_BARS)
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.5 * len(data) + 1)))
    ax.barh(data.index.astype(str), data.values, color=accent, height=0.62)
    ax.invert_yaxis()
    ax.set_title(title); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    for i, v in enumerate(data.values):
        ax.text(v, i, f" {v:,.0f}", va="center", fontsize=9, color="dimgray")
    ax.grid(axis="y", visible=False)
    sns.despine(fig=fig, left=True, bottom=True)
    fig.tight_layout()
    return fig


def pie_chart(data: pd.Series, title: str):
    plt, sns = _mpl()
    if len(data) > 6:
        return bar_chart(data, title, "Value", str(data.index.name or "Category"))
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette("deep", len(data))
    ax.pie(data.values, labels=[str(i) for i in data.index],
           autopct="%1.1f%%", colors=colors,
           wedgeprops={"linewidth": 2, "edgecolor": "white"})
    ax.set_title(title)
    fig.tight_layout()
    return fig


def line_chart(data: pd.Series, title: str, x_label: str, y_label: str):
    plt, sns = _mpl()
    accent = sns.color_palette("deep")[0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(data.index.astype(str), data.values, color=accent, linewidth=2, marker="o", markersize=5)
    ax.set_title(title); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=45)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def histogram(data: pd.Series, title: str, x_label: str, y_label: str):
    plt, sns = _mpl()
    accent = sns.color_palette("deep")[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data, bins=20, color=accent, edgecolor="white", ax=ax)
    mean = float(data.mean())
    ax.axvline(mean, color="dimgray", linestyle="--", linewidth=1.5)
    ax.text(mean, ax.get_ylim()[1] * 0.95, f" mean = {mean:,.2f}", fontsize=9, color="dimgray")
    ax.set_title(title); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


def box_chart(data: pd.Series, title: str, x_label: str, y_label: str):
    plt, sns = _mpl()
    accent = sns.color_palette("deep")[0]
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(x=data, color=accent, ax=ax, width=0.4)
    ax.set_title(title); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    sns.despine(fig=fig, left=True)
    fig.tight_layout()
    return fig


def _render_one(data: pd.Series, kind: str, result: QAResult):
    if kind == "pie":
        return pie_chart(data, result.chart_title)
    if kind == "line":
        return line_chart(data, result.chart_title, result.x_label, result.y_label)
    if kind == "hist":
        return histogram(data, result.chart_title, result.x_label, result.y_label)
    if kind == "box":
        return box_chart(data, f"Spread of {result.x_label}", result.x_label, "")
    return bar_chart(data, result.chart_title, result.x_label, result.y_label)


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
) -> list[tuple]:
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
    charts: list[tuple] = []

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


def make_charts(result: QAResult) -> list[tuple]:
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


# --------------------------------------------------------------------------
# Plotly interactive charts
# --------------------------------------------------------------------------

def plotly_bar(data: pd.Series, title: str, x_label: str, y_label: str) -> go.Figure:
    data = data.head(MAX_BARS).sort_values(ascending=True)
    fig = px.bar(
        x=data.values, y=data.index.astype(str),
        orientation="h", title=title,
        labels={"x": x_label, "y": y_label},
        color=data.values, color_continuous_scale="Blues",
        text=data.values,
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", height=max(300, 40 * len(data)))
    return fig


def plotly_pie(data: pd.Series, title: str) -> go.Figure:
    if len(data) > 8:
        return plotly_bar(data, title, "Value", str(data.index.name or "Category"))
    fig = px.pie(values=data.values, names=data.index.astype(str), title=title,
                 color_discrete_sequence=PLOTLY_COLORS)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plotly_line(data: pd.Series, title: str, x_label: str, y_label: str) -> go.Figure:
    fig = px.line(x=data.index.astype(str), y=data.values, title=title,
                  labels={"x": x_label, "y": y_label}, markers=True,
                  color_discrete_sequence=["#4C72B0"])
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plotly_histogram(data: pd.Series, title: str, x_label: str) -> go.Figure:
    fig = px.histogram(data, title=title, labels={"value": x_label},
                       nbins=20, color_discrete_sequence=["#4C72B0"])
    fig.add_vline(x=float(data.mean()), line_dash="dash", line_color="red",
                  annotation_text=f"mean={data.mean():,.2f}")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=False)
    return fig


def plotly_box(data: pd.Series, title: str, x_label: str) -> go.Figure:
    fig = px.box(data, title=title, labels={"value": x_label},
                 color_discrete_sequence=["#4C72B0"])
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=False)
    return fig


def make_plotly_charts(result: QAResult) -> list[go.Figure]:
    """Build interactive Plotly charts for a QAResult."""
    data = result.supporting_data
    if data is None or result.chart_kind == "none" or len(data) == 0:
        return []
    figs = []
    for kind in _companion_kinds(result):
        if kind == "pie":
            figs.append(plotly_pie(data, result.chart_title))
        elif kind == "line":
            figs.append(plotly_line(data, result.chart_title, result.x_label, result.y_label))
        elif kind == "hist":
            figs.append(plotly_histogram(data, result.chart_title, result.x_label))
        elif kind == "box":
            figs.append(plotly_box(data, f"Spread of {result.x_label}", result.x_label))
        else:
            figs.append(plotly_bar(data, result.chart_title, result.x_label, result.y_label))
    return figs


def make_plotly_overview_charts(df: pd.DataFrame,
                                 max_categorical: int = 2,
                                 max_numeric: int = 2) -> list[go.Figure]:
    """Build interactive overview charts for the uploaded dataset."""
    figs = []
    cat_overview = analysis.get_categorical_overview(df)
    cat_cols = sorted(cat_overview, key=lambda c: len(cat_overview[c]))[:max_categorical]
    for col in cat_cols:
        counts = analysis.get_value_counts(df, col)
        figs.append(plotly_bar(counts, f"Records by {col}", "Count", col))

    num_cols = analysis.numeric_columns(df)[:max_numeric]
    for col in num_cols:
        values = df[col].dropna()
        if values.empty or values.nunique() < 2:
            continue
        figs.append(plotly_histogram(values, f"Distribution of {col}", col))
    return figs


def make_correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """Build an interactive correlation heatmap for numeric columns."""
    num_cols = analysis.numeric_columns(df)
    if len(num_cols) < 2:
        return None
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, title="Correlation Heatmap",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto",
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def make_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                      color_col: str | None = None) -> go.Figure:
    """Build an interactive scatter plot between two numeric columns."""
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                     title=f"{y_col} vs {x_col}",
                     color_discrete_sequence=PLOTLY_COLORS)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig
