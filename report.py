"""Build downloadable CSV and PDF reports summarizing an analysis session.

Both report builders work purely from data already computed elsewhere
(analysis.py summaries, an optional answered question) - they format and
package that data, they never invent numbers.
"""

from __future__ import annotations

import io
from typing import Any, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import analysis

styles = getSampleStyleSheet()


def build_csv_report(df: pd.DataFrame) -> bytes:
    """Build a single CSV report combining dataset summary and statistics.

    Args:
        df: The uploaded dataset.

    Returns:
        UTF-8 encoded CSV bytes with a schema section, a numeric-summary
        section and one section per categorical column's top values.
    """
    buffer = io.StringIO()
    summary = analysis.get_dataset_summary(df)

    buffer.write("Dataset Summary\n")
    buffer.write(f"Rows,{summary['row_count']}\n")
    buffer.write(f"Columns,{summary['column_count']}\n\n")

    buffer.write("Column,Dtype,Missing Values\n")
    for col in summary["columns"]:
        buffer.write(f"{col},{summary['dtypes'][col]},{summary['missing_counts'][col]}\n")
    buffer.write("\n")

    numeric_summary = analysis.get_numeric_overview(df)
    if not numeric_summary.empty:
        buffer.write("Numeric Column Statistics\n")
        buffer.write(numeric_summary.to_csv())
        buffer.write("\n")

    for col, table in analysis.get_categorical_overview(df).items():
        buffer.write(f"Top values - {col}\n")
        buffer.write(table.to_csv())
        buffer.write("\n")

    return buffer.getvalue().encode("utf-8")


def _dataframe_to_table(df: pd.DataFrame, index_label: str = "") -> Table:
    """Convert a small DataFrame into a styled reportlab Table flowable.

    Args:
        df: The table to render (kept small - this is for report display).
        index_label: Header text for the index column.

    Returns:
        A Table flowable with a shaded header row and grid lines.
    """
    header = [index_label] + [str(c) for c in df.columns]
    rows = [[str(i)] + [f"{v:,.2f}" if isinstance(v, float) else str(v) for v in row]
            for i, row in zip(df.index, df.values)]
    table = Table([header] + rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
            ]
        )
    )
    return table


def build_pdf_report(
    df: pd.DataFrame,
    answer: Optional[dict[str, Any]] = None,
    overview_chart_paths: Optional[list[str]] = None,
) -> bytes:
    """Build a PDF report with the dataset summary and the latest Q&A.

    Args:
        df: The uploaded dataset.
        answer: Optional dict with 'question', 'result', 'charts',
            'explanation' - the same shape stored in Streamlit session
            state for the most recent answered question.
        overview_chart_paths: Optional PNG paths of the top overview charts
            to embed.

    Returns:
        PDF file bytes ready for a download button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story: list[Any] = [Paragraph("AI Data Analysis Assistant - Report", styles["Title"]), Spacer(1, 12)]

    summary = analysis.get_dataset_summary(df)
    story.append(Paragraph(
        f"Rows: {summary['row_count']} &nbsp;&nbsp; Columns: {summary['column_count']} &nbsp;&nbsp; "
        f"Missing cells: {sum(summary['missing_counts'].values())}", styles["Normal"],
    ))
    story.append(Spacer(1, 12))

    numeric_summary = analysis.get_numeric_overview(df)
    if not numeric_summary.empty:
        story.append(Paragraph("Numeric column statistics", styles["Heading2"]))
        story.append(_dataframe_to_table(numeric_summary, index_label="column"))
        story.append(Spacer(1, 12))

    for col, table in analysis.get_categorical_overview(df).items():
        story.append(Paragraph(f"Top values - {col}", styles["Heading2"]))
        story.append(_dataframe_to_table(table, index_label=col))
        story.append(Spacer(1, 12))

    for path in overview_chart_paths or []:
        story.append(Image(path, width=5.5 * inch, height=3.2 * inch, kind="proportional"))
        story.append(Spacer(1, 8))

    if answer and "result" in answer:
        story.append(Paragraph("Latest question", styles["Heading2"]))
        story.append(Paragraph(f"Q: {answer['question']}", styles["Normal"]))
        story.append(Paragraph(f"A: {answer['result'].answer_text}", styles["Normal"]))
        story.append(Paragraph(answer.get("explanation", ""), styles["Italic"]))
        story.append(Spacer(1, 8))
        for _, path, _kind in answer.get("charts", []):
            story.append(Image(path, width=5.0 * inch, height=3.0 * inch, kind="proportional"))
            story.append(Spacer(1, 8))

    doc.build(story)
    return buffer.getvalue()
