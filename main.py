"""AI Data Analysis Assistant - entry point.

Run as a web app:      streamlit run main.py
Run from the terminal: python main.py --csv dataset.csv --question "..."

Both paths share the same pipeline: load CSV -> summarize -> answer the
question -> draw a chart -> produce an AI (or offline) explanation.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

import pandas as pd

import ai_explainer
import analysis
import qa_engine
import report
import visualization
from analysis import DatasetError
from qa_engine import QAResult

# Minimal dark-mode stylesheet injected when the top-right toggle is on.
DARK_CSS = """
<style>
.stApp { background-color: #0e1117; color: #e6e6e6; }
[data-testid="stSidebar"] { background-color: #161a23; }
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp li { color: #e6e6e6; }
[data-testid="stMetricValue"] { color: #e6e6e6; }
</style>
"""


def run_pipeline(df: pd.DataFrame, question: str) -> tuple[QAResult, list[tuple], str]:
    """Run the full answer -> charts -> explanation pipeline once.

    The chart type and the LLM provider are both chosen automatically -
    there is no user-facing chart picker or provider picker. The provider
    is controlled solely by AI_PROVIDER in .env (see ai_explainer.py).

    Args:
        df: The loaded dataset.
        question: The user's free-text question.

    Returns:
        Tuple of (QA result, list of (figure, saved PNG path, kind), explanation).
    """
    result = qa_engine.answer_question(df, question)
    if not result.success:
        return result, [], ""
    charts = visualization.make_charts(result)
    explanation = ai_explainer.explain(result)
    return result, charts, explanation


# --------------------------------------------------------------------------
# Streamlit interface
# --------------------------------------------------------------------------

def _running_in_streamlit() -> bool:
    """Detect whether this script was launched via ``streamlit run``.

    Returns:
        True when a Streamlit runtime is active, False for plain Python.
    """
    try:
        from streamlit.runtime import exists

        return exists()
    except Exception:
        return False


def _sidebar_controls() -> tuple[Optional[pd.DataFrame], str, bool, bool]:
    """Render the sidebar: CSV upload, Analyse button, and question box.

    Returns:
        Tuple of (DataFrame or None, question text, analyse_clicked, ask_clicked).
    """
    import streamlit as st

    st.sidebar.title("Data & questions")
    st.sidebar.subheader("Upload your data")
    uploaded = st.sidebar.file_uploader("Upload a CSV file to analyze", type=["csv"])
    df: Optional[pd.DataFrame] = None
    try:
        if uploaded is not None:
            df = analysis.load_dataset(uploaded)
            st.sidebar.success(f"Loaded '{uploaded.name}'")
            if st.session_state.get("_loaded_file") != uploaded.name:
                st.session_state["_loaded_file"] = uploaded.name
                st.session_state.pop("last_answer", None)
                st.session_state.pop("suggested_questions", None)
                st.session_state.pop("analysis_done", None)
        else:
            st.sidebar.info("Upload a CSV file to get started.")
    except DatasetError as exc:
        st.sidebar.error(str(exc))

    analyse_clicked = st.sidebar.button(
        "🔍 Analyse", type="primary", disabled=df is None, key="btn_analyse"
    )
    return df, analyse_clicked


def _render_clickable_questions(
    container: Any, label: str, questions: list[str], key_prefix: str, df: Optional[pd.DataFrame]
) -> None:
    """Render a group of questions as clickable buttons that fill the question box.

    Args:
        container: Streamlit container to render into (e.g. st.sidebar).
        label: Section caption shown above the buttons.
        questions: Question strings to show as buttons.
        key_prefix: Prefix for each button's widget key (keeps keys unique).
        df: The uploaded dataset, used only to disable buttons before upload.
    """
    import streamlit as st

    if not questions:
        return
    container.caption(label)
    for i, q in enumerate(questions):
        if container.button(q, key=f"{key_prefix}_{i}", disabled=df is None, width="stretch"):
            st.session_state["question_main_input"] = q


def _get_suggested_questions(df: pd.DataFrame) -> list[str]:
    """Fetch (and cache) 3 AI-suggested questions for the uploaded dataset.

    Generated once per uploaded file via ai_explainer.suggest_questions -
    a single LLM call with an offline template fallback - and cached in
    session state so it is not re-requested on every rerun.

    Args:
        df: The uploaded dataset.

    Returns:
        Up to 3 suggested question strings.
    """
    import streamlit as st

    if "suggested_questions" not in st.session_state:
        st.session_state["suggested_questions"] = ai_explainer.suggest_questions(df)
    return st.session_state["suggested_questions"]


def _render_summary(df: pd.DataFrame) -> None:
    """Render the automatic dataset summary section.

    Args:
        df: The loaded dataset.
    """
    import streamlit as st

    summary = analysis.get_dataset_summary(df)
    st.subheader("Dataset summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", summary["row_count"])
    col2.metric("Columns", summary["column_count"])
    col3.metric("Missing cells", sum(summary["missing_counts"].values()))
    with st.expander("Columns, types and missing values", expanded=False):
        schema = pd.DataFrame(
            {
                "column": summary["columns"],
                "dtype": [summary["dtypes"][c] for c in summary["columns"]],
                "missing values": [summary["missing_counts"][c] for c in summary["columns"]],
            }
        )
        st.dataframe(schema, width='stretch', hide_index=True)
    with st.expander("Data preview (first 10 rows)"):
        st.dataframe(df.head(10), width='stretch')


def _render_overview_charts(df: pd.DataFrame) -> None:
    """Render interactive Plotly overview charts."""
    import streamlit as st

    figs = visualization.make_plotly_overview_charts(df, max_categorical=2, max_numeric=2)
    if not figs:
        return
    st.subheader("📊 Overview Charts")
    cols = st.columns(min(len(figs), 2))
    for i, fig in enumerate(figs):
        cols[i % 2].plotly_chart(fig, width="stretch", key=f"overview_fig_{i}")


def _render_overview_tables(df: pd.DataFrame) -> None:
    """Render the numeric and categorical summary tables for the dataset.

    Args:
        df: The uploaded dataset.
    """
    import streamlit as st

    st.subheader("Automatic data analysis")

    numeric_summary = analysis.get_numeric_overview(df)
    if not numeric_summary.empty:
        st.markdown("**Numeric columns**")
        st.dataframe(numeric_summary, width='stretch')

    categorical_summary = analysis.get_categorical_overview(df)
    if categorical_summary:
        st.markdown("**Categorical columns**")
        table_cols = st.columns(min(len(categorical_summary), 3))
        for i, (col, table) in enumerate(categorical_summary.items()):
            table_cols[i % len(table_cols)].dataframe(table, width='stretch')


CHAT_CSS = """
<style>
.user-bubble {
    background: #4C72B0; color: white; border-radius: 18px 18px 4px 18px;
    padding: 10px 16px; margin: 6px 0 6px 15%; display: inline-block;
    max-width: 85%; word-wrap: break-word;
}
.ai-bubble {
    background: #f0f2f6; color: #1a1a1a; border-radius: 18px 18px 18px 4px;
    padding: 10px 16px; margin: 6px 15% 6px 0; display: inline-block;
    max-width: 85%; word-wrap: break-word;
}
.bubble-wrap { display: flex; flex-direction: column; }
.user-wrap { align-items: flex-end; }
.ai-wrap { align-items: flex-start; }
</style>
"""


def _run_and_store(df: pd.DataFrame, question: str) -> None:
    """Run pipeline, store result in chat history and last_answer."""
    import streamlit as st

    if not question.strip():
        return
    try:
        result = qa_engine.answer_question(df, question)
        plotly_figs = visualization.make_plotly_charts(result)
        explanation = ai_explainer.explain(result) if result.success else ""
        entry = {
            "question": question,
            "result": result,
            "plotly_figs": plotly_figs,
            "explanation": explanation,
        }
        _append_history(entry)
    except Exception as exc:
        msg = str(exc) or "An unexpected error occurred."
        _append_history({"question": question, "error": f"Analysis failed: {msg}"})


MAX_HISTORY = 10  # Cap stored Q&As so session memory stays bounded on Streamlit Cloud.


def _append_history(entry: dict) -> None:
    """Append a chat entry, trimming old ones (and their figures) beyond the cap."""
    import streamlit as st

    history = st.session_state.setdefault("chat_history", [])
    history.append(entry)
    # Drop figure objects from older entries first, then drop entries entirely.
    for old in history[:-3]:
        old.pop("plotly_figs", None)
    del history[:-MAX_HISTORY]
    st.session_state["last_answer"] = entry


def _render_chat_section(df: pd.DataFrame) -> None:
    """Render chat-style Q&A with history and interactive charts."""
    import streamlit as st

    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    st.subheader("💬 Ask AI")

    # Question input bar
    q_col, btn_col = st.columns([5, 1])
    question = q_col.text_input(
        "question_main",
        key="question_main_input",
        placeholder="e.g. Which category has the highest sales?",
        label_visibility="collapsed",
    )
    if btn_col.button("Ask", type="primary", key="btn_ask_main"):
        if not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("🤖 AI is thinking..."):
                _run_and_store(df, question)

    # Chat history — newest first
    history = st.session_state.get("chat_history", [])
    if not history:
        st.caption("Ask any question about your data above.")
        return

    for n, entry in enumerate(reversed(history)):
        # User bubble
        st.markdown(
            f'<div class="bubble-wrap user-wrap">'
            f'<div class="user-bubble">🧑 {entry["question"]}</div></div>',
            unsafe_allow_html=True,
        )
        # AI bubble
        if "error" in entry:
            st.markdown(
                f'<div class="bubble-wrap ai-wrap">'
                f'<div class="ai-bubble">⚠️ {entry["error"]}</div></div>',
                unsafe_allow_html=True,
            )
            continue

        result: QAResult = entry["result"]
        answer_text = result.answer_text
        st.markdown(
            f'<div class="bubble-wrap ai-wrap">'
            f'<div class="ai-bubble">🤖 {answer_text}</div></div>',
            unsafe_allow_html=True,
        )

        # Plotly charts below bubble
        figs = entry.get("plotly_figs", [])
        if figs:
            chart_cols = st.columns(min(len(figs), 2))
            for i, fig in enumerate(figs):
                chart_cols[i % 2].plotly_chart(fig, width="stretch", key=f"chat_fig_{n}_{i}")

        # AI explanation
        if entry.get("explanation"):
            st.info(f"💡 {entry['explanation']}")

    # Clear history button
    if st.button("🗑️ Clear chat", key="clear_chat"):
        st.session_state["chat_history"] = []
        st.session_state.pop("last_answer", None)
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()


def _render_auto_insights(df: pd.DataFrame) -> None:
    """Show 5 AI-generated key insights after Analyse is clicked."""
    import streamlit as st

    if "auto_insights" not in st.session_state:
        with st.spinner("🤖 Generating AI insights..."):
            st.session_state["auto_insights"] = ai_explainer.generate_insights(df)

    insights = st.session_state.get("auto_insights", [])
    if not insights:
        return
    st.subheader("🔎 Key Insights")
    for insight in insights:
        st.markdown(f"- {insight}")


def _render_data_quality(df: pd.DataFrame) -> None:
    """Show data quality report: missing values, duplicates, column types."""
    import streamlit as st

    with st.expander("🧹 Data Quality Report", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        total_missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())
        num_cols = len(analysis.numeric_columns(df))
        cat_cols = len(analysis.categorical_columns(df))

        col1.metric("Missing Values", total_missing,
                    delta="clean ✅" if total_missing == 0 else f"{total_missing} cells ⚠️",
                    delta_color="normal" if total_missing == 0 else "inverse")
        col2.metric("Duplicate Rows", duplicates,
                    delta="none ✅" if duplicates == 0 else f"{duplicates} found ⚠️",
                    delta_color="normal" if duplicates == 0 else "inverse")
        col3.metric("Numeric Columns", num_cols)
        col4.metric("Categorical Columns", cat_cols)

        if total_missing > 0:
            st.markdown("**Missing values per column:**")
            missing_df = df.isna().sum()
            missing_df = missing_df[missing_df > 0].reset_index()
            missing_df.columns = ["Column", "Missing Count"]
            missing_df["% Missing"] = (missing_df["Missing Count"] / len(df) * 100).round(1)
            st.dataframe(missing_df, width='stretch', hide_index=True)


def _render_correlation(df: pd.DataFrame) -> None:
    """Show interactive correlation heatmap if 2+ numeric columns exist."""
    import streamlit as st

    fig = visualization.make_correlation_heatmap(df)
    if fig is None:
        return
    with st.expander("📈 Correlation Heatmap", expanded=False):
        st.plotly_chart(fig, width="stretch")
        st.caption("Values close to 1 or -1 indicate strong correlation. Close to 0 means little relationship.")


def _render_top_bar() -> None:
    """Render the title with a dark/light mode toggle pinned top-right.

    Returns nothing - the toggle's own widget key drives dark-mode state,
    read wherever DARK_CSS needs to be applied.
    """
    import streamlit as st

    title_col, toggle_col = st.columns([6, 1])
    with title_col:
        st.title("📊 AI Data Analysis Assistant")
    with toggle_col:
        if st.toggle("🌙 Dark", key="dark_mode"):
            st.markdown(DARK_CSS, unsafe_allow_html=True)

    st.markdown(
        """
**Features:**
📂 Upload any CSV &nbsp;|&nbsp; 🤖 AI-powered Q&A with chat history &nbsp;|&nbsp; 📊 Interactive Plotly charts
🔎 Auto insights &nbsp;|&nbsp; 📈 Correlation heatmap &nbsp;|&nbsp; 🧹 Data quality report &nbsp;|&nbsp; 📥 CSV/PDF download
        """,
        unsafe_allow_html=True,
    )


def _render_download_buttons(df: pd.DataFrame, overview_charts: list[tuple]) -> None:
    """Render compact CSV and PDF download buttons inline."""
    import streamlit as st

    dl1, dl2, _ = st.columns([1, 1, 5])

    # CSV: lightweight, build inline
    try:
        csv_bytes = report.build_csv_report(df)
    except Exception:
        csv_bytes = b""

    dl1.download_button(
        "⬇️ CSV",
        data=csv_bytes,
        file_name="analysis_report.csv",
        mime="text/csv",
        disabled=not csv_bytes,
    )

    # PDF: only build when button is clicked, via session state flag
    if dl2.button("⬇️ PDF"):
        try:
            chart_paths = [p for _, p, _, _ in overview_charts if isinstance(p, str)]
            pdf_bytes = report.build_pdf_report(
                df,
                answer=st.session_state.get("last_answer"),
                overview_chart_paths=chart_paths,
            )
            st.session_state["_pdf_cache"] = pdf_bytes
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")
            st.session_state.pop("_pdf_cache", None)

    if "pdf_cache" in st.session_state or "_pdf_cache" in st.session_state:
        pdf_bytes = st.session_state.get("_pdf_cache", b"")
        if pdf_bytes:
            st.download_button(
                "📥 Save PDF",
                data=pdf_bytes,
                file_name="analysis_report.pdf",
                mime="application/pdf",
            )


def _render_question_bar(df: pd.DataFrame) -> None:
    """Render the Ask-a-question row inline in the main area."""
    import streamlit as st

    st.subheader("Ask a question")
    q_col, btn_col = st.columns([5, 1])
    question = q_col.text_input(
        "question_main",
        key="question_main_input",
        placeholder="e.g. Which category has the highest sales?",
        label_visibility="collapsed",
    )
    if btn_col.button("Ask", type="primary", key="btn_ask_main"):
        if not question.strip():
            st.warning("Please type a question before clicking Ask.")
        else:
            with st.spinner("AI is thinking..."):
                _run_and_store(df, question)


def run_streamlit_app() -> None:
    """Compose and run the Streamlit application page."""
    import streamlit as st

    st.set_page_config(page_title="AI Data Analysis Assistant", page_icon="📊", layout="wide")
    df, analyse_clicked = _sidebar_controls()

    _render_top_bar()

    if df is None:
        st.info("⬅️ Upload a CSV file in the sidebar to get started.")
        return

    if analyse_clicked:
        with st.spinner("Analysing dataset..."):
            st.session_state["analysis_done"] = True
            st.session_state.pop("auto_insights", None)  # refresh insights on re-analyse

    if not st.session_state.get("analysis_done"):
        st.info("Click **🔍 Analyse** in the sidebar to begin.")
        return

    # ── Chat Q&A (top, most prominent) ──────────────────────────────────
    _render_chat_section(df)
    st.divider()

    # ── AI Insights ──────────────────────────────────────────────────────
    _render_auto_insights(df)
    st.divider()

    # ── Overview Charts (interactive) ────────────────────────────────────
    _render_overview_charts(df)
    st.divider()

    # ── Correlation Heatmap ──────────────────────────────────────────────
    _render_correlation(df)

    # ── Data Quality ─────────────────────────────────────────────────────
    _render_data_quality(df)

    # ── Dataset Summary ──────────────────────────────────────────────────
    _render_summary(df)
    st.divider()

    # ── Stats Tables ─────────────────────────────────────────────────────
    _render_overview_tables(df)
    st.divider()

    # ── Report Download ──────────────────────────────────────────────────
    _render_download_buttons(df, [])


# --------------------------------------------------------------------------
# CLI fallback
# --------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    """Print the dataset summary block to the terminal.

    Args:
        df: The loaded dataset.
    """
    summary = analysis.get_dataset_summary(df)
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Rows: {summary['row_count']}   Columns: {summary['column_count']}")
    print(f"{'column':<15}{'dtype':<18}{'missing':>7}")
    for col in summary["columns"]:
        print(f"{col:<15}{summary['dtypes'][col]:<18}{summary['missing_counts'][col]:>7}")


def _print_overview(df: pd.DataFrame) -> None:
    """Print the automatic charts + tables analysis, mirroring the Streamlit view.

    Args:
        df: The loaded dataset.
    """
    numeric_summary = analysis.get_numeric_overview(df)
    if not numeric_summary.empty:
        print("\nNUMERIC COLUMNS")
        print(numeric_summary.to_string())

    categorical_summary = analysis.get_categorical_overview(df)
    for col, table in categorical_summary.items():
        print(f"\nTOP VALUES: {col}")
        print(table.to_string())

    charts = visualization.make_overview_charts(df)
    for _, path, kind, col in charts:
        print(f"OVERVIEW CHART: {kind} chart for '{col}' saved to {path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the argparse parser for the CLI fallback.

    Returns:
        The configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="AI Data Analysis Assistant (CLI mode). "
        "Tip: run 'streamlit run main.py' for the web interface."
    )
    parser.add_argument("--csv", required=True, help="Path to the CSV file to analyze.")
    parser.add_argument("--question", help="Free-text question about the dataset.")
    return parser


def run_cli(argv: Optional[list[str]] = None) -> int:
    """Run the same pipeline non-interactively from the terminal.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Process exit code: 0 on success, 1 on any handled error.
    """
    args = _build_arg_parser().parse_args(argv)
    try:
        df = analysis.load_dataset(args.csv)
    except DatasetError as exc:
        print(f"Error: {exc}")
        return 1

    # The answer prints first (mirrors the Streamlit page showing it at the
    # top), then the dataset summary/overview follow underneath.
    exit_code = 0
    if args.question:
        try:
            result, charts, explanation = run_pipeline(df, args.question)
        except Exception as exc:  # Last-resort guard: never show a stack trace.
            print(f"Error while analyzing: {exc}")
            return 1
        print("=" * 60)
        print(f"QUESTION : {args.question}")
        print(f"ANSWER   : {result.answer_text}")
        if not result.success:
            exit_code = 1
        else:
            for _, path, kind in charts:
                print(f"CHART    : {kind} chart saved to {path}")
            print(f"EXPLAIN  : {explanation}")
        print()

    _print_summary(df)
    _print_overview(df)
    if not args.question:
        print("\nNo --question given. Try e.g.:")
        print(f'  python main.py --csv {args.csv} --question "Which category has the highest sales?"')
    return exit_code


if __name__ == "__main__":
    if _running_in_streamlit():
        run_streamlit_app()
    else:
        sys.exit(run_cli())
