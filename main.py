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
        st.dataframe(schema, use_container_width=True, hide_index=True)
    with st.expander("Data preview (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)


def _render_overview_charts(charts: list[tuple]) -> None:
    """Render the two real, data-driven overview charts at the top of the page.

    One bar chart of the most informative categorical column and one
    histogram of the first numeric column - both built directly from the
    uploaded data (never placeholder/dummy figures).

    Args:
        charts: Output of visualization.make_overview_charts(df, ...).
    """
    import streamlit as st

    if not charts:
        return
    st.subheader("Overview charts")
    cols = st.columns(len(charts))
    for col, (fig, _path, _kind, name) in zip(cols, charts):
        col.pyplot(fig)
        col.caption(name)


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
        st.dataframe(numeric_summary, use_container_width=True)

    categorical_summary = analysis.get_categorical_overview(df)
    if categorical_summary:
        st.markdown("**Categorical columns**")
        table_cols = st.columns(min(len(categorical_summary), 3))
        for i, (col, table) in enumerate(categorical_summary.items()):
            table_cols[i % len(table_cols)].dataframe(table, use_container_width=True)


def _run_and_store(df: pd.DataFrame, question: str) -> None:
    """Run the QA pipeline and stash the result in session state.

    Storing the result (rather than rendering it inline) lets the answer
    stay pinned at the top of the page across reruns triggered by other
    widgets (e.g. the dark-mode toggle), until a new question is run or a
    different file is uploaded.

    Args:
        df: The uploaded dataset.
        question: The user's free-text question from the sidebar.
    """
    import streamlit as st

    if not question.strip():
        st.session_state["last_answer"] = {"error": "Please type a question first."}
        return
    try:
        result, charts, explanation = run_pipeline(df, question)
        st.session_state["last_answer"] = {
            "question": question,
            "result": result,
            "charts": charts,
            "explanation": explanation,
        }
    except Exception as exc:
        msg = str(exc) or "An unexpected error occurred."
        st.session_state["last_answer"] = {
            "error": f"Analysis failed: {msg}. Check the dataset and try again."
        }


def _render_answer_top() -> None:
    """Render the most recent question's answer, charts and explanation.

    Shown at the very top of the main page - above the dataset summary and
    overview - since a fresh answer is the thing the user just asked for.
    Every chart here comes straight from the pandas result for this
    question, never a placeholder.
    """
    import streamlit as st

    answer = st.session_state.get("last_answer")
    if not answer:
        return
    st.subheader("Answer")
    if "error" in answer:
        st.warning(answer["error"])
        st.divider()
        return
    result: QAResult = answer["result"]
    st.caption(f"Q: {answer['question']}")
    if not result.success:
        st.error(
            f"Could not answer: {result.answer_text}"
            if "Available columns" not in result.answer_text
            else result.answer_text
        )
        st.divider()
        return
    st.success(f"**Answer:** {result.answer_text}")
    charts = answer["charts"]
    if charts:
        cols = st.columns(len(charts))
        for col, (fig, _path, _kind) in zip(cols, charts):
            col.pyplot(fig)
    if answer.get("explanation"):
        st.info(f"**AI insight:** {answer['explanation']}")
    st.divider()


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
**What this app does:**
- 📂 **Upload any CSV** — no template or sample file required
- 🔍 **Auto-analysis** — overview charts and statistics generated instantly on click
- 💬 **Ask anything** — type any question in plain English; AI answers from your real data
- 📊 **Smart charts** — bar, pie, histogram and box plots chosen automatically per question
- 📥 **Download report** — export a full analysis as CSV or PDF
- 🌙 **Dark / light mode** — toggle top-right anytime
        """,
        unsafe_allow_html=False,
    )


def _render_download_buttons(df: pd.DataFrame, overview_charts: list[tuple]) -> None:
    """Render compact CSV and PDF download buttons inline."""
    import streamlit as st

    chart_paths = [path for _, path, _kind, _name in overview_charts]
    pdf_bytes = report.build_pdf_report(
        df, answer=st.session_state.get("last_answer"), overview_chart_paths=chart_paths
    )
    dl1, dl2, _ = st.columns([1, 1, 5])
    dl1.download_button(
        "⬇️ CSV",
        data=report.build_csv_report(df),
        file_name="analysis_report.csv",
        mime="text/csv",
    )
    dl2.download_button(
        "⬇️ PDF",
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
        st.info("Upload a CSV file in the sidebar to get started.")
        return

    if analyse_clicked:
        with st.spinner("Analysing dataset..."):
            st.session_state["analysis_done"] = True
            st.session_state["overview_charts"] = visualization.make_overview_charts(
                df, max_categorical=1, max_numeric=1
            )

    if not st.session_state.get("analysis_done"):
        st.info("Click **Analyse** in the sidebar to begin exploring your dataset.")
        return

    overview_charts = st.session_state.get("overview_charts", [])

    # Question bar at the top of the main content area.
    _render_question_bar(df)
    st.divider()

    # Answer pinned right below the question bar.
    _render_answer_top()

    # Compact report download buttons.
    _render_download_buttons(df, overview_charts)
    st.divider()

    _render_overview_charts(overview_charts)
    st.divider()
    _render_summary(df)
    st.divider()
    _render_overview_tables(df)


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
