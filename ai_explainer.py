"""One-call LLM wrapper that turns a computed answer into plain language.

Exactly ONE API call is made per question. The provider is swappable via
the ``AI_PROVIDER`` environment variable (or the ``provider`` argument):

* ``openai``  - OpenAI Chat Completions          (OPENAI_API_KEY)
* ``groq``    - Groq's OpenAI-compatible API     (GROQ_API_KEY)
* ``gemini``  - Gemini's OpenAI-compatible API   (GEMINI_API_KEY)
* ``claude``  - Anthropic Messages API           (ANTHROPIC_API_KEY)

Keys are read from the environment / ``.env`` via python-dotenv and are
never hardcoded. When no key is configured (or the call fails), a
template-based explanation is generated purely from the pandas result so
the app still works completely offline.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

import analysis
from qa_engine import QAResult

load_dotenv()  # Pull API keys and settings from a local .env file, if present.

#: Default provider; override with AI_PROVIDER in .env or explain(provider=...).
PROVIDER: str = os.getenv("AI_PROVIDER", "openai").lower()

#: Per-provider connection settings. groq/gemini reuse the OpenAI SDK by
#: pointing it at their OpenAI-compatible endpoints, so one SDK covers three
#: providers; claude uses the anthropic SDK.
PROVIDER_CONFIG: dict[str, dict[str, Optional[str]]] = {
    "openai": {"env_key": "OPENAI_API_KEY", "base_url": None, "model": "gpt-4o-mini"},
    "groq": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
    },
    "claude": {"env_key": "ANTHROPIC_API_KEY", "base_url": None, "model": "claude-haiku-4-5-20251001"},
}

SYSTEM_PROMPT = (
    "You are a data analyst. Given a computed statistic about a dataset, write a "
    "1-2 sentence plain-language explanation for a non-technical reader. Include a "
    "percentage or a comparison to other categories where the context allows it. "
    "State only facts present in the context; do not invent numbers."
)


def explain(result: QAResult, provider: Optional[str] = None) -> str:
    """Explain a computed answer in plain language via one LLM call.

    Args:
        result: The answered question (raw answer + supporting stats).
        provider: Provider name overriding the AI_PROVIDER setting.

    Returns:
        A 1-2 sentence explanation. Falls back to a template built from
        the pandas result when no API key is set or the call fails.
    """
    name = (provider or PROVIDER).lower()
    config = PROVIDER_CONFIG.get(name)
    if config is None:
        return _template_explanation(result) + f" (Unknown AI provider '{name}'.)"
    api_key = os.getenv(config["env_key"] or "")
    if not api_key:
        # No key configured: stay fully offline with the template fallback.
        return _template_explanation(result)
    try:
        prompt = _build_prompt(result)
        model = os.getenv("AI_MODEL") or (config["model"] or "")
        if name == "claude":
            return _call_anthropic(api_key, model, prompt)
        return _call_openai_compatible(api_key, config["base_url"], model, prompt)
    except Exception:
        # Never crash the app on network/API problems - degrade gracefully.
        return _template_explanation(result) + " (AI service unavailable; showing offline summary.)"


def suggest_questions(df: pd.DataFrame, provider: Optional[str] = None) -> list[str]:
    """Suggest 3 questions this dataset can answer, via one LLM call.

    Args:
        df: The uploaded dataset to base suggestions on.
        provider: Provider name overriding the AI_PROVIDER setting.

    Returns:
        Exactly 3 question strings. Falls back to simple template
        questions built from the dataset's own columns when no API key is
        configured or the call fails - so suggestions always appear.
    """
    name = (provider or PROVIDER).lower()
    config = PROVIDER_CONFIG.get(name)
    api_key = os.getenv(config["env_key"] or "") if config else None
    if not config or not api_key:
        return _template_questions(df)
    try:
        prompt = _build_suggestion_prompt(df)
        model = os.getenv("AI_MODEL") or (config["model"] or "")
        if name == "claude":
            text = _call_anthropic(api_key, model, prompt)
        else:
            text = _call_openai_compatible(api_key, config["base_url"], model, prompt)
        questions = _parse_questions(text)
        valid = [q for q in questions if _uses_only_allowed_columns(q, df)]
        return valid if len(valid) == 3 else _template_questions(df)
    except Exception:
        return _template_questions(df)


def _uses_only_allowed_columns(question: str, df: pd.DataFrame) -> bool:
    """Check a suggested question mentions only well-behaved columns.

    Guards against the LLM ignoring the allowed-column instructions and
    picking a technically-real but unusable column (e.g. an order_id
    identifier with almost as many unique values as rows).

    Args:
        question: A single suggested question string.
        df: The uploaded dataset.

    Returns:
        True if at least one column is mentioned and every mentioned
        column is either a "good" categorical column or numeric.
    """
    import qa_engine  # Imported lazily to avoid a module-load-time cycle.

    allowed = set(analysis.get_categorical_overview(df)) | set(analysis.numeric_columns(df))
    mentioned = qa_engine.detect_columns(question, df)
    return bool(mentioned) and all(col in allowed for col in mentioned)


def _build_suggestion_prompt(df: pd.DataFrame) -> str:
    """Assemble the prompt asking the LLM for 3 dataset-specific questions.

    Only columns that make sense to group or average by are offered to the
    model - identifier-like columns (e.g. order_id, near-unique per row)
    are excluded so the LLM cannot suggest an unanswerable question.

    Args:
        df: The uploaded dataset.

    Returns:
        A prompt describing the dataset's usable columns and the question
        styles the app's rule-based engine can answer.
    """
    categorical = list(analysis.get_categorical_overview(df).keys())
    numeric = analysis.numeric_columns(df)
    return (
        f"Category-like columns: {', '.join(categorical) or 'none'}\n"
        f"Numeric columns: {', '.join(numeric) or 'none'}\n\n"
        "Suggest exactly 3 short, distinct natural-language questions a user could "
        "ask about this dataset. Each must fit one of these patterns: "
        "'Which <category column> generated the highest <numeric column>?', "
        "'What is the average <numeric column>?', "
        "'Which <category column> appears most frequently?'. "
        "Only use column names from the two lists above, never any other column. "
        "Reply with exactly 3 lines, one question per line, no numbering or extra text."
    )


def _parse_questions(text: str) -> list[str]:
    """Extract up to 3 clean question strings from an LLM response.

    Args:
        text: Raw model output, expected to be one question per line.

    Returns:
        Up to 3 questions with numbering/bullets and quotes stripped.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = [re.sub(r'^[\d.\-\*\)\s]+', "", ln).strip(' "') for ln in lines]
    return [q for q in cleaned if q][:3]


#: Name fragments suggesting a numeric column is a business "measure"
#: (worth summing/comparing) rather than an attribute like age or year.
_MEASURE_HINTS = ("price", "amount", "revenue", "sales", "total", "cost", "quantity", "value")


def _pick_measure_column(numeric: list[str]) -> Optional[str]:
    """Pick the numeric column most likely to represent a summable measure.

    Args:
        numeric: Candidate numeric column names.

    Returns:
        The first column whose name hints at price/quantity/revenue/etc.,
        or the first numeric column if none match, or None if there are
        no numeric columns at all.
    """
    for col in numeric:
        if any(hint in col.lower() for hint in _MEASURE_HINTS):
            return col
    return numeric[0] if numeric else None


def _template_questions(df: pd.DataFrame) -> list[str]:
    """Build 3 dataset-specific questions purely from column names/dtypes.

    Args:
        df: The uploaded dataset.

    Returns:
        Up to 3 questions using the dataset's own categorical/numeric
        columns, so suggestions work fully offline.
    """
    numeric = analysis.numeric_columns(df)
    # Same cardinality filter as the overview tables: identifier-like
    # columns (e.g. order_id) are excluded, they make unusable questions.
    categorical = list(analysis.get_categorical_overview(df).keys())

    measure = _pick_measure_column(numeric)
    other_numeric = [c for c in numeric if c != measure]
    average_col = other_numeric[0] if other_numeric else measure

    questions: list[str] = []
    if categorical and measure:
        questions.append(f"Which {categorical[0]} generated the highest {measure}?")
    if average_col:
        questions.append(f"What is the average {average_col}?")
    if categorical:
        col = categorical[1] if len(categorical) > 1 else categorical[0]
        questions.append(f"Which {col} appears most frequently?")
    return questions[:3]


def _build_prompt(result: QAResult) -> str:
    """Assemble the user prompt sent to the LLM.

    Args:
        result: The answered question with stats context.

    Returns:
        A compact prompt containing the question, the computed answer and
        the supporting numbers the model may cite.
    """
    lines = [
        f"Question: {result.question}",
        f"Computed answer: {result.answer_text}",
    ]
    if result.stats:
        lines.append(f"Supporting stats: {result.stats}")
    if result.supporting_data is not None and len(result.supporting_data) <= 15:
        # Small groupby tables give the model comparison material; large raw
        # value arrays are skipped to keep the single call cheap.
        lines.append(f"Breakdown:\n{result.supporting_data.to_string()}")
    lines.append("Write the 1-2 sentence explanation now.")
    return "\n".join(lines)


def _call_openai_compatible(
    api_key: str, base_url: Optional[str], model: str, prompt: str
) -> str:
    """Make one chat completion call to an OpenAI-compatible endpoint.

    Args:
        api_key: The provider API key.
        base_url: Custom endpoint for groq/gemini; None for OpenAI itself.
        model: Model name to use.
        prompt: The user prompt.

    Returns:
        The model's explanation text, stripped.
    """
    from openai import OpenAI  # Imported lazily so the app runs without the SDK.

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.3,
    )
    return (response.choices[0].message.content or "").strip()


def _call_anthropic(api_key: str, model: str, prompt: str) -> str:
    """Make one Messages API call to Anthropic (Claude).

    Args:
        api_key: The Anthropic API key.
        model: Claude model name to use.
        prompt: The user prompt.

    Returns:
        The model's explanation text, stripped.
    """
    import anthropic  # Imported lazily so the app runs without the SDK.

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=150,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def generate_insights(df: pd.DataFrame, provider: Optional[str] = None) -> list[str]:
    """Generate 5 key AI insights about the dataset after Analyse is clicked.

    Args:
        df: The uploaded dataset.
        provider: Provider name overriding AI_PROVIDER setting.

    Returns:
        List of 5 insight strings. Falls back to template insights if AI unavailable.
    """
    name = (provider or PROVIDER).lower()
    config = PROVIDER_CONFIG.get(name)
    api_key = os.getenv(config["env_key"] or "") if config else None
    if not config or not api_key:
        return _template_insights(df)
    try:
        numeric = analysis.numeric_columns(df)
        categorical = analysis.categorical_columns(df)
        prompt = (
            f"Dataset: {len(df)} rows, {df.shape[1]} columns.\n"
            f"Numeric columns: {', '.join(numeric) or 'none'}\n"
            f"Categorical columns: {', '.join(categorical) or 'none'}\n"
        )
        if numeric:
            prompt += f"Stats:\n{df[numeric].describe().round(2).to_string()}\n"
        for col in categorical[:3]:
            prompt += f"Top {col}: {dict(df[col].value_counts().head(3))}\n"
        prompt += (
            "\nWrite exactly 5 short, specific, data-driven insights about this dataset. "
            "Each insight must mention actual numbers from the data. "
            "Reply with exactly 5 lines, one insight per line, no numbering."
        )
        model = os.getenv("AI_MODEL") or (config["model"] or "")
        if name == "claude":
            text = _call_anthropic(api_key, model, prompt)
        else:
            text = _call_openai_compatible(api_key, config["base_url"], model, prompt)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[:5] if len(lines) >= 3 else _template_insights(df)
    except Exception:
        return _template_insights(df)


def _template_insights(df: pd.DataFrame) -> list[str]:
    """Build data-driven insights from pandas without AI."""
    insights = []
    insights.append(f"Dataset contains {len(df):,} rows and {df.shape[1]} columns.")
    missing = int(df.isna().sum().sum())
    if missing:
        insights.append(f"There are {missing:,} missing values across the dataset.")
    else:
        insights.append("No missing values found — dataset is complete.")
    numeric = analysis.numeric_columns(df)
    if numeric:
        col = numeric[0]
        insights.append(
            f"{col} ranges from {df[col].min():,.2f} to {df[col].max():,.2f} "
            f"with an average of {df[col].mean():,.2f}."
        )
    cats = list(analysis.get_categorical_overview(df).keys())
    if cats:
        top_val = df[cats[0]].value_counts().index[0]
        top_cnt = int(df[cats[0]].value_counts().iloc[0])
        insights.append(f"Most common {cats[0]} is '{top_val}' appearing {top_cnt:,} times.")
    dups = int(df.duplicated().sum())
    insights.append(f"Duplicate rows: {dups} ({dups/len(df)*100:.1f}% of data)." if dups else "No duplicate rows found.")
    return insights[:5]


def answer_free_question(df: pd.DataFrame, question: str, provider: Optional[str] = None) -> str:
    """Answer any free-text question about the dataset using one LLM call.

    Used as a fallback when the rule-based QA engine cannot match a
    question to a keyword pattern. The model receives the column names,
    data types, a sample of rows, and basic statistics so it can answer
    from real data rather than hallucinating.

    Args:
        df: The uploaded dataset.
        question: The user's free-text question.
        provider: Provider name overriding the AI_PROVIDER setting.

    Returns:
        A plain-language answer string. Returns an empty string when no
        API key is configured so the caller can decide what to show.
    """
    name = (provider or PROVIDER).lower()
    config = PROVIDER_CONFIG.get(name)
    api_key = os.getenv(config["env_key"] or "") if config else None
    if not config or not api_key:
        return ""
    try:
        prompt = _build_free_question_prompt(df, question)
        model = os.getenv("AI_MODEL") or (config["model"] or "")
        if name == "claude":
            return _call_anthropic(api_key, model, prompt)
        return _call_openai_compatible(api_key, config["base_url"], model, prompt)
    except Exception:
        return ""


def _build_free_question_prompt(df: pd.DataFrame, question: str) -> str:
    """Build a prompt that includes enough dataset context for a free answer.

    Args:
        df: The uploaded dataset.
        question: The user's question.

    Returns:
        A prompt string with column info, stats and sample rows.
    """
    numeric = analysis.numeric_columns(df)
    categorical = analysis.categorical_columns(df)
    lines = [
        f"You are a data analyst. A user uploaded a CSV with {len(df)} rows and the following columns:",
        f"  Numeric columns : {', '.join(numeric) or 'none'}",
        f"  Category columns: {', '.join(categorical) or 'none'}",
    ]
    if numeric:
        lines.append("Basic statistics:")
        lines.append(df[numeric].describe().round(2).to_string())
    for col in categorical[:3]:
        top = df[col].value_counts().head(5)
        lines.append(f"Top values in '{col}': {dict(top)}")
    lines.append(f"\nUser question: {question}")
    lines.append(
        "Answer the question directly and concisely (2-4 sentences) using only facts "
        "from the data above. Do not invent numbers that are not in the data."
    )
    return "\n".join(lines)


def _template_explanation(result: QAResult) -> str:
    """Build an offline explanation purely from the pandas result.

    Args:
        result: The answered question with stats context.

    Returns:
        A 1-2 sentence explanation using shares/comparisons when the
        stats provide them, so the app works without any API key.
    """
    stats: dict[str, Any] = result.stats or {}
    if "share_percent" in stats and "top" in stats:
        sentence = (
            f"{stats['top']} leads this comparison, accounting for approximately "
            f"{stats['share_percent']}% of the total"
        )
        if "group_count" in stats:
            sentence += f" across {stats['group_count']} groups"
        sentence += "."
        if "runner_up" in stats:
            sentence += f" The runner-up is {stats['runner_up']} at {stats['runner_up_value']:,.2f}."
        return sentence
    if "overall_average" in stats:
        return (
            f"{result.answer_text}, based on {stats.get('rows_matched', '?')} matching rows, "
            f"compared with an overall average of {stats['overall_average']:,.2f}."
        )
    if "peak_month" in stats:
        return f"{result.answer_text}, the strongest of {stats.get('months', '?')} months in the data."
    return f"{result.answer_text}. This figure was computed directly from the uploaded dataset."
