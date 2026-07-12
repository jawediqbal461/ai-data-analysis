# 📊 AI Data Analysis Assistant

A beginner-friendly Python app that loads any CSV dataset, summarizes it,
answers **natural-language questions** about it, draws a **chart**, and adds a
short **AI-generated plain-language explanation** — available both as a
Streamlit web app and as a plain command-line tool.

## Objective

Show an end-to-end "ask your data" pipeline built from simple, reusable parts:

1. **Load & summarize** — pandas loads the CSV with friendly error handling and
   reports rows, columns, dtypes and missing values.
2. **Understand the question** — a rule-based engine detects intent keywords
   (*highest, average, most frequent, total, trend…*) and the column names
   mentioned in the question. Nothing is hardcoded to one dataset.
3. **Compute** — reusable pandas/numpy functions (group totals, averages,
   value counts, extremes) produce the raw answer plus a supporting table.
4. **Visualize** — matplotlib/seaborn automatically renders a couple of
   complementary, labeled charts for each answer (e.g. a bar chart plus a
   pie chart for category comparisons) and saves each as a timestamped PNG
   in `charts/`. There is no chart-type picker — the pipeline always
   chooses the chart(s) that fit the question.
5. **Explain** — exactly **one** LLM API call turns the numbers into a 1–2
   sentence explanation. With no API key, an offline template explanation is
   generated instead, so the app always works.

## Folder structure

```
project/
├── main.py            # Entry point: Streamlit app + CLI fallback
├── dataset.csv        # Bundled sample data (e-commerce sales, ~180 rows)
├── analysis.py        # CSV loading + reusable statistics functions
├── qa_engine.py       # Rule-based natural-language question routing
├── visualization.py   # Chart building + saving (matplotlib/seaborn)
├── ai_explainer.py    # One-call LLM wrapper + AI question suggestions, with offline fallback
├── report.py          # CSV/PDF report builders for the download buttons
├── requirements.txt   # Pinned dependencies
├── .env.example       # Template for API keys / provider settings
├── README.md
└── charts/            # Auto-created; every chart is saved here as PNG
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) configure an LLM provider
copy .env.example .env          # Windows  (cp on macOS/Linux)
# then edit .env: set AI_PROVIDER and the matching API key
```

Supported providers (set `AI_PROVIDER` in `.env` — this is the **only** place
the provider is chosen; there is no provider picker in the app itself):

| Provider | `AI_PROVIDER` | Key variable        |
|----------|---------------|---------------------|
| OpenAI   | `openai`      | `OPENAI_API_KEY`    |
| Groq     | `groq`        | `GROQ_API_KEY`      |
| Gemini   | `gemini`      | `GEMINI_API_KEY`    |
| Claude   | `claude`      | `ANTHROPIC_API_KEY` |

> **No key? No problem.** Without a key the app produces a template-based
> explanation computed directly from the pandas result, fully offline.

## How to run

**Web app (recommended):**

```bash
streamlit run main.py
```

The app only ever analyzes the CSV **you upload** in the sidebar — there is
no bundled/default dataset loaded automatically.

- **Sidebar**: upload your CSV, then ask a question. Below the upload box
  you'll find clickable **example questions** and 3 **AI-suggested
  questions** generated specifically for your dataset's columns — click
  any of them to auto-fill the question box, or type your own, then hit
  **Run analysis**.
- **Top of the page**: a 🌙 **dark/light mode** toggle sits top-right next
  to the title, and a **Download report** row lets you export the current
  analysis as a **CSV** or a **PDF** (numeric/categorical stats, overview
  charts, and the latest answered question with its chart(s)).
- **Answer first**: once you run analysis, the answer, its charts and the
  AI explanation appear at the very top of the page — above the dataset
  summary — and stay pinned there until you ask a new question or upload a
  different file. Two automatically chosen overview charts (never dummy
  placeholders — always built from your actual data) are shown right below.

**Command line:**

```bash
python main.py --csv dataset.csv --question "Which category generated the highest sales?"
```

`--csv` is required (the CLI has no "upload" step, so you always point it at
a file explicitly). The LLM provider comes only from `AI_PROVIDER` in `.env` -
there is no `--provider` flag. Running without `--question` prints the
dataset summary and example commands.

## Example questions & expected output

| Question | Answer format |
|----------|---------------|
| Which category generated the highest sales? | `Electronics has the highest total revenue: 12,345.67` + bar chart + pie chart + explanation |
| What is the average price of Electronics? | `The average price for Electronics is 84.13` + bar chart of averages per category |
| Which city has the maximum orders? | `New York appears most frequently in 'city': 38 times (21.8% of rows)` + bar + pie chart |
| Which product appears most frequently? | `Cotton T-Shirt appears most frequently in 'product': 19 times (10.6% of rows)` + bar chart |
| What is the total quantity sold? | `The total quantity is 405.00` + per-category breakdown chart |
| How does revenue trend by month? | `Total revenue peaks in 2025-03 at 4,321.00` + line chart + bar chart |

Each question typically generates **two complementary charts** (e.g. a bar
chart plus a pie chart, or a histogram plus a box plot), each saved as its
own timestamped PNG (e.g. `charts/bar_20260711_143000_123456.png`), followed
by a short AI explanation such as:

> *"The Electronics category contributes the highest revenue, accounting for
> approximately 42% of the total across 5 categories."*

## Screenshots

<!-- Add screenshots after running the app -->
| Web app | Chart output |
|---------|--------------|
| *(screenshot placeholder — main page with summary and answer)* | *(screenshot placeholder — generated bar chart)* |

## Notes on the sample dataset

`dataset.csv` contains ~180 synthetic e-commerce orders (columns: `order_id`,
`product`, `category`, `city`, `customer_age`, `quantity`, `price`,
`order_date`). It intentionally includes missing values in `customer_age` and
`city` so the missing-value report has something real to show.
