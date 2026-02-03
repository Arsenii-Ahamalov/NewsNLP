## 1 Overview

**Smart News Summarizer & Analyzer** is an end‑to‑end project that takes raw news articles
from an API and turns them into readable summaries, topics and interactive visualizations.

Phase | Description | Status
----- | ----------- | ------
1 | Fetching & caching articles from The Guardian API | ✅ Complete
2 | Text cleaning, tokenization, stop‑words, lemmatization | ✅ Complete
3 | TextRank summarization (sentence graph + PageRank) | ✅ Complete
4 | Topic modeling with TF‑IDF, PCA & K‑Means | ✅ Complete
5 | Visualizations & Streamlit dashboard | ✅ Complete
6 | Database management & incremental updates | ✅ Complete

The goal is to quickly understand a larger set of news articles:
see the **main topics**, read **short summaries**, and explore data in an interactive dashboard.

## 2 Demo

![Smart News Dashboard Demo](gif.gif)

A short demo of the Streamlit dashboard showing: overview statistics, topic distribution, article browsing by topic, search & filter functionality, and processing new articles from The Guardian API.

## 3 Project Structure

```
SmartNews/
├── README.md
├── requirements.txt
├── run_dashboard.py              # Helper launcher for the Streamlit app
├── config.py                     # API key and basic configuration
├── ahamaars.pdf                  # Generated semester report (current version)
│
├── data/
│   ├── raw/                      # Original downloaded data (if kept)
│   └── processed/
│       ├── articles_with_topics.csv   # Main “database” used by the dashboard
│       ├── topic_modeling_metadata.json
│       └── plots/                     # PNG plots used in report / dashboard
│
├── src/
│   ├── fetch_articles.py         # Guardian API client + basic processing
│   ├── text_cleaning.py         # Cleaning, normalization, statistics
│   ├── utils_nlp.py             # Tokenization, stop‑words, lemmatization
│   ├── textrank.py              # TextRank summarization for single articles / DataFrames
│   ├── topic_modeling.py        # TF‑IDF, PCA, K‑Means, keyword extraction
│   ├── plots.py                 # Matplotlib/Seaborn plots and wordclouds
│   ├── database_manager.py      # Merge new articles, regenerate topics & summaries, stats
│   └── dashboard.py             # Main Streamlit dashboard implementation
│
├── scripts/
│   ├── regenerate_topics.py     # Standalone regeneration of topic modeling
│   └── regenerate_plots.py      # Regenerate all plots in data/processed/plots
│
├── notebooks/                   # Jupyter notebooks for exploration and verification
│   ├── data_verification.ipynb  # Data verification and pipeline testing
│   └── prototype.ipynb          # Prototype and experimentation notebook
│
├── tests/                       # Unit tests for core functionality
│   ├── test_fetch_articles.py
│   ├── test_text_cleaning.py
│   ├── test_utils_nlp.py
│   ├── test_textrank.py
│   ├── test_topic_modeling.py
│   ├── test_database_manager.py
│   ├── test_plots.py
│   └── test_dashboard.py
│
└── .gitignore
```

## 4 Installation

```bash
git clone <this-repo-url>
cd SmartNews

python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows (PowerShell / cmd)

pip install -r requirements.txt
```

On Windows it is recommended to use **WSL** (Ubuntu) and run all commands from there
– přesně tak, jak je projekt vyvíjen.

## 5 Run the project

### 5.1 Start the Streamlit dashboard

```bash
source venv/bin/activate        # or your existing venv
python run_dashboard.py

# alternatively:
streamlit run src/dashboard.py
```

Poté otevři prohlížeč na `http://localhost:8501`.

### 5.2 Run tests

```bash
pytest
# nebo selektivně:
pytest tests/test_textrank.py
pytest tests/test_topic_modeling.py
```

## 6 Data Preparation

- Výsledná data pro dashboard jsou v `data/processed/articles_with_topics.csv`
  a `data/processed/topic_modeling_metadata.json`.
- Během běhu dashboardu lze:
  - stáhnout nové články z Guardian API,
  - vyčistit je, udělat NLP preprocessing,
  - vygenerovat shrnutí a přiřadit témata,
  - sloučit je s existující databází (bez rušení starých dat).

Skripty:
- `scripts/regenerate_topics.py` – kompletně znovu spočítá TF‑IDF, K‑Means a metadata.
- `scripts/regenerate_plots.py` – přegeneruje všechny grafy v `data/processed/plots/`.

## 7 Implemented Pipeline

- **Fetching & caching**
  - Guardian Open Platform API (`src/fetch_articles.py`).
  - Filtrování podle sekce, dotazu a data.
- **Text cleaning & NLP**
  - Odstranění HTML, normalizace textu, statistiky délky (`src/text_cleaning.py`).
  - Tokenizace, stop‑slova, lemmatizace pomocí NLTK + spaCy (`src/utils_nlp.py`).
- **Summarization (TextRank)**
  - Dělení na věty, výpočet podobnosti (Jaccard nad lemmatizovanými tokeny).
  - Graf vět v `networkx`, PageRank, výběr vět do shrnutí (~10 % původního textu).
- **Topic modeling**
  - TF‑IDF (`sklearn.TfidfVectorizer`) s rozšířeným stop‑word seznamem.
  - PCA pro zlepšení stability shlukování.
  - K‑Means s testováním K (pro finální běh fixně 7 témat, aby odpovídal analýze).
  - Extrakce top‑klíčových slov z center clusterů.
- **Visualization**
  - Distribuce témat, top‑klíčová slova, heatmapa, PCA scatter, wordcloudy (`src/plots.py`).
- **Dashboard**
  - Přehled (`Overview`), články podle témat, vizualizace, vyhledávání a filtrování,
  - záložka „Process New Articles“ pro stažení a přidání nových článků do databáze.
- **Database management**
  - Sloučení nových článků s existujícím CSV,
  - automatické doplnění shrnutí,
  - možnost re‑generovat shrnutí i témata (`src/database_manager.py`).

## 8 Road‑map / Status

- Phase 1: Data fetching & exploration – **done**
- Phase 2: Text preprocessing & NLP – **done**
- Phase 3: TextRank summarization – **done**
- Phase 4: Topic modeling + visualization – **done**
- Phase 5: Streamlit dashboard + database update flow – **done**
- Possible future work:
  - neural summarization (BART/T5),
  - multi‑language support,
  - lepší hodnocení shrnutí (ROUGE, BLEU).

_Last updated 2025‑12‑18_