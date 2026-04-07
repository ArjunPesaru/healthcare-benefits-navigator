# Health Benefits Navigator

A Retrieval-Augmented Generation (RAG) system for exploring Massachusetts health insurance plans. Ask plain-English questions about coverage, copays, deductibles, and plan comparisons — and get answers grounded in real 2025 MA Health Connector data.

## Features

- Natural-language search over 28+ MA health plans (HMO, PPO, Bronze → Platinum)
- FAISS vector index for semantic retrieval
- XGBoost re-ranker with 11 query-chunk relevance features
- Mistral LLM for final answer generation
- ConnectorCare subsidy eligibility check based on FPL
- Age-adjusted premium estimates using CMS 2:1 cap curve
- Auto-detects HSA intent in queries for better plan retrieval
- Input length guard prevents oversized prompts to the LLM
- Centralised `PLAN_COLUMNS` schema keeps data definition in one place

## Project Structure

```
├── app.py              # Streamlit frontend
├── config.py           # Plan data, carrier list, ConnectorCare params
├── data_builder.py     # Builds CSVs and text chunks from config
├── setup.py            # One-time setup: build index + train re-ranker
├── rag/
│   ├── __init__.py     # Public API exports
│   ├── embeddings.py   # Sentence-transformer embedding + FAISS index
│   ├── pipeline.py     # End-to-end RAG: encode → retrieve → re-rank → LLM
│   └── reranker.py     # XGBoost re-ranking model
├── data/
│   ├── raw/            # Source plan data (generated)
│   ├── processed/      # Chunked text (generated)
│   └── vectorstore/    # FAISS index files (generated)
└── models/             # Saved XGBoost re-ranker
```

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your Mistral API key**

   Create a `.env` file in the project root:
   ```
   MISTRAL_API_KEY=your_key_here
   ```

3. **Build the index and train the re-ranker (run once)**

   ```bash
   python setup.py
   ```

4. **Launch the app**

   ```bash
   streamlit run app.py
   ```

## Data Sources

- **CMS Benefits & Cost Sharing PUF 2025** — benefit categories and cost-sharing structure
- **MA Health Connector 2025** — ConnectorCare plan parameters (PDF Table 4)
- **MA DOI 2025** — licensed carrier list
- **CMS MA age curve** — age multipliers with 2:1 statutory cap

## Requirements

- Python 3.10+
- Mistral API key (`mistral-small-latest` model)
- See `requirements.txt` for full dependency list
