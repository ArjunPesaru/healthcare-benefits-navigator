# Health Benefits Navigator

A RAG-powered health insurance assistant that helps Massachusetts residents find and compare 2025 health plans. It uses semantic search, an XGBoost re-ranker, and Mistral AI to answer questions and recommend plans based on age, carrier, metal tier, and ConnectorCare eligibility.

---

## What it does

- Recommends MA health insurance plans based on your filters
- Answers questions about copays, premiums, deductibles, and benefits
- Explains ConnectorCare subsidy eligibility
- Compares plans side by side in a chat interface

---

## Requirements

- Python 3.10 or higher
- A Mistral AI API key (free at https://console.mistral.ai)
- Git

---

## Setup

### Step 1 - Clone the repository

```bash
git clone https://github.com/ArjunPesaru/healthcare-benefits-navigator.git
cd healthcare-benefits-navigator
```

### Step 2 - Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 - Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 - Add your Mistral API key

Create a file called `.env` in the project root:

```bash
MISTRAL_API_KEY=your_mistral_api_key_here
```

Get your free API key at https://console.mistral.ai

### Step 5 - Run one-time setup

This builds the FAISS vector index and trains the XGBoost re-ranker. Only needs to be run once.

```bash
python setup.py
```

This takes about 1-2 minutes on first run.

### Step 6 - Launch the app

```bash
streamlit run app.py
```

Open your browser at http://localhost:8501

---

## Project structure

```
healthcare-benefits-navigator/
├── app.py                          Streamlit frontend
├── config.py                       MA carriers, plans, ConnectorCare data
├── data_builder.py                 Generates plan CSVs and RAG text chunks
├── setup.py                        One-time index and model builder
├── requirements.txt                Python dependencies
├── packages.txt                    System packages for cloud deployment
├── rag/
│   ├── pipeline.py                 Full RAG query pipeline
│   ├── embeddings.py               FAISS index build and load
│   └── reranker.py                 XGBoost re-ranker train and load
├── data/
│   └── vectorstore/                FAISS index and chunk metadata
└── models/
    └── xgb_reranker.pkl            Trained re-ranker model
```

---

## How it works

1. Plan data for all 30 MA 2025 health plans is converted into text chunks
2. Chunks are embedded using the all-MiniLM-L6-v2 sentence transformer
3. Embeddings are stored in a FAISS index for fast semantic search
4. At query time, the top 15 matching chunks are retrieved
5. An XGBoost re-ranker narrows them to the top 5
6. Mistral AI generates a natural language answer from the top 5 chunks

---

## Filters available

- Age (18-64)
- Gender
- Metal tier (Bronze, Silver, Gold, Platinum)
- Carrier (8 licensed MA carriers)
- ConnectorCare eligibility (income-based subsidies up to 500% FPL)

---

## Troubleshooting

**App shows "Loading models and index" indefinitely**
Run `python setup.py` first before launching the app.

**MISTRAL_API_KEY not found error**
Make sure you created a `.env` file in the project root with your key.

**Segmentation fault on startup**
This is a known PyTorch and XGBoost conflict on some systems. Make sure you are using Python 3.10 and the exact versions in `requirements.txt`.

**Port 8501 already in use**
Run `streamlit run app.py --server.port 8502` to use a different port.

---

## Deployment

The app can be deployed for free on Streamlit Community Cloud.

1. Push the repository to GitHub
2. Go to https://share.streamlit.io and connect your GitHub account
3. Select the repository, set main file to `app.py`
4. Add `MISTRAL_API_KEY` under Advanced Settings as a secret
5. Click Deploy

---

## Data sources

- CMS State-Based Exchange Public Use Files 2025
- MA Health Connector ConnectorCare Overview 2025 (Table 4)
- MA DOI 2025 licensed carrier filings
