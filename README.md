<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Mistral_AI-FF7000?style=for-the-badge&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-0078D4?style=for-the-badge&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
</p>

<h1 align="center">🏥 Health Insurance Benefits Navigator</h1>

<p align="center">
  <b>A RAG-powered AI chatbot that helps Massachusetts residents navigate health insurance plans, benefits, copays, premiums, and ConnectorCare subsidies.</b>
</p>

<p align="center">
  <a href="#-project-overview">Overview</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-setup--installation">Setup</a> •
  <a href="#-how-to-run">How to Run</a> •
  <a href="#-features">Features</a> •
  <a href="#-data-sources">Sources</a>
</p>

---

## 🎯 Project Overview

The Health Insurance Benefits Navigator is a **Capstone Project** that uses **Retrieval-Augmented Generation (RAG)** to help Massachusetts residents understand and compare health insurance plans available through the MA Health Connector for plan year 2025.

### 💬 Example Questions You Can Ask

```
"What is the copay for a specialist visit on a Silver HMO plan?"
"What is ConnectorCare and do I qualify?"
"Compare Harvard Pilgrim and BCBS Gold plans"
"Which plans are HSA eligible in Massachusetts?"
"What is the monthly premium for a 40-year-old on a Gold plan?"
"What does a Platinum plan cover vs a Bronze plan?"
```

---

## 🏗️ System Architecture

### 📦 Data Preparation

| Step | Process |
|------|---------|
| 1️⃣ | Raw Data — CMS PUF 2025 + MA Health Connector PDF |
| 2️⃣ | Data Cleaning & Extraction — filter to 8 MA carriers |
| 3️⃣ | Data Chunking & Metadata Tagging |
| 4️⃣ | Sentence Embedding — `all-MiniLM-L6-v2` |
| 5️⃣ | FAISS Index — `IndexFlatIP` (cosine similarity) |

### 🔁 RAG Pipeline

| Step | Process |
|------|---------|
| 1️⃣ | User Query |
| 2️⃣ | Query Encoding — enriched with age, tier, carrier filters |
| 3️⃣ | Semantic Search in FAISS — retrieves |
| 4️⃣ | Re-ranking with XGBoost |
| 5️⃣ | LLM Generation — Mistral AI `mistral-small-latest` |
| 6️⃣ | Response to User |
| 7️⃣ | Feedback Loop — logs ratings → retrains XGBoost every 50 entries |

---

## 📊 Dataset

### Data Sources

| # | Source | Content |
|---|--------|---------|
| 1 | **CMS State-Based Exchange PUF 2025** | Plan names, benefits, copays, premiums by age |
| 2 | **MA Health Connector ConnectorCare PDF 2025** | Exact subsidy copay values |
| 3 | **MA DOI Official Plan Filing 2025** | Licensed carriers, plan forms, DOI numbers |
| 4 | **CMS Age Rating Curve (MA-specific)** | Premium multipliers by age (2:1 cap) |

### Coverage

| Category | Details |
|----------|---------|
| 🏢 **Licensed MA Carriers** | BCBS MA, Harvard Pilgrim, Tufts Health, Fallon, Health New England, WellSense, Mass General Brigham, UnitedHealthcare |
| 🏅 **Plan Tiers** | Catastrophic, Bronze, Silver, Gold, Platinum |
| 🏥 **Plan Types** | HMO, PPO, EPO |
| 👤 **Age Range** | 0 – 64 |
| 💰 **ConnectorCare** | 7 income tiers (0 – 500% FPL) |

### Key Policy Fields

| Field | Value | Source |
|-------|-------|--------|
| Gender affects premium | ❌ No | ACA §2701 |
| GLP-1 obesity coverage | ❌ Not covered | MA Health Connector SOA Memo July 2025 |
| Preventive care cost | ✅ $0 | ACA mandate |
| Insulin (generic) | ✅ $0 | PACT Act Ch.342 2024 |
| Insulin (brand) max | ✅ $25 | PACT Act Ch.342 2024 |
| Pediatric dental | ✅ Included | ACA EHB mandate |
| Maternity | ✅ Covered | ACA EHB mandate |

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| 📓 Notebook | Google Colab | Development environment |
| 🐍 Language | Python 3.12 | Core programming language |
| 📊 Data Processing | Pandas, NumPy | Data cleaning & transformation |
| 🔢 Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Text vectorization |
| 🗄️ Vector Store | FAISS (IndexFlatIP) | Semantic search |
| 🎯 Re-ranker | XGBoost (rank:pairwise) | Result re-ranking with 11 features |
| 🤖 LLM | Mistral AI (mistral-small-latest) | Answer generation |
| 🖥️ UI | Streamlit | Chatbot interface |
| 🌐 Tunnel | ngrok / Colab Ports | Public URL for app |
| ☁️ Storage | Google Drive | Persistent file storage |

---

## 📁 Project Structure

```
MA_Health_Navigator/
│
├── 📓 MA_Health_Insurance_Pipeline.ipynb   ← Main Colab notebook
│
├── 📂 data/
│   ├── 📂 raw/
│   │   └── cms_puf/                        ← Downloaded CMS PUF files
│   ├── 📂 processed/
│   │   ├── ma_plans_2025.csv               ← Master plan dataset
│   │   ├── ma_plans_2025.json              ← JSON version
│   │   ├── connectorcare.csv               ← ConnectorCare copay table
│   │   ├── age_multipliers.csv             ← MA age rating curve
│   │   └── chunks.jsonl                    ← RAG text chunks
│   ├── 📂 vectorstore/
│   │   ├── index.faiss                     ← FAISS vector index
│   │   └── chunks_metadata.pkl             ← Chunk metadata
│   └── 📂 feedback/
│       └── feedback_log.csv                ← User feedback log
│
├── 📂 models/
│   └── xgb_reranker.pkl                    ← Trained XGBoost re-ranker
│
├── 📂 app/
│   └── app.py                              ← Streamlit chatbot UI
│
└── 📄 README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- ✅ Google Account (for Colab + Drive)
- ✅ Mistral AI free account → [console.mistral.ai](https://console.mistral.ai)
- ✅ ngrok free account (optional) → [dashboard.ngrok.com](https://dashboard.ngrok.com)

### Step 1 — Open Google Colab

Go to [colab.research.google.com](https://colab.research.google.com) and open `MA_Health_Insurance_Pipeline.ipynb`

### Step 2 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3 — Install Dependencies

```python
!pip install -q requests beautifulsoup4 pandas numpy matplotlib seaborn \
    sentence-transformers faiss-cpu xgboost scikit-learn \
    langchain langchain-community transformers \
    streamlit pyngrok mistralai pdfplumber
```

### Step 4 — Add Mistral API Key

> In Colab left sidebar → 🔑 **Secrets** → Add new secret
> - **Name:** `MISTRAL_API_KEY`
> - **Value:** your key from [console.mistral.ai](https://console.mistral.ai)
> - Toggle **Notebook access → ON**

---

## 🚀 How to Run

Run the notebook cells in this exact order:

| Cell | Phase | Description |
|------|-------|-------------|
| `Cell 1` | ⚙️ Setup | Mount Drive, Install Dependencies, and Import Libraries | 
| `Cell 2` | ⚙️ Setup | Define Project Paths and Directory Structure | 
| `Cell 3` | 📥 Data | Configuration: Data Sources, Carriers, Benefits, and Plan Parameters |
| `Cell 4` | 📥 Data |  Build Realistic MA Synthetic Plan Data | 
| `Cell 5` | 🔄 Processing | Data Loading and Processing |
| `Cell 6` | 📊 EDA  | Exploratory Data Analysis | 
| `Cell 7` | 🧠 RAG | Build RAG Text Chunks |
| `Cell 8` | 🔢 Embedding | Generate Embeddings + Build FAISS Index |
| `Cell 9` | 🎯 Re-ranking | Feature Extraction + XGBoost re-ranker | 
| `Cell 10` | 🤖 LLM | Mistral AI + RAG pipeline | 
| `Cell 11` | 🧪 Testing + 🔄 Feedback | RAG Pipeline Testing + Feedback loop | 
| `Cell 12` | 🌐 App | Streamlit app | 
| `Cell 13` | 🌐 App | Launch Streamlit app via ngrok | 


### Launch the App

```python
# Option 1 — Colab Ports tab (easiest, no account needed)
# Click "Ports" tab at bottom of Colab → Open port 8501

# Option 2 — ngrok (requires free account at dashboard.ngrok.com)
from pyngrok import ngrok
ngrok.set_auth_token("your_authtoken_here")
tunnel = ngrok.connect(8501)
print(tunnel.public_url)
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Smart Semantic Search** | Sentence embeddings + FAISS for accurate retrieval from 400+ plan records |
| 🎯 **XGBoost Re-ranking** | 11-feature re-ranker improves result relevance beyond pure similarity |
| 💬 **Natural Language Q&A** | Mistral AI generates accurate, cited answers using retrieved context |
| 🔧 **Smart Filters** | Filter by age, gender, metal tier, carrier, and ConnectorCare eligibility |
| 💰 **ConnectorCare Subsidies** | 7 income tiers with exact official copay values from MA Health Connector |
| 📈 **Premium by Age** | MA-specific 2:1 age rating curve for accurate premium estimates |
| 🔄 **Feedback Loop** | 1–5 star ratings logged to CSV and triggers XGBoost retraining every 50 entries |

---

## 📚 Data Sources

| Ref | Document | Link |
|-----|----------|------|
| `[1]` | MA Division of Insurance — Insured Health Plans 2025 | [View](https://www.mass.gov/doc/coverages-available-for-individuals-and-small-groups-effective-on-or-after-january-1-2025/download) |
| `[2]` | MA Health Connector — 2026 Seal of Approval Board Memo (July 9, 2025) | [View](https://www.mahealthconnector.org/wp-content/uploads/board_meetings/2025/07-10-25/Board-Memo-2026-Seal-of-Approval-Submission-Summary-070925.pdf) |
| `[3]` | MA Health Connector — ConnectorCare Overview 2025 (Table 4) | [View](https://www.mahealthconnector.org/wp-content/uploads/ConnectorCare-Overview-2025.pdf) |
| `[4]` | CMS State-Based Exchange PUF 2025 | [View](https://www.cms.gov/marketplace/resources/data/state-based-public-use-files) |
| `[5]` | CMS — State-Specific Age Curve Variations (MA 2:1 cap) | [View](https://www.cms.gov/marketplace/resources/data/premium-stabilization-programs) |
| `[6]` | ACA(Affordable Care Act) §2701 — Prohibition on gender rating | [View](https://www.mass.gov/info-details/health-care-reform-for-individuals) |
| `[7]` | PACT Act (Chapter 342, Acts of 2024) — Insulin cost caps | [View](https://malegislature.gov) |

---

## ⚠️ Important Notes

> 🚫 **GLP-1 Drugs:** As of January 1, 2026, all MA carriers removed coverage for GLP-1 drugs for weight loss/obesity per MA DOI and Health Connector directive.

> 👥 **Gender & Premiums:** ACA §2701 prohibits gender-based premium differences. Premiums are **identical for male and female** enrollees. All plans cover maternity as an ACA Essential Health Benefit.

> 💲 **Premium Estimates:** Premiums shown are statewide averages. Actual premiums vary by zip code. Use [mahealthconnector.org](https://www.mahealthconnector.org) for exact quotes.

> 🚨 **Catastrophic Plans:** Available only to adults **under 30** or those with a federal hardship/mandate exemption. Only BCBS MA and Tufts Direct offer Catastrophic plans on-Exchange in 2026.

---

<p align="center">
  Built for the Massachusetts Health Connector community 🏥
  <br/>
  Thank you
</p>
