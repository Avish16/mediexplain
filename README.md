# MediExplain

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

**MediExplain** turns dense medical text into plain-language explanations using a RAG pipeline, modular LLM bots, and a multi-page Streamlit interface.

> **Disclaimer:** For education and demonstration only. Not a medical device. Not a substitute for professional medical advice, diagnosis, or treatment.

---

## Features

- **MediExplain Chatbot** — paste any clinical text and get a plain-English breakdown
- **Synthetic Patient Workflow** — generate realistic patient records: demographics → labs → imaging → notes → PDF export
- **Validator Console** — consistency and safety checks on generated records
- **RAG over biomedical literature** — ChromaDB + `text-embedding-3-small` over local PMC-style HTML articles
- **Dev Container** — one-click environment in GitHub Codespaces or VS Code

---

## Architecture

```
streamlit_app.py          ← Multi-page entry point
│
├── app/
│   ├── bots/             ← Chat + medication RAG bots
│   ├── rag/              ← Ingest, config, retrieval (ChromaDB)
│   └── main_app.py       ← Consent-gated single-page UI
│
├── app_synthetic/
│   └── synthetic_app.py  ← Synthetic patient workflow
│
├── core/                 ← Modular LLM generators (demographics, labs, notes, PDF…)
│
├── tools/                ← Vector store utilities
│
├── html/                 ← (user-provided) PMC HTML articles for RAG
└── mediexplain_chromadb/ ← (generated) Chroma persistent store
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM / Embeddings | OpenAI GPT-4o-mini, `text-embedding-3-small` |
| Vector Store | ChromaDB |
| HTML Parsing | BeautifulSoup4, lxml |
| PDF Export | pypdf |
| Data | pandas, numpy, DuckDB |
| Optional | Google Generative AI |

---

## Prerequisites

- Python **3.11**
- An **OpenAI API key** (`OPENAI_API_KEY`)

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/Avish16/mediexplain.git
cd mediexplain

# 2. Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
# Create .streamlit/secrets.toml (never commit this file):
echo 'OPENAI_API_KEY = "sk-..."' > .streamlit/secrets.toml
# OR export it:
export OPENAI_API_KEY="sk-..."

# 5. Run
streamlit run streamlit_app.py
```

App opens at `http://localhost:8501`.

---

## Adding RAG Data

1. Place PMC-style `.html` files into the `html/` folder.
2. On first run the app indexes them into `mediexplain_chromadb/` automatically.
3. If `html/` is empty, the chatbot still works — RAG features degrade gracefully.

---

## Deployment

### Streamlit Cloud (recommended — free)

1. Push to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **Main file path:** `streamlit_app.py`
4. Add secret: `OPENAI_API_KEY = "sk-..."`
5. Click **Deploy**.

### Vercel (API-only)

See [Phase 3 of the agent roadmap](mediexplain_agent_roadmap.md) for the FastAPI + serverless function setup.

---

## Screenshots

| Chatbot | Synthetic Workflow | Validator |
|---|---|---|
| *(coming soon)* | *(coming soon)* | *(coming soon)* |

Screenshots will be added to `docs/screenshots/` after deployment.

---

## License

[Apache License 2.0](LICENSE)
