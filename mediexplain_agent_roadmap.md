# MediExplain — Claude Code Agent Roadmap
## Complete Guide: Repo Cleanup → Vercel Deployment → Streamlit Cloud

> **Copy this entire file into Claude Code at the start of every session.**
> It gives the agent full context so it never asks obvious questions.

---

## 🧠 PROJECT CONTEXT (Read This First, Agent)

**What this project is:**
MediExplain is a multi-page Streamlit app with RAG + modular LLM bots for medical text explanation. It has three main surfaces:
1. **MediExplain Chatbot** — explains medical text in plain language
2. **Synthetic Patient Workflow** — generates synthetic patient records (demographics → labs → notes → PDF)
3. **Validator Console** — consistency/safety checks on generated records

**Tech stack:**
- Python 3.11, Streamlit (multi-page)
- OpenAI API — `text-embedding-3-small` for embeddings, GPT models for completions
- ChromaDB — persistent vector store under `mediexplain/mediexplain_chromadb/`
- BeautifulSoup / lxml — HTML ingestion for RAG
- DuckDB, pandas, numpy
- Optional: Google Generative AI

**Repo structure (critical to understand):**
```
mediexplain/                  ← git root (what you clone)
├── README.md
├── .gitignore
├── .github/
├── .devcontainer/
└── mediexplain/              ← ALL application code lives here
    ├── streamlit_app.py      ← PRIMARY ENTRY POINT
    ├── requirements.txt
    ├── .streamlit/           ← config.toml + secrets.toml (secrets NOT committed)
    ├── app/
    │   ├── bots/             ← user-facing chat + meds RAG bots
    │   ├── rag/              ← ingest.py, config.py, retrieval
    │   └── main_app.py       ← consent-gated minimal UI
    ├── core/                 ← reusable LLM steps for synthetic records
    ├── app_synthetic/
    │   └── synthetic_app.py
    ├── tools/
    ├── docs/screenshots/
    ├── html/                 ← (user provides) PMC HTML articles for RAG
    └── mediexplain_chromadb/ ← (generated) Chroma persistent store
```

**Secrets needed (never committed to git):**
- `OPENAI_API_KEY` — primary, required for all features
- Set via `mediexplain/mediexplain/.streamlit/secrets.toml` locally
- Set via Vercel env vars or Streamlit Cloud secrets for deployment

---

## 📋 PHASE 1 — Repo Cleanup & Polish

**Goal:** Make the GitHub repo professional, well-documented, and recruiter/interviewer ready.

### 1.1 — Fix .gitignore

Ensure these are ignored (add if missing):
```
# Secrets
.streamlit/secrets.toml
mediexplain/.streamlit/secrets.toml
*.env
.env

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
env/

# ChromaDB (generated, don't commit)
mediexplain/mediexplain_chromadb/
mediexplain/html/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Outputs
*.pdf
*.log
```

### 1.2 — Verify secrets.toml is NOT tracked

Run: `git ls-files | grep secrets`
If anything shows up: `git rm --cached <path>` then commit.

### 1.3 — Add .streamlit/config.toml (if missing)

Create `mediexplain/mediexplain/.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#2563EB"
backgroundColor = "#0F172A"
secondaryBackgroundColor = "#1E293B"
textColor = "#F1F5F9"
font = "sans serif"

[server]
headless = true
port = 8501
```

### 1.4 — Clean up requirements.txt

Open `mediexplain/mediexplain/requirements.txt` and:
- Pin all versions (run `pip freeze` in venv to get exact versions)
- Remove duplicates
- Separate into groups with comments:

```txt
# Core UI
streamlit>=1.35.0

# LLM & Embeddings
openai>=1.30.0
google-generativeai>=0.7.0

# Vector Store
chromadb>=0.5.0

# NLP / ML
sentence-transformers>=3.0.0
numpy>=1.26.0
pandas>=2.2.0

# Parsing
beautifulsoup4>=4.12.0
lxml>=5.2.0
pypdf>=4.2.0

# Database
duckdb>=0.10.0

# Utilities
python-dotenv>=1.0.0
requests>=2.32.0
```

> **On Linux/Mac only** (for ChromaDB compatibility):
> `pysqlite3-binary>=0.5.0`

### 1.5 — Add placeholder folders with .gitkeep

```bash
mkdir -p mediexplain/mediexplain/html
mkdir -p mediexplain/mediexplain/docs/screenshots
touch mediexplain/mediexplain/html/.gitkeep
touch mediexplain/mediexplain/docs/screenshots/.gitkeep
```

### 1.6 — Rewrite README.md

Replace the current README with a polished version that includes:
- Project banner/title with badges (Python version, Streamlit, OpenAI)
- One-line description
- Feature list with emoji bullets
- Architecture diagram (ASCII or Mermaid)
- Prerequisites
- Local setup (step by step)
- How to add RAG data (html/ folder instructions)
- Deployment section (Vercel + Streamlit Cloud)
- Screenshots section (point to docs/screenshots/)
- Tech stack table
- Disclaimer (research/education only, not medical advice)
- License

### 1.7 — Add GitHub Actions CI (optional but impressive)

Create `.github/workflows/lint.yml`:
```yaml
name: Lint & Type Check
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check mediexplain/
```

---

## 📋 PHASE 2 — Get the App Running Locally

**Goal:** Confirm everything works before deployment.

### 2.1 — Setup

```bash
cd mediexplain                          # git root
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install -r mediexplain/requirements.txt
```

### 2.2 — Set your API key

Create `mediexplain/.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-..."
```

### 2.3 — Run the app

```bash
streamlit run mediexplain/streamlit_app.py
```

Expected: app opens at `http://localhost:8501` with 3 pages in sidebar.

### 2.4 — Debug checklist (agent: fix any of these that fail)

- [ ] All imports resolve (no ModuleNotFoundError)
- [ ] `OPENAI_API_KEY` loads correctly — check `st.secrets["OPENAI_API_KEY"]` vs `os.environ.get("OPENAI_API_KEY")`
- [ ] ChromaDB initializes without error (even if html/ is empty)
- [ ] All 3 pages load without crashing
- [ ] Chatbot sends a test message and gets a response
- [ ] Synthetic workflow generates a demo record

### 2.5 — Common fixes agent should apply

**Fix 1 — Secret loading (standardize across all files):**
```python
import os
import streamlit as st

def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.environ.get("OPENAI_API_KEY", "")
```

**Fix 2 — ChromaDB path resolution:**
The config in `app/rag/config.py` should use `__file__` based paths:
```python
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent  # points to mediexplain/ package
HTML_DIR = BASE_DIR / "html"
CHROMA_DIR = BASE_DIR / "mediexplain_chromadb"
```

**Fix 3 — Empty html/ directory graceful handling:**
Any RAG ingest function should check if html/ has files before trying to build index.

---

## 📋 PHASE 3 — Vercel Deployment

> **Important:** Streamlit is a Python server-side framework — it can't be deployed natively on Vercel (which is serverless). The way to deploy MediExplain on Vercel is to build a **FastAPI backend** that wraps the core logic, and a **React/HTML frontend** that calls it.

### Option A — Deploy as FastAPI + React on Vercel (Recommended for portfolio)

**Architecture:**
```
Vercel
├── /api/           ← Python serverless functions (FastAPI via @vercel/python)
│   ├── explain.py  ← POST /api/explain (chatbot endpoint)
│   ├── synthetic.py← POST /api/synthetic (synthetic record endpoint)
│   └── health.py   ← GET /api/health
└── /frontend/      ← Static React build
    └── index.html  ← Single-page app calling the API
```

### 3.1 — Create vercel.json at repo root

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/*.py",
      "use": "@vercel/python"
    },
    {
      "src": "frontend/package.json",
      "use": "@vercel/static-build",
      "config": { "distDir": "build" }
    }
  ],
  "routes": [
    { "src": "/api/(.*)", "dest": "/api/$1" },
    { "src": "/(.*)", "dest": "/frontend/build/$1" }
  ]
}
```

### 3.2 — Create api/ folder at repo root

**api/requirements.txt:**
```txt
fastapi==0.111.0
openai==1.30.0
chromadb==0.5.0
pysqlite3-binary==0.5.3
sentence-transformers==3.0.0
beautifulsoup4==4.12.3
lxml==5.2.2
numpy==1.26.4
pandas==2.2.2
python-dotenv==1.0.1
```

**api/explain.py:**
```python
from http.server import BaseHTTPRequestHandler
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mediexplain'))

from openai import OpenAI

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length))
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are MediExplain, a medical text explainer. Explain in plain language. Always include: this is for education only, not medical advice."},
                {"role": "user", "content": body.get("text", "")}
            ]
        )
        
        result = response.choices[0].message.content
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"explanation": result}).encode())
```

### 3.3 — Set Vercel environment variables

In Vercel dashboard → Project → Settings → Environment Variables:
- `OPENAI_API_KEY` = your key
- `PYTHON_VERSION` = `3.11`

### 3.4 — Deploy

```bash
npm install -g vercel
vercel login
vercel --prod
```

---

## 📋 PHASE 4 — Streamlit Cloud Deployment (Easier, Do This First)

> Streamlit Cloud is the **natural home** for this app and should be deployed alongside Vercel. Deploy Streamlit Cloud first — it's 5 minutes.

### 4.1 — Prepare for Streamlit Cloud

The app is already Streamlit-native. Just need:

**Entry point:** `mediexplain/streamlit_app.py` (relative to repo root)

**Secrets on Streamlit Cloud:**
In Streamlit Cloud → App → Settings → Secrets:
```toml
OPENAI_API_KEY = "sk-..."
```

**requirements.txt must be at:** `mediexplain/requirements.txt`
(Streamlit Cloud auto-detects it if you set the app path correctly)

### 4.2 — RAG on Streamlit Cloud

Since `html/` files are not committed, handle gracefully:
```python
# In any RAG init code:
if not list(HTML_DIR.glob("*.html")):
    st.info("No HTML articles loaded. RAG features unavailable. Upload articles to enable.")
    return None
```

Alternatively, commit a small sample PMC article to `mediexplain/html/sample/` for demo purposes.

### 4.3 — Deploy steps

1. Go to https://share.streamlit.io
2. Connect your GitHub account
3. Select repo: `Avish16/mediexplain`
4. Set **Main file path:** `mediexplain/streamlit_app.py`
5. Add secrets (OPENAI_API_KEY)
6. Click Deploy

**URL format:** `https://avish16-mediexplain-mediexplain-streamlitapp-xxxx.streamlit.app`

---

## 📋 PHASE 5 — Final Polish Before Sharing

### 5.1 — Add demo screenshots

After both deployments are live:
- Screenshot each page
- Save to `mediexplain/docs/screenshots/`
- Update README with actual images

### 5.2 — Update README with live links

```md
## 🚀 Live Demo
- **Streamlit Cloud:** [Open App](https://your-app.streamlit.app)
- **Vercel (API):** [API Docs](https://your-app.vercel.app/api/health)
```

### 5.3 — Add GitHub topics

On GitHub repo page → About (gear icon) → Topics:
`python`, `streamlit`, `openai`, `rag`, `chromadb`, `medical-ai`, `nlp`, `llm`, `healthcare`

### 5.4 — Pin the repo on your GitHub profile

GitHub Profile → Customize → Pin `mediexplain`

---

## 🛑 AGENT INSTRUCTIONS (Critical Rules)

1. **Never commit** `secrets.toml`, `.env`, or any file containing API keys
2. **Always run from repo root** — `streamlit run mediexplain/streamlit_app.py`
3. **ChromaDB path** — use `Path(__file__).parent` based resolution, not hardcoded paths
4. **Secret loading** — standardize all files to try `st.secrets` first, fallback to `os.environ`
5. **RAG graceful degradation** — app must not crash if `html/` is empty
6. **Vercel Python functions** — must use `BaseHTTPRequestHandler` pattern (not FastAPI directly) unless using a proper FastAPI + ASGI adapter
7. **Before pushing** — run `git status` and verify no secrets are staged

---

## 🎯 EXECUTION ORDER

```
Phase 1 → Repo Cleanup     (30 min)
Phase 2 → Local Run        (20 min, fix bugs)
Phase 4 → Streamlit Cloud  (10 min, deploy first)
Phase 3 → Vercel           (45 min, API refactor)
Phase 5 → Final Polish     (20 min)
```

**Total estimated time with Claude Code agent: ~2 hours**

---

## 📎 QUICK REFERENCE

| Command | Purpose |
|---|---|
| `streamlit run mediexplain/streamlit_app.py` | Run locally |
| `git ls-files \| grep secrets` | Check no secrets tracked |
| `vercel --prod` | Deploy to Vercel |
| `pip freeze > requirements-lock.txt` | Lock exact versions |
| `ruff check mediexplain/` | Lint |

---

*Generated for: Avi Sharma — MediExplain project*
*Repo: https://github.com/Avish16/mediexplain*
