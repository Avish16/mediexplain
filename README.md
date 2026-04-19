# MediExplain — RAG & modular AI assistants

**MediExplain** is a research and education prototype that turns dense medical text into plain-language explanations, optionally grounds answers in retrieved literature (RAG), and can generate synthetic patient-style records using a pipeline of small LLM **“bots.”**

**Repository layout:** The only project document at the repository root is this **README.md**. Application code, `requirements.txt`, `LICENSE`, and runtime folders (`html/`, vector stores, etc.) live alongside it, with tooling in dotfolders such as `.gitignore`, `.github/`, and `.devcontainer/`.

---

## Problem statement

Patients and caregivers often struggle to interpret clinical language, lab values, and discharge instructions. Clinicians and researchers also need safe, transparent ways to experiment with retrieval-augmented generation and modular assistants **without** pretending the system is a substitute for licensed care.

This project addresses that gap by:

- Explaining medical text in patient- or caregiver-appropriate language (with explicit disclaimers).
- Using **Chroma** + embeddings over PMC-style HTML articles for literature-grounded retrieval.
- Providing a **multi-step synthetic workflow** (demographics → labs → notes → consolidation → safety/consistency checks → PDF) for demos and teaching.

**Important:** Outputs are for understanding and research only — **not** diagnosis or treatment decisions.

---

## Dataset / source information

| Source | Role |
|--------|------|
| **`html/`** (at repo root) | Expected location for PubMed Central (PMC) or similar HTML exports when using the standalone RAG app (`mediexplain_rag_app.py`). Create this folder and add files before ingestion (see **How to run**). |
| **`app/html/`** | When using **`app/rag/`** ingest/retriever code, paths in `app/rag/config.py` resolve relative to the **`app/`** directory — place HTML here for that pipeline (see note below). |
| **OpenAI API** | Chat completions and `text-embedding-3-small` (and related models as configured) for explanations and embeddings. |
| **Google Generative AI** | Listed in `requirements.txt` for optional integrations (see code paths that import `google.generativeai`). |
| **Medication RAG** | Separate indexing/search under `app/bots/meds_rag_*` using project-specific knowledge stores (see those modules for paths and build steps). |

You are responsible for **licensing and attribution** of any full-text articles you download and index.

---

## How to run

### Prerequisites

- **Python 3.10+** (the dev container uses **3.11**; see `.devcontainer/devcontainer.json`).
- An **OpenAI API key** with access to the models you configure.

### Install

```bash
git clone https://github.com/Avish16/mediexplain.git
cd mediexplain
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On **Windows**, `pysqlite3-binary` is skipped automatically (no wheels); Chroma uses the **stdlib** SQLite. On Linux/macOS, the optional shim is installed for compatibility with some Chroma deployments.

### Secrets (Streamlit)

Set `OPENAI_API_KEY` via an environment variable or Streamlit secrets. A typical local file is:

**`.streamlit/secrets.toml`**

```toml
OPENAI_API_KEY = "sk-..."
```

This path is gitignored — do not commit keys.

---

## Main Streamlit app (multi-page)

**Recommended entry point** (also used by the dev container). From the repository root:

```bash
streamlit run streamlit_app.py
```

Or:

```bash
python -m streamlit run streamlit_app.py
```

Pages include **Synthetic App**, **MediExplain Chatbot**, and **Validator Console**.

---

## Other entry points

| Script | Purpose |
|--------|---------|
| `app/main_app.py` | Minimal consent-gated UI with a simple router (explainer vs labs). Example: `streamlit run app/main_app.py` |
| `mediexplain_rag_app.py` | Standalone PMC HTML → Chroma RAG demo (ingest + query in one file). Run from repo root so `./html` and `./mediexplain_chromadb` resolve as expected. |
| `app_synthetic/synthetic_app.py` | Synthetic patient one-click workflow (also reachable via `streamlit_app.py` navigation). |

---

## RAG indexing

### Standalone app (`mediexplain_rag_app.py`)

1. Place article `.html` files under **`html/`** at the **repository root** (same folder as `streamlit_app.py`).
2. Run `mediexplain_rag_app.py`; Chroma persists under **`./mediexplain_chromadb`** by default.

### Package pipeline (`app/rag/`)

`app/rag/config.py` sets data paths relative to the **`app/`** directory — use **`app/html/`** for sources and **`app/mediexplain_chromadb/`** for the persistent Chroma store when running ingest/retriever from that module layout.

---

## Results / output screenshots

Add screenshots of the Streamlit UI, sample explanations, or PDF outputs under:

**`docs/screenshots/`**

Suggested filenames (optional):

- `synthetic-workflow.png` — synthetic patient pipeline  
- `chatbot.png` — MediExplain chat  
- `rag-query.png` — retrieval + answer  

No images are committed by default; the folder is reserved for your captures.

---

## Tech stack

**Python · Streamlit · OpenAI · Google Generative AI · ChromaDB · Hugging Face · NumPy · pandas · Beautiful Soup · lxml · PyPDF · DuckDB · SQLite · Dev Containers · GitHub Codespaces**

| Layer | Technology |
|-------|------------|
| **UI** | Streamlit (multi-page app) |
| **LLM / embeddings** | OpenAI API (`openai`), models as configured in code; optional Google Generative AI (`google-generativeai`) |
| **Vector store** | ChromaDB + OpenAI embedding function |
| **NLP / ML utilities** | `sentence-transformers`, NumPy, pandas |
| **Parsing** | BeautifulSoup, lxml, pypdf |
| **Optional DB** | DuckDB |
| **SQLite (Chroma)** | `pysqlite3-binary` on Linux/macOS only; stdlib sqlite on Windows (see requirements) |
| **Container** | VS Code Dev Container / GitHub Codespaces (`.devcontainer/`) |

---

## Project structure

Logical layout of this repository:

```text
mediexplain/                    # repository root (clone target)
├── README.md                   # This file
├── LICENSE
├── requirements.txt
├── .gitignore
├── .github/
├── .devcontainer/
├── .streamlit/                 # Optional config.toml / secrets.toml (secrets not committed)
├── streamlit_app.py            # Primary multi-page Streamlit entry
├── mediexplain_rag_app.py
├── download_pdf.py
├── app/
├── core/
├── app_synthetic/
├── tools/
├── docs/
│   └── screenshots/            # Add your UI captures here
├── html/                       # (You provide) PMC HTML for standalone RAG — see RAG indexing
└── mediexplain_chromadb/       # (Generated) Chroma store for standalone app — default path
```

**Design idea:** `core/` holds reusable LLM steps for the synthetic record; `app/bots/` holds user-facing tools for chat and medication search; `app/rag/` centralizes paths and retrieval for the packaged RAG pipeline under `app/`.

---

## Requirements / environment

- **Dependency file:** `requirements.txt`
- **Virtual environment:** Recommended; see **How to run**.
- **Reproducibility:** For papers or demos, optionally `pip freeze > requirements-lock.txt` (not committed by default).

---

## Business impact & takeaway

- **Patient experience:** Clearer explanations can improve engagement and shared decision-making — when delivered alongside clinicians and appropriate guardrails.
- **Research & teaching:** The modular “many small assistants” pattern makes it easier to swap, test, and reason about each step than a single monolithic prompt.
- **Risk awareness:** Any healthcare LLM demo must foreground disclaimers, consent, and human oversight; this repo includes consent UI in select apps and “not medical advice” messaging in explainer-oriented paths.

**Bottom line:** MediExplain is a structured playground for RAG + modular assistants in a medical communication context — **not** a certified medical device.

---

## License

See [LICENSE](LICENSE) (Apache-2.0).
