# MediExplain — RAG & modular AI assistants

MediExplain is a **research and education prototype** that turns dense medical text into **plain-language explanations**, optionally **grounds answers in retrieved literature** (RAG), and can **generate synthetic patient-style records** using a pipeline of small LLM “bots.”

**Repository layout:** The **only project document at the repository root is this `README.md`.** All application code, dependencies, and assets live under **`mediexplain/`** (alongside dotfiles such as `.gitignore`, `.github/`, and `.devcontainer/` for tooling).

If you prefer a longer public name (similar to other forks), you can **rename this repository** on GitHub under **Settings → General** (for example **MediExplain-RAG-modular-AI-assistants**). Your clone folder will then use that new name instead of `mediexplain`.

---

## Problem statement

Patients and caregivers often struggle to interpret clinical language, lab values, and discharge instructions. Clinicians and researchers also need **safe, transparent** ways to experiment with **retrieval-augmented generation** and **modular assistants** without pretending the system is a substitute for licensed care.

This project addresses that gap by:

- Explaining medical text in **patient- or caregiver-appropriate** language (with explicit disclaimers).
- Using **Chroma** + embeddings over **PMC-style HTML** articles for literature-grounded retrieval.
- Providing a **multi-step synthetic workflow** (demographics → labs → notes → consolidation → safety/consistency checks → PDF) for demos and teaching.

**Important:** Outputs are for **understanding and research only**, not diagnosis or treatment decisions.

---

## Dataset / source information

| Source | Role |
|--------|------|
| **`mediexplain/html/`** (local) | Expected location for **PubMed Central (PMC)** or similar **HTML** article exports used to build the Chroma index. Create this folder and add files before running ingestion (see [How to run](#how-to-run)). |
| **OpenAI API** | Chat completions and `text-embedding-3-small` (and related models as configured) for explanations and embeddings. |
| **Google Generative AI** | Listed in `mediexplain/requirements.txt` for optional integrations (see code paths that import `google.generativeai`). |
| **Medication RAG** | Separate indexing/search under `mediexplain/app/bots/meds_rag_*` using project-specific knowledge stores (see those modules for paths and build steps). |

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
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r mediexplain/requirements.txt
```

On **Windows**, `pysqlite3-binary` is omitted automatically (no wheels); Chroma uses **stdlib** SQLite. On Linux/macOS, the optional package may be installed for Chroma compatibility.

### Secrets (Streamlit)

Set **`OPENAI_API_KEY`** via environment variable or **Streamlit secrets**. From the repo root, a typical local path is **`mediexplain/.streamlit/secrets.toml`**.

Example `mediexplain/.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
```

### Main Streamlit app (multi-page)

Recommended entry point (also used by the dev container). Run from the **repository root**:

```bash
streamlit run mediexplain/streamlit_app.py
```

Alternatively:

```bash
cd mediexplain
streamlit run streamlit_app.py
```

Pages include **Synthetic App**, **MediExplain Chatbot**, and **Validator Console**.

### Other entry points

| Script | Purpose |
|--------|---------|
| `mediexplain/app/main_app.py` | Minimal **consent-gated** UI with a simple **router** (explainer vs labs). Example: `streamlit run mediexplain/app/main_app.py` |
| `mediexplain/mediexplain_rag_app.py` | Standalone **PMC HTML → Chroma** RAG demo (ingest + query in one file). |
| `mediexplain/app_synthetic/synthetic_app.py` | Synthetic patient **one-click workflow** (invoked via `streamlit_app.py` navigation). |

### RAG index (when using `app/rag/`)

1. Place article **`.html`** files under **`mediexplain/html/`**.
2. Run your ingest path (for example functions in `mediexplain/app/rag/ingest.py` or the flow in `mediexplain/mediexplain_rag_app.py`) to build **`mediexplain/mediexplain_chromadb/`** (persistent Chroma).

> **Note:** `mediexplain/app/rag/config.py` resolves **`BASE_DIR`** to the **`mediexplain/`** package folder so `html/` and `mediexplain_chromadb/` sit next to `app/`, `core/`, etc.

---

## Results / output screenshots

Add screenshots of the Streamlit UI, sample explanations, or PDF outputs under:

**`mediexplain/docs/screenshots/`**

Suggested filenames (optional):

- `synthetic-workflow.png` — synthetic patient pipeline  
- `chatbot.png` — MediExplain chat  
- `rag-query.png` — retrieval + answer  

_(No images are committed by default; the folder is reserved for your captures.)_

---

## Tech stack

Python · Streamlit · OpenAI · Google Generative AI · ChromaDB · Hugging Face · NumPy · pandas · Beautiful Soup · lxml · PyPDF · DuckDB · SQLite · Dev Containers · GitHub Codespaces

| Layer | Technology |
|-------|------------|
| **UI** | **Streamlit** (multi-page app) |
| **LLM / embeddings** | **OpenAI** API (`openai`), configured models in code / config; optional **Google Generative AI** (`google-generativeai`) |
| **Vector store** | **ChromaDB** + OpenAI embedding function |
| **NLP / ML utilities** | **sentence-transformers**, **numpy**, **pandas** |
| **Parsing** | **BeautifulSoup**, **lxml**, **pypdf** |
| **Optional DB** | **DuckDB** |
| **SQLite (Chroma)** | **`pysqlite3-binary`** on Linux/macOS where applicable; **stdlib sqlite** on Windows (see `mediexplain/requirements.txt`) |
| **Container** | **VS Code Dev Container** / **GitHub Codespaces** (`.devcontainer/`) |

---

## Project structure

Logical layout of the repository:

```
mediexplain/                         # clone root (GitHub repo name may differ if you rename it)
├── README.md                        # This file (only top-level project doc)
├── .gitignore
├── .github/
├── .devcontainer/
└── mediexplain/                     # All application code and assets
    ├── LICENSE
    ├── requirements.txt
    ├── .streamlit/                  # Optional config.toml / secrets.toml (secrets not committed)
    ├── streamlit_app.py           # Primary multi-page Streamlit entry
    ├── mediexplain_rag_app.py
    ├── download_pdf.py
    ├── app/
    ├── core/
    ├── app_synthetic/
    ├── tools/
    ├── docs/
    │   └── screenshots/
    ├── html/                       # (You provide) PMC HTML for RAG indexing
    └── mediexplain_chromadb/       # (Generated) Chroma persistent store
```

**Design idea:** `core/` holds **reusable LLM steps** for the synthetic record; `app/bots/` holds **user-facing** tools for chat and medication search; `app/rag/` centralizes **paths and retrieval** so data lives under **`mediexplain/`**.

---

## Requirements / environment

- **Dependency file:** `mediexplain/requirements.txt`
- **Virtual environment:** Recommended; see [How to run](#how-to-run).
- **Reproducibility:** For papers or demos, export exact versions with `pip freeze > requirements-lock.txt` (optional; not committed by default).

---

## Business impact & takeaway

- **Patient experience:** Clearer explanations can improve engagement and shared decision-making—when delivered **alongside** clinicians and appropriate guardrails.
- **Research & teaching:** The modular “many small assistants” pattern makes it easier to **swap**, **test**, and **reason about** each step than a single monolithic prompt.
- **Risk awareness:** Any healthcare LLM demo must foreground **disclaimers**, **consent**, and **human oversight**; this repo includes consent UI in select apps and repeated “not medical advice” messaging in explainer code paths.

**Bottom line:** MediExplain is a **structured playground** for **RAG + modular assistants** in a medical communication context—not a certified medical device.

---

## License

See [`mediexplain/LICENSE`](mediexplain/LICENSE).
