# MediExplain

**MediExplain** is a Python / [Streamlit](https://streamlit.io/) project that helps turn dense medical text into clearer explanations. It includes a multi-page demo app (synthetic patient records, chat-style Q&A, validation tools), optional RAG over biomedical HTML, and a smaller “router” UI for pasted clinical text.

> **Disclaimer:** This software is for education and demonstration only. It is **not** a medical device and **not** a substitute for professional medical advice, diagnosis, or treatment.

## Features

- **Multi-page Streamlit app** (`streamlit_app.py`): synthetic patient workflow, chatbot, and validator console.
- **Core LLM pipeline** (`core/`): modular “bots” for demographics, labs, imaging, medications, billing, safety labeling, PDF export, and more.
- **RAG prototype** (`mediexplain_rag_app.py`): ChromaDB + embeddings over local HTML (e.g. PMC-style documents); requires OpenAI and prepared data under `html/`.
- **Dev Container** (`.devcontainer/`): ready-to-run environment in GitHub Codespaces or VS Code Dev Containers.

## Requirements

- Python **3.11** (matches the dev container image).
- An **OpenAI API key** for most LLM features (`OPENAI_API_KEY`).
- Additional keys only if you use Google Generative AI paths in your deployment (`google-generativeai` is listed in `requirements.txt`).

## Quick start (local)

```powershell
cd path\to\mediexplain
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Streamlit secrets

Create `.streamlit/secrets.toml` (do **not** commit this file) with at least:

```toml
OPENAI_API_KEY = "sk-..."
```

Alternatively, set `OPENAI_API_KEY` in your environment before `streamlit run`.

### Run the main multi-page app

```powershell
streamlit run streamlit_app.py
```

### Other entry points

| File | Purpose |
|------|--------|
| `app/main_app.py` | Simpler single-page “paste text → explain” flow with consent gate |
| `mediexplain_rag_app.py` | RAG indexing and chat over `html/` + ChromaDB |
| `app_synthetic/synthetic_app.py` | Synthetic patient report generator (also reachable from `streamlit_app.py`) |

## Repository layout

```
mediexplain/
├── app/                 # Main app package (router, RAG helpers, bots, safety)
├── app_synthetic/       # Synthetic data Streamlit apps + validator
├── core/                # LLM-backed generators and orchestration
├── tools/               # Small utilities (vector store checks, etc.)
├── streamlit_app.py     # Recommended entry: navigation + pages
├── mediexplain_rag_app.py
├── requirements.txt
└── LICENSE              # Apache-2.0
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Contributing

Issues and pull requests are welcome. Please keep API keys and local data paths out of git.
