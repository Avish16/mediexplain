# MediExplain

[![Live Demo](https://img.shields.io/badge/Live%20Demo-mediexplain.streamlit.app-00D4FF?style=for-the-badge)](https://mediexplain.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai)](https://openai.com)

> AI-powered medical text explainer with RAG retrieval, 19-bot synthetic patient pipeline, and clinical-grade PDF export.

## Features
- **Medical Chatbot** — explains medical reports in plain language (Patient & Caregiver modes)
- **Synthetic Patient Generator** — 19-bot pipeline generating full clinical records (labs, notes, prescriptions, billing)
- **Validator Console** — pipeline introspection, routing, and safety checks
- **RAG Retrieval** — ChromaDB + OpenAI embeddings over medical literature

## Tech Stack
Python · Streamlit · OpenAI · ChromaDB · ReportLab · BeautifulSoup · DuckDB

## Live Demo
👉 https://mediexplain.streamlit.app

## Local Setup
```bash
git clone https://github.com/Avish16/mediexplain
cd mediexplain
python -m venv .venv
.venv\Scripts\activate
pip install -r mediexplain/requirements.txt
# Add OPENAI_API_KEY to mediexplain/.streamlit/secrets.toml
streamlit run mediexplain/streamlit_app.py
```

## Disclaimer
Research and educational use only. Not a substitute for professional medical advice.
