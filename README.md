# 🏥 MediExplain

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-mediexplain.streamlit.app-00D4FF?style=for-the-badge)](https://mediexplain.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=for-the-badge)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](mediexplain/LICENSE)

> **AI-powered medical intelligence platform** — turns dense clinical text into plain-language explanations, generates clinically realistic synthetic patient records via a 19-bot pipeline, and exports publication-ready PDFs. Built for research, education, and healthcare AI demos.

---

## 🌐 Live Demo

**👉 [mediexplain.streamlit.app](https://mediexplain.streamlit.app)**

> Try generating a full synthetic patient case — demographics, labs, clinical notes, prescriptions, billing, and safety labels — all in one click.

---

## ✨ Features

### 🤖 Medical Chatbot
- Upload a medical report PDF and ask questions in plain English
- Two modes: **Patient Mode** (simple & friendly) and **Caregiver Mode** (technical & clinical)
- RAG-enhanced answers grounded in retrieved medical literature
- Web search toggle for real-time information retrieval
- Consent-gated session with login flow

### 🧬 Synthetic Patient Generator
- **19-bot modular pipeline** — each bot is a specialized AI agent
- Generates complete clinical records end-to-end:
  - Demographics → Diagnosis → Labs → Vitals → Radiology
  - Clinical Notes → Nursing Notes → Medications → Prescriptions
  - Procedures → Pathology → Billing → Safety Labels → Consistency Check
- Exports a **30+ page clinical PDF** with ICD-10 codes, CPT codes, and HCC risk scores
- Graceful error recovery — pipeline never aborts on a single bot failure

### 🔍 Validator Console
- Pipeline introspection dashboard
- Retrieval metrics, routing trace, and safety flags
- Bot output inspection and Q&A history
- Terminal-style UI with color-coded status indicators

---

## 🏗️ Architecture
```
User Input
│
▼
Router Bot ──→ Intent Detection
│
├──→ RAG Retrieval (ChromaDB + text-embedding-3-small)
│
├──→ 19 Specialized Bots (modular, swappable)
│         Demographics · Diagnosis · Labs · Vitals · Radiology
│         Clinical Notes · Nursing · Medications · Prescriptions
│         Procedures · Pathology · Billing · Safety · Consistency
│
▼
Consolidator Bot → Safety Check → PDF Generator
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (multi-page) |
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (persistent) |
| PDF Generation | ReportLab |
| Parsing | BeautifulSoup, lxml, pypdf |
| Data | pandas, numpy, DuckDB |
| Deployment | Streamlit Cloud |

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Avish16/mediexplain
cd mediexplain

# 2. Create virtual environment
python -m venv .venv
.venv\Scriptsctivate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r mediexplain/requirements.txt

# 4. Add your OpenAI API key
# Create file: mediexplain/.streamlit/secrets.toml
# Add line: OPENAI_API_KEY = "sk-..."

# 5. Run the app
streamlit run mediexplain/streamlit_app.py
```

App runs at **http://localhost:8501**

---

## 📁 Project Structure
```
mediexplain/
├── streamlit_app.py          # Main entry point (multi-page)
├── home.py                   # Home page
├── requirements.txt
├── .streamlit/               # Config + secrets (secrets not committed)
├── app/
│   ├── bots/                 # User-facing chat + RAG bots
│   └── rag/                  # Ingest, config, retrieval
├── core/                     # 19 synthetic record bots
├── app_synthetic/
│   ├── synthetic_app.py      # Synthetic generator UI
│   └── validator/            # Validator console
├── tools/
├── html/                     # (User provides) PMC articles for RAG
└── docs/screenshots/         # App screenshots
```

---

## 📊 Sample Output

The Synthetic Patient Generator produces a complete 30+ page clinical record including:

- **Patient Demographics** — name, insurance, contact, occupation, living situation
- **Primary Diagnosis** — ICD-10, SNOMED-CT, HCC category, MDM complexity
- **Lab Results** — CBC, CMP, Lipid Panel, Coagulation, Cultures, Toxicology
- **Vital Signs** — 10-timepoint 24-hour monitoring with NEWS/MEWS scores
- **Clinical Notes** — SOAP note, H&P, ED note, progress notes, discharge summary
- **Prescriptions** — drug interactions, black box warnings, monitoring plans
- **Billing Summary** — DRG grouping, CPT/HCPCS codes, line items, payer breakdown
- **Safety Labels** — medication risks, abnormal lab flags, vital sign alerts
- **Consistency Report** — cross-section validation with suggested fixes

---

## ⚠️ Disclaimer

MediExplain is a **research and educational prototype**. All outputs are AI-generated and entirely fictional. This platform is **not a substitute for professional medical advice**, diagnosis, or treatment. Never make clinical decisions based on AI-generated content. Always consult a qualified healthcare professional.

---

## 📄 License

MIT License — see [LICENSE](mediexplain/LICENSE)

---

*Built with ❤️ using OpenAI, Streamlit, and ChromaDB*
