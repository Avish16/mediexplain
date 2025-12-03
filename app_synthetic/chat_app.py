try:
    # Fix for Chroma + pysqlite3 in Streamlit / Codespaces
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from typing import List
import json
import os
import sys
import traceback

# ---------------------------------------------------------
# Ensure we can import app.bots.*
# ---------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.bots.explainer_bot import run_explainer
from app.bots.labs_bot import run_labs
from app.bots.meds_bot import run_meds
from app.bots.careplan_bot import run_careplan
from app.bots.snapshot_bot import run_snapshot
from app.bots.support_bot import run_support

# =========================================================
# 1. CONFIG & CLIENT
# =========================================================
st.set_page_config(page_title="MediExplain Chatbot", layout="wide")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =========================================================
# 2. CHROMA-BASED LONG-TERM MEMORY
# =========================================================
class ChromaMemoryManager:
    """
    Long-term memory using ChromaDB.

    Each memory is stored as a document with:
    - text: the memory snippet
    - metadata: includes user_id (so multiple users are isolated)
    """

    def __init__(self, path: str = "mediexplain_chroma"):
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection("mediexplain_memory")

    def add_memory(self, user_id: str, text: str):
        text = text.strip()
        if not text:
            return

        doc_id = f"{user_id}_{abs(hash(text))}"
        self.collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[{"user_id": user_id}],
        )

    def retrieve_memory(self, user_id: str, query: str, k: int = 5) -> List[str]:
        try:
            result = self.collection.query(
                query_texts=[query],
                n_results=k,
                where={"user_id": user_id},
            )
            docs = result.get("documents", [[]])[0]
            return docs
        except Exception:
            return []


memory = ChromaMemoryManager()

# =========================================================
# 3. SESSION STATE INIT
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []  # conversation history

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""  # extracted report text

if "user_id" not in st.session_state:
    st.session_state.user_id = None  # login identity (email / ID)

# =========================================================
# 4. LOGIN / USER IDENTIFICATION
# =========================================================
st.sidebar.title("Login")

if st.session_state.user_id is None:
    login_id = st.sidebar.text_input(
        "Enter your email or patient ID",
        placeholder="e.g. john.doe@example.com",
    )
    if st.sidebar.button("Continue") and login_id.strip():
        st.session_state.user_id = login_id.strip()
        st.rerun()
else:
    st.sidebar.success(f"Logged in as: {st.session_state.user_id}")
    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.session_state.messages = []
        st.session_state.pdf_text = ""
        st.rerun()

# If not logged in, stop here
if st.session_state.user_id is None:
    st.title("🩺 MediExplain – Your Medical Report Companion")
    st.info("Please log in using your email or patient ID in the sidebar to continue.")
    st.stop()

user_id = st.session_state.user_id

# =========================================================
# 5. HEADER & MODE SWITCH
# =========================================================
st.title("🩺 MediExplain – Your Medical Report Companion")

mode = st.radio(
    "Choose AI Explanation Mode:",
    [
        "Patient Mode (Simple & Friendly)",
        "Caregiver Mode (Technical & Clinical)",
    ],
)

st.markdown(
    """
> **Important:** MediExplain does **not** replace a doctor.  
> It helps you understand your reports but cannot diagnose or provide medical treatment.
"""
)

# =========================================================
# 6. PDF UPLOAD & EXTRACTION
# =========================================================
uploaded_pdf = st.file_uploader("Upload your medical report (PDF)", type=["pdf"])

if uploaded_pdf is not None:
    reader = PdfReader(uploaded_pdf)
    extracted = ""
    for page in reader.pages:
        try:
            extracted += (page.extract_text() or "") + "\n"
        except Exception:
            pass

    st.session_state.pdf_text = extracted.strip()
    st.success("PDF uploaded and processed successfully!")

    with st.expander("View extracted report text"):
        if st.session_state.pdf_text:
            st.write(st.session_state.pdf_text)
        else:
            st.write("_No text could be extracted from this PDF._")

# ---------------------------------------------------------
# 7. MEMORY SNIPPET EXTRACTOR
# ---------------------------------------------------------
def extract_memory_snippet(user_input: str, assistant_reply: str) -> str:
    """
    Ask the model to summarize long-term relevant info from this exchange.
    """
    prompt = f"""
From the following conversation between a user and MediExplain,
extract ONLY clinically or personally meaningful long-term facts that
should be stored in this user's profile.

Examples of useful memory items:
- Long-term conditions or diagnoses
- Ongoing symptoms or key clinical findings
- Important lab or imaging results
- Medications, allergies, or treatments
- Patient or caregiver preferences (e.g., prefers simple explanations)

If nothing is worth storing, reply with an empty line.

User said:
{user_input}

Assistant replied:
{assistant_reply}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    return text

# ---------------------------------------------------------
# 8. ROUTER: DECIDE WHICH BOT TO CALL
# ---------------------------------------------------------
def route_to_specialist_bot(
    mode: str,
    question: str,
    pdf_text: str,
    long_term_memory: List[str],
) -> str:
    """
    Uses a small GPT call to choose which specialist bot to call.

    Available bots:
    - EXPLAINER  : general explanation of report / conditions
    - LABS       : questions about lab values, blood tests, panels
    - MEDS       : questions about medications, doses, side-effects
    - CAREPLAN   : follow-up care, monitoring, lifestyle, red-flags
    - SNAPSHOT   : high-level summary of the whole case
    - SUPPORT    : emotional support, coping, communication tips
    """

    ltm_str = "\n- ".join(long_term_memory) if long_term_memory else "None"

    router_system = """
You are a routing agent for MediExplain.

Your job:
Given the user question and context, decide which internal specialist
bot is the **single best fit** to answer.

Return STRICT JSON only, no commentary, using this schema:

{
  "bot": "EXPLAINER | LABS | MEDS | CAREPLAN | SNAPSHOT | SUPPORT",
  "reason": "short explanation of why this bot is appropriate"
}

Guidelines:
- Use LABS for questions about blood work, lab values, reference ranges,
  abnormal / high / low results, or interpretation of specific tests
  like CBC, CMP, lipid panel, HbA1c, troponin, etc.
- Use MEDS for medication names, doses, timing, interactions, side effects,
  what a drug is for, how long to take it, etc.
- Use CAREPLAN for follow-up schedule, monitoring, lifestyle changes,
  red-flag symptoms, rehabilitation, diet / exercise advice (linked to the
  medical condition), and care-coordination questions.
- Use SNAPSHOT when the user asks for an overall summary of the case,
  “what is going on with me?”, “big picture”, or wants a step-by-step
  story of the illness and treatment.
- Use SUPPORT when the main focus is emotions, anxiety, fear,
  talking to family, how to ask the doctor questions, or general reassurance.
- In all other cases (or if you are unsure), use EXPLAINER.
"""

    user_payload = f"""
MODE: {mode}

USER QUESTION:
{question}

UPLOADED REPORT (if any):
{pdf_text[:4000]}

LONG-TERM MEMORY SNIPPETS:
- {ltm_str}
"""

    router_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": router_system},
            {"role": "user", "content": user_payload},
        ],
    )

    raw = router_resp.choices[0].message.content.strip()

    # Try to parse JSON safely
    try:
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        bot = data.get("bot", "EXPLAINER").upper()
    except Exception:
        bot = "EXPLAINER"

    if bot not in {"EXPLAINER", "LABS", "MEDS", "CAREPLAN", "SNAPSHOT", "SUPPORT"}:
        bot = "EXPLAINER"

    return bot

# ---------------------------------------------------------
# 9. GENERAL ORCHESTRATOR
# ---------------------------------------------------------
def generate_orchestrated_response(user_input: str, mode: str) -> str:
    """
    1. Pull long-term memory for this user.
    2. Ask router which specialist bot to use.
    3. Call the corresponding local bot function.
    """

    pdf_text = st.session_state.pdf_text or ""
    ltm_snippets = memory.retrieve_memory(user_id, user_input, k=5)

    chosen_bot = route_to_specialist_bot(mode, user_input, pdf_text, ltm_snippets)

    try:
        if chosen_bot == "LABS":
            reply = run_labs(user_input, mode, pdf_text, ltm_snippets)
        elif chosen_bot == "MEDS":
            reply = run_meds(user_input, mode, pdf_text, ltm_snippets)
        elif chosen_bot == "CAREPLAN":
            reply = run_careplan(user_input, mode, pdf_text, ltm_snippets)
        elif chosen_bot == "SNAPSHOT":
            reply = run_snapshot(user_input, mode, pdf_text, ltm_snippets)
        elif chosen_bot == "SUPPORT":
            reply = run_support(user_input, mode, pdf_text, ltm_snippets)
        else:  # EXPLAINER or fallback
            reply = run_explainer(user_input, mode, pdf_text, ltm_snippets)

        # Add a tiny footer so you can debug routing if needed (optional)
        reply += f"\n\n---\n_(Answered by: {chosen_bot} bot)_"

        return reply

    except Exception as e:
        # If specialist bot crashes, fall back to a simple explainer using GPT directly
        traceback.print_exc()
        fallback_prompt = f"""
You are MediExplain. The specialist pipeline failed, so you must answer
directly.

MODE: {mode}

User question:
{user_input}

Report (if any):
{pdf_text}

Answer in a way that matches the MODE, and remind the user that this
does not replace a doctor's advice.
"""
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": fallback_prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

# =========================================================
# 10. CHAT HISTORY UI
# =========================================================
st.write("Conversation")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# =========================================================
# 11. CHAT INPUT & FLOW
# =========================================================
user_input = st.chat_input(
    "Ask a question about your medical report, labs, medications, or care plan..."
)

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("MediExplain is thinking..."):
        assistant_reply = generate_orchestrated_response(user_input, mode)

    # Add assistant reply to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )

    # Display reply
    st.chat_message("assistant").markdown(assistant_reply)

    # Try to store a long-term memory snippet
    try:
        memory_snippet = extract_memory_snippet(user_input, assistant_reply)
        if memory_snippet:
            memory.add_memory(user_id, memory_snippet)
    except Exception:
        pass

# =========================================================
# 12. CLEAR CONVERSATION BUTTON
# =========================================================
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
