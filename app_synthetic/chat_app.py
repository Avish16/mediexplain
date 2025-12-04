# =========================================================
# 0. SQLITE FIX FOR CHROMA
# =========================================================
try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

# =========================================================
# IMPORTS
# =========================================================
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

# Make bots importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# BOT IMPORTS
from app.bots.explainer_bot import run_explainer
from app.bots.labs_bot import run_labs, explain_labs
from app.bots.meds_bot import run_meds, explain_medications
from app.bots.careplan_bot import run_careplan, generate_care_plan
from app.bots.snapshot_bot import run_snapshot, generate_snapshot
from app.bots.support_bot import run_support, generate_support_message
from app.bots.prescription_bot import explain_prescriptions, run_prescriptions


# =========================================================
# 1. CONFIG & OPENAI CLIENT
# =========================================================
st.set_page_config(page_title="MediExplain Chatbot", layout="wide")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# =========================================================
# 2. MEMORY MANAGER (FINAL – FIXED)
# =========================================================
class ChromaMemoryManager:
    def __init__(self):
        # Use in-memory Chroma → avoids all tenant/database errors
        self.client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
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

    def retrieve_memory(self, user_id: str, query: str, k: int = 5):
        try:
            result = self.collection.query(
                query_texts=[query],
                n_results=k,
                where={"user_id": user_id},
            )
            docs = result.get("documents", [[]])[0]
            return docs
        except:
            return []

memory = ChromaMemoryManager()

# =========================================================
# 3. SESSION STATE INIT
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "file_id" not in st.session_state:
    st.session_state.file_id = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None


# =========================================================
# 4. LOGIN
# =========================================================
st.sidebar.title("Login")

if st.session_state.user_id is None:
    login_id = st.sidebar.text_input("Enter your email or patient ID")

    if st.sidebar.button("Continue") and login_id.strip():
        st.session_state.user_id = login_id.strip()
        st.rerun()
else:
    st.sidebar.success(f"Logged in as: {st.session_state.user_id}")

    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.session_state.messages = []
        st.session_state.pdf_text = ""
        st.session_state.file_id = None
        st.rerun()

if st.session_state.user_id is None:
    st.title("🩺 MediExplain – Your Medical Report Companion")
    st.info("Please log in to continue.")
    st.stop()

user_id = st.session_state.user_id


# =========================================================
# 5. HEADER + MODE
# =========================================================
st.title("🩺 MediExplain – Your Medical Report Companion")

mode = st.radio(
    "Choose explanation mode:",
    ["Patient Mode (Simple & Friendly)", "Caregiver Mode (Technical & Clinical)"],
)

# =========================================================
# 6. PDF UPLOAD + VECTOR STORE REGISTER (FINAL VERSION)
# =========================================================

uploaded_pdf = st.file_uploader("Upload your medical report (PDF)", type=["pdf"])

if uploaded_pdf is not None:

    # Extract text
    reader = PdfReader(uploaded_pdf)
    extracted = ""
    for page in reader.pages:
        try:
            extracted += (page.extract_text() or "") + "\n"
        except:
            pass

    st.session_state.pdf_text = extracted.strip()

    # Create vector store (once)
    vector_store = client.vector_stores.create(name="mediexplain_vs")
    st.session_state.vector_store_id = vector_store.id

    # Upload PDF into vector store
    client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=[uploaded_pdf]
    )

    st.success("PDF indexed into vector store for file_search!")

    # ---- Show extracted report ----
    with st.expander("View extracted report text"):
        if st.session_state.pdf_text:
            st.write(st.session_state.pdf_text)
        else:
            st.write("_No text could be extracted from this PDF._")
# =========================================================
# 7. FILE SEARCH HELPER (FIXED)
# =========================================================
def search_pdf_context(query: str) -> str:
    vector_store_id = st.session_state.get("vector_store_id")
    if not vector_store_id:
        return ""  # No PDF uploaded

    prompt = f"""
Search the uploaded medical report for text relevant to:
\"{query}\"

Return a fused summary (2–4 paragraphs) ONLY using info from the PDF.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],  # 👈 TOP-LEVEL FIELD
                # You can add more options later if the SDK supports them
            }
        ],
        max_output_tokens=800,
    )

    return response.output_text or ""

# =========================================================
# 8. MEMORY SNIPPET EXTRACTOR (FINAL VERSION)
# =========================================================

def extract_memory_snippet(user_input: str, assistant_reply: str) -> str:
    """
    Ask the model to extract only long-term clinically meaningful facts.
    Returns an empty string if nothing should be stored.
    """
    prompt = f"""
From the conversation below, extract ONLY long-term clinically meaningful details
that should be saved in the user's memory profile.

Examples of valid memory items:
- Diagnoses, chronic conditions
- Medication allergies or long-term prescriptions
- Baseline vitals, lab abnormalities
- Critical medical history
- Patient preferences (e.g., 'prefers simple explanations')

If nothing is appropriate, return an empty string.

USER: {user_input}
ASSISTANT: {assistant_reply}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return resp.choices[0].message.content.strip()

# =========================================================
# 9. ROUTER (FINAL VERSION)
# =========================================================

def route_to_specialist_bot(mode, question, pdf_text, long_term_memory):
    """
    Determines which internal specialist bot to call.
    Returns JSON: {"bot": "...", "reason": "..."}
    """

    system_prompt = """
You are MediExplain's routing agent.

Your job: choose ONE best bot for answering the user's question.
Return STRICT JSON: {"bot": "...", "reason": "..."}

Valid bots:
- EXPLAINER
- LABS
- MEDS
- CAREPLAN
- SNAPSHOT
- SUPPORT
- PRESCRIPTIONS
"""

    user_payload = f"""
MODE: {mode}
QUESTION: {question}

REPORT TEXT (first 3000 chars):
{pdf_text[:3000]}

USER MEMORY:
{long_term_memory}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ]
    ).choices[0].message.content

    # Parse JSON safely
    try:
        clean = resp.replace("```json", "").replace("```", "").strip()
        bot_name = json.loads(clean).get("bot", "EXPLAINER").upper()
    except:
        bot_name = "EXPLAINER"

    if bot_name not in {
        "EXPLAINER", "LABS", "MEDS", "CAREPLAN",
        "SNAPSHOT", "SUPPORT", "PRESCRIPTIONS"
    }:
        bot_name = "EXPLAINER"

    return bot_name
# =========================================================
# 10. ORCHESTRATOR (FINAL VERSION)
# =========================================================

def generate_orchestrated_response(user_input: str, mode: str) -> str:
    """
    1. Retrieve long-term memory
    2. Route to the correct specialist bot
    3. Pull contextual evidence via vector-store-search
    4. Call the bot
    5. If bot fails → safe fallback answer
    """

    pdf_text = st.session_state.pdf_text or ""
    long_term_memory = memory.retrieve_memory(user_id, user_input, k=5)

    # Determine which bot to use
    chosen_bot = route_to_specialist_bot(
        mode, user_input, pdf_text, long_term_memory
    )

    # Retrieve evidence from the PDF using vector store
    pdf_context = search_pdf_context(user_input)

    try:
        # Route to correct bot
        if chosen_bot == "LABS":
            reply = run_labs(user_input, mode, pdf_context, long_term_memory)

        elif chosen_bot == "MEDS":
            reply = run_meds(user_input, mode, pdf_context, long_term_memory)

        elif chosen_bot == "CAREPLAN":
            reply = run_careplan(user_input, mode, pdf_context, long_term_memory)

        elif chosen_bot == "SNAPSHOT":
            reply = run_snapshot(user_input, mode, pdf_context, long_term_memory)

        elif chosen_bot == "SUPPORT":
            reply = run_support(user_input, mode, pdf_context, long_term_memory)

        elif chosen_bot == "PRESCRIPTIONS":
            reply = run_prescriptions(user_input, mode, pdf_context, long_term_memory)

        else:
            # Default EXPLAINER bot
            reply = run_explainer(
                mode=mode,
                report_text=pdf_context or pdf_text,
                user_question=user_input,
            )

        return reply + f"\n\n---\n_Answered by: **{chosen_bot} bot**_"

    except Exception as e:
        traceback.print_exc()

        # Safe fallback answer
        fallback_prompt = f"""
A specialist bot failed. Give a safe, simple explanation.

QUESTION:
{user_input}

CONTEXT FROM REPORT:
{pdf_context}
"""

        fallback = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": fallback_prompt}],
            temperature=0.3
        )

        return fallback.choices[0].message.content.strip()


# =========================================================
# 11. CHAT UI
# =========================================================
st.write("Conversation")

for msg in st.session_state.messages:
    role = msg["role"]
    st.chat_message(role).markdown(msg["content"])

user_input = st.chat_input(
    "Ask a question about your medical report, labs, medications, or care plan..."
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        assistant_reply = generate_orchestrated_response(user_input, mode)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
    st.chat_message("assistant").markdown(assistant_reply)

    # memory write
    snippet = extract_memory_snippet(user_input, assistant_reply)
    if snippet:
        memory.add_memory(user_id, snippet)


# =========================================================
# 12. CLEAR BUTTON
# =========================================================
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

