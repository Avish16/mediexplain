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
import json
import os
import sys
import traceback

# Make bots importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# BOT IMPORTS
from app.bots.explainer_bot import run_explainer
from app.bots.labs_bot import run_labs
from app.bots.meds_bot import run_meds
from app.bots.careplan_bot import run_careplan
from app.bots.snapshot_bot import run_snapshot
from app.bots.support_bot import run_support
from app.bots.prescription_bot import run_prescriptions
from app.bots.meds_rag_search import search_meds_knowledge


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
        # In-memory Chroma (no disk / tenant issues)
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
        except Exception:
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

if "vector_store_id" not in st.session_state:
    st.session_state.vector_store_id = None

if "user_choice" not in st.session_state:
    st.session_state.user_choice = None

if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False

if "latest_web_refs" not in st.session_state:
    st.session_state.latest_web_refs = None

if "latest_meds_rag_chunks" not in st.session_state:
    st.session_state.latest_meds_rag_chunks = []


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
        st.session_state.vector_store_id = None
        st.rerun()

if st.session_state.user_id is None:
    st.title("🩺 MediExplain – Your Medical Report Companion")
    st.info("Please log in to continue.")
    st.stop()

user_id = st.session_state.user_id


# =========================================================
# 5. HEADER + MODE (PATIENT ONLY)
# =========================================================
st.title("🩺 MediExplain – Your Medical Report Companion")

# For now, keep a simple toggle but treat everything as “patient” persona
# mode = st.radio(
#     "Choose explanation mode:",
#     ["Patient Mode (Simple & Friendly)"],
# )
# # Internally just use the word "patient"
# mode_internal = "patient"

# UI label for user
mode_label = st.radio(
    "Choose explanation mode:",
    ["Patient Mode (Simple & Friendly)", "Caregiver Mode (Technical & Clinical)"],
)

# Internal value used by bots
mode = "caregiver" if "Caregiver" in mode_label else "patient"



# =========================================================
# 6. PDF UPLOAD + VECTOR STORE REGISTER (OPENAI FILE_SEARCH)
# =========================================================
uploaded_pdf = st.file_uploader("Upload your medical report (PDF)", type=["pdf"])

if uploaded_pdf is not None:
    # Extract text for display / fallback
    reader = PdfReader(uploaded_pdf)
    extracted = ""
    for page in reader.pages:
        try:
            extracted += (page.extract_text() or "") + "\n"
        except Exception:
            pass

    st.session_state.pdf_text = extracted.strip()

    # Create vector store (new Responses API)
    vs = client.vector_stores.create(name="mediexplain_vs")
    st.session_state.vector_store_id = vs.id

    # Upload PDF into vector store
    client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vs.id,
        files=[uploaded_pdf],
    )

    st.success("✅ PDF indexed into vector store for file_search!")

    with st.expander("📄 View extracted report text"):
        if st.session_state.pdf_text:
            st.write(st.session_state.pdf_text)
        else:
            st.write("_No text could be extracted from this PDF._")
elif not st.session_state.pdf_text:
    st.info("Please upload a medical report PDF to get the best explanations.")


# =========================================================
# 7. FILE SEARCH HELPER (OPENAI RESPONSES + FILE_SEARCH)
# =========================================================
def search_pdf_context(query: str) -> str:
    """
    Use OpenAI Responses + file_search tool over the vector store
    created from the uploaded report.
    """
    vector_store_id = st.session_state.get("vector_store_id")
    if not vector_store_id:
        return ""

    prompt = f"""
Search the uploaded medical report for content relevant to:
\"{query}\"

Return a fused summary (2–4 paragraphs), ONLY using information from the PDF.
Do not invent new data.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        tools=[
            {
                "type": "file_search",
                # ✅ vector_store_ids MUST be top-level under the tool
                "vector_store_ids": [vector_store_id],
            }
        ],
        max_output_tokens=800,
    )

    return response.output_text or ""


# =========================================================
# 8. MEMORY SNIPPET EXTRACTOR
# =========================================================
def extract_memory_snippet(user_input: str, assistant_reply: str) -> str:
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
        temperature=0,
    )

    return resp.choices[0].message.content.strip()


# =========================================================
# 9. ROUTER (WITH OUT_OF_SCOPE + MEDICATION RULES)
# =========================================================
def route_to_specialist_bot(mode: str, question: str, pdf_text: str, long_term_memory):
    system_prompt = """
You are MediExplain’s routing agent.

Your ONLY job is to choose ONE bot for the user query.
Return STRICT JSON: {"bot": "...", "reason": "..."}

---------------------
### SCOPE RULES
---------------------
MediExplain ONLY answers medical-report-related questions.

If the user asks about ANYTHING outside healthcare or their report (examples):
- politics (“who is the president”),
- celebrities,
- sports,
- cooking,
- gaming,
- geography,
- weather,
- homework help,
- general knowledge,
→ RETURN bot="OUT_OF_SCOPE"

Do NOT attempt to answer. This must be caught BY YOU, the router.

---------------------
### MEDICATION ROUTING RULES
---------------------
If the question mentions ANY of these:
- drug names (ibuprofen, atenolol, furosemide, amiodarone, etc.)
- “side effects”, “interactions”, “risks”
- “why am I taking this”
- “is this safe to take”
- dose questions (“increase? reduce? skip?”)
→ ALWAYS choose bot="MEDS"

If the question references:
- discharge prescriptions
- sig instructions (take twice daily, etc.)
- medication list in the PDF
→ ALWAYS choose bot="PRESCRIPTIONS"

NEVER route medication questions to EXPLAINER.

---------------------
### OTHER ROUTES
---------------------
If about lab values → bot="LABS"
If about vital signs, symptoms → bot="SNAPSHOT"
If about care plan → bot="CAREPLAN"
If emotional distress → bot="SUPPORT"
If general explanation → bot="EXPLAINER"

---------------------
### IMPORTANT:
You MUST choose ONLY from:
["EXPLAINER","LABS","MEDS","CAREPLAN","SNAPSHOT","SUPPORT","PRESCRIPTIONS","OUT_OF_SCOPE"]

Return STRICT JSON. Never write anything else.
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
            {"role": "user", "content": user_payload},
        ],
    ).choices[0].message.content

    try:
        clean = resp.replace("```json", "").replace("```", "").strip()
        bot_name = json.loads(clean).get("bot", "EXPLAINER").upper()
    except Exception:
        bot_name = "EXPLAINER"

    if bot_name not in {
        "EXPLAINER",
        "LABS",
        "MEDS",
        "CAREPLAN",
        "SNAPSHOT",
        "SUPPORT",
        "PRESCRIPTIONS",
        "OUT_OF_SCOPE",
    }:
        bot_name = "EXPLAINER"

    return bot_name


# =========================================================
# 10. SIMPLE WEB SEARCH HELPER (TOGGLE-CONTROLLED)
# =========================================================
def run_websearch(query: str) -> str:
    """
    Optional web-search using OpenAI Responses API.
    If web_search is not enabled on the account, this may error,
    but the function itself is minimal.
    """
    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Use web search to answer:\n{query}",
            tools=[{"type": "web_search"}],
            max_output_tokens=800,
        )
        return resp.output_text or "No web search result text was returned."
    except Exception as e:
        return f"Web search is not available in this environment: {e}"

def get_conversation_history(limit=20):
    """
    Returns the most recent conversation messages in formatted text.
    """

    history = st.session_state.messages[-limit:]
    formatted = ""

    for msg in history:
        role = msg["role"]
        content = msg["content"]
        formatted += f"{role.upper()}: {content}\n"

    return formatted.strip()


# =========================================================
# 11. ORCHESTRATOR (PDF + MEMORY + MEDS RAG + WEBSEARCH)
# =========================================================
def generate_orchestrated_response(user_input: str, mode: str) -> str:
    """
    1. Optional web-search (if toggle ON)
    2. Retrieve long-term memory
    3. Route to correct specialist bot
    4. Pull contextual evidence from PDF (file_search)
    5. For MEDS / PRESCRIPTIONS, also pull medication RAG
    6. Call bot
    7. Fallback if something fails
    """
    conversation_history = get_conversation_history(limit=12)
    # 1) WEB SEARCH (if enabled)
    if st.session_state.get("web_search_enabled", False):
        webresult = run_websearch(user_input)
        st.session_state.latest_web_refs = webresult
        return f"### 🌐 Web Search Result\n\n{webresult}"

    # 2) MEMORY + PDF TEXT
    pdf_text = st.session_state.pdf_text or ""
    long_term_memory = memory.retrieve_memory(user_id, user_input, k=5)

    # 3) ROUTE
    chosen_bot = route_to_specialist_bot(
        mode, user_input, pdf_text, long_term_memory
    )

    if chosen_bot == "OUT_OF_SCOPE":
        return (
            "I'm MediExplain — I can only help with *your medical report*, "
            "your labs, medications, care plan, or clinical explanations.\n\n"
            "This question appears to be outside that scope. "
            "Please ask something related to the provided medical report."
        )

    # 4) PDF CONTEXT
    pdf_context = search_pdf_context(user_input)

    # 5) MEDICATION RAG (only for MEDS / PRESCRIPTIONS)
    meds_rag_text = ""
    if chosen_bot in {"MEDS", "PRESCRIPTIONS"}:
        try:
            rag = search_meds_knowledge(user_input, top_k=5)
            meds_rag_text = rag.get("rag_text", "") or ""
            st.session_state.latest_meds_rag_chunks = rag.get("chunks", [])
        except Exception as e:
            st.session_state.latest_meds_rag_chunks = []
            meds_rag_text = f"(Medication RAG lookup failed: {e})"

    # Combine PDF context + meds RAG if present
    combined_context = pdf_context or pdf_text
    if meds_rag_text:
        combined_context += (
            "\n\n---\n"
            "### Evidence from medication research literature (RAG)\n"
            f"{meds_rag_text}\n"
        )

    # 6) CALL BOT
    try:
        if chosen_bot == "LABS":
            reply = run_labs(
                user_input,
                mode,
                combined_context,
                long_term_memory,
                conversation_history=conversation_history
            )


        elif chosen_bot == "MEDS":
            reply = run_meds(
                user_input,
                mode,
                combined_context,
                long_term_memory,
                conversation_history=conversation_history
            )

        elif chosen_bot == "CAREPLAN":
            reply = run_careplan(
                user_input,
                mode,
                combined_context,
                long_term_memory,
                conversation_history=conversation_history
            )

        elif chosen_bot == "SNAPSHOT":
            reply = run_snapshot(
                user_input,
                mode,
                combined_context,
                long_term_memory,
                conversation_history=conversation_history
            )

        elif chosen_bot == "SUPPORT":
            reply = run_support(
                user_input,
                mode,
                combined_context,
                long_term_memory,
                conversation_history=conversation_history
            )

        elif chosen_bot == "PRESCRIPTIONS":
            reply = run_prescriptions(
                user_input,
                mode,
                combined_context,
                long_term_memory,
                conversation_history=conversation_history
            )

        else:
            # EXPLAINER default
            reply = run_explainer(
                user_input,
                mode,
                combined_context,
                long_term_memory,
                conversation_history=conversation_history
            )

        return reply + f"\n\n---\n_Answered by: **{chosen_bot} bot**_"

    except Exception:
        traceback.print_exc()

        fallback_prompt = f"""
A specialist bot failed. Give a safe, simple explanation.

QUESTION:
{user_input}

CONTEXT FROM REPORT:
{combined_context}
"""

        fallback = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": fallback_prompt}],
            temperature=0.3,
        )

        return fallback.choices[0].message.content.strip()


# =========================================================
# 12. PATIENT WELCOME PANEL + BUTTON HANDLER
# =========================================================
def show_patient_welcome(name: str):
    st.markdown(
        f"""
        ### 👋 Welcome, **{name}**  
        I'm here to help you understand your medical report in simple language.  
        What would you like to do next?
        """
    )

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        if st.button("📄 Explain my report"):
            st.session_state.user_choice = "explain"

    with col2:
        if st.button("🧪 Explain my labs"):
            st.session_state.user_choice = "labs"

    with col3:
        if st.button("💊 Explain my medications"):
            st.session_state.user_choice = "meds"

    with col4:
        if st.button("📝 Create a 1-week care plan"):
            st.session_state.user_choice = "careplan"

    if st.button("🤝 I feel overwhelmed or anxious"):
        st.session_state.user_choice = "support_me"


def handle_welcome_choice(mode: str):
    """
    Execute actions when a welcome button is clicked.
    This must be called AFTER generate_orchestrated_response is defined.
    """
    choice = st.session_state.get("user_choice")
    if not choice:
        return

    # All of these auto-questions use the orchestrator
    if choice == "explain":
        auto_q = "Please explain my medical report in simple terms."
        with st.spinner("Explaining your report..."):
            reply = generate_orchestrated_response(auto_q, mode)
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    elif choice == "labs":
        auto_q = "Please explain my lab results."
        with st.spinner("Analyzing labs..."):
            reply = generate_orchestrated_response(auto_q, mode)
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    elif choice == "meds":
        auto_q = "Explain all medications and their side effects."
        with st.spinner("Reviewing medications and side effects..."):
            reply = generate_orchestrated_response(auto_q, mode)
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    elif choice == "careplan":
        auto_q = "Create a one-week care plan based on my report."
        with st.spinner("Preparing a one-week care plan..."):
            reply = generate_orchestrated_response(auto_q, mode)
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    elif choice == "support_me":
        auto_q = "I feel overwhelmed. Please help me feel better."
        with st.spinner("Connecting you with a supportive explanation..."):
            reply = generate_orchestrated_response(auto_q, mode)
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    # Reset after handling so buttons work again next click
    st.session_state.user_choice = None


# =========================================================
# 13. WEB SEARCH TOGGLE (SIDEBAR)
# =========================================================
st.sidebar.markdown("### 🌐 Web Search (optional)")
st.sidebar.toggle(
    "Enable web search for current question",
    key="web_search_enabled",
)

with st.sidebar.expander("🔗 Web References (Latest)"):
    refs = st.session_state.get("latest_web_refs")
    if refs:
        st.write(refs)
    else:
        st.caption("No web search done yet.")


# =========================================================
# 14. WELCOME PANEL (ONLY IF PDF EXISTS)
# =========================================================
if st.session_state.pdf_text:
    show_patient_welcome(st.session_state.user_id)
    handle_welcome_choice(mode)
else:
    st.info("Upload a medical report PDF to unlock guided options.")


# =========================================================
# 15. CHAT UI
# =========================================================
st.markdown("### Conversation")

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

    snippet = extract_memory_snippet(user_input, assistant_reply)
    if snippet:
        memory.add_memory(user_id, snippet)


# =========================================================
# 16. CLEAR BUTTON
# =========================================================
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()
