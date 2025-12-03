try:
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

# =========================================================
# 1. CONFIG & CLIENT
# =========================================================
st.set_page_config(page_title="MediExplain Chatbot", layout="wide")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =========================================================
# 2. CHROMA-BASED LONG-TERM MEMORY MANAGER
# =========================================================
class ChromaMemoryManager:
    """
    Long-term memory using ChromaDB.

    Each memory is stored as a document with:
    - text: the memory snippet
    - metadata: includes user_id (so multiple users are isolated)
    """

    def __init__(self, path: str = "mediexplain_chroma"):
        # Persistent client: uses local sqlite + disk
        self.client = chromadb.PersistentClient(
            path=path,
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
            metadatas=[{"user_id": user_id}]
        )

    def retrieve_memory(self, user_id: str, query: str, k: int = 5) -> List[str]:
        """
        Retrieve the top-k memories for this user based on semantic similarity.
        """
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
        placeholder="e.g. john.doe@example.com"
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
# 5. MAIN HEADER & MODE SWITCH
# =========================================================
st.title("🩺 MediExplain – Your Medical Report Companion")

mode = st.radio(
    "Choose AI Explanation Mode:",
    ["Patient Mode (Simple & Friendly)", "Caregiver Mode (Technical & Clinical)"],
    horizontal=False,
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


# =========================================================
# 7. MEMORY-AWARE UTILS
# =========================================================
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


def generate_response(user_input: str, mode: str) -> str:
    """
    Generate a MediExplain response using:
    - Persona (patient vs caregiver)
    - Long-term memory (ChromaDB)
    - Uploaded PDF context
    - Short-term chat history
    """
    # Persona
    if "Patient Mode" in mode:
        persona = (
            "Use very simple, friendly language. Avoid medical jargon where possible. "
            "Explain step-by-step, be reassuring, and focus on clarity."
        )
    else:
        persona = (
            "Use clinically accurate language appropriate for a caregiver or clinician. "
            "You may use medical terminology and provide more detailed reasoning, "
            "but still stay concise and clear."
        )

    # Retrieve long-term memory from Chroma
    ltm_snippets = memory.retrieve_memory(user_id, user_input, k=5)
    long_term_context = "\n- ".join(ltm_snippets) if ltm_snippets else "None available."

    system_content = f"""
You are MediExplain, an AI assistant that helps people understand their medical
information in a safe, responsible way.

User ID: {user_id}

Persona:
{persona}

Long-term memory for this user (important past facts, if any):
- {long_term_context}

Current uploaded medical report (if any):
--------------------
{st.session_state.pdf_text or "No report uploaded for this session."}
--------------------

Always:
- Be clear about what you know from the report vs what you are inferring.
- Encourage the user to discuss important findings with their doctor.
- Do NOT provide definitive diagnoses or treatment plans.
"""

    messages = [{"role": "system", "content": system_content}]

    # Short-term conversation history
    for m in st.session_state.messages:
        messages.append({"role": m["role"], "content": m["content"]})

    # Latest user message
    messages.append({"role": "user", "content": user_input})

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
    )

    reply = response.choices[0].message.content
    return reply


# =========================================================
# 8. CHAT HISTORY UI
# =========================================================
st.write("Conversation")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])


# =========================================================
# 9. CHAT INPUT & FLOW
# =========================================================
user_input = st.chat_input("Ask a question about your medical report or health history...")

if user_input:
    # Add user message to short-term memory
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("MediExplain is thinking..."):
        assistant_reply = generate_response(user_input, mode)

    # Add assistant reply to short-term memory
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    # Display reply
    st.chat_message("assistant").markdown(assistant_reply)

    # Extract and store long-term memory into Chroma
    try:
        memory_snippet = extract_memory_snippet(user_input, assistant_reply)
        if memory_snippet:
            memory.add_memory(user_id, memory_snippet)
    except Exception:
        # Don't break the app if memory extraction fails
        pass


# =========================================================
# 10. CLEAR CONVERSATION BUTTON
# =========================================================
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
