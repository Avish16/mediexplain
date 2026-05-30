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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from styles import inject_global_css

from app.bots.explainer_bot import run_explainer
from app.bots.labs_bot import run_labs
from app.bots.meds_bot import run_meds
from app.bots.careplan_bot import run_careplan
from app.bots.snapshot_bot import run_snapshot
from app.bots.support_bot import run_support
from app.bots.prescription_bot import run_prescriptions
from app.bots.meds_rag_search import search_meds_knowledge

# =========================================================
# 1. STYLES + CONFIG
# =========================================================
inject_global_css()

st.markdown("""
<style>
/* ── Chat message overrides ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-role="user"] [data-testid="stMarkdownContainer"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(0,102,255,0.1)) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    border-radius: 16px 4px 16px 16px !important;
    padding: 12px 16px !important;
    color: #E2E8F0 !important;
    max-width: 80%;
    margin-left: auto;
}

/* AI bubble */
[data-testid="stChatMessage"][data-role="assistant"] [data-testid="stMarkdownContainer"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(0,212,255,0.1) !important;
    border-radius: 4px 16px 16px 16px !important;
    padding: 12px 16px !important;
    color: #CBD5E1 !important;
}

/* Avatar */
[data-testid="stChatMessageAvatarUser"] {
    background: rgba(0,212,255,0.15) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: rgba(0,102,255,0.15) !important;
    border: 1px solid rgba(0,102,255,0.3) !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.1) !important;
}

/* Mode pills: radio styled as toggle */
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 8px !important;
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(0,212,255,0.15) !important;
    border-radius: 100px !important;
    padding: 4px !important;
    width: fit-content !important;
}
div[data-testid="stRadio"] label {
    border-radius: 100px !important;
    padding: 5px 18px !important;
    border: none !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    cursor: pointer;
    transition: all 0.2s ease !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,102,255,0.15)) !important;
    border: none !important;
    color: #00D4FF !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(0,212,255,0.2) !important;
    border-radius: 12px !important;
}

/* Typing indicator dots */
@keyframes blink {
  0%,80%,100% { opacity: 0.2; transform: scale(0.8); }
  40%          { opacity: 1;   transform: scale(1.1); }
}
.typing-dot { animation: blink 1.4s ease-in-out infinite; display:inline-block; }
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. OPENAI CLIENT
# =========================================================
def _get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        st.error("OPENAI_API_KEY not found. Set it in .streamlit/secrets.toml or as an env var.")
        st.stop()
    return key

client = OpenAI(api_key=_get_api_key())

# =========================================================
# 3. MEMORY MANAGER
# =========================================================
class ChromaMemoryManager:
    def __init__(self):
        self.client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection("mediexplain_memory")

    def add_memory(self, user_id: str, text: str):
        text = text.strip()
        if not text:
            return
        doc_id = f"{user_id}_{abs(hash(text))}"
        self.collection.add(ids=[doc_id], documents=[text], metadatas=[{"user_id": user_id}])

    def retrieve_memory(self, user_id: str, query: str, k: int = 5):
        try:
            result = self.collection.query(
                query_texts=[query], n_results=k, where={"user_id": user_id}
            )
            return result.get("documents", [[]])[0]
        except Exception:
            return []


memory = ChromaMemoryManager()

# =========================================================
# 4. SESSION STATE
# =========================================================
for key, default in [
    ("messages", []),
    ("pdf_text", ""),
    ("file_id", None),
    ("user_id", None),
    ("vector_store_id", None),
    ("user_choice", None),
    ("web_search_enabled", False),
    ("latest_web_refs", None),
    ("latest_meds_rag_chunks", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =========================================================
# 5. SIDEBAR – LOGIN + SETTINGS
# =========================================================
st.sidebar.markdown("""
<div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:600;
            color:#00D4FF;text-transform:uppercase;letter-spacing:1.5px;
            padding:8px 0 12px;border-bottom:1px solid rgba(0,212,255,0.1);margin-bottom:16px;">
  Session
</div>
""", unsafe_allow_html=True)

if st.session_state.user_id is None:
    login_id = st.sidebar.text_input("Email or Patient ID", placeholder="your@email.com")
    if st.sidebar.button("Continue", use_container_width=True) and login_id.strip():
        st.session_state.user_id = login_id.strip()
        st.rerun()
else:
    st.sidebar.markdown(f"""
    <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);
                border-radius:8px;padding:10px 12px;margin-bottom:12px;">
      <div style="font-size:11px;color:#10B981;font-weight:600;letter-spacing:1px;
                  text-transform:uppercase;margin-bottom:2px;">Logged in</div>
      <div style="font-size:13px;color:#E2E8F0;word-break:break-all;">{st.session_state.user_id}</div>
    </div>
    """, unsafe_allow_html=True)
    if st.sidebar.button("Logout", use_container_width=True):
        for k in ("user_id","messages","pdf_text","file_id","vector_store_id"):
            st.session_state[k] = None if k != "messages" else []
        st.rerun()

# Web search toggle
st.sidebar.markdown("""
<div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:600;
            color:#64748B;text-transform:uppercase;letter-spacing:1.5px;
            padding:12px 0 8px;border-top:1px solid rgba(0,212,255,0.08);margin-top:8px;">
  Options
</div>
""", unsafe_allow_html=True)

st.sidebar.toggle("Web Search", key="web_search_enabled")

with st.sidebar.expander("Web References"):
    refs = st.session_state.get("latest_web_refs")
    if refs:
        st.write(refs)
    else:
        st.caption("No web search done yet.")

# =========================================================
# 6. GATE: REQUIRE LOGIN
# =========================================================
if st.session_state.user_id is None:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                padding:80px 20px;text-align:center;">
      <div style="font-size:48px;margin-bottom:20px;">💬</div>
      <h2 style="font-family:'Space Grotesk',sans-serif;color:#F1F5F9;margin-bottom:8px;">
        MediExplain Chatbot
      </h2>
      <p style="color:#64748B;font-size:15px;max-width:400px;line-height:1.6;">
        Enter your email or patient ID in the sidebar to get started.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

user_id = st.session_state.user_id

# =========================================================
# 7. PAGE HEADER + MODE PILLS
# =========================================================
st.markdown("""
<div style="padding:28px 0 20px;animation:fadeIn 0.5s ease;">
  <h1 style="font-family:'Space Grotesk',sans-serif;font-size:30px;font-weight:700;
             color:#F1F5F9;margin:0 0 6px;">Medical Chatbot</h1>
  <p style="font-size:14px;color:#64748B;margin:0;">
    Ask anything about your medical report, labs, medications, or care plan.
  </p>
</div>
""", unsafe_allow_html=True)

mode_label = st.radio(
    "Explanation mode",
    ["Patient Mode", "Caregiver Mode"],
    horizontal=True,
    label_visibility="collapsed",
)
mode = "caregiver" if "Caregiver" in mode_label else "patient"

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# =========================================================
# 8. PDF UPLOAD
# =========================================================
with st.expander("Upload Medical Report PDF", expanded=not bool(st.session_state.pdf_text)):
    uploaded_pdf = st.file_uploader("Drop your PDF here", type=["pdf"], label_visibility="collapsed")
    if uploaded_pdf is not None:
        reader = PdfReader(uploaded_pdf)
        extracted = ""
        for page in reader.pages:
            try:
                extracted += (page.extract_text() or "") + "\n"
            except Exception:
                pass
        st.session_state.pdf_text = extracted.strip()
        vs = client.vector_stores.create(name="mediexplain_vs")
        st.session_state.vector_store_id = vs.id
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vs.id, files=[uploaded_pdf]
        )
        st.success("PDF indexed and ready for questions.")
        if st.session_state.pdf_text:
            with st.expander("View extracted text"):
                st.write(st.session_state.pdf_text)

if not st.session_state.pdf_text:
    st.markdown("""
    <div style="background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.15);
                border-radius:10px;padding:12px 16px;font-size:13px;color:#64748B;
                display:flex;gap:8px;align-items:center;margin-bottom:8px;">
      <span>ℹ</span> Upload a medical report PDF for best results. You can still ask general questions without it.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# 9. HELPER FUNCTIONS
# =========================================================
def search_pdf_context(query: str) -> str:
    vector_store_id = st.session_state.get("vector_store_id")
    if not vector_store_id:
        return ""
    prompt = f'Search the uploaded medical report for: "{query}"\nReturn a 2–4 paragraph summary using ONLY information from the PDF.'
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
        max_output_tokens=800,
    )
    return response.output_text or ""


def extract_memory_snippet(user_input: str, assistant_reply: str) -> str:
    prompt = f"""Extract ONLY long-term clinically meaningful facts to save in the user's memory profile.
Examples: diagnoses, allergies, medications, lab abnormalities, patient preferences.
Return an empty string if nothing is appropriate.
USER: {user_input}
ASSISTANT: {assistant_reply}"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def route_to_specialist_bot(mode: str, question: str, pdf_text: str, long_term_memory):
    system_prompt = """You are MediExplain's routing agent. Return STRICT JSON: {"bot": "...", "reason": "..."}

SCOPE: Only answer medical-report questions. If the user asks about politics, celebrities, sports, cooking, gaming, geography, weather, or general knowledge, return bot="OUT_OF_SCOPE".

ROUTING:
- drug names / side effects / interactions / "is this safe" → bot="MEDS"
- discharge prescriptions / sig instructions → bot="PRESCRIPTIONS"
- lab values → bot="LABS"
- vital signs / symptoms → bot="SNAPSHOT"
- care plan → bot="CAREPLAN"
- emotional distress → bot="SUPPORT"
- general explanation → bot="EXPLAINER"

Choose ONLY from: ["EXPLAINER","LABS","MEDS","CAREPLAN","SNAPSHOT","SUPPORT","PRESCRIPTIONS","OUT_OF_SCOPE"]
Return STRICT JSON only."""

    user_payload = f"MODE: {mode}\nQUESTION: {question}\nREPORT TEXT (first 3000 chars):\n{pdf_text[:3000]}\nUSER MEMORY:\n{long_term_memory}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_payload}],
    ).choices[0].message.content
    try:
        clean = resp.replace("```json", "").replace("```", "").strip()
        bot_name = json.loads(clean).get("bot", "EXPLAINER").upper()
    except Exception:
        bot_name = "EXPLAINER"
    valid = {"EXPLAINER","LABS","MEDS","CAREPLAN","SNAPSHOT","SUPPORT","PRESCRIPTIONS","OUT_OF_SCOPE"}
    return bot_name if bot_name in valid else "EXPLAINER"


def run_websearch(query: str) -> str:
    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Use web search to answer:\n{query}",
            tools=[{"type": "web_search"}],
            max_output_tokens=800,
        )
        return resp.output_text or "No web search result returned."
    except Exception as e:
        return f"Web search unavailable: {e}"


def get_conversation_history(limit=20):
    history = st.session_state.messages[-limit:]
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history).strip()


def generate_orchestrated_response(user_input: str, mode: str) -> str:
    if st.session_state.get("web_search_enabled", False):
        webresult = run_websearch(user_input)
        st.session_state.latest_web_refs = webresult
        return f"### Web Search Result\n\n{webresult}"

    pdf_text = st.session_state.pdf_text or ""
    long_term_memory = memory.retrieve_memory(user_id, user_input, k=5)
    chosen_bot = route_to_specialist_bot(mode, user_input, pdf_text, long_term_memory)

    if chosen_bot == "OUT_OF_SCOPE":
        return ("I'm MediExplain — I can only help with your medical report, labs, medications, "
                "care plan, or clinical explanations. This question is outside that scope.")

    pdf_context = search_pdf_context(user_input)
    meds_rag_text = ""
    if chosen_bot in {"MEDS", "PRESCRIPTIONS"}:
        try:
            rag = search_meds_knowledge(user_input, top_k=5)
            meds_rag_text = rag.get("rag_text", "") or ""
            st.session_state.latest_meds_rag_chunks = rag.get("chunks", [])
        except Exception as e:
            meds_rag_text = f"(Medication RAG lookup failed: {e})"

    combined_context = pdf_context or pdf_text
    if meds_rag_text:
        combined_context += f"\n\n---\n### Medication Research Evidence\n{meds_rag_text}\n"

    conversation_history = get_conversation_history(limit=12)
    bot_map = {
        "LABS": run_labs, "MEDS": run_meds, "CAREPLAN": run_careplan,
        "SNAPSHOT": run_snapshot, "SUPPORT": run_support, "PRESCRIPTIONS": run_prescriptions,
    }
    try:
        fn = bot_map.get(chosen_bot, run_explainer)
        reply = fn(user_input, mode, combined_context, long_term_memory, conversation_history=conversation_history)
        return reply + f"\n\n---\n*Answered by: **{chosen_bot} bot***"
    except Exception:
        traceback.print_exc()
        fallback = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Give a safe, simple explanation.\nQUESTION: {user_input}\nCONTEXT: {combined_context}"}],
            temperature=0.3,
        )
        return fallback.choices[0].message.content.strip()

# =========================================================
# 10. QUICK-ACTION BUTTONS (only when PDF is loaded)
# =========================================================
if st.session_state.pdf_text:
    st.markdown("""
    <div style="font-size:12px;color:#475569;font-weight:600;text-transform:uppercase;
                letter-spacing:1px;margin-bottom:8px;">Quick Actions</div>
    """, unsafe_allow_html=True)
    qa_cols = st.columns(5)
    quick_actions = [
        ("📄", "Explain Report"),
        ("🧪", "Explain Labs"),
        ("💊", "My Medications"),
        ("📝", "Care Plan"),
        ("🤝", "I Feel Overwhelmed"),
    ]
    prompts = [
        "Please explain my medical report in simple terms.",
        "Please explain my lab results.",
        "Explain all my medications and their side effects.",
        "Create a one-week care plan based on my report.",
        "I feel overwhelmed. Please help me feel better.",
    ]
    for col, (icon, label), prompt in zip(qa_cols, quick_actions, prompts):
        with col:
            if st.button(f"{icon} {label}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.spinner(""):
                    reply = generate_orchestrated_response(prompt, mode)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                snippet = extract_memory_snippet(prompt, reply)
                if snippet:
                    memory.add_memory(user_id, snippet)
                st.rerun()

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# =========================================================
# 11. CHAT HISTORY
# =========================================================
st.markdown("""
<div style="font-size:12px;color:#475569;font-weight:600;text-transform:uppercase;
            letter-spacing:1px;margin-bottom:4px;margin-top:8px;">Conversation</div>
""", unsafe_allow_html=True)

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:40px 20px;color:#334155;">
      <div style="font-size:32px;margin-bottom:10px;">💬</div>
      <div style="font-size:14px;">Ask a question about your medical report to get started.</div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================================
# 12. CHAT INPUT
# =========================================================
user_input = st.chat_input("Ask about your report, labs, medications, or care plan...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Typing indicator
    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div style="display:flex;gap:5px;align-items:center;padding:4px 0;">
          <span class="typing-dot" style="width:7px;height:7px;background:#00D4FF;
                border-radius:50;display:inline-block;">&#8203;</span>
          <span class="typing-dot" style="width:7px;height:7px;background:#00D4FF;
                border-radius:50%;display:inline-block;">&#8203;</span>
          <span class="typing-dot" style="width:7px;height:7px;background:#00D4FF;
                border-radius:50%;display:inline-block;">&#8203;</span>
        </div>
        """, unsafe_allow_html=True)

        assistant_reply = generate_orchestrated_response(user_input, mode)
        typing_placeholder.markdown(assistant_reply)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    snippet = extract_memory_snippet(user_input, assistant_reply)
    if snippet:
        memory.add_memory(user_id, snippet)

# =========================================================
# 13. CLEAR
# =========================================================
if st.session_state.messages:
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    if st.button("Clear Conversation", use_container_width=False):
        st.session_state.messages = []
        st.rerun()
