import os
import sys
import glob
import logging

# --- Fix for ChromaDB sqlite3 issue (must be BEFORE import chromadb) ---
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
HTML_FOLDER = "html"                   # folder with PMCxxxx.html
CHROMA_PATH = "./mediexplain_chromadb"
COLLECTION_NAME = "MediExplainPMC"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"            # or gpt-4.1-mini

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

logging.basicConfig(level=logging.INFO)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def load_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("⚠️ OpenAI API key not found in Streamlit secrets.")
        st.stop()


def extract_text_from_html(path: str) -> str:
    """Turn an HTML article into clean plain text."""
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


def chunk_text(text: str):
    """Simple character-based overlapping chunks."""
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0

    return chunks


def create_vectorDB(api_key: str, chroma_client):
    """
    Create / load Chroma collection and embed ALL html/ files.
    Only runs once per Streamlit session.
    """
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    # Only embed on first run: if collection has any ids, assume it's done
    existing = collection.count()
    if existing > 0:
        st.sidebar.info(f"Vector DB already has {existing} chunks. Skipping re-embed.")
        return collection

    if not os.path.isdir(HTML_FOLDER):
        st.sidebar.error(f"HTML folder '{HTML_FOLDER}' not found.")
        return collection

    html_paths = glob.glob(os.path.join(HTML_FOLDER, "*.html"))
    if not html_paths:
        st.sidebar.error(f"No .html files found in '{HTML_FOLDER}'.")
        return collection

    st.sidebar.write(f"Found {len(html_paths)} HTML articles. Embedding...")

    for path in html_paths:
        filename = os.path.basename(path)          # e.g. PMC123456.html
        pmcid = filename.replace(".html", "")

        try:
            text = extract_text_from_html(path)
            if not text.strip():
                continue

            chunks = chunk_text(text)
            ids = [f"{pmcid}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {"source": pmcid, "chunk_index": i} for i in range(len(chunks))
            ]

            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
            )
            st.sidebar.success(f"✅ Embedded {pmcid} ({len(chunks)} chunks)")
        except Exception as e:
            st.sidebar.error(f"❌ Error processing {filename}: {e}")

    st.sidebar.success("✅ Finished building vector DB.")
    return collection


def build_rag_prompt(user_question: str, retrieved_docs: list[str]) -> str:
    context = "\n\n---\n\n".join(retrieved_docs)
    return f"""
You are an AI assistant for medical article exploration.

Use ONLY the following context from open-access medical articles to answer the question.
If the answer is not clearly supported by the context, say you do not know.

CONTEXT:
{context}

QUESTION:
{user_question}

Give a clear, concise explanation at graduate level.
Always add the sentence: "This is not medical advice."
"""


# -------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------
st.title("🧬 MediExplain – PMC RAG Explorer")

st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Choose mode:",
    ["Part A – Retrieval Test", "Part B – RAG Chatbot"],
)

# ✅ API key + OpenAI client
api_key = load_api_key()
client = OpenAI(api_key=api_key)

# ✅ Chroma client (now that sqlite is patched)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# ✅ Initialize vector DB once per session
if "MediExplain_vectorDB" not in st.session_state:
    st.session_state["MediExplain_vectorDB"] = create_vectorDB(api_key, chroma_client)

collection = st.session_state["MediExplain_vectorDB"]

if collection is None:
    st.error("Vector DB not initialized. Check sidebar errors.")
    st.stop()

# -------------------------------------------------
# PART A – RETRIEVAL TEST
# -------------------------------------------------
if mode == "Part A – Retrieval Test":
    st.subheader("🔍 Retrieval Test (Top 3 Chunks)")

    test_queries = [
        "immune checkpoint inhibitor adverse effects",
        "management of type 2 diabetes",
        "cardiovascular risk factors",
        "kidney function monitoring",
    ]
    query_choice = st.selectbox("Pick a test query:", test_queries)

    if st.button("Run Test Search"):
        results = collection.query(
            query_texts=[query_choice],
            n_results=3,
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        st.write(f"### Top 3 chunks for **{query_choice}**")
        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            st.markdown(f"**{i}. Source:** `{meta['source']}` – chunk {meta['chunk_index']}")
            st.text_area(
                label=f"Preview {i}",
                value=doc[:800] + ("..." if len(doc) > 800 else ""),
                height=200,
            )

# -------------------------------------------------
# PART B – RAG CHATBOT
# -------------------------------------------------
else:
    st.subheader("💬 MediExplain RAG Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask a question about the medical articles..."):
        # Save + display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 🔎 Retrieve top 4 chunks
        results = collection.query(
            query_texts=[user_input],
            n_results=4,
        )
        retrieved_docs = results["documents"][0]
        sources = results["metadatas"][0]

        # 🧠 Build RAG prompt
        prompt = build_rag_prompt(user_input, retrieved_docs)

        # Call LLM with streaming
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            response = st.write_stream(stream)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        # Show which articles were used
        with st.expander("📂 Sources used"):
            for m in sources:
                st.write(f"- {m['source']} (chunk {m['chunk_index']})")
