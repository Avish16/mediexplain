import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
#from app.bots.meds_rag_search import search_meds_knowledge

import pandas as pd
import streamlit as st


# =========================
# CONSTANTS
# =========================

SAFETY_LEVELS = {
    "safe": "✅ Safe – no safety issues detected.",
    "transform": "⚠️ Transformed – unsafe content softened or rewritten.",
    "block": "⛔ Blocked – unsafe to answer.",
}

DEFAULT_TOP_K = 5


# =========================
# DATA MODELS
# =========================

@dataclass
class RetrievedChunk:
    rank: int
    score: float
    source: str
    doc_id: str
    snippet: str


@dataclass
class RetrievalDiagnostics:
    latency_ms: float
    top_k: int
    num_returned: int
    index_name: str
    strategy: str
    chunks: List[RetrievedChunk]


@dataclass
class RoutingTraceStep:
    step: str
    detail: str
    meta: Dict[str, Any]


@dataclass
class RoutingDiagnostics:
    detected_intent: str
    selected_bot: str
    confidence: float
    trace: List[RoutingTraceStep]


@dataclass
class SafetyDiagnostics:
    decision: str
    policy_flags: List[str]
    notes: str


@dataclass
class BotOutputs:
    final_answer: str
    model_name: str
    temperature: float
    raw_completion: str
    reasoning_notes: str


@dataclass
class SyntheticPatientSnapshot:
    patient_id: str
    demographics: Dict[str, Any]
    vitals: Dict[str, Any]
    labs: Dict[str, Any]
    medications: Dict[str, Any]
    clinical_notes: Dict[str, Any]


@dataclass
class ValidatorResult:
    query: str
    timestamp: float
    retrieval: RetrievalDiagnostics
    routing: RoutingDiagnostics
    safety: SafetyDiagnostics
    bot_outputs: BotOutputs
    synthetic_patient: Optional[SyntheticPatientSnapshot]


# NEW: one entry per question–answer pair stored in memory/history
@dataclass
class ConversationTurn:
    timestamp: float
    query: str
    answer: str


# =========================
# DEMO PIPELINE OUTPUT
# =========================
from app.bots.meds_rag_search import search_meds_knowledge

MEDS_VECTOR_STORE_ID = "vs_6930ffbfc0188191997f62a2ebe5daf5"  # <--- your vector store ID

def _demo_result(user_query: str, top_k: int) -> ValidatorResult:
    start = time.time()

    # ---- REAL RAG SEARCH ----
    # rag_text = search_meds_knowledge(
    #     query=user_query,
    #     vector_store_id=MEDS_VECTOR_STORE_ID,
    #     k=top_k,
    # )

    latency = (time.time() - start) * 1000

    # ---- Build diagnostics ----
    chunks = []
    for i, chunk in enumerate(rag_text.split("\n\n")):
        if not chunk.strip():
            continue
        chunks.append(
            RetrievedChunk(
                rank=i + 1,
                score=0.90 - i * 0.03,
                source="meds_vector_store",
                doc_id=f"rag_chunk_{i}",
                snippet=chunk.strip()[:400],
            )
        )

    retrieval = RetrievalDiagnostics(
        latency_ms=latency,
        top_k=top_k,
        num_returned=len(chunks),
        index_name="openai_vector_store",
        strategy="file_search (embeddings)",
        chunks=chunks,
    )

    # Everything else stays SAME
    # Safety, routing, synthetic patient, outputs — unchanged mock
    result = ValidatorResult(
        query=user_query,
        timestamp=time.time(),
        retrieval=retrieval,
        routing=_demo_routing(),
        safety=_demo_safety(),
        bot_outputs=_demo_bot_output(rag_text),
        synthetic_patient=_demo_patient(),
    )

    return result

# =========================
# UI RENDER HELPERS
# =========================

def _render_overview(result: ValidatorResult) -> None:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Retrieval latency (ms)", f"{result.retrieval.latency_ms:.1f}")
    with col2:
        st.metric(
            "Evidence chunks",
            f"{result.retrieval.num_returned}/{result.retrieval.top_k}",
        )
    with col3:
        st.metric("Intent confidence", f"{result.routing.confidence:.2f}")
    with col4:
        st.metric("Safety decision", result.safety.decision)

    st.markdown("---")
    st.subheader("High-level pipeline summary")
    st.write(
        f"- **Detected intent:** `{result.routing.detected_intent}`  \n"
        f"- **Selected bot:** `{result.routing.selected_bot}`  \n"
        f"- **Retrieval index:** `{result.retrieval.index_name}` "
        f"using `{result.retrieval.strategy}`  \n"
        f"- **Safety flags:** {', '.join(result.safety.policy_flags) or 'None'}"
    )

    st.markdown("### Full query")
    st.write(result.query)


def _render_retrieval_panel(result: ValidatorResult) -> None:
    st.subheader("Retrieved evidence")
    df = pd.DataFrame(
        [
            {
                "Rank": c.rank,
                "Score": round(c.score, 4),
                "Source": c.source,
                "Doc ID": c.doc_id,
                "Snippet": c.snippet,
            }
            for c in result.retrieval.chunks
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Raw retrieval diagnostics"):
        st.json(
            {
                "latency_ms": result.retrieval.latency_ms,
                "top_k": result.retrieval.top_k,
                "num_returned": result.retrieval.num_returned,
                "index_name": result.retrieval.index_name,
                "strategy": result.retrieval.strategy,
            }
        )


def _render_routing_panel(result: ValidatorResult) -> None:
    st.subheader("Router decisions & trace")
    st.write(
        f"**Detected intent:** `{result.routing.detected_intent}`  \n"
        f"**Selected bot:** `{result.routing.selected_bot}`  \n"
        f"**Confidence:** `{result.routing.confidence:.2f}`"
    )
    st.markdown("### Routing steps")
    for step in result.routing.trace:
        with st.container(border=True):
            st.markdown(f"**{step.step}**")
            st.write(step.detail)
            if step.meta:
                with st.expander("Metadata"):
                    st.json(step.meta)


def _render_safety_panel(result: ValidatorResult) -> None:
    st.subheader("Safety & guardrails")
    desc = SAFETY_LEVELS.get(result.safety.decision, "Unknown safety state.")
    if result.safety.decision == "safe":
        st.success(desc)
    elif result.safety.decision == "transform":
        st.warning(desc)
    elif result.safety.decision == "block":
        st.error(desc)
    else:
        st.info(desc)

    st.write("**Policy flags:**", ", ".join(result.safety.policy_flags) or "None")
    st.markdown("**Notes:**")
    st.write(result.safety.notes)


def _render_bot_outputs_panel(result: ValidatorResult) -> None:
    st.subheader("Bot answer & reasoning")

    st.markdown("### Final answer shown to user")
    st.write(result.bot_outputs.final_answer)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model:**", result.bot_outputs.model_name)
    with col2:
        st.write("**Temperature:**", result.bot_outputs.temperature)

    with st.expander("Reasoning notes (developer only)"):
        st.write(result.bot_outputs.reasoning_notes)

    with st.expander("Raw completion payload"):
        st.text(result.bot_outputs.raw_completion)


def _render_synthetic_patient_panel(result: ValidatorResult) -> None:
    st.subheader("Synthetic patient context")
    patient = result.synthetic_patient
    if patient is None:
        st.info("No synthetic patient attached to this query.")
        return

    st.write(f"**Patient ID:** `{patient.patient_id}`")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Demographics")
        st.json(patient.demographics)
        st.markdown("#### Vitals")
        st.json(patient.vitals)
    with col2:
        st.markdown("#### Labs")
        st.json(patient.labs)
        st.markdown("#### Medications")
        st.json(patient.medications)

    st.markdown("#### Clinical notes")
    st.json(patient.clinical_notes)


def _render_raw_json_panel(result: ValidatorResult) -> None:
    st.subheader("Raw ValidatorResult payload")
    st.json(asdict(result))


# NEW: history renderer
def _render_history_panel(history: List[ConversationTurn]) -> None:
    st.subheader("Validator Q&A history (this session)")
    if not history:
        st.info("No previous runs recorded yet.")
        return

    # Most recent first
    rows = [
        {
            "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t.timestamp)),
            "Question": t.query[:120] + ("…" if len(t.query) > 120 else ""),
            "Answer (preview)": t.answer[:120] + ("…" if len(t.answer) > 120 else ""),
        }
        for t in reversed(history)
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Full questions and answers")
    for idx, t in enumerate(reversed(history), start=1):
        with st.expander(f"Run {idx} – {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t.timestamp))}"):
            st.markdown("**Question:**")
            st.write(t.query)
            st.markdown("**Answer:**")
            st.write(t.answer)


# =========================
# MAIN ENTRYPOINT
# =========================

def run_validator_page() -> None:
    st.title("🩺 MediExplain – Validator Console")
    st.caption(
        "Developer-facing console for diagnostics on retrieval, routing, "
        "safety, final bot output, synthetic patient context, and Q&A history."
    )

    st.sidebar.header("Validator controls")

    default_query = (
        "Is it safe to increase my blood pressure medication dose "
        "if I still have headaches?"
    )
    user_query = st.sidebar.text_area(
        "User query",
        value=default_query,
        height=120,
    )

    top_k = st.sidebar.slider(
        "Top-K documents", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1
    )

    reuse_last = st.sidebar.checkbox(
        "Reuse last result (don’t re-run pipeline)", value=False
    )

    run_btn = st.sidebar.button("Run validation", type="primary")

    # session state for last result + history
    if "validator_last_result" not in st.session_state:
        st.session_state.validator_last_result = None
    if "validator_history" not in st.session_state:
        st.session_state.validator_history: List[ConversationTurn] = []

    if run_btn or (
        not reuse_last and st.session_state.validator_last_result is None
    ):
        if not user_query.strip():
            st.warning("Please enter a user query first.")
            return

        with st.spinner("Running mock MediExplain pipeline…"):
            result = _demo_result(user_query.strip(), top_k=top_k)

        st.session_state.validator_last_result = result

        # push Q&A into in-memory history
        st.session_state.validator_history.append(
            ConversationTurn(
                timestamp=result.timestamp,
                query=result.query,
                answer=result.bot_outputs.final_answer,
            )
        )

    result: Optional[ValidatorResult] = st.session_state.validator_last_result

    if result is None:
        st.info("Enter a query in the sidebar and click **Run validation**.")
        return

    history: List[ConversationTurn] = st.session_state.validator_history

    tabs = st.tabs(
        [
            "Overview",
            "Retrieval",
            "Routing",
            "Bot outputs",
            "Safety",
            "Synthetic patient",
            "Q&A History",
            "Raw JSON",
        ]
    )

    (
        overview_tab,
        retrieval_tab,
        routing_tab,
        bot_tab,
        safety_tab,
        patient_tab,
        history_tab,
        json_tab,
    ) = tabs

    with overview_tab:
        _render_overview(result)
    with retrieval_tab:
        _render_retrieval_panel(result)
    with routing_tab:
        _render_routing_panel(result)
    with bot_tab:
        _render_bot_outputs_panel(result)
    with safety_tab:
        _render_safety_panel(result)
    with patient_tab:
        _render_synthetic_patient_panel(result)
    with history_tab:
        _render_history_panel(history)
    with json_tab:
        _render_raw_json_panel(result)


# When Streamlit runs this file as a page via st.Page, __name__ == "__main__".
if __name__ == "__main__":
    run_validator_page()
