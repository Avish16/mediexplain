import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from styles import inject_global_css
from app.bots.meds_rag_search import search_meds_knowledge
from app_synthetic.router import route_to_specialist_bot

GLOBAL_MED_RAG_VECTORSTORE_ID = "vs_6930ffbfc0188191997f62a2ebe5daf5"

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
# 🔥 Global RAG Vector Store ID for medication research papers
GLOBAL_MED_RAG_VECTORSTORE_ID = "vs_6930ffbfc0188191997f62a2ebe5daf5"


def _demo_result(user_query: str, top_k: int) -> ValidatorResult:
    """REAL RAG retrieval now replaces demo retrieval."""

    # --- RAG retrieval using real vector store ---
    rag = search_meds_knowledge(
        query=user_query,
        top_k=top_k,
        vector_store_id=GLOBAL_MED_RAG_VECTORSTORE_ID,
    )

    # Extract chunks for the UI
    chunks = []
    for i, c in enumerate(rag.get("chunks", [])):
        chunks.append(
            RetrievedChunk(
                rank=c.get("rank", i + 1),
                score=c.get("score", 1.0),
                source=c.get("source", "unknown"),
                doc_id=c.get("doc_id", f"doc_{i+1}"),
                snippet=c.get("snippet", "")
            )
        )

    retrieval = RetrievalDiagnostics(
        latency_ms=12.4,        
        top_k=top_k,
        num_returned=len(chunks),
        index_name="medication_rag_vectorstore",
        strategy="vector_search_only",
        chunks=chunks,
    )

    # --- Routing (mocked for validator UI) ---
    routing = RoutingDiagnostics(
        detected_intent="medication_question",
        selected_bot="MEDS",
        confidence=1.0,
        trace=[
            RoutingTraceStep(
                step="router_mock",
                detail="Validator uses RAG-only mode; routing mocked.",
                meta={}
            )
        ]
    )

    # --- Safety diagnostics ---
    safety = SafetyDiagnostics(
        decision="safe",
        policy_flags=[],
        notes="No unsafe medical instructions detected."
    )

    # --- Final Answer ---
    final_answer = rag.get("answer", "No medical evidence found in the index.")

    bot_outputs = BotOutputs(
        final_answer=final_answer,
        model_name="gpt-4.1-mini",
        temperature=0.2,
        raw_completion=final_answer,
        reasoning_notes="Generated from RAG evidence."
    )

    return ValidatorResult(
        query=user_query,
        timestamp=time.time(),
        retrieval=retrieval,
        routing=routing,
        safety=safety,
        bot_outputs=bot_outputs,
        synthetic_patient=None,
    )

# =========================
# UI RENDER HELPERS
# =========================

def _render_overview(result: ValidatorResult) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Latency (ms)", f"{result.retrieval.latency_ms:.1f}")
    with c2:
        st.metric("Evidence Chunks", f"{result.retrieval.num_returned}/{result.retrieval.top_k}")
    with c3:
        st.metric("Confidence", f"{result.routing.confidence:.2f}")
    with c4:
        st.metric("Safety", result.safety.decision.upper())

    safety_color = "t-ok" if result.safety.decision == "safe" else ("t-warn" if result.safety.decision == "transform" else "t-err")
    flags = ", ".join(result.safety.policy_flags) if result.safety.policy_flags else "none"

    st.markdown(f"""
    <div class="term-block" style="margin-top:16px;">
      <span class="t-dim">────── pipeline trace ──────</span><br>
      <span class="t-key">detected_intent</span>  <span class="t-dim">=</span>  <span class="t-val">{result.routing.detected_intent}</span><br>
      <span class="t-key">selected_bot    </span>  <span class="t-dim">=</span>  <span class="t-info">{result.routing.selected_bot}</span><br>
      <span class="t-key">retrieval_index </span>  <span class="t-dim">=</span>  <span class="t-val">{result.retrieval.index_name}</span><br>
      <span class="t-key">strategy        </span>  <span class="t-dim">=</span>  <span class="t-val">{result.retrieval.strategy}</span><br>
      <span class="t-key">safety_decision </span>  <span class="t-dim">=</span>  <span class="{safety_color}">{result.safety.decision}</span><br>
      <span class="t-key">policy_flags    </span>  <span class="t-dim">=</span>  <span class="t-val">{flags}</span><br>
      <span class="t-dim">────── query ──────</span><br>
      <span class="t-label">&gt; </span><span class="t-val">{result.query}</span>
    </div>
    """, unsafe_allow_html=True)


def _render_retrieval_panel(result: ValidatorResult) -> None:
    # Terminal header row
    rows_html = ""
    for c in result.retrieval.chunks:
        score_color = "t-ok" if c.score >= 0.8 else ("t-warn" if c.score >= 0.5 else "t-err")
        snippet_safe = (c.snippet[:120] + "...") if len(c.snippet) > 120 else c.snippet
        rows_html += f"""
        <div style="border-bottom:1px solid rgba(16,185,129,0.08);padding:8px 0;">
          <span class="t-dim">[{c.rank:02d}]</span>
          <span class="{score_color}" style="margin:0 10px;">score={c.score:.4f}</span>
          <span class="t-key">{c.source}</span>
          <span class="t-dim"> / {c.doc_id}</span><br>
          <span class="t-val" style="margin-left:28px;font-size:11px;">{snippet_safe}</span>
        </div>"""

    st.markdown(f"""
    <div class="term-block">
      <span class="t-ok">RETRIEVAL</span>
      <span class="t-dim"> chunks={result.retrieval.num_returned}/{result.retrieval.top_k}
      latency={result.retrieval.latency_ms:.1f}ms
      index={result.retrieval.index_name}</span>
      {rows_html if rows_html else '<br><span class="t-warn">no chunks returned</span>'}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Raw diagnostics JSON"):
        st.json({
            "latency_ms": result.retrieval.latency_ms,
            "top_k": result.retrieval.top_k,
            "num_returned": result.retrieval.num_returned,
            "index_name": result.retrieval.index_name,
            "strategy": result.retrieval.strategy,
        })


def _render_routing_panel(result: ValidatorResult) -> None:
    conf_color = "t-ok" if result.routing.confidence >= 0.8 else ("t-warn" if result.routing.confidence >= 0.5 else "t-err")
    trace_rows = ""
    for i, step in enumerate(result.routing.trace):
        trace_rows += f"""
        <div style="padding:6px 0;border-bottom:1px solid rgba(16,185,129,0.06);">
          <span class="t-dim">[step-{i}]</span>
          <span class="t-info" style="margin:0 8px;">{step.step}</span>
          <span class="t-val">{step.detail}</span>
        </div>"""

    st.markdown(f"""
    <div class="term-block">
      <span class="t-ok">ROUTING</span><br>
      <span class="t-key">intent    </span> <span class="t-dim">=</span> <span class="t-val">{result.routing.detected_intent}</span><br>
      <span class="t-key">bot       </span> <span class="t-dim">=</span> <span class="t-info">{result.routing.selected_bot}</span><br>
      <span class="t-key">confidence</span> <span class="t-dim">=</span> <span class="{conf_color}">{result.routing.confidence:.2f}</span><br>
      <span class="t-dim">── trace ──</span>
      {trace_rows}
    </div>
    """, unsafe_allow_html=True)


def _render_safety_panel(result: ValidatorResult) -> None:
    decision = result.safety.decision
    decision_map = {
        "safe":      ("t-ok",   "[PASS]", "No safety issues detected."),
        "transform": ("t-warn", "[WARN]", "Unsafe content transformed/softened."),
        "block":     ("t-err",  "[FAIL]", "Unsafe — response blocked."),
    }
    css_cls, badge, default_note = decision_map.get(decision, ("t-info", "[INFO]", ""))
    flags = ", ".join(result.safety.policy_flags) if result.safety.policy_flags else "none"

    st.markdown(f"""
    <div class="term-block">
      <span class="{css_cls}">{badge} SAFETY AUDIT</span><br>
      <span class="t-key">decision     </span> <span class="t-dim">=</span> <span class="{css_cls}">{decision.upper()}</span><br>
      <span class="t-key">policy_flags </span> <span class="t-dim">=</span> <span class="t-val">{flags}</span><br>
      <span class="t-key">notes        </span> <span class="t-dim">=</span> <span class="t-val">{result.safety.notes or default_note}</span>
    </div>
    """, unsafe_allow_html=True)


def _render_bot_outputs_panel(result: ValidatorResult) -> None:
    st.markdown(f"""
    <div class="term-block">
      <span class="t-ok">[BOT OUTPUT]</span>
      <span class="t-dim"> model={result.bot_outputs.model_name}
      temp={result.bot_outputs.temperature}</span><br>
      <span class="t-dim">──────────────────────────────────────</span><br>
      <span class="t-val" style="white-space:pre-wrap;word-break:break-word;">{result.bot_outputs.final_answer[:1200]}{"..." if len(result.bot_outputs.final_answer) > 1200 else ""}</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Reasoning notes"):
        st.markdown(f"<div class='term-block'><span class='t-dim'>{result.bot_outputs.reasoning_notes}</span></div>", unsafe_allow_html=True)

    with st.expander("Raw completion payload"):
        st.markdown(f"<div class='term-block'><span class='t-val' style='white-space:pre-wrap;'>{result.bot_outputs.raw_completion[:2000]}</span></div>", unsafe_allow_html=True)


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
    import json as _json
    raw = _json.dumps(asdict(result), indent=2, ensure_ascii=False)
    st.markdown(f"""
    <div class="term-block" style="max-height:500px;overflow-y:auto;">
      <span class="t-dim">// raw ValidatorResult payload</span><br>
      <pre style="margin:0;color:#94A3B8;font-size:11px;white-space:pre-wrap;word-break:break-all;">{raw}</pre>
    </div>
    """, unsafe_allow_html=True)


# NEW: history renderer
def _render_history_panel(history: List[ConversationTurn]) -> None:
    if not history:
        st.markdown("""
        <div class="term-block">
          <span class="t-warn">// no runs recorded in this session</span>
        </div>
        """, unsafe_allow_html=True)
        return

    rows_html = ""
    for i, t in enumerate(reversed(history)):
        ts = time.strftime("%H:%M:%S", time.localtime(t.timestamp))
        q = (t.query[:80] + "...") if len(t.query) > 80 else t.query
        a = (t.answer[:80] + "...") if len(t.answer) > 80 else t.answer
        rows_html += f"""
        <div style="padding:6px 0;border-bottom:1px solid rgba(16,185,129,0.06);">
          <span class="t-dim">[{ts}]</span>
          <span class="t-info" style="margin:0 8px;">#{i+1}</span>
          <span class="t-label">Q:</span> <span class="t-val">{q}</span><br>
          <span style="margin-left:40px;"><span class="t-ok">A:</span> <span class="t-dim">{a}</span></span>
        </div>"""

    st.markdown(f"""
    <div class="term-block">
      <span class="t-ok">Q&amp;A HISTORY</span>
      <span class="t-dim"> ({len(history)} runs this session)</span>
      {rows_html}
    </div>
    """, unsafe_allow_html=True)

    for idx, t in enumerate(reversed(history), start=1):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t.timestamp))
        with st.expander(f"[{ts}] Run #{idx}"):
            st.markdown(f"<div class='term-block'><span class='t-label'>QUERY</span><br><span class='t-val'>{t.query}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='term-block' style='margin-top:6px;'><span class='t-ok'>ANSWER</span><br><span class='t-val' style='white-space:pre-wrap;'>{t.answer}</span></div>", unsafe_allow_html=True)


# =========================
# MAIN ENTRYPOINT
# =========================

def run_validator_page() -> None:
    inject_global_css()

    # Terminal-specific CSS
    st.markdown("""
    <style>
    /* Terminal panel */
    .term-block {
        background: #030712;
        border: 1px solid rgba(16,185,129,0.2);
        border-radius: 10px;
        padding: 16px 20px;
        font-family: ‘Courier New’, ‘Consolas’, monospace;
        font-size: 12px;
        line-height: 1.7;
        overflow-x: auto;
    }
    .term-block .t-ok     { color: #10B981; }
    .term-block .t-err    { color: #EF4444; }
    .term-block .t-warn   { color: #F59E0B; }
    .term-block .t-info   { color: #00D4FF; }
    .term-block .t-dim    { color: #374151; }
    .term-block .t-label  { color: #6EE7B7; font-weight: bold; }
    .term-block .t-val    { color: #E2E8F0; }
    .term-block .t-key    { color: #7DD3FC; }
    .term-prompt::before  { content: ‘$ ‘; color: #10B981; }

    /* Override tab styling for terminal theme */
    .validator-page .stTabs [data-baseweb="tab"] {
        font-family: ‘Courier New’, monospace !important;
        font-size: 12px !important;
        letter-spacing: 0.5px !important;
    }

    /* Dataframe monospace */
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
        font-family: ‘Courier New’, monospace !important;
        font-size: 12px !important;
    }
    </style>
    <div class="validator-page">
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style="padding:28px 0 20px;animation:fadeIn 0.5s ease;">
      <div style="display:inline-block;background:rgba(16,185,129,0.08);
                  border:1px solid rgba(16,185,129,0.2);border-radius:100px;
                  padding:4px 14px;font-size:11px;color:#10B981;
                  font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;
                  font-family:’Courier New’,monospace;">DEV CONSOLE</div>
      <h1 style="font-family:’Space Grotesk’,sans-serif;font-size:30px;font-weight:700;
                 color:#F1F5F9;margin:0 0 6px;">Validator Console</h1>
      <p style="font-size:13px;color:#475569;margin:0;font-family:’Courier New’,monospace;">
        // pipeline introspection: retrieval · routing · safety · bot outputs · Q&amp;A history
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("""
    <div style="font-family:’Courier New’,monospace;font-size:12px;font-weight:600;
                color:#10B981;text-transform:uppercase;letter-spacing:1.5px;
                padding:8px 0 12px;border-bottom:1px solid rgba(16,185,129,0.15);margin-bottom:16px;">
      &gt; Controls
    </div>
    """, unsafe_allow_html=True)

    default_query = (
        "Is it safe to increase my blood pressure medication dose "
        "if I still have headaches?"
    )
    user_query = st.sidebar.text_area(
        "Query",
        value=default_query,
        height=120,
    )

    top_k = st.sidebar.slider(
        "Top-K documents", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1
    )

    reuse_last = st.sidebar.checkbox(
        "Reuse last result", value=False
    )

    run_btn = st.sidebar.button(">> Run Validation", type="primary", use_container_width=True)

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
        st.markdown("""
        <div class="term-block">
          <span class="t-ok">mediexplain-validator</span><span class="t-dim"> v1.0 ready</span><br>
          <span class="t-dim">────────────────────────────────────────</span><br>
          <span class="t-warn">⚠</span> <span class="t-val">No query run yet.</span><br>
          <span class="t-dim">&gt; Enter a query in the sidebar and click</span> <span class="t-info">&gt;&gt; Run Validation</span>
        </div>
        """, unsafe_allow_html=True)
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
