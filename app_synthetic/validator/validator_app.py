import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

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


# =========================
# DEMO PIPELINE OUTPUT
# =========================

def _demo_result(user_query: str, top_k: int) -> ValidatorResult:
    """Temporary mock implementation so the Validator UI works
    even before the real pipeline is wired up.
    """

    # --- Retrieval demo ---
    chunks = [
        RetrievedChunk(
            rank=i + 1,
            score=0.95 - i * 0.05,
            source="NIH — Hypertension Guidelines (2023)",
            doc_id=f"nih_htn_{i+1}",
            snippet=(
                "Hypertension is defined as a sustained blood pressure "
                "≥130/80 mmHg in adults, confirmed on at least two separate visits."
            ),
        )
        for i in range(top_k)
    ]

    retrieval = RetrievalDiagnostics(
        latency_ms=115.3,
        top_k=top_k,
        num_returned=len(chunks),
        index_name="faiss_medical_vector_index",
        strategy="hybrid (BM25 + embeddings)",
        chunks=chunks,
    )

    # --- Routing demo ---
    routing = RoutingDiagnostics(
        detected_intent="medication_question",
        selected_bot="MedicationBot",
        confidence=0.89,
        trace=[
            RoutingTraceStep(
                step="Pre-processing",
                detail="Lowercased text, removed PHI-like patterns, normalized units.",
                meta={"tokens": 42},
            ),
            RoutingTraceStep(
                step="Intent classification",
                detail="Model predicted 'medication_question' with 0.89 confidence.",
                meta={"model": "router-small-medical-v1"},
            ),
            RoutingTraceStep(
                step="Bot selection",
                detail="Selected MedicationBot as primary responder.",
                meta={"candidate_bots": ["MedicationBot", "ExplainerBot"]},
            ),
        ],
    )

    # --- Safety demo ---
    safety = SafetyDiagnostics(
        decision="transform",
        policy_flags=["medical_advice", "dosage_request"],
        notes=(
            "User asked about changing anti-hypertensive dose. "
            "Answer reframed to educational guidance with clear disclaimer; "
            "no direct dosage instructions provided."
        ),
    )

    # --- Bot output demo ---
    final_answer = (
        "It sounds like you are asking whether it is safe to change the dose of your "
        "blood pressure medication. Only a clinician who knows your full medical history "
        "and all of your medications can safely make that decision. Sudden dose changes "
        "may cause your blood pressure to become too high or too low, which can lead to "
        "symptoms such as dizziness, fainting or chest pain.\n\n"
        "Please contact your doctor or local clinic before adjusting the medication yourself. "
        "If you experience severe chest pain, trouble breathing, or feel like you might faint, "
        "seek emergency care immediately."
    )

    bot_outputs = BotOutputs(
        final_answer=final_answer,
        model_name="gpt-5.1-medical-tuned",
        temperature=0.2,
        raw_completion=final_answer,
        reasoning_notes=(
            "Combined retrieved hypertension guideline snippets with safety policy rules "
            "to produce a non-prescriptive, education-only answer and clear escalation advice."
        ),
    )

    # --- Synthetic patient demo ---
    synthetic = SyntheticPatientSnapshot(
        patient_id="syn_demo_001",
        demographics={
            "age": 62,
            "sex": "Female",
            "race": "Synthetic",
            "insurance": "Medicare (synthetic)",
        },
        vitals={
            "bp_latest": "148/92 mmHg",
            "heart_rate": "78 bpm",
            "resp_rate": "16 /min",
            "temperature": "36.8 °C",
        },
        labs={
            "creatinine": "0.9 mg/dL",
            "eGFR": "78 mL/min/1.73m²",
            "HbA1c": "6.4 %",
        },
        medications={
            "antihypertensives": [
                {"name": "Lisinopril", "dose": "20 mg daily"},
                {"name": "Amlodipine", "dose": "5 mg daily"},
            ],
            "lipid_lowering": [{"name": "Atorvastatin", "dose": "20 mg nightly"}],
        },
        clinical_notes={
            "summary": (
                "Lives alone in an apartment. Limited social support, relies on phone "
                "check-ins with niece. Mild difficulty understanding medication labels; "
                "occasionally misses evening doses."
            ),
            "recent_visit": (
                "Primary concern: elevated home BP readings despite perceived adherence."
            ),
        },
    )

    return ValidatorResult(
        query=user_query,
        timestamp=time.time(),
        retrieval=retrieval,
        routing=routing,
        safety=safety,
        bot_outputs=bot_outputs,
        synthetic_patient=synthetic,
    )


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


# =========================
# MAIN ENTRYPOINT
# =========================

def run_validator_page() -> None:
    st.title("🩺 MediExplain – Validator Console")
    st.caption(
        "Developer-facing console for diagnostics on retrieval, routing, "
        "safety, final bot output and synthetic patient context."
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

    if "validator_last_result" not in st.session_state:
        st.session_state.validator_last_result = None

    if run_btn or (
        not reuse_last and st.session_state.validator_last_result is None
    ):
        if not user_query.strip():
            st.warning("Please enter a user query first.")
            return

        with st.spinner("Running mock MediExplain pipeline…"):
            result = _demo_result(user_query.strip(), top_k=top_k)
        st.session_state.validator_last_result = result

    result: Optional[ValidatorResult] = st.session_state.validator_last_result

    if result is None:
        st.info("Enter a query in the sidebar and click **Run validation**.")
        return

    tabs = st.tabs(
        [
            "Overview",
            "Retrieval",
            "Routing",
            "Bot outputs",
            "Safety",
            "Synthetic patient",
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
    with json_tab:
        _render_raw_json_panel(result)


# When Streamlit runs this file as a page via st.Page, __name__ == "__main__".
if __name__ == "__main__":
    run_validator_page()
