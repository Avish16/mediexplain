# ============================================================
# SYNTHETIC PATIENT REPORT – PIPELINE UI
# app_synthetic/synthetic_app.py
# ============================================================
import os
import sys
import time
import json
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from styles import inject_global_css

from core.synthetic_demographics import generate_demographics_llm
from core.diagnosis_bot import generate_diagnosis_llm
from core.timeline_bot import generate_timeline_llm
from core.lab_bot import generate_lab_report_llm
from core.vitals_bot import generate_vitals_llm
from core.radiology_bot import generate_radiology_studies_llm
from core.procedure_bot import generate_procedures_llm
from core.pathology_bot import generate_pathology_report_llm
from core.medication_bot import generate_medication_plan_llm
from core.nursing_notes_bot import generate_nursing_notes_llm
from core.clinical_notes_bot import generate_clinical_notes_llm
from core.prescription_bot import generate_prescriptions_llm
from core.billing_bot import generate_billing_summary_llm
from core.consolidator_bot import consolidate_patient_record
from core.safety_labeler_bot import label_safety_llm
from core.consistency_checker_bot import check_consistency_llm
from core.renderer_bot import render_patient_record
from core.composer_bot import compose_final_document
from core.pdf_generator import generate_pdf

inject_global_css()

# ── PIPELINE STEP DEFINITIONS ────────────────────────────────────
PIPELINE_STEPS = [
    {"name": "Demographics Bot",    "icon": "👤"},
    {"name": "Diagnosis Bot",       "icon": "🔬"},
    {"name": "Timeline Bot",        "icon": "📅"},
    {"name": "Lab Bot",             "icon": "🧪"},
    {"name": "Vitals Bot",          "icon": "💓"},
    {"name": "Radiology Bot",       "icon": "🩻"},
    {"name": "Procedure Bot",       "icon": "🩹"},
    {"name": "Pathology Bot",       "icon": "🔭"},
    {"name": "Medication Bot",      "icon": "💊"},
    {"name": "Nursing Notes Bot",   "icon": "📋"},
    {"name": "Clinical Notes Bot",  "icon": "📝"},
    {"name": "Prescription Bot",    "icon": "📄"},
    {"name": "Billing Bot",         "icon": "💰"},
    {"name": "Consolidator Bot",    "icon": "🗂"},
    {"name": "Safety Labeler",      "icon": "🛡"},
    {"name": "Consistency Checker", "icon": "✔"},
    {"name": "Renderer Bot",        "icon": "🎨"},
    {"name": "Composer Bot",        "icon": "📖"},
    {"name": "PDF Generator",       "icon": "📑"},
]


# ── NATIVE PIPELINE RENDERER ─────────────────────────────────────
def _render_pipeline(placeholder, steps, latest_name="", latest_output=None):
    """Render the pipeline using native Streamlit components only."""
    done    = sum(1 for s in steps if s["status"] in ("done", "skipped"))
    running = sum(1 for s in steps if s["status"] == "running")
    total   = len(steps)

    with placeholder.container():
        st.markdown(
            "<p style='color:#00D4FF; font-size:11px; font-weight:600;"
            " letter-spacing:2px; text-transform:uppercase; margin:0 0 6px;'>"
            "AI PIPELINE</p>",
            unsafe_allow_html=True,
        )
        progress_val = (done + running * 0.5) / total if total else 0
        st.progress(min(progress_val, 1.0), text=f"{done}/{total} complete")

        for step in steps:
            status = step["status"]
            name   = step["name"]
            t      = f"`{step['time']:.1f}s`" if step.get("time") else ""

            if status == "done":
                st.markdown(f"✅ &nbsp; **{name}** &nbsp; {t}", unsafe_allow_html=True)
            elif status == "running":
                st.markdown(
                    f"<span style='color:#00D4FF'>⚡ &nbsp; **{name}** — running...</span>",
                    unsafe_allow_html=True,
                )
            elif status == "error":
                st.markdown(
                    f"<span style='color:#EF4444'>❌ &nbsp; **{name}** — FAILED</span>",
                    unsafe_allow_html=True,
                )
            elif status == "skipped":
                st.markdown(
                    f"<span style='color:#F59E0B'>⏭ &nbsp; ~~{name}~~ — skipped</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span style='color:#475569; font-size:13px;'>○ &nbsp; {name}</span>",
                    unsafe_allow_html=True,
                )

        # Output preview for latest completed step
        if latest_name and latest_output is not None:
            st.divider()
            st.caption(f"Latest output: **{latest_name}**")
            if isinstance(latest_output, str):
                preview = latest_output
            else:
                try:
                    preview = json.dumps(latest_output, indent=2, ensure_ascii=False)
                except Exception:
                    preview = str(latest_output)
            st.code(preview[:600] + ("..." if len(preview) > 600 else ""), language=None)


def _show_summary(placeholder, total_time, done, total, errors):
    """Show post-pipeline summary using native Streamlit components."""
    with placeholder.container():
        if errors == 0:
            st.success(f"**Pipeline Complete** — {done}/{total} bots · {total_time:.1f}s total")
        else:
            st.warning(
                f"**Pipeline finished with {errors} error(s)** — {done}/{total} bots · {total_time:.1f}s"
            )
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Time", f"{total_time:.1f}s")
        with m2:
            st.metric("Bots Run", f"{done}/{total}")
        with m3:
            st.metric("Status", "All Clear" if errors == 0 else f"{errors} Error(s)")


# ── SESSION STATE ─────────────────────────────────────────────────
for key, default in [
    ("pipeline_steps", [
        {"name": s["name"], "icon": s["icon"], "status": "pending", "time": None, "output": None}
        for s in PIPELINE_STEPS
    ]),
    ("pipeline_done", False),
    ("pipeline_summary", None),
    ("output_pdf_path", "synthetic_patient_report.pdf"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── SIDEBAR ───────────────────────────────────────────────────────
st.sidebar.markdown(
    "<p style='color:#00D4FF; font-size:12px; font-weight:600; letter-spacing:2px;"
    " text-transform:uppercase; border-bottom:1px solid rgba(0,212,255,0.1);"
    " padding-bottom:8px; margin-bottom:12px;'>Patient Configuration</p>",
    unsafe_allow_html=True,
)
age       = st.sidebar.number_input("Patient Age", min_value=1, max_value=110, value=45)
gender    = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"], index=0)
logo_path = st.sidebar.text_input("Hospital Logo Path (optional)", value="assets/hospital_logo.png")

st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox("Single-bot debug mode")

if debug_mode:
    bot_to_run = st.sidebar.selectbox("Choose bot:", [
        "Demographics", "Diagnosis", "Timeline", "Lab", "Vitals",
        "Radiology", "Procedures", "Pathology", "Medications",
        "Nursing Notes", "Clinical Notes", "Prescriptions", "Billing",
    ])
    if st.sidebar.button("Run Selected Bot"):
        st.write(f"### Debug: {bot_to_run}")
        dummy_dx = {"primary_diagnosis": "Test Condition", "icd10_code": "T00.00"}
        dummy_tl = {"timeline_table": []}
        try:
            if bot_to_run == "Demographics":
                st.write(generate_demographics_llm(age, gender))
            elif bot_to_run == "Diagnosis":
                st.write(generate_diagnosis_llm(age, gender))
            elif bot_to_run == "Timeline":
                st.write(generate_timeline_llm(age, gender, dummy_dx))
            elif bot_to_run == "Lab":
                st.write(generate_lab_report_llm(age, gender, dummy_dx, dummy_tl))
            elif bot_to_run == "Vitals":
                st.write(generate_vitals_llm(age, gender, dummy_dx, dummy_tl))
            elif bot_to_run == "Radiology":
                st.write(generate_radiology_studies_llm(age, gender, dummy_dx, dummy_tl))
            elif bot_to_run == "Procedures":
                st.write(generate_procedures_llm(age, gender, dummy_dx, dummy_tl, {}, {}))
            elif bot_to_run == "Pathology":
                st.write(generate_pathology_report_llm(age, gender, dummy_dx, {}, {}, {}))
            elif bot_to_run == "Medications":
                st.write(generate_medication_plan_llm(age, gender, dummy_dx, dummy_tl, {}, {}))
            elif bot_to_run == "Nursing Notes":
                st.write(generate_nursing_notes_llm(age, gender, {}, dummy_dx, {}, {}, dummy_tl))
            elif bot_to_run == "Clinical Notes":
                st.write(generate_clinical_notes_llm(age, gender, {}, dummy_dx, dummy_tl, {}, {}, {}))
            elif bot_to_run == "Prescriptions":
                st.write(generate_prescriptions_llm(age, gender, dummy_dx, {}, {}, {}))
            elif bot_to_run == "Billing":
                st.write(generate_billing_summary_llm(age, gender, {}, dummy_dx, {}, {}, {}, {}))
        except Exception as e:
            st.error(f"Bot failed: {e}")
    st.stop()

# ── PAGE HEADER ───────────────────────────────────────────────────
st.markdown(
    "<p style='color:#00D4FF; font-size:11px; font-weight:600; letter-spacing:2px;"
    " text-transform:uppercase; margin-top:16px;'>19-BOT PIPELINE</p>",
    unsafe_allow_html=True,
)
st.title("Synthetic Patient Generator")
st.caption(
    "Generate a complete, clinically realistic medical record — "
    "demographics through discharge PDF."
)

# ── GENERATE BUTTON ───────────────────────────────────────────────
gen_col, _ = st.columns([3, 7])
with gen_col:
    generate_clicked = st.button(
        "⚡  Generate Full Synthetic Case",
        type="primary",
        use_container_width=True,
    )

# ── PIPELINE LAYOUT ───────────────────────────────────────────────
left_col, right_col = st.columns([4, 6], gap="large")

with left_col:
    pipeline_placeholder = st.empty()

with right_col:
    status_placeholder  = st.empty()
    summary_placeholder = st.empty()
    download_placeholder = st.empty()

# Render current saved state on every page load
steps = st.session_state.pipeline_steps
_render_pipeline(pipeline_placeholder, steps)

if st.session_state.pipeline_done and st.session_state.pipeline_summary:
    s = st.session_state.pipeline_summary
    _show_summary(summary_placeholder, s["total_time"], s["done"], s["total"], s["errors"])
    if os.path.exists(st.session_state.output_pdf_path):
        with open(st.session_state.output_pdf_path, "rb") as f:
            download_placeholder.download_button(
                label="⬇️  Download Synthetic Medical Record PDF",
                data=f,
                file_name="synthetic_patient_record.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# ── PIPELINE EXECUTION ────────────────────────────────────────────
if generate_clicked:
    st.session_state.pipeline_done    = False
    st.session_state.pipeline_summary = None
    steps = [
        {"name": s["name"], "icon": s["icon"], "status": "pending", "time": None, "output": None}
        for s in PIPELINE_STEPS
    ]
    st.session_state.pipeline_steps = steps

    pipeline_start = time.time()
    # Mutable container avoids nonlocal (plain if-block is not a function scope)
    ctx = {"errors": 0, "latest_name": "", "latest_output": None}

    def _set(idx, status, result=None, elapsed=None):
        steps[idx]["status"] = status
        steps[idx]["time"]   = elapsed
        steps[idx]["output"] = result

    def _render(idx=None):
        if idx is not None and steps[idx].get("output") is not None:
            ctx["latest_name"]   = steps[idx]["name"]
            ctx["latest_output"] = steps[idx]["output"]
        _render_pipeline(pipeline_placeholder, steps, ctx["latest_name"], ctx["latest_output"])

    def run(idx, fn, *args):
        _set(idx, "running")
        _render()
        t0 = time.time()
        try:
            result = fn(*args)
            _set(idx, "done", result, time.time() - t0)
            _render(idx)
            return result
        except Exception as e:
            _set(idx, "error", None, time.time() - t0)
            _render()
            ctx["errors"] += 1
            status_placeholder.error(f"**{steps[idx]['name']} failed:** {e}")
            print(f"[ERROR] {steps[idx]['name']}:", e)
            raise

    try:
        # 0 — Demographics
        demographics = run(0, generate_demographics_llm, age, gender)

        # 1 — Diagnosis
        diagnosis = run(1, generate_diagnosis_llm, age, gender)
        if isinstance(diagnosis, str):
            diagnosis = {"primary_diagnosis": diagnosis, "icd10_code": "", "snomed_code": ""}

        # 2 — Timeline
        timeline = run(2, generate_timeline_llm, age, gender, diagnosis)
        if isinstance(timeline, str):
            import re
            from datetime import datetime
            summary = (timeline.split("TIMELINE SUMMARY:")[1].split("TIMELINE TABLE:")[0].strip()
                       if "TIMELINE SUMMARY:" in timeline else timeline[:200])
            blocks = re.split(r"\n(?=\d+\.)", timeline)
            events = [
                {"date": datetime.now().strftime("%Y-%m-%d"), "event_type": "Event",
                 "description": b.split("\n")[0].strip()}
                for b in blocks if b.strip()
            ]
            timeline = {"timeline_summary": summary, "timeline_table": events}

        # 3 — Labs
        labs = run(3, generate_lab_report_llm, age, gender, diagnosis, timeline)

        # 4 — Vitals
        vitals = run(4, generate_vitals_llm, age, gender, diagnosis, timeline)

        # 5 — Radiology (skipped)
        _set(5, "skipped", {}, 0.0)
        _render()
        radiology            = {}
        radiology_image_urls = []
        status_placeholder.info("Radiology Bot is paused — imaging generation skipped.")

        # 6 — Procedures
        procedures = run(6, generate_procedures_llm, age, gender, diagnosis, timeline, labs, radiology)

        # 7 — Pathology
        pathology = run(7, generate_pathology_report_llm, age, gender, diagnosis, procedures, radiology, labs)

        # 8 — Medications
        medications = run(8, generate_medication_plan_llm, age, gender, diagnosis, timeline, labs, vitals)

        # 9 — Nursing Notes
        nursing_notes = run(9, generate_nursing_notes_llm,
                            age, gender, demographics, diagnosis, vitals, labs, timeline)

        # 10 — Clinical Notes
        clinical_notes = run(10, generate_clinical_notes_llm,
                             age, gender, demographics, diagnosis, timeline, labs, vitals, radiology)

        # 11 — Prescriptions
        prescriptions = run(11, generate_prescriptions_llm,
                            age, gender, diagnosis, medications, vitals, labs)

        # 12 — Billing
        billing = run(12, generate_billing_summary_llm,
                      age, gender, demographics, diagnosis, procedures, labs, radiology, medications)

        # 13 — Consolidator
        patient_record = run(13, consolidate_patient_record,
                             demographics, diagnosis, timeline, labs, vitals, radiology,
                             procedures, pathology, clinical_notes, nursing_notes,
                             medications, prescriptions, billing)

        # 14 — Safety Labels
        safety_labels = run(14, label_safety_llm, patient_record)

        # 15 — Consistency
        consistency = run(15, check_consistency_llm, patient_record)

        # 16 — Renderer
        rendered_text = run(16, render_patient_record, patient_record, safety_labels, consistency)

        # 17 — Composer
        final_text = run(17, compose_final_document, rendered_text)

        # 18 — PDF
        _set(18, "running")
        _render()
        t0       = time.time()
        logo_arg = logo_path if logo_path and os.path.exists(logo_path) else None
        generate_pdf(
            report_text=final_text,
            radiology_images=radiology_image_urls,
            output_file=st.session_state.output_pdf_path,
            logo_path=logo_arg,
        )
        _set(18, "done", st.session_state.output_pdf_path, time.time() - t0)
        _render(18)

        # ── Summary ──────────────────────────────────────────────
        total_time = time.time() - pipeline_start
        done_count = sum(1 for s in steps if s["status"] == "done")
        st.session_state.pipeline_done    = True
        st.session_state.pipeline_summary = {
            "total_time": total_time,
            "done":       done_count,
            "total":      len(steps),
            "errors":     ctx["errors"],
        }
        st.session_state.pipeline_steps = steps

        status_placeholder.empty()
        _show_summary(summary_placeholder, total_time, done_count, len(steps), ctx["errors"])

        if os.path.exists(st.session_state.output_pdf_path):
            with open(st.session_state.output_pdf_path, "rb") as f:
                download_placeholder.download_button(
                    label="⬇️  Download Synthetic Medical Record PDF",
                    data=f,
                    file_name="synthetic_patient_record.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    except Exception as e:
        st.session_state.pipeline_steps = steps
        status_placeholder.error(f"Pipeline aborted: {e}")
        print("[FATAL] Pipeline aborted:", e)
