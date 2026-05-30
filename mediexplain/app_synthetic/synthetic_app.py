# ============================================================
# SYNTHETIC PATIENT REPORT – PIPELINE UI
# app_synthetic/synthetic_app.py
# ============================================================
import os
import sys
import time
import json
import traceback
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

# ── Extra page-level styles ──────────────────────────────────────
st.markdown("""
<style>
.generate-btn > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,102,255,0.2)) !important;
    border: 1px solid rgba(0,212,255,0.5) !important;
    color: #00D4FF !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 14px 32px !important;
    border-radius: 10px !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
}
.generate-btn > button:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.35), rgba(0,102,255,0.35)) !important;
    border-color: #00D4FF !important;
    box-shadow: 0 0 30px rgba(0,212,255,0.3), 0 4px 20px rgba(0,212,255,0.15) !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── PIPELINE STEP DEFINITIONS ────────────────────────────────────
PIPELINE_STEPS = [
    {"name": "Demographics Bot",     "icon": "👤"},
    {"name": "Diagnosis Bot",        "icon": "🔬"},
    {"name": "Timeline Bot",         "icon": "📅"},
    {"name": "Lab Bot",              "icon": "🧪"},
    {"name": "Vitals Bot",           "icon": "💓"},
    {"name": "Radiology Bot",        "icon": "🩻"},
    {"name": "Procedure Bot",        "icon": "🩹"},
    {"name": "Pathology Bot",        "icon": "🔭"},
    {"name": "Medication Bot",       "icon": "💊"},
    {"name": "Nursing Notes Bot",    "icon": "📋"},
    {"name": "Clinical Notes Bot",   "icon": "📝"},
    {"name": "Prescription Bot",     "icon": "📄"},
    {"name": "Billing Bot",          "icon": "💰"},
    {"name": "Consolidator Bot",     "icon": "🗂"},
    {"name": "Safety Labeler",       "icon": "🛡"},
    {"name": "Consistency Checker",  "icon": "✔"},
    {"name": "Renderer Bot",         "icon": "🎨"},
    {"name": "Composer Bot",         "icon": "📖"},
    {"name": "PDF Generator",        "icon": "📑"},
]

_S = {
    "pending": {"color": "#334155", "label": "PENDING", "border": "#334155", "text": "#475569"},
    "running": {"color": "#00D4FF", "label": "RUNNING", "border": "#00D4FF", "text": "#00D4FF"},
    "done":    {"color": "#10B981", "label": "DONE",    "border": "#10B981", "text": "#10B981"},
    "error":   {"color": "#EF4444", "label": "ERROR",   "border": "#EF4444", "text": "#EF4444"},
    "skipped": {"color": "#F59E0B", "label": "SKIP",    "border": "#F59E0B", "text": "#F59E0B"},
}


def _fmt_output(output) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        text = output
    else:
        try:
            text = json.dumps(output, indent=2, ensure_ascii=False)
        except Exception:
            text = str(output)
    return text[:800] + ("\n\n[...truncated]" if len(text) > 800 else "")


def render_pipeline_html(steps: list, latest_name: str = "", latest_output=None) -> str:
    rows = []
    for step in steps:
        status = step.get("status", "pending")
        s = _S[status]
        t = f"{step['time']:.1f}s" if step.get("time") else ""
        opacity = "0.3" if status == "pending" else "1.0"

        if status == "done":
            dot_ch = "✓"
        elif status == "error":
            dot_ch = "✗"
        elif status == "running":
            dot_ch = "⚡"
        elif status == "skipped":
            dot_ch = "—"
        else:
            dot_ch = step["icon"]

        glow = f"box-shadow:0 0 12px {s['color']}80;" if status == "running" else ""
        row = f"""
        <div style="display:flex;align-items:center;gap:10px;padding:5px 0;
                    opacity:{opacity};transition:all 0.3s ease;">
          <div style="width:28px;height:28px;border-radius:50%;border:2px solid {s['border']};
                      background:#0A0F1E;display:flex;align-items:center;justify-content:center;
                      font-size:11px;flex-shrink:0;color:{s['color']};{glow}font-weight:700;">
            {dot_ch}
          </div>
          <div style="flex:1;min-width:0;">
            <div style="font-size:12px;color:{'#E2E8F0' if status != 'pending' else '#475569'};
                        font-family:'Inter',sans-serif;font-weight:500;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
              {step['name']}
            </div>
            {f'<div style="font-size:10px;color:{s["text"]};opacity:0.8;">{t}</div>' if t else ''}
          </div>
          <span style="font-size:10px;color:{s['color']};background:{s['color']}18;
                       padding:2px 7px;border-radius:100px;font-weight:600;
                       letter-spacing:0.5px;flex-shrink:0;white-space:nowrap;">
            {s['label']}
          </span>
        </div>"""
        rows.append(row)

    preview_html = ""
    if latest_name:
        out_text = _fmt_output(latest_output)
        preview_html = f"""
        <div style="margin-top:16px;border-top:1px solid rgba(0,212,255,0.1);padding-top:14px;">
          <div style="font-size:11px;color:#00D4FF;font-weight:600;
                      letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">
            Latest: {latest_name}
          </div>
          <pre style="font-family:'Courier New',monospace;font-size:10px;color:#64748B;
                      white-space:pre-wrap;word-break:break-all;margin:0;
                      max-height:180px;overflow:hidden;line-height:1.5;">{out_text}</pre>
        </div>"""

    return f"""
    <style>
    @keyframes pulse-run {{
      0%,100% {{ box-shadow: 0 0 6px rgba(0,212,255,0.5); }}
      50%      {{ box-shadow: 0 0 18px rgba(0,212,255,1), 0 0 32px rgba(0,212,255,0.3); }}
    }}
    </style>
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(0,212,255,0.12);
                border-radius:14px;padding:18px 16px;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:11px;font-weight:700;
                  color:#00D4FF;text-transform:uppercase;letter-spacing:2px;
                  margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid rgba(0,212,255,0.1);">
        AI Pipeline
      </div>
      <div style="position:relative;">
        <div style="position:absolute;left:13px;top:14px;bottom:14px;width:1px;
                    background:linear-gradient(to bottom,rgba(0,212,255,0.35),rgba(0,212,255,0.04));"></div>
        {"".join(rows)}
      </div>
      {preview_html}
    </div>"""


def render_summary_card(total_time: float, done: int, total: int, errors: int) -> str:
    ok = errors == 0
    status_color = "#10B981" if ok else "#EF4444"
    status_label = "All Systems Go" if ok else f"{errors} Error(s)"
    return f"""
    <div style="background:linear-gradient(135deg,rgba(16,185,129,0.06),rgba(0,212,255,0.04));
                border:1px solid rgba(16,185,129,0.3);border-radius:16px;
                padding:28px 32px;text-align:center;margin:20px 0;animation:fadeIn 0.5s ease;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:700;
                  color:#F1F5F9;margin-bottom:6px;">Pipeline Complete</div>
      <div style="color:#64748B;font-size:14px;margin-bottom:24px;font-family:'Inter',sans-serif;">
        Full synthetic medical record generated
      </div>
      <div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;">
        <div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:32px;font-weight:700;
                      color:#00D4FF;">{total_time:.1f}s</div>
          <div style="font-size:12px;color:#475569;margin-top:3px;">Total Time</div>
        </div>
        <div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:32px;font-weight:700;
                      color:#00D4FF;">{done}/{total}</div>
          <div style="font-size:12px;color:#475569;margin-top:3px;">Bots Run</div>
        </div>
        <div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:32px;font-weight:700;
                      color:{status_color};">{status_label}</div>
          <div style="font-size:12px;color:#475569;margin-top:3px;">Status</div>
        </div>
      </div>
    </div>"""


# ── SESSION STATE INIT ───────────────────────────────────────────
if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = [
        {"name": s["name"], "icon": s["icon"], "status": "pending", "time": None, "output": None}
        for s in PIPELINE_STEPS
    ]
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "pipeline_summary" not in st.session_state:
    st.session_state.pipeline_summary = None
if "output_pdf_path" not in st.session_state:
    st.session_state.output_pdf_path = "synthetic_patient_report.pdf"

# ── SIDEBAR ──────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;
            color:#00D4FF;text-transform:uppercase;letter-spacing:1.5px;
            padding:8px 0 12px;border-bottom:1px solid rgba(0,212,255,0.1);margin-bottom:16px;">
  Patient Configuration
</div>
""", unsafe_allow_html=True)

age = st.sidebar.number_input("Patient Age", min_value=1, max_value=110, value=45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"], index=0)
logo_path = st.sidebar.text_input("Hospital Logo Path (optional)", value="assets/hospital_logo.png")

st.sidebar.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:600;
            color:#64748B;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">
  Debug Tools
</div>
""", unsafe_allow_html=True)

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

# ── PAGE HEADER ──────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 24px; animation:fadeIn 0.5s ease;">
  <div style="display:inline-block;background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);
              border-radius:100px;padding:4px 14px;font-size:11px;color:#00D4FF;
              font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;">
    19-Bot Pipeline
  </div>
  <h1 style="font-family:'Space Grotesk',sans-serif;font-size:36px;font-weight:700;
             color:#F1F5F9;margin:0 0 8px;">Synthetic Patient Generator</h1>
  <p style="font-size:15px;color:#64748B;margin:0;font-family:'Inter',sans-serif;">
    Generate a complete, clinically realistic medical record from demographics through discharge PDF.
  </p>
</div>
""", unsafe_allow_html=True)

# ── GENERATE BUTTON ──────────────────────────────────────────────
gen_col, _ = st.columns([3, 7])
with gen_col:
    st.markdown('<div class="generate-btn">', unsafe_allow_html=True)
    generate_clicked = st.button("⚡  Generate Full Synthetic Case", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── PIPELINE LAYOUT ──────────────────────────────────────────────
left_col, right_col = st.columns([4, 6], gap="large")

with left_col:
    pipeline_placeholder = st.empty()

with right_col:
    status_placeholder = st.empty()
    summary_placeholder = st.empty()
    download_placeholder = st.empty()

# Render existing state (persistent after run)
steps = st.session_state.pipeline_steps
pipeline_placeholder.markdown(
    render_pipeline_html(steps), unsafe_allow_html=True
)

if st.session_state.pipeline_done and st.session_state.pipeline_summary:
    s = st.session_state.pipeline_summary
    summary_placeholder.markdown(
        render_summary_card(s["total_time"], s["done"], s["total"], s["errors"]),
        unsafe_allow_html=True,
    )
    if os.path.exists(st.session_state.output_pdf_path):
        with open(st.session_state.output_pdf_path, "rb") as f:
            download_placeholder.download_button(
                label="⬇️  Download Synthetic Medical Record PDF",
                data=f,
                file_name="synthetic_patient_record.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# ── PIPELINE EXECUTION ───────────────────────────────────────────
if generate_clicked:
    # Reset state
    st.session_state.pipeline_done = False
    st.session_state.pipeline_summary = None
    steps = [
        {"name": s["name"], "icon": s["icon"], "status": "pending", "time": None, "output": None}
        for s in PIPELINE_STEPS
    ]
    st.session_state.pipeline_steps = steps

    pipeline_start = time.time()
    errors = 0
    latest_name = ""
    latest_output = None

    def _set(idx, status, result=None, elapsed=None):
        steps[idx]["status"] = status
        steps[idx]["time"] = elapsed
        steps[idx]["output"] = result

    def _render(idx=None):
        nonlocal latest_name, latest_output
        if idx is not None and steps[idx].get("output") is not None:
            latest_name = steps[idx]["name"]
            latest_output = steps[idx]["output"]
        pipeline_placeholder.markdown(
            render_pipeline_html(steps, latest_name, latest_output),
            unsafe_allow_html=True,
        )

    def run(idx, fn, *args):
        nonlocal errors
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
            errors += 1
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
            summary = timeline.split("TIMELINE SUMMARY:")[1].split("TIMELINE TABLE:")[0].strip() \
                if "TIMELINE SUMMARY:" in timeline else timeline[:200]
            blocks = re.split(r"\n(?=\d+\.)", timeline)
            events = [{"date": datetime.now().strftime("%Y-%m-%d"), "event_type": "Event",
                       "description": b.split("\n")[0].strip()} for b in blocks if b.strip()]
            timeline = {"timeline_summary": summary, "timeline_table": events}

        # 3 — Labs
        labs = run(3, generate_lab_report_llm, age, gender, diagnosis, timeline)

        # 4 — Vitals
        vitals = run(4, generate_vitals_llm, age, gender, diagnosis, timeline)

        # 5 — Radiology (skipped)
        _set(5, "skipped", {}, 0.0)
        _render()
        radiology = {}
        radiology_image_urls = []
        status_placeholder.info("Radiology Bot is paused — imaging generation skipped.")

        # 6 — Procedures
        procedures = run(6, generate_procedures_llm, age, gender, diagnosis, timeline, labs, radiology)

        # 7 — Pathology
        pathology = run(7, generate_pathology_report_llm, age, gender, diagnosis, procedures, radiology, labs)

        # 8 — Medications
        medications = run(8, generate_medication_plan_llm, age, gender, diagnosis, timeline, labs, vitals)

        # 9 — Nursing Notes
        nursing_notes = run(9, generate_nursing_notes_llm, age, gender, demographics, diagnosis, vitals, labs, timeline)

        # 10 — Clinical Notes
        clinical_notes = run(10, generate_clinical_notes_llm, age, gender, demographics, diagnosis, timeline, labs, vitals, radiology)

        # 11 — Prescriptions
        prescriptions = run(11, generate_prescriptions_llm, age, gender, diagnosis, medications, vitals, labs)

        # 12 — Billing
        billing = run(12, generate_billing_summary_llm, age, gender, demographics, diagnosis, procedures, labs, radiology, medications)

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
        t0 = time.time()
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
        st.session_state.pipeline_done = True
        st.session_state.pipeline_summary = {
            "total_time": total_time, "done": done_count,
            "total": len(steps), "errors": errors,
        }
        st.session_state.pipeline_steps = steps

        status_placeholder.empty()
        summary_placeholder.markdown(
            render_summary_card(total_time, done_count, len(steps), errors),
            unsafe_allow_html=True,
        )
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
        total_time = time.time() - pipeline_start
        done_count = sum(1 for s in steps if s["status"] == "done")
        st.session_state.pipeline_steps = steps
        status_placeholder.error(f"Pipeline aborted: {e}")
        print("[FATAL] Pipeline aborted:", e)
