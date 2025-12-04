# ============================================================
# SYNTHETIC PATIENT REPORT – ONE CLICK WORKFLOW
# app_synthetic/synthetic_app.py
# ============================================================

import os
import sys
import traceback
import streamlit as st

# Make project root importable (for core.* modules)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------- Core bot imports (your modules) ----------
from core.synthetic_demographics import generate_demographics_llm
from core.diagnosisapp/rag_bot import generate_diagnosis_llm
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


# ============================================================
# Small helper to run each bot with clear labeling
# ============================================================

def run_step(label: str, fn, *args, **kwargs):
    """
    Run a single bot step with a clear label.
    Shows success / error in Streamlit and raises if failed.
    """
    with st.spinner(f"Running {label}..."):
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ {label} FAILED")
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            # Also print to console for Codespaces logs
            print(f"[ERROR] {label} failed:", e)
            raise
        else:
            st.success(f"✅ {label} completed")
            return result


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Synthetic Patient Report Generator", layout="wide")

st.title("🧬 Synthetic Patient Report Generator – One Click")

st.sidebar.header("Patient Inputs")
age = st.sidebar.number_input("Patient Age", min_value=1, max_value=110, value=45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"], index=0)

logo_path = st.sidebar.text_input(
    "Hospital Logo Path (optional)",
    value="assets/hospital_logo.png"
)

# ------------------------------------------------------------
# Debug mode: run bots individually
# ------------------------------------------------------------
st.sidebar.subheader("⚙️ Debug Mode")

debug_mode = st.sidebar.checkbox("Run single bot (debug mode)")

if debug_mode:
    bot_to_run = st.sidebar.selectbox(
        "Choose bot to run:",
        [
            "Demographics",
            "Diagnosis",
            "Timeline",
            "Lab",
            "Vitals",
            "Radiology",
            "Procedures",
            "Pathology",
            "Medications",
            "Nursing Notes",
            "Clinical Notes",
            "Prescriptions",
            "Billing",
        ],
    )

    if st.sidebar.button("🚀 Run Selected Bot"):
        st.write(f"### Debug Run: {bot_to_run}")

        # --- Create dummy inputs for dependent bots ---
        dummy_demo = {"age": age, "gender": gender}
        dummy_dx = {"primary_diagnosis": "Test Condition", "icd10_code": "T00.00"}
        dummy_timeline = {"timeline_table": []}
        dummy_labs = {}
        dummy_vitals = {}

        try:
            # NOTE: use st.write for *everything* so it works for dicts AND plain text
            if bot_to_run == "Demographics":
                st.write(generate_demographics_llm(age, gender))

            elif bot_to_run == "Diagnosis":
                st.write(generate_diagnosis_llm(age, gender))

            elif bot_to_run == "Timeline":
                st.write(generate_timeline_llm(age, gender, dummy_dx))

            elif bot_to_run == "Lab":
                st.write(generate_lab_report_llm(age, gender, dummy_dx, dummy_timeline))

            elif bot_to_run == "Vitals":
                st.write(generate_vitals_llm(age, gender, dummy_dx, dummy_timeline))

            elif bot_to_run == "Radiology":
                st.write(generate_radiology_studies_llm(age, gender, dummy_dx, dummy_timeline))

            elif bot_to_run == "Procedures":
                st.write(
                    generate_procedures_llm(
                        age, gender, dummy_dx, dummy_timeline, dummy_labs, {}
                    )
                )

            elif bot_to_run == "Pathology":
                st.write(
                    generate_pathology_report_llm(
                        age, gender, dummy_dx, {}, {}, {}
                    )
                )

            elif bot_to_run == "Medications":
                st.write(
                    generate_medication_plan_llm(
                        age, gender, dummy_dx, dummy_timeline, dummy_labs, dummy_vitals
                    )
                )

            elif bot_to_run == "Nursing Notes":
                st.write(
                    generate_nursing_notes_llm(
                        age,
                        gender,
                        dummy_demo,
                        dummy_dx,
                        dummy_vitals,
                        dummy_labs,
                        dummy_timeline,
                    )
                )

            elif bot_to_run == "Clinical Notes":
                st.write(
                    generate_clinical_notes_llm(
                        age,
                        gender,
                        dummy_demo,
                        dummy_dx,
                        dummy_timeline,
                        dummy_labs,
                        dummy_vitals,
                        {},
                    )
                )

            elif bot_to_run == "Prescriptions":
                st.write(
                    generate_prescriptions_llm(
                        age, gender, dummy_dx, {}, dummy_vitals, dummy_labs
                    )
                )

            elif bot_to_run == "Billing":
                st.write(
                    generate_billing_summary_llm(
                        age, gender, dummy_demo, dummy_dx, {}, {}, {}, {}
                    )
                )

        except Exception as e:
            st.error("❌ Bot failed")
            st.code(str(e))

output_pdf_path = "synthetic_patient_report.pdf"

st.markdown("---")

# ============================================================
# FULL PIPELINE
# ============================================================
if st.button("🚀 Generate FULL Synthetic Case"):
    try:
        # 1) DEMOGRAPHICS BOT
        demographics = run_step(
            "Demographics Bot",
            generate_demographics_llm,
            age,
            gender,
        )

        # 2) DIAGNOSIS BOT
        diagnosis = run_step(
            "Diagnosis Bot",
            generate_diagnosis_llm,
            age,
            gender,
        )

        # 3) TIMELINE BOT
        timeline = run_step(
            "Timeline Bot",
            generate_timeline_llm,
            age,
            gender,
            diagnosis,
        )
        
        # =============== FIX TIMELINE STRING → TIMELINE DICT ===============
        from datetime import datetime

        if isinstance(timeline, str):

            try:
                # Extract summary
                if "TIMELINE SUMMARY:" in timeline:
                    summary = timeline.split("TIMELINE SUMMARY:")[1]
                    summary = summary.split("TIMELINE TABLE:")[0].strip()
                else:
                    summary = timeline[:200]

                timeline_dict = {
                    "timeline_summary": summary,
                    "timeline_table": []
                }

                # Extract events by regex
                import re
                event_blocks = re.split(r"\n(?=\d+\.)", timeline)
                for block in event_blocks:
                    block = block.strip()
                    if not block:
                        continue
                    # first line "1. 2023-05-02 – ED Visit"
                    first_line = block.split("\n")[0]
                    timeline_dict["timeline_table"].append({
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "event_type": "Event",
                        "description": first_line
                    })

                timeline = timeline_dict

            except Exception as e:
                st.error("❌ Timeline → Dict conversion failed")
                st.code(str(e))
                raise e



        # 4) LAB BOT
        labs = run_step(
            "Lab Bot",
            generate_lab_report_llm,
            age,
            gender,
            diagnosis,
            timeline,
        )

        # 5) VITALS BOT
        vitals = run_step(
            "Vitals Bot",
            generate_vitals_llm,
            age,
            gender,
            diagnosis,
            timeline,
        )

        # 6) RADIOLOGY BOT (includes image generation)
        # radiology = 'NONE' 

        # ====================================================
        # 6) RADIOLOGY BOT (PAUSED)
        # ====================================================
        radiology = {}          # empty placeholder so downstream bots don't break
        radiology_image_urls = []   # no images for now
        st.info("🛑 Radiology Bot is currently paused – skipping imaging generation.")

        # run_step(
        #     "Radiology Bot",
        #     generate_radiology_studies_llm,
        #     age,
        #     gender,
        #     diagnosis,
        #     timeline,
        # )

        # # Collect URLs for PDF later (if present)
        # radiology_image_urls = []
        # if isinstance(radiology, dict):
        #     for study in radiology.get("studies", []):
        #         url = study.get("image_url")
        #         if url:
        #             radiology_image_urls.append(url)

        # 7) PROCEDURE BOT
        procedures = run_step(
            "Procedure Bot",
            generate_procedures_llm,
            age,
            gender,
            diagnosis,
            timeline,
            labs,
            radiology,
        )

        # 8) PATHOLOGY BOT
        pathology = run_step(
            "Pathology Bot",
            generate_pathology_report_llm,
            age,
            gender,
            diagnosis,
            procedures,
            radiology,
            labs,
        )

        # 9) MEDICATION BOT
        medications = run_step(
            "Medication Bot",
            generate_medication_plan_llm,
            age,
            gender,
            diagnosis,
            timeline,
            labs,
            vitals,
        )

        # 10) NURSING NOTES BOT
        nursing_notes = run_step(
            "Nursing Notes Bot",
            generate_nursing_notes_llm,
            age,
            gender,
            demographics,
            diagnosis,
            vitals,
            labs,
            timeline,
        )

        # 11) CLINICAL NOTES BOT
        clinical_notes = run_step(
            "Clinical Notes Bot",
            generate_clinical_notes_llm,
            age,
            gender,
            demographics,
            diagnosis,
            timeline,
            labs,
            vitals,
            radiology,
        )

        # 12) PRESCRIPTION BOT
        prescriptions = run_step(
            "Prescription Bot",
            generate_prescriptions_llm,
            age,
            gender,
            diagnosis,
            medications,
            vitals,
            labs,
        )

        # 13) BILLING BOT
        billing = run_step(
            "Billing Bot",
            generate_billing_summary_llm,
            age,
            gender,
            demographics,
            diagnosis,
            procedures,
            labs,
            radiology,
            medications,
        )

        # 14) CONSOLIDATOR BOT
        patient_record = run_step(
            "Consolidator Bot",
            consolidate_patient_record,
            demographics,
            diagnosis,
            timeline,
            labs,
            vitals,
            radiology,
            procedures,
            pathology,
            clinical_notes,
            nursing_notes,
            medications,
            prescriptions,
            billing,
        )

        # 15) SAFETY LABELER BOT
        safety_labels = run_step(
            "Safety Labeler Bot",
            label_safety_llm,
            patient_record,
        )

        # 16) CONSISTENCY CHECKER BOT
        consistency = run_step(
            "Consistency Checker Bot",
            check_consistency_llm,
            patient_record,
        )

        # 17) RENDERER BOT (TEXT)
        rendered_text = run_step(
            "Renderer Bot",
            render_patient_record,
            patient_record,
            safety_labels,
            consistency,
        )

        # 18) COMPOSER BOT (WRAP HEADER/FOOTER)
        final_text = run_step(
            "Composer Bot",
            compose_final_document,
            rendered_text,
        )

        # 19) PDF GENERATOR
        st.info("📄 Generating PDF...")
        logo_arg = logo_path if logo_path and os.path.exists(logo_path) else None

        generate_pdf(
            report_text=final_text,
            radiology_images=radiology_image_urls,
            output_file=output_pdf_path,
            logo_path=logo_arg,
        )
        st.success("✅ PDF generated")

        # DOWNLOAD LINK
        with open(output_pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Synthetic Medical Record PDF",
                data=f,
                file_name="synthetic_patient_record.pdf",
                mime="application/pdf",
            )

        st.success("🎉 Full pipeline completed successfully.")

    except Exception as e:
        st.error("🚨 Pipeline aborted due to an error above.")
        print("[FATAL] Pipeline aborted:", e)
