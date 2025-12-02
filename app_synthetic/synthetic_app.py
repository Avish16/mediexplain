import sys
import os
import json
import streamlit as st

# Make sure Python can see the repo root and core/ package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.synthetic_demographics import generate_demographics_llm

st.title("Synthetic Patient Report Generator")

st.header("Demographics Builder")

age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

if st.button("Generate Demographics"):
    try:
        demographics = generate_demographics_llm(age=age, gender=gender)

        st.subheader("Generated Demographics (LLM)")
        st.json(demographics)

        st.download_button(
            label="Download JSON",
            data=json.dumps(demographics, indent=2),
            file_name="synthetic_demographics.json",
            mime="application/json"
        )
    except Exception as e:
        st.error(f"Error generating demographics: {e}")



# DIAGNOSIS 
from core.diagnosis_bot import generate_diagnosis_llm

st.header("Diagnosis Generator")

if st.button("Generate Diagnosis"):
    diagnosis = generate_diagnosis_llm(age, gender)
    st.subheader("Generated Diagnosis")
    st.json(diagnosis)

    st.download_button(
        "Download Diagnosis JSON",
        data=json.dumps(diagnosis, indent=2),
        file_name="synthetic_diagnosis.json",
        mime="application/json"
    )


# TIMELINE 
from core.timeline_bot import generate_timeline_llm

st.header("Clinical Timeline Generator")

if st.button("Generate Clinical Timeline"):

    diagnosis = generate_diagnosis_llm(age, gender)
    timeline = generate_timeline_llm(age, gender, diagnosis)

    st.subheader("Timeline Summary")
    st.write(timeline["timeline_summary"])

    st.subheader("Timeline Table")
    st.table(timeline["timeline_table"])

    st.download_button(
        "Download Timeline JSON",
        data=json.dumps(timeline, indent=2),
        file_name="synthetic_timeline.json",
        mime="application/json"
    )


# LAB BOT 

from core.lab_bot import generate_lab_report_llm

st.header("Laboratory Report Generator")

if st.button("Generate Lab Report"):

    diagnosis = st.session_state.get("diagnosis") or generate_diagnosis_llm(age, gender)
    timeline = st.session_state.get("timeline") or generate_timeline_llm(age, gender, diagnosis)

    lab_report = generate_lab_report_llm(age, gender, diagnosis, timeline)

    st.subheader("Lab Report")
    st.json(lab_report)

    st.download_button(
        "Download Lab Report JSON",
        data=json.dumps(lab_report, indent=2),
        file_name="synthetic_lab_report.json",
        mime="application/json"
    )

# VITALS BOT 

from core.vitals_bot import generate_vitals_llm

st.header("Vitals Report Generator")

if st.button("Generate Vitals Report"):

    diagnosis = st.session_state.get("diagnosis") or generate_diagnosis_llm(age, gender)
    timeline = st.session_state.get("timeline") or generate_timeline_llm(age, gender, diagnosis)

    vitals = generate_vitals_llm(age, gender, diagnosis, timeline)

    st.subheader("Vitals Report")
    st.json(vitals)

    st.download_button(
        "Download Vitals JSON",
        data=json.dumps(vitals, indent=2),
        file_name="synthetic_vitals.json",
        mime="application/json"
    )

# RADIOLOGY BOT 

from core.radiology_bot import generate_radiology_studies_llm

st.header("Radiology Bot (Imaging + Findings)")

if st.button("Generate Radiology Studies"):

    diagnosis = st.session_state.get("diagnosis") or generate_diagnosis_llm(age, gender)
    timeline = st.session_state.get("timeline") or generate_timeline_llm(age, gender, diagnosis)

    rads = generate_radiology_studies_llm(age, gender, diagnosis, timeline)

    st.subheader("Radiology Summary")
    st.write(rads.get("radiology_summary", ""))

    st.subheader("Studies")
    for study in rads.get("studies", []):
        st.markdown(f"**Role:** {study.get('role')} | **Date:** {study.get('study_date')} | **Modality:** {study.get('modality')}")
        st.write(study.get("impression", ""))
        if "image_url" in study:
            st.image(study["image_url"], caption=f"{study.get('modality')} – {study.get('body_region')} ({study.get('role')})")

    st.download_button(
        "Download Radiology JSON",
        data=json.dumps(rads, indent=2),
        file_name="synthetic_radiology.json",
        mime="application/json"
    )


# CLINICAL BOT 

from core.clinical_notes_bot import generate_clinical_notes_llm

st.header("Clinical Notes Generator")

if st.button("Generate Clinical Notes"):

    # re-use or regenerate all upstream components
    demographics = generate_demographics_llm(age=age, gender=gender)

    diagnosis = st.session_state.get("diagnosis") or generate_diagnosis_llm(age, gender)
    timeline = st.session_state.get("timeline") or generate_timeline_llm(age, gender, diagnosis)

    lab_report = generate_lab_report_llm(age, gender, diagnosis, timeline)
    vitals = generate_vitals_llm(age, gender, diagnosis, timeline)
    radiology = generate_radiology_studies_llm(age, gender, diagnosis, timeline)

    clinical_notes = generate_clinical_notes_llm(
        age=age,
        gender=gender,
        demographics=demographics,
        diagnosis=diagnosis,
        timeline=timeline,
        labs=lab_report,
        vitals=vitals,
        radiology=radiology
    )

    st.subheader("Clinical Notes (JSON)")
    st.json(clinical_notes)

    st.download_button(
        "Download Clinical Notes JSON",
        data=json.dumps(clinical_notes, indent=2),
        file_name="synthetic_clinical_notes.json",
        mime="application/json"
    )


# NURSING NOTES BOT 

from core.nursing_notes_bot import generate_nursing_notes_llm

st.header("Nursing Notes Generator")

if st.button("Generate Nursing Notes"):
    demographics = generate_demographics_llm(age=age, gender=gender)
    diagnosis = generate_diagnosis_llm(age, gender)
    timeline = generate_timeline_llm(age, gender, diagnosis)
    labs = generate_lab_report_llm(age, gender, diagnosis, timeline)
    vitals = generate_vitals_llm(age, gender, diagnosis, timeline)

    nursing_notes = generate_nursing_notes_llm(
        age, gender, demographics, diagnosis, vitals, labs, timeline
    )

    st.subheader("Nursing Notes")
    st.json(nursing_notes)

    st.download_button(
        "Download Nursing Notes JSON",
        data=json.dumps(nursing_notes, indent=2),
        file_name="synthetic_nursing_notes.json",
        mime="application/json"
    )


# MEDICATION BOTfrom core.medication_bot import generate_medication_plan_llm

st.header("Medication Plan Generator")

if st.button("Generate Medication Plan"):

    diagnosis = st.session_state.get("diagnosis") or generate_diagnosis_llm(age, gender)
    timeline = st.session_state.get("timeline") or generate_timeline_llm(age, gender, diagnosis)
    lab_report = generate_lab_report_llm(age, gender, diagnosis, timeline)
    vitals = generate_vitals_llm(age, gender, diagnosis, timeline)

    meds = generate_medication_plan_llm(
        age=age,
        gender=gender,
        diagnosis=diagnosis,
        timeline=timeline,
        labs=lab_report,
        vitals=vitals
    )

    st.subheader("Medication Plan")
    st.json(meds)

    st.download_button(
        "Download Medication JSON",
        data=json.dumps(meds, indent=2),
        file_name="synthetic_medications.json",
        mime="application/json"
    )


# PRESCRITION BOT 

from core.prescription_bot import generate_prescriptions_llm

st.header("Prescription Generator")

if st.button("Generate Prescriptions"):

    diagnosis = generate_diagnosis_llm(age, gender)
    timeline = generate_timeline_llm(age, gender, diagnosis)
    labs = generate_lab_report_llm(age, gender, diagnosis, timeline)
    vitals = generate_vitals_llm(age, gender, diagnosis, timeline)
    meds = generate_medication_plan_llm(age, gender, diagnosis, timeline, labs, vitals)

    prescriptions = generate_prescriptions_llm(
        age, gender, diagnosis, meds, vitals, labs
    )

    st.subheader("Prescriptions JSON")
    st.json(prescriptions)

    st.download_button(
        "Download Prescriptions JSON",
        data=json.dumps(prescriptions, indent=2),
        file_name="synthetic_prescriptions.json",
        mime="application/json"
    )


# PROCEDURE BOT 

from core.procedure_bot import generate_procedures_llm

st.header("Procedure History Generator")

if st.button("Generate Procedures"):

    diagnosis = st.session_state.get("diagnosis") or generate_diagnosis_llm(age, gender)
    timeline = st.session_state.get("timeline") or generate_timeline_llm(age, gender, diagnosis)
    lab_report = generate_lab_report_llm(age, gender, diagnosis, timeline)
    radiology = generate_radiology_studies_llm(age, gender, diagnosis, timeline)

    procedures = generate_procedures_llm(
        age=age,
        gender=gender,
        diagnosis=diagnosis,
        timeline=timeline,
        labs=lab_report,
        radiology=radiology
    )

    st.subheader("Procedures JSON")
    st.json(procedures)

    st.download_button(
        "Download Procedures JSON",
        data=json.dumps(procedures, indent=2),
        file_name="synthetic_procedures.json",
        mime="application/json"
    )

# PATHOLOGY BOT 

from core.pathology_bot import generate_pathology_report_llm

st.header("Pathology Report Generator")

if st.button("Generate Pathology Report"):

    diagnosis = generate_diagnosis_llm(age, gender)
    timeline = generate_timeline_llm(age, gender, diagnosis)
    labs = generate_lab_report_llm(age, gender, diagnosis, timeline)
    radiology = generate_radiology_studies_llm(age, gender, diagnosis, timeline)
    procedures = generate_procedures_llm(age, gender, diagnosis, timeline, labs, radiology)

    path_report = generate_pathology_report_llm(
        age, gender, diagnosis, procedures, radiology, labs
    )

    st.subheader("Pathology Report JSON")
    st.json(path_report)

    st.download_button(
        "Download Pathology Report JSON",
        data=json.dumps(path_report, indent=2),
        file_name="synthetic_pathology.json",
        mime="application/json"
    )


# BILLING BOT 

from core.billing_bot import generate_billing_summary_llm

st.header("Billing & Coding Summary Generator")

if st.button("Generate Billing Summary"):

    demographics = generate_demographics_llm(age=age, gender=gender)
    diagnosis = st.session_state.get("diagnosis") or generate_diagnosis_llm(age, gender)
    timeline = st.session_state.get("timeline") or generate_timeline_llm(age, gender, diagnosis)
    lab_report = generate_lab_report_llm(age, gender, diagnosis, timeline)
    radiology = generate_radiology_studies_llm(age, gender, diagnosis, timeline)
    procedures = generate_procedures_llm(age, gender, diagnosis, timeline, lab_report, radiology)
    meds = generate_medication_plan_llm(age, gender, diagnosis, timeline, lab_report, vitals={})

    billing = generate_billing_summary_llm(
        age=age,
        gender=gender,
        demographics=demographics,
        diagnosis=diagnosis,
        procedures=procedures,
        labs=lab_report,
        radiology=radiology,
        medications=meds,
        length_of_stay_days=5
    )

    st.subheader("Billing JSON")
    st.json(billing)

    st.download_button(
        "Download Billing JSON",
        data=json.dumps(billing, indent=2),
        file_name="synthetic_billing.json",
        mime="application/json"
    )


from core.consolidator_bot import consolidate_patient_record
from core.consistency_checker_bot import check_consistency_llm
from core.safety_labeler_bot import label_safety_llm
from core.renderer_bot import render_patient_record
from core.composer_bot import compose_final_document
from core.pdf_generator import generate_pdf

st.header("Generate Full Medical Report PDF")

if st.button("Generate Full Report"):
    # gather all existing outputs
    record = consolidate_patient_record(
        demographics, diagnosis, timeline, labs, vitals,
        radiology, procedures, pathology, clinical_notes,
        nursing_notes, medications, prescriptions, billing
    )

    consistency = check_consistency_llm(record)
    safety = label_safety_llm(record)

    rendered = render_patient_record(record, safety, consistency)
    final_text = compose_final_document(rendered)

    # gather radiology images
    rad_images = [s["image_url"] for s in radiology["studies"]]

    output_path = "final_patient_report.pdf"
    generate_pdf(final_text, rad_images, output_path)

    with open(output_path, "rb") as f:
        st.download_button("Download Medical Report PDF", f, file_name="patient_report.pdf")


# ------------------------------------------
# GENERATE EVERYTHING AT ONCE
# ------------------------------------------

import os
from core.consolidator_bot import consolidate_patient_record
from core.consistency_checker_bot import check_consistency_llm
from core.safety_labeler_bot import label_safety_llm
from core.renderer_bot import render_patient_record
from core.composer_bot import compose_final_document
from core.pdf_generator import generate_pdf

st.header("✨ Generate Complete Medical Record")

if st.button("Generate Full Synthetic Report"):

    with st.spinner("Generating full patient record…"):

        # Step 1: Generate each bot output
        demographics = generate_demographics_llm(age=age, gender=gender)
        diagnosis = generate_diagnosis_llm(age, gender)
        timeline = generate_timeline_llm(age, gender, diagnosis)
        labs = generate_lab_report_llm(age, gender, diagnosis, timeline)
        vitals = generate_vitals_llm(age, gender, diagnosis, timeline)
        radiology = generate_radiology_studies_llm(age, gender, diagnosis, timeline)
        procedures = generate_procedures_llm(age, gender, diagnosis, timeline, labs, radiology)
        pathology = generate_pathology_llm(age, gender, diagnosis, labs, radiology)
        clinical_notes = generate_clinical_notes_llm(age, gender, diagnosis, timeline)
        nursing_notes = generate_nursing_notes_llm(age, gender, diagnosis, timeline)
        medications = generate_medication_plan_llm(age, gender, diagnosis, timeline, labs, vitals)
        prescriptions = generate_prescriptions_llm(age, gender, medications)
        billing = generate_billing_summary_llm(
            age=age,
            gender=gender,
            demographics=demographics,
            diagnosis=diagnosis,
            procedures=procedures,
            labs=labs,
            radiology=radiology,
            medications=medications,
            length_of_stay_days=5
        )

        # Step 2: Consolidate full patient record
        record = consolidate_patient_record(
            demographics, diagnosis, timeline, labs, vitals,
            radiology, procedures, pathology, clinical_notes,
            nursing_notes, medications, prescriptions, billing
        )

        # Step 3: Consistency Checker
        consistency = check_consistency_llm(record)

        # Step 4: Safety Labeler
        safety = label_safety_llm(record)

        # Step 5: Renderer
        rendered_text = render_patient_record(record, safety, consistency)

        # Step 6: Compose final text
        final_text = compose_final_document(rendered_text)

        # Step 7: Prepare radiology images
        radiology_imgs = [s["image_url"] for s in radiology["studies"]]

        # Step 8: Add hospital logo
        hospital_logo_path = os.path.join("assets", "hospital_logo.png")

        # Step 9: Generate full PDF
        output_pdf = "full_medical_record.pdf"
        generate_pdf(
            report_text=final_text,
            radiology_images=radiology_imgs,
            output_file=output_pdf,
            logo_path=hospital_logo_path
        )

        st.success("Full synthetic medical record generated!")

        with open(output_pdf, "rb") as f:
            st.download_button(
                "⬇️ Download Full Medical Report PDF",
                f,
                file_name="patient_report.pdf",
                mime="application/pdf"
            )
