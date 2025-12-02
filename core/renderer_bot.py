def render_section(title: str, content: str) -> str:
    border = "=" * 80
    return f"{border}\n{title}\n{border}\n{content}\n\n"


def render_patient_record(patient_record: dict, safety_labels: dict, consistency: dict) -> str:
    """
    Converts JSON into readable medical text sections.
    """
    out = ""

    pr = patient_record["patient_record"]

    # DEMOGRAPHICS
    demo = pr["demographics"]
    demo_text = "\n".join([f"{k}: {v}" for k, v in demo.items()])
    out += render_section("PATIENT DEMOGRAPHICS", demo_text)

    # DIAGNOSIS
    dx = pr["diagnosis"]
    dx_text = "\n".join([f"{k}: {v}" for k, v in dx.items()])
    out += render_section("PRIMARY DIAGNOSIS", dx_text)

    # TIMELINE
    tl = pr["timeline"]
    tl_text = tl["timeline_summary"] + "\n\nEvents:\n"
    for ev in tl["timeline_table"]:
        tl_text += f"- {ev['date']} | {ev['event_type']} | {ev['description']}\n"
    out += render_section("CLINICAL TIMELINE", tl_text)

    # LABS
    labs = pr["labs"]
    labs_text = json.dumps(labs, indent=2)
    out += render_section("LABORATORY RESULTS", labs_text)

    # VITALS
    vit = pr["vitals"]
    vit_text = json.dumps(vit, indent=2)
    out += render_section("VITAL SIGNS", vit_text)

    # RADIOLOGY
    rad = pr["radiology"]
    rad_text = rad["radiology_summary"]
    out += render_section("RADIOLOGY INTERPRETATION", rad_text)

    # PROCEDURES
    procs = pr["procedures"]
    procs_text = json.dumps(procs, indent=2)
    out += render_section("PROCEDURES", procs_text)

    # PATHOLOGY
    path = pr["pathology"]
    path_text = json.dumps(path, indent=2)
    out += render_section("PATHOLOGY REPORT", path_text)

    # CLINICAL NOTES
    notes = pr["clinical_notes"]
    notes_text = json.dumps(notes, indent=2)
    out += render_section("PHYSICIAN CLINICAL NOTES", notes_text)

    # NURSING NOTES
    nn = pr["nursing_notes"]
    nn_text = json.dumps(nn, indent=2)
    out += render_section("NURSING NOTES", nn_text)

    # MEDICATIONS
    meds = pr["medications"]
    meds_text = json.dumps(meds, indent=2)
    out += render_section("MEDICATION PLAN", meds_text)

    # PRESCRIPTIONS
    rx = pr["prescriptions"]
    rx_text = json.dumps(rx, indent=2)
    out += render_section("PRESCRIPTIONS", rx_text)

    # BILLING
    bill = pr["billing"]
    bill_text = json.dumps(bill, indent=2)
    out += render_section("BILLING SUMMARY", bill_text)

    # SAFETY LABELS
    safe_text = json.dumps(safety_labels, indent=2)
    out += render_section("SAFETY LABELS", safe_text)

    # CONSISTENCY
    cons_text = json.dumps(consistency, indent=2)
    out += render_section("CONSISTENCY REPORT", cons_text)

    return out
