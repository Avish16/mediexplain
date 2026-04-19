import json
import textwrap

PAGE_BREAK = "\f"   # Form-feed — universally respected as a new-page marker


# ----------------------------------------------------
# Utility Formatting
# ----------------------------------------------------
def _header(title: str) -> str:
    border = "=" * 90
    return f"{border}\n{title}\n{border}\n"


def _table_block(title: str, data: dict) -> str:
    out = _header(title)
    if not isinstance(data, dict):
        return out + "(no data)\n"

    longest_key = max(len(k) for k in data.keys())
    for k, v in data.items():
        out += f"{k:<{longest_key}} : {v}\n"
    return out


def _json_block(title: str, obj: dict) -> str:
    pretty = json.dumps(obj, indent=2, ensure_ascii=False)
    return f"{_header(title)}{pretty}\n"


def _text_block(title: str, text: str) -> str:
    wrapped = textwrap.fill(text, width=90)
    return f"{_header(title)}{wrapped}\n"


def _page(content: str) -> str:
    return content + "\n" + PAGE_BREAK + "\n"


# ----------------------------------------------------
# RENDERER CORE
# ----------------------------------------------------
def render_patient_record(patient_record: dict, safety_labels: dict, consistency: dict) -> str:
    """
    NEW PAGE-PER-SECTION RENDERER
    Clean formatting for a medical PDF.
    """

    out = []
    pr = patient_record.get("patient_record", {})

    # ----------------------------------------------------
    # 1) DEMOGRAPHICS — table format
    # ----------------------------------------------------
    demo = pr.get("demographics", {})
    out.append(_page(_table_block("PATIENT DEMOGRAPHICS", demo)))

    # ----------------------------------------------------
    # 2) DIAGNOSIS — narrative text
    # ----------------------------------------------------
    dx = pr.get("diagnosis", {})
    dx_text = "\n".join([f"{k}: {v}" for k, v in dx.items()])
    out.append(_page(_text_block("PRIMARY DIAGNOSIS", dx_text)))

    # ----------------------------------------------------
    # 3) TIMELINE — narrative + table
    # ----------------------------------------------------
    tl = pr.get("timeline", {})
    timeline_summary = tl.get("timeline_summary", "No summary available.")

    tl_events = ""
    for ev in tl.get("timeline_table", []):
        tl_events += f"- {ev.get('date')} | {ev.get('event_type')} | {ev.get('description')}\n"

    tl_full = f"{timeline_summary}\n\nEVENTS:\n{tl_events}"
    out.append(_page(_text_block("CLINICAL TIMELINE", tl_full)))

    # ----------------------------------------------------
    # 4) LAB RESULTS — JSON block
    # ----------------------------------------------------
    labs = pr.get("labs", {})
    out.append(_page(_json_block("LABORATORY RESULTS", labs)))

    # ----------------------------------------------------
    # 5) VITAL SIGNS — JSON block
    # ----------------------------------------------------
    vitals = pr.get("vitals", {})
    out.append(_page(_json_block("VITAL SIGNS", vitals)))

    # ----------------------------------------------------
    # 6) RADIOLOGY — narrative
    # ----------------------------------------------------
    rad = pr.get("radiology", {})
    out.append(_page(_text_block("RADIOLOGY SUMMARY", rad.get("radiology_summary", ""))))

    # ----------------------------------------------------
    # 7) PROCEDURES — JSON block
    # ----------------------------------------------------
    procs = pr.get("procedures", {})
    out.append(_page(_json_block("PROCEDURES", procs)))

    # ----------------------------------------------------
    # 8) PATHOLOGY — JSON block (long narrative inside)
    # ----------------------------------------------------
    pathology = pr.get("pathology", {})
    out.append(_page(_json_block("PATHOLOGY REPORT", pathology)))

    # ----------------------------------------------------
    # 9) CLINICAL NOTES — JSON block
    # ----------------------------------------------------
    notes = pr.get("clinical_notes", {})
    out.append(_page(_json_block("CLINICAL NOTES", notes)))

    # ----------------------------------------------------
    # 10) NURSING NOTES — JSON block
    # ----------------------------------------------------
    nursing = pr.get("nursing_notes", {})
    out.append(_page(_json_block("NURSING NOTES", nursing)))

    # ----------------------------------------------------
    # 11) MEDICATION PLAN — JSON block
    # ----------------------------------------------------
    meds = pr.get("medications", {})
    out.append(_page(_json_block("MEDICATION PLAN", meds)))

    # ----------------------------------------------------
    # 12) PRESCRIPTIONS — JSON block
    # ----------------------------------------------------
    rx = pr.get("prescriptions", {})
    out.append(_page(_json_block("PRESCRIPTIONS", rx)))

    # ----------------------------------------------------
    # 13) BILLING — JSON block
    # ----------------------------------------------------
    billing = pr.get("billing", {})
    out.append(_page(_json_block("BILLING SUMMARY", billing)))

    # ----------------------------------------------------
    # 14) SAFETY LABELS — JSON block
    # ----------------------------------------------------
    out.append(_page(_json_block("SAFETY LABELS", safety_labels)))

    # ----------------------------------------------------
    # 15) CONSISTENCY REPORT — JSON block
    # ----------------------------------------------------
    out.append(_page(_json_block("CONSISTENCY REPORT", consistency)))

    # ----------------------------------------------------
    # MERGE ALL
    # ----------------------------------------------------
    return "".join(out)
