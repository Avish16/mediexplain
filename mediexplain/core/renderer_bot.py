import json
import textwrap

PAGE_BREAK = "\f"   # Form-feed — universally respected as a new-page marker


# ----------------------------------------------------
# Utility Formatting
# ----------------------------------------------------
def _header(title: str) -> str:
    border = "=" * 90
    return f"{border}\n{title}\n{border}\n"


def _table_block(title: str, data) -> str:
    """
    Render a key-value table.
    Accepts a dict, or a plain-text string with 'Key: Value' lines
    (the format returned by synthetic_demographics).
    """
    out = _header(title)

    # If a plain-text string, parse Key: Value lines into a dict
    if isinstance(data, str) and data.strip():
        parsed = {}
        for line in data.splitlines():
            line = line.strip()
            if line.startswith("=") or not line:
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip()
                if key and val:
                    parsed[key] = val
        data = parsed if parsed else data   # fall back to raw string if no pairs found

    if isinstance(data, str):
        return out + (data.strip() or "(no data)") + "\n"

    if not isinstance(data, dict) or not data:
        return out + "(no data)\n"

    longest_key = max(len(k) for k in data.keys())
    for k, v in data.items():
        out += f"{k:<{longest_key}} : {v}\n"
    return out


def _json_block(title: str, obj) -> str:
    """
    Render structured data.
    If obj is a plain-text string (e.g. vitals narrative) render it as
    wrapped text instead of JSON to avoid quoting it as a JSON string.
    """
    if isinstance(obj, str):
        return _text_block(title, obj)
    if not obj:
        return _header(title) + "(no data)\n"
    pretty = json.dumps(obj, indent=2, ensure_ascii=False)
    return f"{_header(title)}{pretty}\n"


def _text_block(title: str, text: str) -> str:
    if not text or not str(text).strip():
        return _header(title) + "(no data)\n"
    # Preserve intentional newlines; wrap each paragraph individually
    paragraphs = str(text).split("\n")
    wrapped = "\n".join(
        textwrap.fill(p, width=90) if len(p) > 90 else p
        for p in paragraphs
    )
    return f"{_header(title)}{wrapped}\n"


def _page(content: str) -> str:
    return content + "\n" + PAGE_BREAK + "\n"


# ----------------------------------------------------
# RENDERER CORE
# ----------------------------------------------------
def render_patient_record(patient_record: dict, safety_labels: dict, consistency: dict) -> str:
    """
    Page-per-section renderer.  Handles both dict and plain-text bot outputs.
    """

    out = []
    pr = patient_record.get("patient_record", {})

    # 1) DEMOGRAPHICS — table (bot returns plain text "Key: Value" lines)
    demo = pr.get("demographics", {})
    out.append(_page(_table_block("PATIENT DEMOGRAPHICS", demo)))

    # 2) DIAGNOSIS — narrative
    dx = pr.get("diagnosis", {})
    if isinstance(dx, dict):
        dx_text = "\n".join(f"{k}: {v}" for k, v in dx.items())
    else:
        dx_text = str(dx)
    out.append(_page(_text_block("PRIMARY DIAGNOSIS", dx_text)))

    # 3) TIMELINE — summary + events table
    tl = pr.get("timeline", {})
    if isinstance(tl, dict):
        timeline_summary = tl.get("timeline_summary", str(tl) if tl else "No summary available.")
        tl_events = "".join(
            f"- {ev.get('date','?')} | {ev.get('event_type','?')} | {ev.get('description','')}\n"
            for ev in tl.get("timeline_table", [])
        )
        tl_full = f"{timeline_summary}\n\nEVENTS:\n{tl_events}" if tl_events else timeline_summary
    else:
        tl_full = str(tl)
    out.append(_page(_text_block("CLINICAL TIMELINE", tl_full)))

    # 4) LAB RESULTS
    labs = pr.get("labs", {})
    out.append(_page(_json_block("LABORATORY RESULTS", labs)))

    # 5) VITAL SIGNS — bot returns plain text narrative
    vitals = pr.get("vitals", {})
    out.append(_page(_json_block("VITAL SIGNS", vitals)))

    # 6) RADIOLOGY — bot may be paused (empty dict) or return a summary
    rad = pr.get("radiology", {})
    if not rad:
        rad_text = "(Radiology generation paused — no imaging data for this record)"
    elif isinstance(rad, str):
        rad_text = rad
    else:
        rad_text = (rad.get("radiology_summary")
                    or rad.get("summary")
                    or json.dumps(rad, indent=2, ensure_ascii=False))
    out.append(_page(_text_block("RADIOLOGY SUMMARY", rad_text)))

    # 7) PROCEDURES
    procs = pr.get("procedures", {})
    out.append(_page(_json_block("PROCEDURES", procs)))

    # 8) PATHOLOGY
    pathology = pr.get("pathology", {})
    out.append(_page(_json_block("PATHOLOGY REPORT", pathology)))

    # 9) CLINICAL NOTES
    notes = pr.get("clinical_notes", {})
    out.append(_page(_json_block("CLINICAL NOTES", notes)))

    # 10) NURSING NOTES
    nursing = pr.get("nursing_notes", {})
    out.append(_page(_json_block("NURSING NOTES", nursing)))

    # 11) MEDICATION PLAN
    meds = pr.get("medications", {})
    out.append(_page(_json_block("MEDICATION PLAN", meds)))

    # 12) PRESCRIPTIONS
    rx = pr.get("prescriptions", {})
    out.append(_page(_json_block("PRESCRIPTIONS", rx)))

    # 13) BILLING
    billing = pr.get("billing", {})
    out.append(_page(_json_block("BILLING SUMMARY", billing)))

    # 14) SAFETY LABELS
    out.append(_page(_json_block("SAFETY LABELS", safety_labels)))

    # 15) CONSISTENCY REPORT
    out.append(_page(_json_block("CONSISTENCY REPORT", consistency)))

    return "".join(out)
