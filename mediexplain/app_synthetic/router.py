"""
Routing logic extracted into a standalone module so validator_app.py
can import route_to_specialist_bot without triggering the chat page UI.
"""
import json
import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None

_client_cache = None


def _client():
    global _client_cache
    if _client_cache is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and st is not None:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                pass
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        _client_cache = OpenAI(api_key=api_key)
    return _client_cache


def route_to_specialist_bot(mode: str, question: str, pdf_text: str, long_term_memory) -> str:
    system_prompt = """You are MediExplain's routing agent. Return STRICT JSON: {"bot": "...", "reason": "..."}

SCOPE: Only answer medical-report questions. If the user asks about politics, celebrities, sports,
cooking, gaming, geography, weather, or general knowledge, return bot="OUT_OF_SCOPE".

ROUTING:
- drug names / side effects / interactions / "is this safe" -> bot="MEDS"
- discharge prescriptions / sig instructions -> bot="PRESCRIPTIONS"
- lab values -> bot="LABS"
- vital signs / symptoms -> bot="SNAPSHOT"
- care plan -> bot="CAREPLAN"
- emotional distress -> bot="SUPPORT"
- general explanation -> bot="EXPLAINER"

Choose ONLY from: ["EXPLAINER","LABS","MEDS","CAREPLAN","SNAPSHOT","SUPPORT","PRESCRIPTIONS","OUT_OF_SCOPE"]
Return STRICT JSON only."""

    user_payload = (
        f"MODE: {mode}\n"
        f"QUESTION: {question}\n"
        f"REPORT TEXT (first 3000 chars):\n{pdf_text[:3000]}\n"
        f"USER MEMORY:\n{long_term_memory}"
    )
    resp = _client().chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_payload},
        ],
    ).choices[0].message.content

    try:
        clean = resp.replace("```json", "").replace("```", "").strip()
        bot_name = json.loads(clean).get("bot", "EXPLAINER").upper()
    except Exception:
        bot_name = "EXPLAINER"

    valid = {"EXPLAINER", "LABS", "MEDS", "CAREPLAN", "SNAPSHOT", "SUPPORT", "PRESCRIPTIONS", "OUT_OF_SCOPE"}
    return bot_name if bot_name in valid else "EXPLAINER"
