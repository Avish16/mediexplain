import os
import re
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


_client = None

# =========================================================
# OPENAI CLIENT
# =========================================================
def _get_openai_client() -> OpenAI:
    """Shared lazy-initialized OpenAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and st is not None:
            api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _client = OpenAI(api_key=api_key)
    return _client


# =========================================================
# PERSONA
# =========================================================
def _persona_block(mode: str) -> str:
    """Tone for patient vs caregiver."""
    mode = (mode or "").lower()
    if mode == "caregiver":
        return (
            "Provide supportive guidance to a caregiver with some clinical understanding.\n"
            "- Validate emotional burden & logistical stress.\n"
            "- Suggest practical steps (communication, organization, red flags).\n"
            "- Do NOT provide therapy or mental-health treatment.\n"
        )
    return (
        "Provide supportive, empathetic guidance to a patient.\n"
        "- Validate feelings in a gentle, non-clinical tone.\n"
        "- Offer simple steps like writing questions, bringing a supporter, etc.\n"
        "- Avoid medical instructions or therapy.\n"
    )


_DISCLAIMER = (
    "This is emotional and educational support only — not medical or crisis care. "
    "In an emergency, contact local emergency services immediately."
)


# =========================================================
# ZIP CODE HELPERS
# =========================================================
_ZIP_REGEX = re.compile(r"\b\d{5}(?:-\d{4})?\b")

def _extract_zip_from_text(*texts: str) -> str | None:
    for t in texts:
        if not t:
            continue
        m = _ZIP_REGEX.search(t)
        if m:
            return m.group(0)
    return None


# =========================================================
# CLASSIFY CRISIS
# =========================================================
def _classify_crisis_level(user_text: str) -> str:
    """
    Return EXACTLY: CRISIS / DISTRESS / SAFE
    """
    client = _get_openai_client()

    prompt = f"""
Classify the emotional risk level of this message.

MESSAGE:
\"\"\"{user_text}\"\"\"

Return EXACTLY:
CRISIS      → suicidal/self-harm intent
DISTRESS    → strong emotion but no explicit harm
SAFE        → no emotional red flags
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    label = resp.choices[0].message.content.strip().upper()
    return label if label in {"CRISIS", "DISTRESS", "SAFE"} else "SAFE"


# =========================================================
# NEARBY RESOURCE WEB SEARCH
# =========================================================
def _search_local_mental_health_resources(zip_code: str) -> str:
    client = _get_openai_client()

    prompt = f"""
Use web search to find 3–5 mental-health clinics or counseling centers near ZIP {zip_code}.
Return:
- Name
- Type (clinic / hotline / center)
- Address (if available)
- Phone
- Website
Format as markdown bullet points.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "user", "content": prompt}],
        tools=[{"type": "web_search"}],
        max_output_tokens=700,
    )

    return (response.output_text or "").strip()


# =========================================================
# NON-CRISIS SUPPORT
# =========================================================
def _build_standard_support_message(
    mode: str,
    context: str,
    user_input: str,
    conversation_history: str,
    model="gpt-4.1-mini",
    max_tokens=800,
) -> str:

    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – a calm, compassionate assistant.

{persona}

### Conversation So Far
{conversation_history}

You MUST:
- Validate feelings without diagnosing.
- Encourage connection with clinical team & trusted support.
- Avoid promises or medical instructions.

End with a paraphrased reminder:
{_DISCLAIMER}
"""

    user_content = f"""
Context from the report:
--------------------
{context}
--------------------

User message:
--------------------
{user_input}
--------------------
"""

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_output_tokens=max_tokens,
    )

    return (resp.output_text or "").strip()


# =========================================================
# CRISIS SUPPORT (NO LLM NEEDED)
# =========================================================
def _build_crisis_support_message(
    user_input: str,
    pdf_text: str,
    memory_snippets,
) -> str:

    crisis_header = (
        "I’m really sorry you’re feeling this way. Your safety matters.\n\n"
        "**If you feel you might act on these thoughts or are in immediate danger, please contact emergency services right now** "
        "(911 in the U.S.) or go to the nearest emergency room.\n\n"
        "You can also call or text **988** for 24/7 support from the Suicide & Crisis Lifeline."
    )

    memory_text = "\n".join(memory_snippets or [])
    zip_code = _extract_zip_from_text(user_input, pdf_text, memory_text)

    # If ZIP available → perform web search
    resources_block = ""
    ask_block = ""

    if zip_code:
        try:
            results = _search_local_mental_health_resources(zip_code)
            if results:
                resources_block = (
                    f"\n\n---\n### Nearby mental-health resources near `{zip_code}`\n"
                    f"{results}"
                )
        except Exception:
            resources_block = (
                "\n\n---\nI tried looking for nearby resources but ran into a technical issue. "
                "You can still reach out to local hospitals or 988."
            )
    else:
        ask_block = (
            "\n\n---\nIf you feel comfortable, you can share your **ZIP code** so I can look up nearby clinics "
            "on your next message. This is optional.\n"
        )

    safety_tail = (
        "\n\n---\nThis space can offer support but cannot replace real crisis care. "
        "Please reach out to professionals who can help you right now."
    )

    return crisis_header + resources_block + ask_block + safety_tail


# =========================================================
# PUBLIC API — USED BY ORCHESTRATOR
# =========================================================
def run_support(
    user_input,
    mode,
    pdf_text,
    memory_snippets,
    conversation_history="",
):
    """
    Called by the orchestrator:
    - Detect crisis level
    - If crisis → return hard-coded crisis-safe response
    - Else → generate emotional support using LLM
    """

    risk = _classify_crisis_level(user_input)

    if risk == "CRISIS":
        return _build_crisis_support_message(
            user_input=user_input,
            pdf_text=pdf_text,
            memory_snippets=memory_snippets,
        )

    # NORMAL SUPPORT
    return _build_standard_support_message(
        mode=mode,
        context=pdf_text,
        user_input=user_input,
        conversation_history=conversation_history,
    )
