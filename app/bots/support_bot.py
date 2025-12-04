import os
import re
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


_client = None


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


def _persona_block(mode: str) -> str:
    """Tone for patient vs caregiver (used in non-crisis responses)."""
    mode = (mode or "").lower()
    if mode == "caregiver":
        return (
            "Provide supportive guidance to a caregiver who already has some\n"
            "medical understanding but may feel overwhelmed.\n"
            "- Acknowledge emotional burden and logistical stress.\n"
            "- Offer practical tips about communication with the care team,\n"
            "  organizing information, and watching for red flags.\n"
            "- Do not provide therapy or mental-health treatment, just supportive\n"
            "  education and validation.\n"
        )
    return (
        "Provide supportive, empathetic guidance directly to the patient.\n"
        "- Validate feelings (worry, confusion, frustration) in a non-clinical tone.\n"
        "- Offer simple, concrete steps for preparing questions, bringing a friend,\n"
        "  or writing things down.\n"
        "- Avoid toxic positivity; stay realistic but hopeful.\n"
    )


_DISCLAIMER = (
    "This is emotional and educational support only, not crisis counseling or "
    "medical advice. In an emergency, the person must contact local emergency services."
)


# =========================================================
# CRISIS DETECTION + LOCATION HELPERS
# =========================================================

def _classify_crisis_level(user_text: str) -> str:
    """
    Classify the user message into one of:
    - 'CRISIS'    : clear suicidal intent / self-harm risk
    - 'DISTRESS'  : strong negative emotions but no explicit intent
    - 'SAFE'      : neutral / routine support
    """
    client = _get_openai_client()

    prompt = f"""
Read this message and classify the level of emotional risk.

MESSAGE:
\"\"\"{user_text}\"\"\"

Return EXACTLY one of these labels:
- CRISIS    (clear suicidal intent or plans, self-harm, 'want to die', etc.)
- DISTRESS  (major distress but no explicit self-harm intent)
- SAFE      (no obvious crisis or self-harm language)
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    label = resp.choices[0].message.content.strip().upper()
    if label not in {"CRISIS", "DISTRESS", "SAFE"}:
        label = "SAFE"
    return label


_ZIP_REGEX = re.compile(r"\b\d{5}(?:-\d{4})?\b")


def _extract_zip_from_text(*texts: str) -> str | None:
    """Try to find a US-style ZIP code in any of the provided texts."""
    for t in texts:
        if not t:
            continue
        match = _ZIP_REGEX.search(t)
        if match:
            return match.group(0)
    return None


def _search_local_mental_health_resources(zip_code: str) -> str:
    """
    Use OpenAI web_search tool to find nearby mental-health / counseling centers.
    Returns markdown-list text that can be appended to the crisis response.
    """
    client = _get_openai_client()

    prompt = f"""
Use web search to find 3–5 mental-health or counseling resources near ZIP/postal code {zip_code}.
Prefer outpatient counseling centers, crisis lines, and community mental-health clinics.

For each result, return:
- Name
- Type (e.g., 'Counseling center', 'Hospital-based clinic', 'Hotline')
- Address (if available)
- Phone number
- Website (if available)

Format the answer as markdown bullet points.
"""

    # IMPORTANT: With the Responses API, web_search is enabled by declaring the tool.
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "user", "content": prompt}],
        tools=[{"type": "web_search"}],
        max_output_tokens=700,
    )

    return (response.output_text or "").strip()


# =========================================================
# MAIN SUPPORT GENERATORS
# =========================================================

def _build_standard_support_message(
    mode: str,
    brief_context_text: str,
    user_input: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 800,
) -> str:
    """Original non-crisis support behavior (with persona + disclaimer)."""
    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – a calm, compassionate assistant.

{persona}

You MUST:
- Avoid making promises about outcomes.
- Encourage the person to lean on their clinical team and social support.
- Gently remind them that online tools cannot replace real clinicians.

End with a short paragraph that rephrases:

{_DISCLAIMER}
"""

    user_content = (
        "Here is the brief context about the situation:\n"
        "--------------------\n"
        f"{brief_context_text}\n"
        "--------------------\n\n"
        "Here is what the person just said or asked:\n"
        "--------------------\n"
        f"{user_input}\n"
        "--------------------\n"
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_output_tokens=max_tokens,
    )

    return (response.output_text or "").strip()


def _build_crisis_support_message(
    mode: str,
    brief_context_text: str,
    user_input: str,
    pdf_text: str,
    memory_snippets,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 900,
) -> str:
    """
    Crisis-safe response:
    - Acknowledge feelings
    - Strongly recommend immediate real-world help (988 / emergency)
    - Optionally add local resource search if a ZIP is already known
    - Ask for ZIP only if not available (no forced sharing)
    """
    client = _get_openai_client()

    # 1) Always construct a core crisis message
    crisis_header = (
        "I’m really sorry you’re feeling this way. You’re not alone, and your safety matters.\n\n"
        "I am **not** a crisis service or a clinician, but I can help you find real people who "
        "can support you right now.\n\n"
        "**If you feel you might act on these thoughts or are in immediate danger,** "
        "please contact your local emergency services (such as 911 in the United States) "
        "or go to the nearest emergency room.\n\n"
        "In the U.S., you can also call or text **988** to reach the Suicide & Crisis Lifeline. "
        "They are available 24/7."
    )

    # 2) Try to find a ZIP from the current inputs (user text, report, memory)
    memory_text = "\n".join(memory_snippets or [])
    zip_code = _extract_zip_from_text(user_input, pdf_text, memory_text)

    local_resources_block = ""
    ask_for_zip_block = ""

    if zip_code:
        # 3a) If we already have a ZIP, use web_search to find nearby clinics
        try:
            resources_markdown = _search_local_mental_health_resources(zip_code)
            if resources_markdown:
                local_resources_block = (
                    f"\n\n---\n\n"
                    f"### Nearby mental-health resources around ZIP `{zip_code}`\n\n"
                    f"{resources_markdown}\n"
                )
        except Exception:
            # If web search fails, we still keep the core safety message
            local_resources_block = (
                "\n\n---\n\n"
                "I tried to look up nearby clinics but ran into a technical problem. "
                "You can still reach out to your local hospital, primary-care clinic, or 988 for guidance."
            )
    else:
        # 3b) No ZIP → gently ask for it (for next turn), without forcing
        ask_for_zip_block = (
            "\n\n---\n\n"
            "If you feel comfortable and it’s safe, you can share your **ZIP or postal code** here. "
            "On your next message, I can use it to look up nearby counseling or mental-health centers "
            "so you have more options close to where you are.\n\n"
            "If you don’t want to share that, it’s completely okay — you can still contact 988 "
            "or your local emergency services directly."
        )

    # 4) Assemble final message (no extra model call needed here)
    safety_tail = (
        "\n\n---\n\n"
        "This space is only for support and information. It cannot replace care from a real doctor, therapist, "
        "or crisis team. If your thoughts of self-harm get stronger, please reach out to **988**, "
        "your local crisis line, or emergency services right away."
    )

    return crisis_header + local_resources_block + ask_for_zip_block + safety_tail


# =========================================================
# PUBLIC API
# =========================================================

def generate_support_message(
    mode: str,
    brief_context_text: str,
    user_input: str,
    memory_snippets=None,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 800,
) -> str:
    """
    Main entry point used by the app.

    - Detects crisis level from the *user_input*
    - If CRISIS → crisis-specific flow + optional web_search
    - Else → original supportive explainer behavior
    """
    memory_snippets = memory_snippets or []
    risk_level = _classify_crisis_level(user_input)

    if risk_level == "CRISIS":
        # Crisis-safe branch with optional local resource lookup
        return _build_crisis_support_message(
            mode=mode,
            brief_context_text=brief_context_text,
            user_input=user_input,
            pdf_text=brief_context_text,
            memory_snippets=memory_snippets,
            model=model,
            max_tokens=max_tokens,
        )

    # Non-crisis or moderate distress → normal support behavior
    return _build_standard_support_message(
        mode=mode,
        brief_context_text=brief_context_text,
        user_input=user_input,
        model=model,
        max_tokens=max_tokens,
    )


def run_support(user_input: str, mode: str, pdf_text: str, memory_snippets):
    """
    Wrapper used by chat_app.generate_orchestrated_response(...)

    - `user_input`   : what the user just typed
    - `pdf_text`     : context from the medical report (often a summary)
    - `memory_snippets`: long-term memory for this user (may include location)
    """
    return generate_support_message(
        mode=mode,
        brief_context_text=pdf_text,
        user_input=user_input,
        memory_snippets=memory_snippets,
    )