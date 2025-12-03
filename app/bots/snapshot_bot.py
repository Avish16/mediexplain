import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


_client = None


def _get_openai_client() -> OpenAI:
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
    mode = (mode or "").lower()
    if mode == "caregiver":
        return (
            "Generate a concise but information-dense snapshot suitable for an\n"
            "experienced caregiver to quickly understand the case.\n"
            "- Use a mini-problem list, key labs/imaging findings, and current therapy.\n"
            "- You may mention ICD-10 style categories in parentheses.\n"
            "- Output should be like a brief handoff note.\n"
        )
    return (
        "Generate a one-page style snapshot for a patient.\n"
        "- Use sections: 'Big Picture', 'What is Going On', 'What is Being Done',\n"
        "  and 'What to Watch For'.\n"
        "- Keep it very readable and not overwhelming.\n"
    )


_DISCLAIMER = (
    "This snapshot is just a summary of the report and should not be used as a "
    "stand-alone medical record."
)


def generate_snapshot(
    mode: str,
    full_case_text: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 900,
) -> str:
    """
    Snapshot bot: condensed overview.

    Parameters
    ----------
    mode : 'patient' or 'caregiver'
    full_case_text : entire extracted report / EMR section

    Returns
    -------
    Short markdown snapshot.
    """
    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – your task is to build a very compact 'snapshot' of this case.

{persona}

The snapshot must fit conceptually on about one page of text and avoid
unnecessary detail.

Always close with one sentence reminding that:

{_DISCLAIMER}
"""

    user_content = (
        "Here is the full report to condense into a snapshot:\n"
        "--------------------\n"
        f"{full_case_text}\n"
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

