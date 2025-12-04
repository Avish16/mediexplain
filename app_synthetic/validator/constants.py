# app/app_synthetic/validator/constants.py

INTENT_LABELS = [
    "medication_question",
    "lab_explanation",
    "general_health",
    "follow_up",
    "unknown",
]

BOT_NAMES = [
    "ExplainerBot",
    "MedicationBot",
    "LabsBot",
    "GeneralHealthBot",
    "FallbackBot",
]

SAFETY_LEVELS = {
    "safe": "✅ Safe – no safety issues detected.",
    "transform": "⚠️ Transformed – unsafe content softened or rewritten.",
    "block": "⛔ Blocked – unsafe to answer.",
}

DEFAULT_TOP_K = 5
