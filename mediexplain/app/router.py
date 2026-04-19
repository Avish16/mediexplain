from app.bots.explainer_bot import ExplainerBot
from app.bots.labs_bot import LabsBot

class RouterBot:
    def __init__(self):
        self.explainer = ExplainerBot()
        self.labs = LabsBot()

    def route(self, text):
        text_lower = text.lower()

        # If labs info detected
        if any(keyword in text_lower for keyword in ["lab", "glucose", "mg/dl", "cbc", "hemoglobin"]):
            return self.labs.explain_labs(text)

        # Default to explainer bot
        return self.explainer.explain(text)
