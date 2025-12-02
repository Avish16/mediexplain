import random
import uuid

def generate_demographics():
    names = ["John Carter", "Sarah Greene", "Michael Patel", "Emily Johnson"]
    hospitals = ["Syracuse General Hospital", "Upstate Medical Center", "Crouse Health"]
    doctors = ["Dr. Alan Smith", "Dr. Priya Sen", "Dr. Gabriel Lee"]

    return {
        "name": random.choice(names),
        "age": random.randint(25, 85),
        "gender": random.choice(["Male", "Female"]),
        "mrn": str(uuid.uuid4())[:8],
        "hospital": random.choice(hospitals),
        "physician": random.choice(doctors)
    }

def generate_vitals():
    return {
        "heart_rate": random.randint(60, 110),
        "blood_pressure": f"{random.randint(100, 150)}/{random.randint(60, 95)}",
        "respiratory_rate": random.randint(12, 22),
        "temperature": round(random.uniform(97.0, 102.5), 1),
        "spo2": random.randint(90, 100)
    }

def generate_labs():
    return {
        "cbc": {
            "WBC": round(random.uniform(4.0, 12.0), 1),
            "RBC": round(random.uniform(3.5, 5.8), 1),
            "Hemoglobin": round(random.uniform(10.0, 17.0), 1)
        },
        "lft": {
            "AST": random.randint(10, 60),
            "ALT": random.randint(10, 60),
            "Bilirubin": round(random.uniform(0.3, 2.5), 1)
        },
        "renal": {
            "Creatinine": round(random.uniform(0.6, 2.2), 1),
            "BUN": random.randint(7, 30)
        }
    }

def generate_medications():
    meds = [
        {"name": "Amlodipine", "dose": "5 mg", "frequency": "Once daily"},
        {"name": "Metformin", "dose": "500 mg", "frequency": "Twice daily"},
        {"name": "Atorvastatin", "dose": "20 mg", "frequency": "Once daily"}
    ]
    return random.sample(meds, random.randint(1, 3))

def generate_clinical_notes():
    templates = [
        "Patient presents with fatigue and intermittent dizziness. No acute distress. Continue monitoring vitals and adjust medications if symptoms worsen.",
        "Patient reports shortness of breath during exertion. Ordered chest X-ray and CBC panel for further evaluation.",
        "Patient recovering well post-discharge. Vitals stable. Continue current medication regimen and schedule follow-up in 2 weeks."
    ]
    return random.choice(templates)

def generate_synthetic_patient():
    return {
        "demographics": generate_demographics(),
        "vitals": generate_vitals(),
        "labs": generate_labs(),
        "medications": generate_medications(),
        "clinical_notes": generate_clinical_notes()
    }
