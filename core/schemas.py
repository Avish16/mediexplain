synthetic_patient_schema = {
    "demographics": {
        "name": str,
        "age": int,
        "gender": str,
        "mrn": str,
        "hospital": str,
        "physician": str
    },
    "vitals": {
        "heart_rate": int,
        "blood_pressure": str,
        "respiratory_rate": int,
        "temperature": float,
        "spo2": int
    },
    "labs": {
        "cbc": dict,
        "lft": dict,
        "renal": dict
    },
    "medications": list,
    "clinical_notes": str
}
