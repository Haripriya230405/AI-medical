import difflib
import joblib

def process_text(symptom_string):
    user_symptoms = [s.strip().lower() for s in symptom_string.split(",")]
    corrected_symptoms = []
    unknown_symptoms = []

    known_symptoms = joblib.load("symptom_columns.pkl")

    for symptom in user_symptoms:
        closest = difflib.get_close_matches(symptom, known_symptoms, n=1, cutoff=0.7)
        if closest:
            corrected_symptoms.append(closest[0])
        else:
            unknown_symptoms.append(symptom)

    return corrected_symptoms, unknown_symptoms
# venv\Scripts\activate