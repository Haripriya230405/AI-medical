import difflib
import joblib

def process_text(symptom_string):
    """
    Process the user input symptom string:
    - Splits by commas
    - Cleans and lowercases symptoms
    - Matches against known symptoms with fuzzy matching
    - Returns (corrected_symptoms, unknown_symptoms)
    """
    # Load known symptoms from file
    known_symptoms = joblib.load("symptom_columns.pkl")

    # Clean input
    user_symptoms = [s.strip().lower() for s in symptom_string.split(",") if s.strip()]

    corrected_symptoms = []
    unknown_symptoms = []

    for symptom in user_symptoms:
        # Fuzzy match
        closest = difflib.get_close_matches(symptom, known_symptoms, n=1, cutoff=0.7)
        if closest:
            corrected_symptoms.append(closest[0])
        else:
            unknown_symptoms.append(symptom)

    return corrected_symptoms, unknown_symptoms
