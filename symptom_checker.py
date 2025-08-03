import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")

# Get symptom columns from training data
df = pd.read_csv("Training.csv")
all_symptoms = df.columns.tolist()
all_symptoms.remove("prognosis")

# Solutions for each disease
disease_solutions = {
    "Flu": "Rest, stay hydrated, take paracetamol for fever.",
    "Common Cold": "Drink warm fluids, rest, and avoid cold drinks.",
    "Migraine": "Take prescribed medicine, rest in a dark room.",
    "Food Poisoning": "Drink ORS, avoid spicy food, rest well.",
    "Asthma": "Use inhalers, avoid dust, and seek medical help if it worsens.",
    "Paralysis (Hemorrhage)": "Seek emergency care immediately and follow neurorehab.",
}

def predict_disease(symptoms_list):
    # Create binary input vector
    input_vector = [1 if symptom in symptoms_list else 0 for symptom in all_symptoms]
    
    # Predict disease
    prediction = model.predict([input_vector])[0]
    
    # Get suggestion
    suggestion = disease_solutions.get(prediction, "Please consult a medical professional.")
    
    return prediction, suggestion
