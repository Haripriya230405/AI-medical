import joblib
import numpy as np

# ----------------- Load Model and Features -----------------
try:
    model = joblib.load("model.pkl")   # Default model
except FileNotFoundError:
    model = None
    print("⚠️ Warning: model.pkl not found. Train your model first using train_model.py")

try:
    symptoms_list = joblib.load("symptom_columns.pkl")
except FileNotFoundError:
    symptoms_list = []
    print("⚠️ Warning: symptom_columns.pkl not found. Make sure it's created during training.")

# ----------------- Symptom / Disease Tips -----------------
symptom_tips = {
    "itching": [
        "Apply anti-itch creams or ointments.",
        "Avoid scratching the affected area.",
        "Take antihistamines if recommended."
    ],
    "shivering": [
        "Keep yourself warm with blankets or warm clothing.",
        "Drink warm fluids.",
        "Rest and monitor your temperature."
    ],
    "skin_rash": [
        "Apply soothing lotions or creams like aloe vera.",
        "Avoid harsh soaps and irritants.",
        "Consult a doctor if the rash persists or spreads."
    ],
    "mood_swings": [
        "Practice relaxation techniques like meditation or deep breathing.",
        "Engage in light physical activity.",
        "Maintain a healthy sleep and diet routine."
    ],
    "weight_loss": [
        "Ensure you have a balanced and nutritious diet.",
        "Consult a healthcare professional for underlying causes.",
        "Monitor your weight regularly."
    ],
    "sweating": [
        "Wear breathable clothing and stay hydrated.",
        "Avoid spicy foods and caffeine if it triggers sweating.",
        "Practice good hygiene to prevent skin irritation."
    ],

    "cough": [
        "Drink warm fluids like soup or tea.",
        "Use cough syrups or lozenges as needed.",
        "Avoid cold drinks and dusty environments."
    ],

    "headache": [
        "Rest in a quiet, dark room.",
        "Take mild pain relief like paracetamol.",
        "Stay hydrated and avoid screen time."
    ],
    "fatigue": [
        "Get 7–8 hours of sleep.",
        "Eat fruits and iron-rich food.",
        "Reduce physical and mental exertion."
    ],
    "vomiting": [
        "Stay hydrated with ORS or coconut water.",
        "Avoid solid food until vomiting subsides.",
        "Consult a doctor if it persists more than 24 hours."
    ],
    "nausea": [
        "Sip ginger tea or peppermint water.",
        "Eat light, bland meals.",
        "Avoid strong odors and spicy food."
    ],
    "diarrhea": [
        "Take oral rehydration solution (ORS).",
        "Avoid dairy and greasy food temporarily.",
        "Visit a doctor if it lasts more than 2 days."
    ],
    "shortness of breath": [
        "Sit upright and breathe slowly.",
        "Avoid physical strain.",
        "Seek emergency medical care if severe."
    ],
    "fungal infection": [
        "Keep the affected area clean and dry.",
        "Avoid tight or synthetic clothing.",
        "Apply antifungal creams as directed.",
        "Don’t share towels or personal items."
    ],
    "allergy": [
        "Avoid known allergens (dust, pollen, foods).",
        "Take antihistamines if prescribed.",
        "Keep your surroundings clean and dust-free.",
        "Use air purifiers if needed."
    ],
    "GERD": [
        "Eat small, frequent meals.",
        "Avoid spicy and fatty foods.",
        "Do not lie down immediately after eating.",
        "Elevate head during sleep."
    ],
    "chronic cholestasis": [
        "Follow a low-fat, high-fiber diet.",
        "Avoid alcohol and hepatotoxic substances.",
        "Stay hydrated.",
        "Monitor liver function regularly."
    ],
    "drug reaction": [
        "Discontinue the suspected medication immediately.",
        "Consult a healthcare provider.",
        "Avoid self-medication.",
        "Wear a medical alert bracelet if needed."
    ],
    "peptic ulcer disease": [
        "Avoid NSAIDs and spicy food.",
        "Eat bland and small meals.",
        "Avoid smoking and alcohol.",
        "Take antacids as prescribed."
    ],
    "AIDS": [
        "Adhere strictly to antiretroviral therapy (ART).",
        "Eat a balanced and nutritious diet.",
        "Avoid infections and maintain hygiene.",
        "Get regular medical check-ups."
    ],
    "diabetes": [
        "Monitor blood sugar levels regularly.",
        "Exercise regularly and maintain a healthy weight.",
        "Follow a diabetic-friendly diet.",
        "Take medications/insulin on time."
    ],
    "gastroenteritis": [
        "Stay hydrated using ORS or fluids.",
        "Avoid oily or dairy foods.",
        "Eat light, soft meals.",
        "Maintain hygiene and wash hands frequently."
    ],
    "bronchial asthma": [
        "Avoid dust, smoke, and strong odors.",
        "Use inhalers as prescribed.",
        "Practice breathing exercises.",
        "Avoid cold environments if sensitive."
    ],
    "migraine": [
        "Avoid triggers like bright lights and strong smells.",
        "Stay hydrated and take breaks from screens.",
        "Use cold compress and rest in a dark room.",
        "Take prescribed medications at onset."
    ],
    "cervical spondylosis": [
        "Maintain correct posture.",
        "Do neck strengthening exercises.",
        "Avoid prolonged use of mobile or laptop.",
        "Use cervical pillow if needed."
    ],
    "paralysis (brain hemorrhage)": [
        "Follow rehabilitation and physiotherapy.",
        "Manage blood pressure and sugar levels.",
        "Take medications consistently.",
        "Avoid smoking and alcohol."
    ],
    "jaundice": [
        "Avoid oily and spicy food.",
        "Stay well hydrated.",
        "Avoid alcohol completely.",
        "Get liver function tests regularly."
    ],
    "malaria": [
        "Use mosquito nets and repellents.",
        "Remove stagnant water nearby.",
        "Complete the prescribed medication course.",
        "Stay hydrated and rest."
    ],
    "chicken pox": [
        "Avoid scratching, use calamine lotion.",
        "Wear loose clothing.",
        "Rest and avoid contact with others.",
        "Keep fingernails trimmed."
    ],
    "dengue": [
        "Take rest and stay hydrated.",
        "Avoid aspirin, use paracetamol for fever.",
        "Monitor platelet count if advised.",
        "Use mosquito protection."
    ],
    "typhoid": [
        "Take antibiotics as prescribed.",
        "Eat soft, bland food.",
        "Avoid outside food and untreated water.",
        "Rest adequately."
    ],
    "hepatitis A": [
        "Take proper rest and avoid strenuous activity.",
        "Eat a liver-friendly diet.",
        "Drink boiled or purified water.",
        "Avoid alcohol and junk food."
    ],
    "common cold": [
        "Rest and drink warm fluids.",
        "Use steam inhalation and nasal drops.",
        "Avoid cold environments.",
        "Gargle with warm salt water."
    ],
    "pneumonia": [
        "Complete the full course of antibiotics.",
        "Drink fluids and rest.",
        "Avoid cold air and smoke.",
        "Seek medical help for severe cases."
    ],
    "heart attack": [
        "Seek emergency help immediately.",
        "Chew an aspirin if advised.",
        "Rest and avoid exertion.",
        "Adopt a heart-healthy lifestyle afterward."
    ],
    "varicose veins": [
        "Avoid standing for long periods.",
        "Elevate legs while resting.",
        "Wear compression stockings.",
        "Exercise regularly (e.g., walking)."
    ],
    "hypothyroidism": [
        "Take thyroid medication daily.",
        "Exercise regularly.",
        "Eat a diet rich in iodine and selenium.",
        "Avoid raw cruciferous vegetables in excess."
    ],
    "osteoarthritis": [
        "Do gentle exercise like walking or swimming.",
        "Maintain a healthy weight.",
        "Apply hot or cold compress for pain relief.",
        "Avoid repetitive stress on joints."
    ],
    "acne": [
        "Wash your face with a gentle cleanser twice daily.",
        "Avoid picking or popping pimples.",
        "Use non-comedogenic skincare products.",
        "Stay hydrated and avoid oily food."
    ],
    "urinary tract infection": [
        "Drink plenty of water.",
        "Do not hold urine for long periods.",
        "Urinate after sexual activity.",
        "Complete the prescribed course of antibiotics."
    ],
    "psoriasis": [
        "Moisturize skin daily.",
        "Avoid harsh soaps and cold environments.",
        "Follow stress-reducing routines.",
        "Take prescribed medications or phototherapy."
    ],
    "hypertension": [
        "Reduce salt intake.",
        "Exercise regularly (e.g., brisk walking).",
        "Monitor blood pressure at home.",
        "Take medications on time and avoid stress."
    ]
}

# ----------------- Helper Functions -----------------
def get_symptom_tips(user_symptoms):
    """Return health tips based on user symptoms."""
    tips = []
    for symptom in user_symptoms:
        symptom = symptom.lower().strip()
        if symptom in symptom_tips:
            tips.extend(symptom_tips[symptom])
    return tips if tips else ["Please consult a doctor for proper diagnosis and treatment."]

def predict_disease(user_symptoms):
    """Predict disease based on user symptoms using the trained model."""
    if model is None or not symptoms_list:
        return "Model not available", ["Please train the model first."], []

    # Convert symptoms to binary vector
    input_vector = [1 if symptom in user_symptoms else 0 for symptom in symptoms_list]
    input_vector = np.array(input_vector).reshape(1, -1)

    # Make prediction
    predicted_disease = model.predict(input_vector)[0]

    # Get tips
    tips = get_symptom_tips(user_symptoms)

    return predicted_disease, tips, tips
