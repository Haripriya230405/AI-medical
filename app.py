from flask import Flask, request, render_template
from symptom_checker import predict_disease
from nlp_utils import process_text

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms', '').strip()

    if not symptoms:
        return render_template('index.html', prediction="❌ No symptoms provided.", suggestion="Please enter at least one symptom.")

    # Step 1: Clean input
    processed = process_text(symptoms)

    # Step 2: Predict disease + suggestion
    disease, tip = predict_disease(processed)

    # Step 3: Send both to the template
    return render_template('index.html', prediction=disease, suggestion=tip)

if __name__ == '__main__':
    app.run(debug=True)
