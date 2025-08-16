# AI-medical

# 🩺 AI Symptom Checker

An intelligent web-based application that predicts possible diseases based on user-reported symptoms using machine learning and natural language processing.

---

## 🚀 Features

- Predicts diseases based on symptoms using a trained ML model  
- Cleans and processes user input with NLP techniques  
- Simple and intuitive web interface built with Flask and HTML  
- Easily extendable with new data or models  

---

## 🧠 How It Works

1. User enters symptoms via the web interface.  
2. Input is cleaned and processed using `nlp_utils.py`.  
3. The processed symptoms are passed to the trained model (`model.pkl`).  
4. The model predicts the most likely disease.  
5. Result is displayed back to the user.  

---

## 📁 Project Structure

AI_Symptom_Checker/ 
│ ├── backend/ │ 
  ├── app.py # Flask server │
  ├── symptom_checker.py # Disease prediction logic │
  ├── nlp_utils.py # Text preprocessing │
  ├── model.pkl # Trained ML model │
  ├── train_model.py # Model training script │
  ├── symptom_disease_dataset.csv # Dataset used for training │
  └── requirements.txt # Python dependencies │ 
├── templates/ │
  └── index.html # Web interface

1.Code

2.Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3.Install dependencies:
pip install -r requirements.txt

4.Train the model (optional if model.pkl is already present):
python train_model.py

5.Run the Flask app:
python app.py

6.Open your browser and go to:
http://localhost:xxxxxxx

