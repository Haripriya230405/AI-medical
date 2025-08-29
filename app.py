from flask import Flask, request, render_template, redirect, url_for, session
from nlp_utils import process_text
from symptom_checker import get_symptom_tips
import os, csv, json
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

app = Flask(__name__, template_folder='templates')
app.secret_key = "your_secret_key"

# ---------------------- Utility Functions ----------------------
def get_accuracy():
    try:
        if not os.path.exists("model_results.json") or not os.path.exists("best_model.txt"):
            return "Unavailable", "Unavailable"

        with open("model_results.json", "r") as f:
            results = json.load(f)

        with open("best_model.txt", "r") as f:
            best_model = f.read().strip()

        if best_model not in results:
            return best_model, "Unavailable"

        acc = results[best_model].get("accuracy", None)
        if acc is None:
            return best_model, "Unavailable"

        # Convert decimals like 0.94 → 94.00
        if acc <= 1:
            acc = acc * 100

        return best_model, f"{acc:.2f}"
    except Exception as e:
        print("⚠️ Accuracy fetch error:", e)
        return "Unavailable", "Unavailable"

def save_history(symptoms, disease, tips):
    with open("history.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([now, ", ".join(symptoms), disease, " | ".join(tips)])

# ---------------------- Routes ----------------------
@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        mobile = request.form['mobile']
        address = request.form['address']

        session['user'] = {
            'name': name,
            'age': age,
            'mobile': mobile,
            'address': address
        }
        session['user_name'] = name

        with open("users.csv", mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([now, name, age, mobile, address])

        return redirect(url_for('symptom_checker'))
    return render_template("login.html")

@app.route('/symptom')
def symptom_checker():
    if 'user_name' not in session:
        return redirect(url_for('login'))

    best_model, accuracy = get_accuracy()

    return render_template(
        "index.html",
        accuracy=accuracy,
        user_name=session.get("user_name"),
        best_model=best_model
    )

@app.route('/predict', methods=['POST'])
def predict():
    symptoms_raw = request.form.get('symptoms', '').strip()
    best_model, accuracy = get_accuracy()
    selected_model = best_model

    if not symptoms_raw:
        return render_template(
            'index.html',
            prediction="❌ No symptoms provided.",
            suggestion=["Please enter at least one symptom."],
            accuracy=accuracy,
            user_name=session.get("user_name"),
            best_model=best_model
        )

    processed, unknown = process_text(symptoms_raw)

    if unknown:
        return render_template(
            'index.html',
            prediction="❌ Unknown symptoms detected.",
            suggestion=[f"'{sym}' not recognized." for sym in unknown] +
                       ["Please correct your input or try similar terms."],
            accuracy=accuracy,
            user_symptoms=symptoms_raw,
            user_name=session.get("user_name"),
            best_model=best_model
        )

    # ------------------ Load best model ------------------
    all_symptoms = joblib.load("symptom_columns.pkl")
    label_enc = joblib.load("label_encoder.pkl")

    input_vector = [1 if symptom in processed else 0 for symptom in all_symptoms]
    input_df = pd.DataFrame([input_vector], columns=all_symptoms)

    try:
        model = joblib.load(f"{selected_model}.pkl")
    except:
        model = joblib.load("ensemble.pkl") if os.path.exists("ensemble.pkl") else joblib.load("model.pkl")

    # Predict
    try:
        probs = model.predict_proba(input_df)[0]
        pred_idx = np.argmax(probs)
        disease = label_enc.inverse_transform([pred_idx])[0]
    except:
        pred_idx = model.predict(input_df)[0]
        disease = label_enc.inverse_transform([pred_idx])[0] if isinstance(pred_idx, (int, np.integer)) else str(pred_idx)

    tips = get_symptom_tips(processed)
    save_history(processed, disease, tips)

    return render_template(
        "index.html",
        prediction=disease,
        suggestion=tips,
        accuracy=accuracy,
        user_symptoms=symptoms_raw,
        user_name=session.get("user_name"),
        best_model=selected_model
    )

@app.route('/history')
def view_history():
    history_data = []
    try:
        with open("history.csv", mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            history_data = list(reader)
    except FileNotFoundError:
        pass
    return render_template("history.html", history=history_data)

@app.route('/delete_history')
def delete_history():
    with open("history.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Symptoms", "Predicted Disease", "Suggested Tips"])
    return render_template("history.html", history=[])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---------------------- Visualization ----------------------
@app.route('/visualization')
def visualization():
    df = pd.read_csv("Training.csv")
    symptom_counts = df.drop(columns=["prognosis"]).sum().sort_values(ascending=False)
    top_10_symptoms = symptom_counts.head(10).index.tolist()
    df_top = df[top_10_symptoms]
    corr = df_top.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap - Top 10 Most Common Symptoms", fontsize=16)

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template("visualization.html", plot_url=plot_url)

@app.route('/confusion')
def confusion_view():
    df = pd.read_csv("Training.csv")
    all_symptoms = joblib.load("symptom_columns.pkl")
    model = joblib.load("ensemble.pkl")
    label_enc = joblib.load("label_encoder.pkl")

    X = df.drop(columns=["prognosis"])
    y = df["prognosis"].astype(str)

    for c in all_symptoms:
        if c not in X.columns:
            X[c] = 0
    X = X[all_symptoms]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    if np.issubdtype(y_pred.dtype, np.number):
        y_pred = label_enc.inverse_transform(y_pred)

    acc = accuracy_score(y_test, y_pred)

    labels = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=False, cmap="Blues", fmt="d", cbar=True)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")

    return render_template(
        "confusion.html",
        plot_url=plot_url,
        accuracy=f"{acc:.4f}"
    )

# ---------------------- Init ----------------------
if __name__ == '__main__':
    if not os.path.exists("history.csv"):
        with open("history.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Symptoms", "Predicted Disease", "Suggested Tips"])

    if not os.path.exists("users.csv"):
        with open("users.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Name", "Age", "Mobile", "Address"])

    app.run(debug=True)
