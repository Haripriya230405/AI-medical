from flask import Flask, request, render_template, redirect, url_for, session
from symptom_checker import predict_disease
from nlp_utils import process_text
import os
import csv
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

app = Flask(__name__, template_folder='templates')
app.secret_key = "your_secret_key"

# ---------------------- Utility Functions ----------------------
def get_accuracy():
    try:
        with open("accuracy.txt", "r") as f:
            return f.read().strip()
    except:
        return "Not Available"

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
    return render_template("index.html", accuracy=get_accuracy(), user_name=session.get("user_name"))

@app.route('/predict', methods=['POST'])
def predict():
    symptoms_raw = request.form.get('symptoms', '').strip()

    if not symptoms_raw:
        return render_template(
            'index.html',
            prediction="❌ No symptoms provided.",
            suggestion=["Please enter at least one symptom."],
            accuracy=get_accuracy(),
            user_name=session.get("user_name")
        )

    processed, unknown = process_text(symptoms_raw)

    if unknown:
        return render_template(
            'index.html',
            prediction="❌ Unknown symptoms detected.",
            suggestion=[f"'{sym}' not recognized." for sym in unknown] + ["Please correct your input or try similar terms."],
            accuracy=get_accuracy(),
            user_symptoms=symptoms_raw,
            user_name=session.get("user_name")
        )

    disease, tips, _ = predict_disease(processed)
    save_history(processed, disease, tips)

    return render_template(
        "index.html",
        prediction=disease,
        suggestion=tips,
        accuracy=get_accuracy(),
        user_symptoms=symptoms_raw,
        user_name=session.get("user_name")
    )

@app.route('/history')
def view_history():
    history_data = []
    try:
        with open("history.csv", mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
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

# ---------------------- Visualization Route (Correlation Heatmap) ----------------------
@app.route('/visualization')
def visualization():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    import base64

    # Load dataset
    df = pd.read_csv("Training.csv")

    # Count symptoms (excluding prognosis column)
    symptom_counts = df.drop(columns=["prognosis"]).sum().sort_values(ascending=False)

    # Take top 10 symptoms
    top_10_symptoms = symptom_counts.head(10).index.tolist()

    # Filter dataframe to top 10 symptoms only
    df_top = df[top_10_symptoms]

    # Create correlation matrix
    corr = df_top.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap - Top 10 Most Common Symptoms", fontsize=16)

    # Save to image buffer
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template("visualization.html", plot_url=plot_url)

# ---------------------- Confusion Matrix Route ----------------------
@app.route('/confusion')
def confusion_view():
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import pandas as pd
    import joblib
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Load dataset & model
    df = pd.read_csv("Training.csv")
    all_symptoms = joblib.load("symptom_columns.pkl")
    model = joblib.load("model.pkl")

    X = df.drop(columns=["prognosis"])
    y = df["prognosis"].astype(str)

    # Ensure all symptom columns exist
    for c in all_symptoms:
        if c not in X.columns:
            X[c] = 0
    X = X[all_symptoms]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Predictions
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Confusion matrix
    labels = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Convert confusion matrix to DataFrame for better labeling
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=False, cmap="Blues", fmt="d", cbar=True)
    plt.title(f"Confusion Matrix (Accuracy: {acc*100:.2f}%)", fontsize=16)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Save to image buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Classification report
    report_text = classification_report(y_test, y_pred, zero_division=0)

    return render_template(
        "confusion.html",
        plot_url=plot_url,
        accuracy=f"{acc*100:.2f}",
        report_text=report_text
    )

# ---------------------- Initialize CSVs ----------------------
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

