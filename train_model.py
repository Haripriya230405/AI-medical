# train_model.py

import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ----------------- Load Dataset -----------------
print("📂 Loading dataset...")
df = pd.read_csv("Training.csv")

X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Encode labels
label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)

# Save encoder & symptom columns
joblib.dump(label_enc, "label_encoder.pkl")
joblib.dump(list(X.columns), "symptom_columns.pkl")

# ----------------- Train/Test Split -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
)

# ----------------- Define Models with tuned parameters -----------------
models = {
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "rf": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42),
    "logreg": LogisticRegression(max_iter=500, random_state=42),
    "gb": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    "svm_linear": SVC(kernel="linear", probability=True, random_state=42)
}

# ----------------- Train & Evaluate -----------------
results = {}
classification_reports = {}

for name, model in models.items():
    try:
        print(f"⚡ Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"accuracy": float(acc)}
        
        # Decode to original class names for classification report
        y_test_labels = label_enc.inverse_transform(y_test)
        y_pred_labels = label_enc.inverse_transform(y_pred)
        report = classification_report(y_test_labels, y_pred_labels)
        classification_reports[name] = report
        
        # Save model
        joblib.dump(model, f"{name}.pkl")
        print(f"✅ {name} trained, Accuracy: {acc*100:.2f}%")
    except Exception as e:
        print(f"❌ {name} failed:", e)

# ----------------- Ensemble Model -----------------
print("⚡ Training ensemble model...")
ensemble_models = [(n, m) for n, m in models.items() if n in results]
ensemble = VotingClassifier(estimators=ensemble_models, voting="soft")
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)
results["ensemble"] = {"accuracy": float(acc)}

# Classification report for ensemble
y_test_labels = label_enc.inverse_transform(y_test)
y_pred_labels = label_enc.inverse_transform(y_pred)
classification_reports["ensemble"] = classification_report(y_test_labels, y_pred_labels)

# Save ensemble model
joblib.dump(ensemble, "ensemble.pkl")
print(f"🏆 Ensemble trained, Accuracy: {acc*100:.2f}%")

# ----------------- Save Best Model -----------------
best_model = max(results, key=lambda k: results[k]["accuracy"])
with open("best_model.txt", "w") as f:
    f.write(best_model)

# ----------------- Save Results & Classification Reports -----------------
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=4)

with open("classification_report.txt", "w") as f:
    for model_name, report in classification_reports.items():
        f.write(f"--- {model_name} ---\n")
        f.write(report)
        f.write("\n\n")

print(f"✅ Training complete. Best Model: {best_model} ({results[best_model]['accuracy']*100:.2f}%)")
print("✅ Classification reports saved to classification_report.txt")
