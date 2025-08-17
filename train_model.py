import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("Training.csv")
X = data.drop(columns=["prognosis"])
y = data["prognosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "symptom_columns.pkl")


with open("accuracy.txt", "w") as f:
    f.write(f"{accuracy:.2f}")
