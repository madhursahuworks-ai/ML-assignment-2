import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("data.csv")

# ✅ 1) Drop empty columns (e.g., 'Unnamed: 32')
df = df.dropna(axis=1, how="all")

# ✅ 2) Split features/target
X = df.drop(["id", "diagnosis"], axis=1)
y = df["diagnosis"].map({"M": 1, "B": 0})

# ✅ 3) Build preprocessing pipeline: impute -> scale
preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

results = []
joblib.dump(list(X.columns), "model/feature_columns.pkl")  # keep column order for app

# Fit preprocess once on training set, transform both
X_train_p = preprocess.fit_transform(X_train)
X_test_p = preprocess.transform(X_test)
joblib.dump(preprocess, "model/preprocess.pkl")

for name, model in models.items():
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)

    # some models may not have predict_proba; here all do
    y_prob = model.predict_proba(X_test_p)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

    joblib.dump(model, f"model/{name.replace(' ', '_')}.pkl")

pd.DataFrame(results).to_csv("results.csv", index=False)
print(pd.DataFrame(results))
