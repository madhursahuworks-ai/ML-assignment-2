import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Breast Cancer Classification App")

uploaded_file = st.file_uploader("Upload CSV Test Data", type="csv")

model_name = st.selectbox("Select Model", [
    "Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"
])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # drop empty columns
    df = df.dropna(axis=1, how="all")

    # ensure expected columns
    feature_cols = joblib.load("model/feature_columns.pkl")
    X = df[feature_cols]
    y = df["diagnosis"].map({"M": 1, "B": 0})

    preprocess = joblib.load("model/preprocess.pkl")
    X_p = preprocess.transform(X)

    model = joblib.load(f"model/{model_name.replace(' ', '_')}.pkl")
    preds = model.predict(X_p)

    st.subheader("Classification Report")
    st.text(classification_report(y, preds))

    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
