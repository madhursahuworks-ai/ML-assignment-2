import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# -----------------------------
# UI: Title + Project Details
# -----------------------------
st.title("Breast Cancer Classification App")

st.markdown(
    """
**Project Summary (BITS ML Assignment-2)**  
This app demonstrates an end-to-end Machine Learning workflow for **binary classification** on the
**Breast Cancer Wisconsin (Diagnostic)** dataset. It supports:

-  Uploading a **CSV test dataset**
-  Selecting one of **6 classification models**
-  Showing **all required evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC
-  Visualizing results via **Confusion Matrix** and **Classification Report**

**Expected CSV format**
- Must contain column: `diagnosis` with values `M` (malignant) or `B` (benign)
- Must contain the same feature columns used during training
"""
)

st.divider()

# -----------------------------
# Inputs
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV Test Data", type="csv")

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"],
)

# -----------------------------
# Helpers
# -----------------------------
def _show_dataset_preview(df: pd.DataFrame, n: int = 5) -> None:
    st.subheader("Dataset Preview")
    st.caption(f"Showing first {n} rows")
    st.dataframe(df.head(n), use_container_width=True)


def _fail_with_details(msg: str, details: str | None = None) -> None:
    st.error(msg)
    if details:
        st.code(details)
    st.stop()


# -----------------------------
# Main
# -----------------------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        _fail_with_details(" Could not read the uploaded CSV. Please upload a valid CSV file.", str(e))

    if df is None or df.empty:
        _fail_with_details(" Uploaded CSV is empty. Please upload a non-empty dataset.")

    # Drop columns that are fully NaN (e.g., 'Unnamed: 32' in some versions)
    df = df.dropna(axis=1, how="all")

    _show_dataset_preview(df)

    # Load training-time expected columns
    try:
        feature_cols = joblib.load("model/feature_columns.pkl")
        preprocess = joblib.load("model/preprocess.pkl")
    except Exception as e:
        _fail_with_details(
            " Model artifacts not found. Make sure you ran training and have these files:\n"
            "- model/feature_columns.pkl\n- model/preprocess.pkl\n- model/<ModelName>.pkl",
            str(e),
        )

    # Validate required columns
    required_cols = set(feature_cols) | {"diagnosis"}
    present_cols = set(df.columns)

    missing_required = sorted(list(required_cols - present_cols))
    extra_cols = sorted(list(present_cols - required_cols))  # not fatal, just info

    if missing_required:
        _fail_with_details(
            " Uploaded dataset is missing required columns.",
            "Missing columns:\n- " + "\n- ".join(missing_required),
        )

    # Validate diagnosis values
    diag_raw = df["diagnosis"]
    if diag_raw.isna().any():
        _fail_with_details(
            " Column 'diagnosis' contains missing values (NaN). Please remove/replace them.",
            f"NaN count in diagnosis: {int(diag_raw.isna().sum())}",
        )

    allowed_diag = {"M", "B"}
    invalid_diag = sorted(set(diag_raw.astype(str).unique()) - allowed_diag)
    if invalid_diag:
        _fail_with_details(
            " Column 'diagnosis' has invalid values. Allowed values are only 'M' and 'B'.",
            "Invalid values found:\n- " + "\n- ".join(invalid_diag),
        )

    # Build X, y in the same column order used during training
    X = df[feature_cols]
    y = df["diagnosis"].map({"M": 1, "B": 0})

    # Check for non-numeric columns in features
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        _fail_with_details(
            " Some feature columns are not numeric. Please ensure all feature values are numbers.",
            "Non-numeric feature columns:\n- " + "\n- ".join(non_numeric),
        )

    # Quick NaN summary (pre-imputation)
    nan_counts = X.isna().sum()
    total_nans = int(nan_counts.sum())
    if total_nans > 0:
        st.warning(
            f" Uploaded data contains {total_nans} missing feature values. "
            "They will be handled by the saved preprocessing pipeline (median imputation)."
        )
        with st.expander("See NaN counts per column"):
            st.dataframe(nan_counts[nan_counts > 0].sort_values(ascending=False), use_container_width=True)

    # Transform using saved preprocessing pipeline
    try:
        X_p = preprocess.transform(X)
    except Exception as e:
        _fail_with_details(
            " Preprocessing failed. This can happen if the uploaded data has incompatible types or values.",
            str(e),
        )

    # Load chosen model
    model_path = f"model/{model_name.replace(' ', '_')}.pkl"
    try:
        model = joblib.load(model_path)
    except Exception as e:
        _fail_with_details(f" Could not load model file: {model_path}", str(e))

    # Predict
    try:
        preds = model.predict(X_p)
    except Exception as e:
        _fail_with_details(" Prediction failed. Please check your uploaded data.", str(e))

    # Metrics (on uploaded data)
    try:
        auc = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_p)[:, 1]
            auc = roc_auc_score(y, y_prob)

        metrics = {
            "Accuracy": accuracy_score(y, preds),
            "AUC": auc,
            "Precision": precision_score(y, preds),
            "Recall": recall_score(y, preds),
            "F1": f1_score(y, preds),
            "MCC": matthews_corrcoef(y, preds),
        }
    except Exception as e:
        _fail_with_details(" Metric calculation failed.", str(e))

    st.subheader("Evaluation Metrics (on uploaded data)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['Precision']:.4f}")
    c3.metric("Recall", f"{metrics['Recall']:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("F1", f"{metrics['F1']:.4f}")
    c5.metric("MCC", f"{metrics['MCC']:.4f}")
    c6.metric("AUC", "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}")

    st.table(pd.DataFrame([metrics]))

    # Helpful info about column matching (extra cols are okay)
    with st.expander("Column check details"):
        st.write(f" Required columns present: {len(required_cols)}")
        st.write(f" Feature columns used: {len(feature_cols)}")
        if extra_cols:
            st.info(
                "ℹ️ Your uploaded CSV contains extra columns not used by the model. "
                "That’s okay; they were ignored.\n\nExtra columns:\n- " + "\n- ".join(extra_cols)
            )
        else:
            st.write("No extra columns found.")

    # Classification report + confusion matrix
    st.subheader("Classification Report")
    st.text(classification_report(y, preds))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
