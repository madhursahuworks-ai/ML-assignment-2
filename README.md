a. Problem Statement

The objective of this project is to build, evaluate, and deploy multiple Machine Learning classification models to predict whether a breast tumor is malignant (M) or benign (B) using the Breast Cancer Wisconsin (Diagnostic) dataset.
The project demonstrates an end-to-end ML workflow including data preprocessing, model training, evaluation using multiple metrics, building an interactive Streamlit UI, and cloud deployment.

b. Dataset Description

Dataset: Breast Cancer Wisconsin (Diagnostic)
Source: UCI Machine Learning Repository / Kaggle

Description:
The dataset contains diagnostic measurements of breast mass tissue extracted from digitized images of fine needle aspirate (FNA) of breast masses.

Key Details:

Attribute	Value
Total Samples	569
Total Features	30
Target Variable	diagnosis (M = Malignant, B = Benign)
Feature Type	Numeric (mean, standard error, worst values of cell nucleus properties)
Missing Values	Handled using median imputation
Problem Type	Binary Classification

c. Models & Evaluation Metrics
The following 6 classification models were implemented and evaluated on the same dataset:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Naive Bayes (Gaussian)
Random Forest (Ensemble)
XGBoost (Ensemble)

Evaluation Metrics Used :
Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)

| ML Model Name       | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
| ------------------- | -------- | ---- | --------- | ------ | ---- | ---- |
| Logistic Regression | 0.97     | 0.99 | 0.98      | 0.96   | 0.97 | 0.94 |
| Decision Tree       | 0.93     | 0.93 | 0.93      | 0.92   | 0.92 | 0.85 |
| KNN                 | 0.96     | 0.98 | 0.97      | 0.95   | 0.96 | 0.92 |
| Naive Bayes         | 0.94     | 0.96 | 0.95      | 0.92   | 0.93 | 0.88 |
| Random Forest       | 0.98     | 0.99 | 0.99      | 0.97   | 0.98 | 0.96 |
| XGBoost             | 0.99     | 0.99 | 0.99      | 0.98   | 0.99 | 0.97 |


d. Observations

| ML Model Name       | Observation about Model Performance                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Strong baseline model with good interpretability and high accuracy on this dataset.                      |
| Decision Tree       | Simple to interpret but slightly prone to overfitting; lower generalization compared to ensemble models. |
| KNN                 | Performs well after feature scaling; sensitive to choice of k and distance metric.                       |
| Naive Bayes         | Fast and simple but assumes feature independence, which limits performance.                              |
| Random Forest       | Excellent performance due to ensemble averaging; robust to noise and overfitting.                        |
| XGBoost             | Best overall performance with high accuracy and MCC; powerful gradient boosting ensemble.                |
