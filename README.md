a. Problem Statement

Build and deploy multiple classification models to predict whether a tumor is malignant or benign.

b. Dataset Description

Breast Cancer Wisconsin Dataset with 569 samples and 30 numeric features.

c. Models & Evaluation Metrics
Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.97	0.99	0.98	0.96	0.97	0.94
Decision Tree	0.93	0.93	0.93	0.92	0.92	0.85
KNN	0.96	0.98	0.97	0.95	0.96	0.92
Naive Bayes	0.94	0.96	0.95	0.92	0.93	0.88
Random Forest	0.98	0.99	0.99	0.97	0.98	0.96
XGBoost	0.99	0.99	0.99	0.98	0.99	0.97
d. Observations
Model	Observation
Logistic Regression	Strong baseline, interpretable
Decision Tree	Overfits slightly
KNN	Sensitive to scaling
Naive Bayes	Fast but simplistic
Random Forest	Best tradeoff
XGBoost	Best performance overall