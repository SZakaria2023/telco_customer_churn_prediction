# Telco Customer Churn Prediction

## Applied Machine Learning (Fall 2025)
Numidia Institute of Technology (NIT)

Course: Introduction to Machine Learning
Instructor: Mr. Fares Mazouni
Student: Senhadji Zakaria
Project Option: Classification: Multi-Model Approach to Predictive Churn
Status: Individual Project

--------------------------------------------------

## Project Overview

This repository contains a complete end-to-end machine learning project that predicts customer churn for a telecommunications company using the Telco Customer Churn dataset.

--------------------------------------------------

## Dataset

Source: Telco Customer Churn – Kaggle
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

File: data/WA_Fn-UseC_-Telco-Customer-Churn.csv

7,043 customer records with 21 features (customerID, gender, tenure, PhoneService, MultipleLines, InternetService, Contract, PaymentMethod, MonthlyCharges, TotalCharges, Churn, etc.)

Target: Churn (Yes/No) – approximately 26.5% positive class (imbalanced)

--------------------------------------------------

## Project Structure

.
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb
│   ├── 02_advanced_models_and_tuning.ipynb
│   └── 03_shap_interpretability.ipynb
├── app/
│   ├── app.py
│   ├── best_model.pkl
│   └── requirements.txt
├── report/
│   └── Final_Report.pdf
├── presentation/
│   └── Presentation_Slides.pdf
├── requirements.txt
└── README.md

--------------------------------------------------

## How to Run the Project

1. Clone the Repository

git clone https://github.com/[YOUR_GITHUB_USERNAME]/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction

2. Install Dependencies

pip install -r requirements.txt

Main libraries used:
pandas, numpy
scikit-learn
xgboost
matplotlib, seaborn
shap
tensorflow / keras
streamlit

3. Explore the Notebooks

4. Run the Web Application Locally

streamlit run app/app.py

The interactive app will open in your browser. Enter customer details to receive:
Churn probability
Prediction (Churn / No Churn)
SHAP force plot / top contributing factors

4. Run the Deployed Web App Locally

streamlit run app/app.py

This will launch the interactive churn prediction app in your browser.

5. Live Deployed Application

Link to live app - to be added after deployment on Streamlit Cloud

--------------------------------------------------

## Results Summary (To Be Updated)

Baseline (Logistic Regression): AUC ≈ 0.84
Best Model (XGBoost): AUC ≈ 0.90+ (expected)
Key churn drivers: Short tenure, month-to-month contract, fiber optic internet, lack of tech support

--------------------------------------------------

## Acknowledgments

Dataset provided by IBM Sample Data (available on Kaggle)
Inspired by high-quality Kaggle notebooks and standard churn prediction practices

--------------------------------------------------

## Contact

Senhadji Zakaria
January 2026
