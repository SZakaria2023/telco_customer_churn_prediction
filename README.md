# Telco Customer Churn Prediction

## Applied Machine Learning (Fall 2025)
Numidia Institute of Technology (NIT)

Course: Introduction to Machine Learning  
Instructor: Mr. Fares Mazouni  
Student: Senhadji Zakaria  
Project Option: Classification: Multi-Model Approach to Predictive Churn  
Status: Individual Project

---

## Project Overview

This repository contains a complete end-to-end machine learning project that predicts customer churn for a telecommunications company using the Telco Customer Churn dataset. The project implements multiple classification models, compares their performance, and provides an interactive web application for real-time predictions with model explainability.

---

## Dataset

**Source:** Telco Customer Churn – Kaggle  
**Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**File:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

- **7,032 customer records** (after cleaning)
- **20 features** (after removing customerID)
- **Target:** Churn (Yes/No) – approximately **26.6% positive class** (imbalanced dataset)

**Key Features:**
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Account: tenure, Contract, PaperlessBilling, PaymentMethod
- Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Charges: MonthlyCharges, TotalCharges

---

## Project Structure

```
.
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebook/
│   ├── eda_and_models.ipynb      # EDA, model training, and evaluation
│   └── shap.ipynb                 # SHAP analysis notebook
├── app/
│   ├── app.py                     # Streamlit web application
│   ├── preprocessing.py           # Preprocessing utilities
│   ├── best_model.pkl             # Trained best model
│   ├── target_encoder.pkl         # Label encoder for target
│   ├── feature_names.pkl          # Feature names for consistency
│   └── requirements.txt           # App-specific requirements
├── report/
│   └── Final_Report.md            # Project report (Markdown)
├── presentation/
│   └── Presentation_Slides.md     # Presentation slides (Markdown)
├── tests/
│   └── test_preprocessing.py       # Unit tests
├── requirements.txt               # Main project dependencies
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/[YOUR_GITHUB_USERNAME]/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Main libraries:**
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- shap
- streamlit
- joblib

---

## How to Run

### 1. Explore the Notebooks

Open and run the Jupyter notebooks:
- `notebook/eda_and_models.ipynb` - Complete EDA, model training, and evaluation
- `notebook/shap.ipynb` - SHAP explainability analysis

### 2. Run the Web Application

```bash
streamlit run app/app.py
```

The interactive app will open in your browser (typically at `http://localhost:8501`).

**Features:**
- Input all customer features through an intuitive sidebar form
- Get real-time churn predictions with probability scores
- View SHAP explanations showing top contributing factors
- Visualize feature importance for each prediction

### 3. Run Tests

```bash
pytest tests/
```

---

## Model Performance

### Models Implemented

1. **Logistic Regression** (Baseline)
2. **XGBoost** (Gradient Boosting)
3. **Neural Network (MLP)** (Multi-Layer Perceptron)

### Evaluation Metrics

The best model is automatically selected based on ROC-AUC score. All models are evaluated using:

- **ROC-AUC Score** (primary metric for imbalanced data)
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **Cross-Validation** (5-fold stratified)

### Results Summary

*Note: Results will be updated after running the notebook with the tuned models.*

**Key Findings:**
- Models are evaluated on a stratified 80/20 train-test split
- Cross-validation ensures robust performance estimates
- Feature importance analysis reveals key churn drivers

**Top Churn Drivers:**
- Short tenure (< 12 months)
- Month-to-month contracts
- Fiber optic internet service
- Lack of tech support
- Electronic check payment method

---

## Features

### Data Preprocessing
- Missing value handling (TotalCharges)
- One-hot encoding for categorical variables
- Feature engineering (interactions, binning)
- Target encoding

### Model Training
- Multiple model families (linear, tree-based, neural network)
- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust evaluation
- Automatic best model selection

### Model Explainability
- SHAP (SHapley Additive exPlanations) values
- Feature importance visualizations
- Individual prediction explanations
- Global model insights

### Web Application
- User-friendly interface
- Real-time predictions
- Interactive feature input
- Model explanations
- Error handling and validation

---

## Project Deliverables

- ✅ Complete EDA with visualizations
- ✅ Multiple trained models with comparison
- ✅ Interactive Streamlit web application
- ✅ Model explainability (SHAP)
- ✅ Comprehensive evaluation metrics
- ✅ Preprocessing pipeline
- ✅ Unit tests
- 📄 Final Report (see `report/` folder)
- 📊 Presentation Slides (see `presentation/` folder)

---

## Technical Details

### Preprocessing Pipeline

The preprocessing module (`app/preprocessing.py`) handles:
- Data cleaning (missing values, type conversion)
- Target encoding (Yes/No → 1/0)
- Feature encoding (one-hot encoding)
- Feature alignment for predictions

### Model Selection

The best model is automatically selected based on test set ROC-AUC score and saved for deployment.

### Deployment

The Streamlit app can be deployed on:
- Streamlit Cloud
- Heroku
- AWS/GCP/Azure
- Local server

---

## Future Improvements

- [ ] Add more advanced models (Random Forest, LightGBM, CatBoost)
- [ ] Implement SMOTE for handling class imbalance
- [ ] Add model retraining pipeline
- [ ] Deploy to cloud platform
- [ ] Add batch prediction capability
- [ ] Implement model versioning

---

## Acknowledgments

- Dataset provided by IBM Sample Data (available on Kaggle)
- Inspired by high-quality Kaggle notebooks and standard churn prediction practices
- Built with open-source ML libraries

---

## Contact

**Senhadji Zakaria**  
January 2026

---

## License

This project is for educational purposes as part of the Applied Machine Learning course at NIT.
