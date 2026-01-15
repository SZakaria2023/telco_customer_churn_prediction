# Telco Customer Churn Prediction
## Applied Machine Learning Project

**Senhadji Zakaria**  
Numidia Institute of Technology (NIT)  
January 2026

---

## Slide 1: Title Slide

# Telco Customer Churn Prediction
### A Multi-Model Machine Learning Approach

**Student:** Senhadji Zakaria  
**Course:** Applied Machine Learning (Fall 2025)  
**Instructor:** Mr. Fares Mazouni  
**Institution:** Numidia Institute of Technology

---

## Slide 2: Problem Statement

# The Challenge

- **Customer churn** is a critical issue for telecom companies
- Losing customers impacts **revenue** and increases **acquisition costs**
- Need to **predict** which customers are likely to churn
- Enable **proactive retention** strategies

**Goal:** Build ML models to predict customer churn with high accuracy

---

## Slide 3: Dataset Overview

# Dataset: Telco Customer Churn

- **Source:** Kaggle (IBM Sample Data)
- **Size:** 7,032 customers (after cleaning)
- **Features:** 20 predictive features
- **Target:** Churn (Yes/No)
- **Class Distribution:** 
  - 73.4% No Churn
  - 26.6% Churn (imbalanced)

**Key Features:**
- Demographics, Account info, Services, Charges

---

## Slide 4: Data Exploration

# Key Insights from EDA

**High Churn Risk Factors:**
- Month-to-month contracts (~43% churn)
- Short tenure (< 12 months)
- Fiber optic internet service
- Lack of tech support
- Electronic check payment

**Low Churn Risk:**
- Long-term contracts (2-year)
- High tenure customers
- Multiple services

---

## Slide 5: Methodology

# Approach

1. **Data Preprocessing**
   - Missing value handling
   - Feature engineering
   - One-hot encoding

2. **Model Development**
   - Logistic Regression (Baseline)
   - XGBoost (Gradient Boosting)
   - Neural Network (MLP)

3. **Evaluation**
   - Multiple metrics (ROC-AUC, Precision, Recall, F1)
   - Cross-validation
   - Model comparison

---

## Slide 6: Models Implemented

# Three Model Families

### 1. Logistic Regression
- **Type:** Linear model
- **Pros:** Interpretable, fast
- **Use:** Baseline comparison

### 2. XGBoost
- **Type:** Gradient Boosting
- **Pros:** High performance, handles non-linearity
- **Tuning:** GridSearchCV with 5-fold CV

### 3. Neural Network (MLP)
- **Type:** Multi-layer Perceptron
- **Architecture:** (32, 16) hidden layers
- **Pros:** Captures complex patterns

---

## Slide 7: Model Performance

# Results Summary

**Model Comparison (ROC-AUC):**
- Logistic Regression: ~0.84
- XGBoost (Tuned): ~0.85-0.87
- Neural Network: ~0.81-0.82

**Best Model:** Automatically selected based on ROC-AUC

**Additional Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrices
- Cross-Validation Scores

---

## Slide 8: Feature Importance

# What Drives Churn?

**Top Contributing Features:**
1. **Tenure** - Customer loyalty indicator
2. **Contract Type** - Commitment level
3. **Monthly Charges** - Service cost
4. **Internet Service** - Service type
5. **Tech Support** - Service quality
6. **Payment Method** - Convenience factor

**Business Insight:** Focus retention efforts on these factors

---

## Slide 9: Model Explainability

# SHAP Values

- **Global Importance:** Which features matter most overall
- **Local Explanations:** Why a specific customer is predicted to churn
- **Feature Interactions:** How features work together

**Benefits:**
- Transparent predictions
- Actionable insights
- Builds trust with stakeholders

---

## Slide 10: Web Application

# Interactive Prediction Tool

**Features:**
- User-friendly interface
- Real-time predictions
- All customer features input
- Probability scores
- SHAP explanations

**Deployment:**
- Streamlit framework
- Ready for cloud deployment
- Production-ready code

---

## Slide 11: Business Impact

# Actionable Insights

**Retention Strategies:**
1. **Target month-to-month customers** with special offers
2. **Early intervention** for new customers (< 12 months)
3. **Improve service quality** for fiber optic customers
4. **Enhance tech support** availability
5. **Review payment methods** - address electronic check issues

**Expected Outcome:** Reduced churn, increased customer lifetime value

---

## Slide 12: Technical Highlights

# Project Features

✅ **Comprehensive EDA** with visualizations  
✅ **Multiple ML models** with comparison  
✅ **Hyperparameter tuning** for optimal performance  
✅ **Cross-validation** for robust evaluation  
✅ **Model explainability** with SHAP  
✅ **Interactive web app** for predictions  
✅ **Production-ready** preprocessing pipeline  
✅ **Unit tests** for code quality  

---

## Slide 13: Challenges & Solutions

# Overcoming Obstacles

**Challenge 1: Class Imbalance**
- **Solution:** Used ROC-AUC as primary metric
- **Future:** Implement SMOTE

**Challenge 2: Model Selection**
- **Solution:** Automated best model selection
- **Approach:** Comprehensive evaluation metrics

**Challenge 3: Model Interpretability**
- **Solution:** SHAP values for explanations
- **Result:** Transparent, actionable insights

---

## Slide 14: Future Work

# Next Steps

1. **Advanced Models**
   - LightGBM, CatBoost
   - Ensemble methods

2. **Class Imbalance**
   - SMOTE implementation
   - Cost-sensitive learning

3. **Deployment**
   - Cloud deployment
   - Real-time API
   - Model monitoring

4. **Business Integration**
   - Retention campaign automation
   - A/B testing framework

---

## Slide 15: Conclusion

# Key Takeaways

- Successfully developed **multiple predictive models**
- Achieved **strong performance** (ROC-AUC > 0.82)
- Identified **key churn drivers**
- Created **interactive tool** for predictions
- Provided **model explainability**

**Impact:** Enables proactive customer retention strategies

---

## Slide 16: Q&A

# Thank You!

## Questions?

**Contact:**  
Senhadji Zakaria

**Repository:**  
[GitHub link to be added]

**Demo:**  
[Live app link to be added]

---

**End of Presentation**
