# Setup Guide

This guide will help you set up and run the Telco Customer Churn Prediction project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning)

## Step-by-Step Setup

### 1. Clone or Download the Project

If using Git:
```bash
git clone <repository-url>
cd telco_customer_churn_prediction
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, streamlit, shap; print('All packages installed successfully!')"
```

### 5. Run the Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open and run:
   - `notebook/eda_and_models.ipynb` - This will train models and save them
   - `notebook/shap.ipynb` - SHAP analysis

**Important:** Make sure to run `eda_and_models.ipynb` first to generate the model files!

### 6. Run the Web Application

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

### 7. Run Tests (Optional)

```bash
pip install pytest
pytest tests/
```

## Troubleshooting

### Issue: Model files not found

**Solution:** Run the `notebook/eda_and_models.ipynb` notebook first to generate:
- `app/best_model.pkl`
- `app/target_encoder.pkl`
- `app/feature_names.pkl`

### Issue: Import errors

**Solution:** Make sure you're in the project root directory and virtual environment is activated.

### Issue: SHAP visualization errors

**Solution:** Some models (like Logistic Regression) may not support TreeExplainer. The app includes error handling for this.

### Issue: Port already in use

**Solution:** Streamlit will automatically use the next available port, or specify one:
```bash
streamlit run app/app.py --server.port 8502
```

## Project Structure

```
telco_customer_churn_prediction/
├── data/                    # Dataset
├── notebook/                # Jupyter notebooks
├── app/                     # Streamlit application
│   ├── app.py              # Main app file
│   ├── preprocessing.py    # Preprocessing utilities
│   └── *.pkl               # Model files (generated)
├── report/                  # Project report
├── presentation/            # Presentation slides
├── tests/                   # Unit tests
├── requirements.txt         # Dependencies
└── README.md               # Project documentation
```

## Next Steps

1. Explore the notebooks to understand the analysis
2. Run the web app and test predictions
3. Review the report and presentation materials
4. Customize models or add features as needed

## Support

For issues or questions, refer to the README.md or contact the project author.
