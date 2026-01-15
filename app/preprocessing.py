"""
Preprocessing module for Telco Customer Churn Prediction
Handles data cleaning and feature encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def clean_data(df):
    """
    Clean the raw dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset
    
    Returns:
    --------
    df : pandas.DataFrame
        Cleaned dataset
    """
    # Convert TotalCharges to numeric (has spaces/blanks → NaN)
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing TotalCharges
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)
    
    # Remove customerID (no predictive power)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    return df


def encode_target(df, le=None):
    """
    Encode target variable (Yes/No → 1/0)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with Churn column
    le : LabelEncoder, optional
        Fitted LabelEncoder (for consistency)
    
    Returns:
    --------
    df : pandas.DataFrame
        Dataset with encoded Churn
    le : LabelEncoder
        Fitted LabelEncoder
    """
    df = df.copy()
    
    if le is None:
        le = LabelEncoder()
        df['Churn'] = le.fit_transform(df['Churn'])
    else:
        df['Churn'] = le.transform(df['Churn'])
    
    return df, le


def encode_features(df, feature_names=None):
    """
    One-hot encode categorical features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with categorical features
    feature_names : list, optional
        List of expected feature names (for consistency)
    
    Returns:
    --------
    df_encoded : pandas.DataFrame
        One-hot encoded dataset
    """
    df = df.copy()
    
    # Remove target if present
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)
    
    # Convert categorical columns to proper types for get_dummies
    # Ensure bin columns are treated as categorical
    for col in ['tenure_bin', 'monthly_bin']:
        if col in df.columns:
            # Convert to int first, then to category
            df[col] = df[col].fillna(0).astype(int).astype('category')
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Reindex to match training features if provided
    if feature_names is not None:
        df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    
    return df_encoded


def create_engineered_features(df):
    """
    Create engineered features matching the notebook
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with tenure and MonthlyCharges
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with engineered features added
    """
    df = df.copy()
    
    # Feature engineering: interaction term
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['tenure_monthly_interaction'] = df['tenure'] * df['MonthlyCharges']
    
    # Feature engineering: tenure binning
    if 'tenure' in df.columns:
        df['tenure_bin'] = pd.cut(
            df['tenure'], 
            bins=[0, 12, 24, 48, 72], 
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(float)  # Convert to float to handle NaN if needed
    
    # Feature engineering: monthly charges binning
    if 'MonthlyCharges' in df.columns:
        df['monthly_bin'] = pd.cut(
            df['MonthlyCharges'], 
            bins=[0, 50, 75, 100, 120], 
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(float)  # Convert to float to handle NaN if needed
    
    return df


def preprocess_input(raw_input_dict, feature_names, le=None):
    """
    Preprocess a single input for prediction
    
    Parameters:
    -----------
    raw_input_dict : dict
        Dictionary with feature values
    feature_names : list
        List of expected feature names from training
    le : LabelEncoder, optional
        Fitted LabelEncoder for target (not used here but kept for consistency)
    
    Returns:
    --------
    input_encoded : pandas.DataFrame
        Preprocessed input ready for model
    """
    # Convert to DataFrame
    raw_input = pd.DataFrame([raw_input_dict])
    
    # Create engineered features (matching notebook)
    raw_input = create_engineered_features(raw_input)
    
    # Encode features
    input_encoded = encode_features(raw_input, feature_names)
    
    return input_encoded
