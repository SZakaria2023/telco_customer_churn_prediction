"""
Unit tests for preprocessing module
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.preprocessing import clean_data, encode_target, encode_features, preprocess_input


class TestCleanData:
    """Test data cleaning functions"""
    
    def test_clean_data_removes_customerid(self):
        """Test that customerID is removed"""
        df = pd.DataFrame({
            'customerID': ['A', 'B', 'C'],
            'tenure': [1, 2, 3],
            'TotalCharges': ['100', '200', '300']
        })
        result = clean_data(df)
        assert 'customerID' not in result.columns
    
    def test_clean_data_converts_totalcharges(self):
        """Test that TotalCharges is converted to numeric"""
        df = pd.DataFrame({
            'TotalCharges': ['100.5', '200.3', ' '],
            'tenure': [1, 2, 3]
        })
        result = clean_data(df)
        assert pd.api.types.is_numeric_dtype(result['TotalCharges'])
    
    def test_clean_data_drops_missing_totalcharges(self):
        """Test that rows with missing TotalCharges are dropped"""
        df = pd.DataFrame({
            'TotalCharges': ['100', '200', ' '],
            'tenure': [1, 2, 3]
        })
        result = clean_data(df)
        assert len(result) == 2  # One row with missing TotalCharges removed


class TestEncodeTarget:
    """Test target encoding functions"""
    
    def test_encode_target_creates_encoder(self):
        """Test that encoder is created when not provided"""
        df = pd.DataFrame({'Churn': ['Yes', 'No', 'Yes']})
        result_df, encoder = encode_target(df)
        assert isinstance(encoder, LabelEncoder)
        assert set(result_df['Churn'].unique()) == {0, 1}
    
    def test_encode_target_uses_existing_encoder(self):
        """Test that existing encoder is used"""
        df1 = pd.DataFrame({'Churn': ['Yes', 'No']})
        df1_encoded, encoder = encode_target(df1)
        
        df2 = pd.DataFrame({'Churn': ['Yes', 'No']})
        df2_encoded, _ = encode_target(df2, le=encoder)
        
        # Should produce same encoding
        assert (df1_encoded['Churn'] == df2_encoded['Churn']).all()


class TestEncodeFeatures:
    """Test feature encoding functions"""
    
    def test_encode_features_one_hot_encoding(self):
        """Test that categorical features are one-hot encoded"""
        df = pd.DataFrame({
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'tenure': [1, 2, 3]
        })
        result = encode_features(df)
        # Should have columns for Contract_One year and Contract_Two year (drop_first=True)
        assert 'Contract_One year' in result.columns or 'Contract_Two year' in result.columns
    
    def test_encode_features_reindexes_to_feature_names(self):
        """Test that features are reindexed to match training features"""
        df = pd.DataFrame({
            'Contract': ['Month-to-month'],
            'tenure': [1]
        })
        feature_names = ['tenure', 'Contract_One year', 'Contract_Two year', 'Other_Feature']
        result = encode_features(df, feature_names=feature_names)
        assert set(result.columns) == set(feature_names)
        assert result['Other_Feature'].iloc[0] == 0  # Missing feature filled with 0


class TestPreprocessInput:
    """Test input preprocessing for predictions"""
    
    def test_preprocess_input_creates_dataframe(self):
        """Test that input dict is converted to DataFrame"""
        input_dict = {
            'tenure': 12,
            'Contract': 'Month-to-month',
            'MonthlyCharges': 50.0
        }
        feature_names = ['tenure', 'MonthlyCharges', 'Contract_One year', 'Contract_Two year']
        result = preprocess_input(input_dict, feature_names)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_preprocess_input_matches_feature_names(self):
        """Test that output matches expected feature names"""
        input_dict = {
            'tenure': 12,
            'Contract': 'Month-to-month'
        }
        feature_names = ['tenure', 'Contract_One year', 'Contract_Two year']
        result = preprocess_input(input_dict, feature_names)
        assert set(result.columns) == set(feature_names)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
