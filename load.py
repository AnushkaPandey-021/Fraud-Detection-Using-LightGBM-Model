import joblib
import pandas as pd
import os
import numpy as np
from pandas.errors import EmptyDataError
from .preprocessing import preprocess_fraud_data

# Define paths for model and features
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "lightgbm_model.pkl")
features_path = os.path.join(current_dir, "model_features.pkl")

# Load model and expected features
try:
    model = joblib.load(model_path)
    expected_features = joblib.load(features_path)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# Define required columns for CSV validation
REQUIRED_COLUMNS = [
    'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
    'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'
]

def validate_csv(file):
    """Validate CSV structure and content"""
    try:
        file.seek(0)
        df_sample = pd.read_csv(file, nrows=1)
        file.seek(0)
        
        missing = [col for col in REQUIRED_COLUMNS if col not in df_sample.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        if df_sample.empty:
            raise EmptyDataError("File contains no data")
            
    except pd.errors.ParserError:
        raise ValueError("Invalid CSV format")
    finally:
        file.seek(0)

def predict_from_csv(uploaded_file, threshold=0.2):
    """Process CSV and generate predictions"""
    try:
        if uploaded_file.size == 0:
            raise ValueError("Empty file uploaded")

        validate_csv(uploaded_file)

        chunk_size = 10000
        results = []

        for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size, low_memory=True):
            if 'isFlaggedFraud' not in chunk.columns:
                chunk['isFlaggedFraud'] = 0

            # Preprocess the chunk using expected features
            processed = preprocess_fraud_data(chunk, expected_features)

            # Align features by adding missing ones as zero columns
            missing_features = set(expected_features) - set(processed.columns)
            for feature in missing_features:
                processed[feature] = 0
            processed = processed[expected_features]  # Ensure column order matches

            # Check if processed input has non-zero values
            if processed.sum().sum() == 0:
                raise RuntimeError("All features are zero after preprocessing – check input data and encoding.")

            # Predict fraud probability and make binary fraud prediction
            probabilities = model.predict_proba(processed)[:, 1]
            predictions = (probabilities > threshold).astype(int)

            # Append results to the list
            chunk['fraud_probability'] = probabilities
            chunk['fraud_prediction'] = predictions
            results.append(chunk)

        return pd.concat(results, ignore_index=True)

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")
    finally:
        uploaded_file.seek(0)
