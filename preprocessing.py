import pandas as pd

def create_features(df):
    """Feature engineering with type preservation"""
    df = df.copy()
    
    # --- Existing Numerical Features ---
    # Create balance ratios
    df['orig_balance_ratio'] = df['newbalanceOrig'] / (df['oldbalanceOrg'] + 1e-6)
    df['dest_balance_ratio'] = df['newbalanceDest'] / (df['oldbalanceDest'] + 1e-6)
    
    # Create movement features
    df['net_movement'] = (df['newbalanceOrig'] - df['oldbalanceOrg']) + \
                        (df['newbalanceDest'] - df['oldbalanceDest'])
    df['amount_to_balance'] = df['amount'] / (df['oldbalanceOrg'] + df['oldbalanceDest'] + 1e-6)
    
    # Time-based features
    df['hour'] = df['step'] % 24
    df['is_offpeak'] = ((df['hour'] >= 0) & (df['hour'] <= 5)) | (df['hour'] >= 22)

    # --- Transaction Type Handling ---
    if 'type' in df.columns:
        # Preserve original type for visualization
        df['transaction_type'] = df['type'].copy()
        # Create dummy variables for modeling
        df = pd.get_dummies(df, columns=['type'], drop_first=True)

    return df

def preprocess_fraud_data(df, expected_features):
    """Full preprocessing pipeline"""
    # Create features
    df = create_features(df)
    
    # Drop unused columns but keep transaction_type
    drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step', 'hour']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Ensure feature alignment
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
            
    # Return features + preserved transaction_type
    return df[expected_features + ['transaction_type']]