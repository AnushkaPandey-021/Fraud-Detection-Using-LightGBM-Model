import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from Folder.preprocessing import create_features
# ---- Config ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Parent of "Folder"
MODEL_PATH = os.path.join(BASE_DIR, 'fraud_model_combined.txt')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler_combined.pkl')
THRESHOLD_PATH = os.path.join(BASE_DIR, 'threshold_combined.npy')
NEW_DATASET_PATH = os.path.join(BASE_DIR, 'Folder', 'cleaned_fraudTrain.csv')

# ---- Validation Checks ----
def validate_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Critical file missing: {path}")

validate_path(MODEL_PATH)
validate_path(SCALER_PATH)
validate_path(THRESHOLD_PATH)
validate_path(NEW_DATASET_PATH)

# ---- Load artifacts ----
model = lgb.Booster(model_file=MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
threshold = np.load(THRESHOLD_PATH)

# ---- Load and preprocess data ----
df = pd.read_csv(NEW_DATASET_PATH)

# ---- Column mapping ----
rename_map = {
    'unix_time': 'step',
    'amt': 'amount',
    'category': 'type',
    'cc_num': 'nameOrig',
    'merchant': 'nameDest',
    'is_fraud': 'isFraud'
}
df = df.rename(columns=rename_map)

# ---- Feature engineering ----
def create_features(df):
    df = df.copy()
    # Add missing PaySim columns if needed
    for col in ['newbalanceOrig', 'oldbalanceOrg', 'newbalanceDest', 'oldbalanceDest']:
        if col not in df.columns:
            df[col] = 0.0  # Initialize with zeros if missing
    
    # Original feature calculations
    df['orig_balance_ratio'] = df['newbalanceOrig'] / (df['oldbalanceOrg'] + 1e-6)
    df['dest_balance_ratio'] = df['newbalanceDest'] / (df['oldbalanceDest'] + 1e-6)
    df['net_movement'] = (df['newbalanceOrig'] - df['oldbalanceOrg']) + (df['newbalanceDest'] - df['oldbalanceDest'])
    df['amount_to_balance'] = df['amount'] / (df['oldbalanceOrg'] + df['oldbalanceDest'] + 1e-6)
    df['hour'] = df['step'] % 24
    df['is_offpeak'] = ((df['hour'] >= 0) & (df['hour'] <= 5)) | (df['hour'] >= 22)
    
    # Handle transaction types like training did
    if 'type' in df.columns:
        df = pd.get_dummies(df, columns=['type'], drop_first=True)
    
    return df

df = create_features(df)

# ---- Drop unnecessary columns ----
drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step', 'hour']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# ---- Align features ----
expected_cols = scaler.feature_names_in_
for col in expected_cols:
    if col not in df.columns:
        df[col] = 0  # Add missing features with default value
df = df[expected_cols]

# ---- Scale features ----
df_scaled = pd.DataFrame(scaler.transform(df), columns=expected_cols)

# ---- Predict ----
probs = model.predict(df_scaled)
preds = (probs >= threshold).astype(int)

# ---- Evaluation ----
if 'isFraud' in df.columns:
    y_true = df['isFraud']
    print("\nClassification Report:")
    print(classification_report(y_true, preds))
    print(f"AUC: {roc_auc_score(y_true, probs):.4f}")
else:
    print(f"Predicted frauds: {sum(preds)}")

# ---- Save results ----
output_df = df.copy()
output_df['fraud_probability'] = probs
output_df['predicted_fraud'] = preds
output_df.to_csv("fraud_results.csv", index=False)
print("\nResults saved to fraud_results.csv")