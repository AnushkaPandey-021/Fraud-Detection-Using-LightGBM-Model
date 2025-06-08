import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from lightgbm import early_stopping, log_evaluation
import pickle

# 1. Load dataset efficiently
print("\nLoading and optimizing dataset...")
dtypes = {
    'step': 'int16', 'amount': 'float32',
    'oldbalanceOrg': 'float32', 'newbalanceOrig': 'float32',
    'oldbalanceDest': 'float32', 'newbalanceDest': 'float32',
    'isFraud': 'int8'
}
file_path = 'C:/Users/anapa/OneDrive/Desktop/trial 1/Folder/balanced_fraud_data.csv'
data = pd.read_csv(file_path, dtype=dtypes)

# 2. Feature Engineering
def create_features(df):
    df = df.copy()
    df['orig_balance_ratio'] = df['newbalanceOrig'] / (df['oldbalanceOrg'] + 1e-6)
    df['dest_balance_ratio'] = df['newbalanceDest'] / (df['oldbalanceDest'] + 1e-6)
    df['net_movement'] = (df['newbalanceOrig'] - df['oldbalanceOrg']) + (df['newbalanceDest'] - df['oldbalanceDest'])
    df['amount_to_balance'] = df['amount'] / (df['oldbalanceOrg'] + df['oldbalanceDest'] + 1e-6)
    df['is_reciprocal'] = (df['oldbalanceOrg'] > 0) & (df['newbalanceDest'] > df['oldbalanceDest'])
    df['hour'] = df['step'] % 24
    df['is_offpeak'] = ((df['hour'] >= 0) & (df['hour'] <= 5)) | (df['hour'] >= 22)
    if 'type' in df.columns:
        df['type_amount_interaction'] = df['amount'] * (df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int))
    return df

print("\nEngineering features...")
data = create_features(data)
data = data.drop(['nameOrig', 'nameDest', 'isFlaggedFraud', 'step'], axis=1, errors='ignore')
if 'type' in data.columns:
    data = pd.get_dummies(data, columns=['type'], drop_first=True)

# 3. Train-Test Split
y = data['isFraud']
X = data.drop('isFraud', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# 5. LightGBM Model Training
print("\nTraining LightGBM model...")
model = lgb.LGBMClassifier(
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=100,
    reg_alpha=0.01,
    reg_lambda=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=2000,
    scale_pos_weight=2,
    boosting_type='gbdt',
    objective='binary',
    metric='auc',
    importance_type='gain'
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    eval_metric='auc',
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=50)]
)

# 6. Isolation Forest
print("\nTraining Isolation Forest for ensemble...")
fraud_rate = y_train.mean()
iso_forest = IsolationForest(contamination=fraud_rate, random_state=42, n_estimators=200)
iso_forest.fit(X_train_scaled)

# Convert to probabilities
iso_scores = iso_forest.decision_function(X_test_scaled)
iso_probs_fraud = 1 - (1 / (1 + np.exp(-iso_scores)))

# 7. Combine Probabilities
val_probs = model.predict_proba(X_test_scaled)[:, 1]
combined_probs = np.where(
    val_probs > 0.5,
    0.7 * val_probs + 0.3 * iso_probs_fraud,
    0.3 * val_probs + 0.7 * iso_probs_fraud
)

# 8. Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, combined_probs)
min_precision = 0.85
viable_thresholds = [t for p, t in zip(precision[:-1], thresholds) if p >= min_precision]
optimal_threshold = (
    viable_thresholds[np.argmax([recall[i] for i, p in enumerate(precision[:-1]) if p >= min_precision])]
    if viable_thresholds else np.median(combined_probs)
)

# 9. Evaluation
final_preds = (combined_probs >= optimal_threshold).astype(int)
print("\nFinal Ensemble Evaluation:")
print(classification_report(y_test, final_preds))
print("ROC AUC:", roc_auc_score(y_test, combined_probs))
print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))

# 10. Feature Importance
plt.figure(figsize=(12, 8))
lgb.plot_importance(model, importance_type='gain', max_num_features=15)
plt.title("LightGBM Feature Importance (Gain)")
plt.tight_layout()
plt.show()

# 11. Save only correct artifacts
print("\nSaving model and feature list...")
with open('lightgbm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model_features.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

print("\n✅ Saved:")
print("- lightgbm_model.pkl (trained model)")
print("- model_features.pkl (list of expected features)")
