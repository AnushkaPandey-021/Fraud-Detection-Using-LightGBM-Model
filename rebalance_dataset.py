import pandas as pd
from sklearn.utils import resample

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('C:\\Users\\anapa\\OneDrive\\Desktop\\trial 1\\Folder\\PS_20174392719_1491204439457_log.csv')

# Basic cleaning
df = df.dropna(subset=['isFraud'])
df['isFraud'] = df['isFraud'].astype(int)
df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, errors='ignore')

# Show original class distribution
print("\nOriginal class distribution:")
print(df['isFraud'].value_counts())

# Separate classes
fraud = df[df['isFraud'] == 1]
non_fraud = df[df['isFraud'] == 0]

# Set new size constraints
max_total = 1_500_000  # keep under 2M
fraud_ratio = 0.3
n_fraud = int(max_total * fraud_ratio)
n_non_fraud = max_total - n_fraud

# Resample fraud (up) and non-fraud (down)
fraud_upsampled = resample(fraud, replace=True, n_samples=n_fraud, random_state=42)
non_fraud_downsampled = resample(non_fraud, replace=False, n_samples=n_non_fraud, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([fraud_upsampled, non_fraud_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Show new class distribution
print("\nNew class distribution:")
print(balanced_df['isFraud'].value_counts())

# Save rebalanced dataset
balanced_df.to_csv('C:\\Users\\anapa\\OneDrive\\Desktop\\trial 1\\Folder\\balanced_fraud_data.csv', index=False)
print("\nBalanced dataset saved successfully!")
