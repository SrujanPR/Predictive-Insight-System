import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
DATA_PATH = "data/customer_churn_data.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\n✔ Dataset Loaded Successfully")
print(df.head())

# ---------------------------------------------------------
# 2. CREATE NEW FEATURES
# ---------------------------------------------------------

# ---- SAFE TENURE BINNING ----
tenure_max = df['tenure'].max()

# Create raw bins (may include duplicates if tenure_max = 72)
raw_bins = [-1, 6, 12, 24, 48, 72, tenure_max]

# Remove duplicates
unique_bins = sorted(set(raw_bins))

# Generate matching number of labels automatically
labels = []
for i in range(len(unique_bins) - 1):
    low = unique_bins[i] + 1
    high = unique_bins[i + 1]
    labels.append(f"{low}-{high}")

df['tenure_bin'] = pd.cut(
    df['tenure'],
    bins=unique_bins,
    labels=labels,
    include_lowest=True
)


# Total Charges can be blank → convert to float safely
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Numeric features that may contain missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

print("\n✔ Feature Engineering Completed")
print(df[['tenure', 'tenure_bin', 'TotalCharges']].head())

# ---------------------------------------------------------
# 3. DEFINE FEATURE GROUPS
# ---------------------------------------------------------
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_bin'
]

# ---------------------------------------------------------
# 4. PREPROCESSING PIPELINE
# ---------------------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

print("\n✔ Preprocessor Pipeline Created Successfully")

# ---------------------------------------------------------
# 5. SAVE ENGINEERED DATASET
# ---------------------------------------------------------
OUTPUT_PATH = "./data/fe_customer_churn.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✔ Engineered dataset saved to: {OUTPUT_PATH}")

# ---------------------------------------------------------
# 5. Export preprocessed data (optional)
# ---------------------------------------------------------

X = df.drop('Churn', axis=1)
y = df['Churn'].map({"Yes": 1, "No": 0})  # binary encoding

print("\n✔ Shapes:")
print("X:", X.shape)
print("y:", y.shape)

# ---------------------------------------------------------
# 6. Fit-transform the dataset
# ---------------------------------------------------------

X_processed = preprocessor.fit_transform(X)

# ---------------------------------------------------------
# 8. Save Preprocessor
# ---------------------------------------------------------
import joblib
os.makedirs("artifacts", exist_ok=True)

joblib.dump(preprocessor, "artifacts/preprocessor.joblib")

print("\n✔ Preprocessor saved to artifacts/preprocessor.joblib")


print("\n✔ Preprocessing Completed")
print("Processed X shape:", X_processed.shape)
