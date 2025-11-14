# Step 2: cleaning_pipeline.py
import pandas as pd
import numpy as np
import os

DATA_PATH = './data/customer_churn_data.csv'
OUT_PATH = 'data'
os.makedirs(OUT_PATH, exist_ok=True)
ARTIFACTS = 'artifacts'
os.makedirs(ARTIFACTS, exist_ok=True)

def cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # lower/strip column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # drop duplicates
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Dropped duplicates: {before-after}")

    # Convert 'totalcharges' or 'total_charges' to numeric if present
    for candidate in ['totalcharges', 'total_charges', 'total_charges ']:
        if candidate in df.columns:
            df[candidate] = pd.to_numeric(df[candidate], errors='coerce')

    # Normalize churn column to 0/1 if present
    if 'churn' in df.columns:
        df['churn'] = df['churn'].astype(str).str.strip().str.lower().map({'yes':1,'no':0})
    
    # Handle missing values:
    # - If numeric: fill with median
    # - If categorical (object): fill with 'Unknown'
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in num_cols:
        if df[c].isna().sum() > 0:
            df[c] = df[c].fillna(df[c].median())
    for c in obj_cols:
        if df[c].isna().sum() > 0:
            df[c] = df[c].fillna('Unknown')

    # Outlier handling (IQR winsorization for numeric columns with many unique values)
    for c in num_cols:
        if df[c].nunique() > 10:
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5*iqr
            upper = q3 + 1.5*iqr
            df[c] = df[c].clip(lower, upper)

    # Ensure churn is integer 0/1 when available
    if 'churn' in df.columns:
        df['churn'] = df['churn'].fillna(0).astype(int)

    return df

# Execute cleaning
raw = pd.read_csv(DATA_PATH)
df_clean = cleaning_pipeline(raw)
print("\nAfter cleaning shape:", df_clean.shape)
print("\nMissing per column:\n", df_clean.isna().sum())
# Save cleaned file
clean_path = os.path.join(OUT_PATH, 'clean_customer_churn.csv')
df_clean.to_csv(clean_path, index=False)
print("\nCleaned dataset saved to:", clean_path)
