import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ARTIFACTS = 'artifacts'
os.makedirs(ARTIFACTS, exist_ok=True)

df = pd.read_csv('./data/clean_customer_churn.csv')

# 1) Distribution of tenure by churn
col1 = 'tenure' if 'tenure' in df.columns else df.select_dtypes('number').columns[0]
plt.figure(figsize=(8,4))
sns.histplot(data=df, x=col1, hue='churn', multiple='stack', bins=30)
plt.title(f'Distribution of {col1} by churn')
plt.tight_layout()
p1 = os.path.join(ARTIFACTS, f'dist_{col1}.png')
plt.savefig(p1)
print("Saved:", p1)
plt.clf()

# 2) Correlation heatmap of numeric features
num_cols = df.select_dtypes(include=['number']).columns.tolist()
if len(num_cols) >= 2:
    plt.figure(figsize=(10,8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu', square=True)
    plt.title('Correlation heatmap (numeric features)')
    plt.tight_layout()
    p2 = os.path.join(ARTIFACTS, 'corr_heatmap.png')
    plt.savefig(p2)
    print("Saved:", p2)
    plt.clf()

# 3) Categorical counts for a top categorical column (e.g., contract)
cat_candidates = [c for c in df.columns if df[c].dtype == 'object']
col3 = 'contract' if 'contract' in df.columns else (cat_candidates[0] if cat_candidates else None)
if col3:
    plt.figure(figsize=(8,4))
    order = df[col3].value_counts().index
    sns.countplot(data=df, x=col3, hue='churn', order=order)
    plt.xticks(rotation=45)
    plt.title(f'{col3} counts by churn')
    plt.tight_layout()
    p3 = os.path.join(ARTIFACTS, f'cat_counts_{col3}.png')
    plt.savefig(p3)
    print("Saved:", p3)
    plt.clf()

print("EDA complete. Files in artifacts/")
