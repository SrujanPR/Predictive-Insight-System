import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
FE_PATH = "./data/fe_customer_churn.csv"
PREPROCESSOR_PATH = "./artifacts/preprocessor.joblib"

print("✔ Loading FE dataset...")

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
if not os.path.exists(FE_PATH):
    raise FileNotFoundError(f"FE dataset not found at: {FE_PATH}")

df = pd.read_csv(FE_PATH)
print("✔ FE dataset loaded!")

# ---------------------------------------------------------
# 2. Load preprocessor just for demonstration (optional)
# ---------------------------------------------------------
if os.path.exists(PREPROCESSOR_PATH):
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("✔ Preprocessor loaded!")
else:
    print("⚠ Preprocessor not found, skipping load.")

# ---------------------------------------------------------
# 3. VISUALIZATIONS
# ---------------------------------------------------------

print("✔ Creating visualizations...")

# --- Plot 1: Tenure distribution ---
plt.figure(figsize=(8, 4))
sns.histplot(df["tenure"], bins=30)
plt.title("Distribution of Customer Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("artifacts/dist_tenure.png")
plt.close()

# --- Plot 2: Contract type counts ---
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="Contract")
plt.title("Contract Type Distribution")
plt.tight_layout()
plt.savefig("artifacts/cat_counts_contract.png")
plt.close()

# --- Plot 3: Correlation Heatmap ---
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

plt.figure(figsize=(6, 4))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("artifacts/corr_heatmap.png")
plt.close()

print("✔ Visualization Completed! Saved to /artifacts/")
