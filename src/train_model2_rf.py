import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------------
# Load Feature-Engineered Dataset
# -------------------------------------------------------
DATA_PATH = "./data/fe_customer_churn.csv"
PREPROCESSOR_PATH = "./artifacts/preprocessor.joblib"

df = pd.read_csv(DATA_PATH)

target = "Churn"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found.")

print("✔ Loaded FE dataset")

# -------------------------------------------------------
# Prepare X and y
# -------------------------------------------------------
X = df.drop(columns=[target, "customerID"], errors="ignore")

# Correct Yes/No → 1/0 encoding
y = df[target].map({"Yes": 1, "No": 0})

if y.isna().sum() > 0:
    raise ValueError("Churn column contains unexpected values!")

# Load preprocessor
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Preprocess features
X_processed = preprocessor.transform(X)

# -------------------------------------------------------
# Train-test split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------
# Train Random Forest
# -------------------------------------------------------
model_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42
)

model_rf.fit(X_train, y_train)

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
y_pred = model_rf.predict(X_test)

print("\n✔ Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------------------------------
# Save Model
# -------------------------------------------------------
joblib.dump(model_rf, "artifacts/model_rf.joblib")
print("\n✔ Random Forest Model Saved → artifacts/model_rf.joblib")
