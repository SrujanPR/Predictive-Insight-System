import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------------
# Load Feature-Engineered Dataset
# -------------------------------------------------------
DATA_PATH = "./data/fe_customer_churn.csv"
PREPROCESSOR_PATH = "./artifacts/preprocessor.joblib"

df = pd.read_csv(DATA_PATH)

# Target column
target = "Churn"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found. Available: {df.columns}")

print("✔ Loaded FE dataset")

# -------------------------------------------------------
# Prepare X and y
# -------------------------------------------------------
X = df.drop(columns=[target, "customerID"], errors="ignore")

# Correct mapping of Yes/No to 1/0
y = df[target].map({"Yes": 1, "No": 0})

if y.isna().sum() > 0:
    raise ValueError("Churn column contains values other than 'Yes' or 'No'")

# Load preprocessing pipeline
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Transform features
X_processed = preprocessor.transform(X)

# -------------------------------------------------------
# Train-test split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------
# Train Logistic Regression
# -------------------------------------------------------
model = LogisticRegression(max_iter=300, class_weight='balanced')
model.fit(X_train, y_train)

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
y_pred = model.predict(X_test)

print("\n✔ Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------------------------------
# Save Model
# -------------------------------------------------------
joblib.dump(model, "./artifacts/model_logreg.joblib")
print("\n✔ Logistic Regression Model Saved → artifacts/model_logreg.joblib")
