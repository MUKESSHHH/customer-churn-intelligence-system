
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_churn.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Drop customer ID if present
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)


df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# handle missing values

df.replace(" ", np.nan, inplace=True)

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())


label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

print("Total columns after encoding:", df.shape[1])

# feature & target split

X = df.drop("Churn", axis=1)
y = df["Churn"]

# train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# model training

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=4

)

rf.fit(X_train, y_train)



y_probs = rf.predict_proba(X_test)[:, 1]
y_pred = [1 if p >= 0.30 else 0 for p in y_probs]
report = classification_report(y_test, y_pred)

result_path = os.path.join(RESULTS_DIR, "model_results.txt")
with open(result_path, "w") as f:
    f.write("RANDOM FOREST - PROFESSIONAL THRESHOLD (0.30) REPORT\n")
    f.write(report)

print("Model evaluation (0.30 Threshold) saved to:", result_path)

# churn drivers

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_path = os.path.join(RESULTS_DIR, "churn_drivers.csv")
feature_importance.to_csv(importance_path, index=False)

print("Top churn drivers saved to:", importance_path)

# churn prediction

all_probabilities = rf.predict_proba(X)[:, 1]
all_predictions = ["YES" if p >= 0.30 else "NO" for p in all_probabilities]

risk_levels = []
for p in all_probabilities:
    if p >= 0.70: risk_levels.append("CRITICAL")
    elif p >= 0.30: risk_levels.append("MEDIUM")
    else: risk_levels.append("LOW")

customer_results = pd.DataFrame({
    "CustomerIndex": X.index,
    "WillChurn": all_predictions,
    "RiskLevel": risk_levels,
    "ChurnProbability": all_probabilities.round(2),
    "MonthlyCharges": df.loc[X.index, "MonthlyCharges"]
})


customer_output_path = os.path.join(
    RESULTS_DIR, "customer_churn_predictions.csv"
)

customer_results.to_csv(customer_output_path, index=False)

print("Customer-level churn predictions saved to:", customer_output_path)

# sample output

sample_customer = X_test.iloc[:1]
prediction = rf.predict(sample_customer)[0]
probability = rf.predict_proba(sample_customer)[0][1]

print("\n==============================")
print("SAMPLE CUSTOMER PREDICTION")
print("==============================")
print("Will churn:", "YES" if prediction == 1 else "NO")
print("Churn probability:", round(probability, 2))
