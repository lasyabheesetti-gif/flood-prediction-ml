# =========================================================
# RISING WATERS: MACHINE LEARNING APPROACH TO FLOOD PREDICTION
# APSCHE SMARTBRIDGE MAJOR PROJECT
# Google Colab Compatible - Single Script
# =========================================================

# -----------------------------
# 1. Import Required Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 2. Create Dataset
# -----------------------------
np.random.seed(42)

data = {
    "Rainfall_mm": np.random.randint(50, 300, 500),
    "River_Level_m": np.random.uniform(2, 10, 500),
    "Soil_Moisture": np.random.uniform(0.2, 0.9, 500),
    "Temperature_C": np.random.randint(20, 40, 500)
}

df = pd.DataFrame(data)

# Flood condition (label creation)
df["Flood"] = (
    (df["Rainfall_mm"] > 180) &
    (df["River_Level_m"] > 6.5) &
    (df["Soil_Moisture"] > 0.6)
).astype(int)

print("Dataset Preview:")
print(df.head())

# -----------------------------
# 3. Data Exploration
# -----------------------------
print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nFlood Distribution:")
print(df["Flood"].value_counts())

# -----------------------------
# 4. Feature & Target Separation
# -----------------------------
X = df.drop("Flood", axis=1)
y = df["Flood"]

# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 7. Logistic Regression Model
# -----------------------------
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_lr))

# -----------------------------
# 8. Decision Tree Model
# -----------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy:",
      accuracy_score(y_test, y_pred_dt))

# -----------------------------
# 9. Random Forest Model (Main)
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:",
      accuracy_score(y_test, y_pred_rf))

# -----------------------------
# 10. Classification Report
# -----------------------------
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# -----------------------------
# 11. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Flood Prediction")
plt.show()

# -----------------------------
# 12. Feature Importance
# -----------------------------
feature_importance = rf.feature_importances_
features = X.columns

plt.figure(figsize=(6, 4))
plt.barh(features, feature_importance)
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.show()

# -----------------------------
# 13. Flood Risk Prediction (Simulation)
# -----------------------------
new_input = pd.DataFrame({
    "Rainfall_mm": [220],
    "River_Level_m": [7.8],
    "Soil_Moisture": [0.75],
    "Temperature_C": [28]
})

new_input_scaled = scaler.transform(new_input)
prediction = rf.predict(new_input_scaled)
probability = rf.predict_proba(new_input_scaled)

print("\nNew Data Prediction:")
if prediction[0] == 1:
    print("⚠️ HIGH FLOOD RISK DETECTED")
else:
    print("✅ NO FLOOD RISK")

print("Flood Probability:", probability)

# -----------------------------
# 14. Risk Level Classification
# -----------------------------
risk_prob = probability[0][1]

if risk_prob < 0.3:
    risk = "LOW RISK"
elif risk_prob < 0.6:
    risk = "MEDIUM RISK"
else:
    risk = "HIGH RISK"

print("Flood Risk Level:", risk)

# -----------------------------
# 15. Project Completed
# -----------------------------
print("\n✅ Flood Prediction ML Project Execution Completed Successfully")
