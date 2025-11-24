# -----------------------------------------------------------
#        MACHINE LEARNING : DISEASE PREDICTOR (DIABETES)
# -----------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------------

# Download Pima Diabetes Dataset from:
# https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv

data = pd.read_csv("diabetes.csv")   # Place dataset in same folder

# Display first few rows
print("\nDataset Preview:")
print(data.head())

# -----------------------------------------------------------
# 2. SPLIT DATA INTO FEATURES & LABEL
# -----------------------------------------------------------

X = data.drop("Outcome", axis=1)   # Features
y = data["Outcome"]                # Target label (0 = No, 1 = Yes)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------------------------------------
# 3. TRAIN MODEL
# -----------------------------------------------------------

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -----------------------------------------------------------
# 4. EVALUATE MODEL
# -----------------------------------------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# -----------------------------------------------------------
# 5. USER INPUT FOR PREDICTION
# -----------------------------------------------------------

print("\nEnter patient details for disease prediction:\n")

pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose Level: "))
bp = float(input("Blood Pressure: "))
skin = float(input("Skin Thickness: "))
insulin = float(input("Insulin Level: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

# Combine input into list
input_data = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]

# Make prediction
prediction = model.predict(input_data)

# -----------------------------------------------------------
# 6. DISPLAY RESULT
# -----------------------------------------------------------

if prediction[0] == 1:
    print("\n⚠️ DISEASE DETECTED: The patient is likely to have diabetes.")
else:
    print("\n✔ Healthy: The patient is unlikely to have diabetes.")

print("\n--- END OF PROGRAM ---")# machine-learning-disease-prediction
