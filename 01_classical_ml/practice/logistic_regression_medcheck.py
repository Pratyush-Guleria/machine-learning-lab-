import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# User Data
data = {
    "age": [25, 30, 35, 40, 45, 50, 55, 60, 28, 33, 48, 52, 58, 22, 38],
    "bmi": [22, 24, 27, 29, 31, 33, 35, 37, 21, 26, 32, 34, 36, 20, 28],
    "has_diabetes_risk": [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)


# --- Separating independent features (Matrix X) and the dependent target variable (vector y) ---
X = df[["age", "bmi"]]
y = df["has_diabetes_risk"]


# --- Standard Split (80% train 20% test) --- 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# =================================================================
                # --- Model Training (LogesticRegression) ---
#==================================================================

logistic_model_model = LogisticRegression()
logistic_model_model.fit(X_train, y_train)


# Prediction on the Testing Data
logesticregression_model_pred = logistic_model_model.predict(X_test)


# Prediction on the new Data 
new_patient = pd.DataFrame(
    [[42, 30]],
    columns = ["age", "bmi"]
)
new_patient_test_pred = logistic_model_model.predict(new_patient)[0]
new_patient_test_pred_confidence = logistic_model_model.predict_proba(new_patient)[0]

print(f"✨ Predicted for the New Patient [Algo used : LogesticRegression] ✨\n")

print("Age                             : 42")
print("BMI                             : 30")
print(f"[Prediction] Has Diabetes_risk : {"Yes" if new_patient_test_pred == 1 else "No"}")
print(f"Model Confidence               : {new_patient_test_pred_confidence}")


# -------------------------------------------------------------------------
# Comparing Training Score and Testing Score
# -------------------------------------------------------------------------
print("\n🔩 Comparing Training Score and Testing Score")

train_score = logistic_model_model.score(X_train, y_train)
test_score = logistic_model_model.score(X_test, y_test)

print(f"Training Score :{train_score * 100:.2f}%")
print(f"Testing Score  :{test_score * 100:.2f}%\n")


new_batch = pd.DataFrame({
    "age" : [27, 44, 51, 36, 60],
    "bmi" : [23, 30, 34, 26, 38]
})

new_batch_test_pred = logistic_model_model.predict_proba(new_batch[["age", "bmi"]])
new_batch['Diabetic_Probability_%'] = (new_batch_test_pred[:, 1] * 100).round(2)

new_batch['Final_Prediction'] = logistic_model_model.predict(new_batch[['age', 'bmi']])
new_batch['Has_Diabetes_Risk'] = new_batch["Final_Prediction"].map({0 : "No", 1 : "Yes"})
print(new_batch)