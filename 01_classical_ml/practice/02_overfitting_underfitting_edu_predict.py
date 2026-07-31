import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression


# --- Data Preparation ---

data = {
    "hours_studied": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10],
    "score": [35, 40, 38, 50, 48, 55, 58, 62, 65, 70, 68, 75, 78, 82, 88, 95]
}

df = pd.DataFrame(data)


# --- Separating independent features (Matrix X) and the dependent target variable (vector y) ---
X = df[["hours_studied"]]
y = df["score"]


# --- Standard Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 42)


# --- Model Training ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Prediction on Test Data ---
y_pred = model.predict(X_test)

new_student = pd.DataFrame(
    [[9]],
    columns=["hours_studied"]
)
new_score = model.predict(new_student)[0]

print(f"\n✨ Predicted for the new student score (9 hours studied): {new_score:.1f} Marks\n")

# =========================================================================
# Comparing Training Score and Testing Score
# =========================================================================
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Score :{train_score * 100:.2f}%")
print(f"Testing Score  :{test_score * 100:.2f}%\n")

if train_score > 0.9 and (train_score - test_score) > 0.15:
    print("⚠️ Likely Overfitting — big gap between train and test")
elif train_score < 0.6 and test_score < 0.6:
    print("⚠️ Likely Underfitting — both scores are low")
else:
    print("Model training score and test score both are good ")
    print("✅ Looks like a good fit\n")
    

# =========================================================================
# FULL REPORT OF THE MODEL WITH ALL RELEVANT NUMBERS
# =========================================================================
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = MSE ** 0.5                      # Standard way to find RMSE 
R_square = r2_score(y_test, y_pred)

print("============ 📜 MODEL PERFORMANCE REPORT ============")
print(f"🔹 Mean Absolute Error (MAE)               : {MAE:.4f} Score")
print(f"🔹 Mean Squared Error (MSE)                : {MSE:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE)          : {RMSE:.4f} Score")
print(f"🔹 R Square                                :{R_square * 100:.4f}%")
print("="*50)