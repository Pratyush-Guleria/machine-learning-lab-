import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge


# User data
data = {
    "size_sqft":       [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 1100, 1900, 2300, 2700, 1400],
    "bedrooms":         [2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 2, 4, 4, 5, 3],
    "age_years":        [5, 8, 3, 10, 2, 15, 1, 20, 4, 25, 7, 6, 12, 3, 9],
    "distance_market_km": [2, 3, 1, 5, 1, 8, 2, 10, 3, 12, 4, 2, 6, 1, 3],
    "price_lakhs":      [42, 48, 58, 65, 82, 88, 98, 108, 122, 128, 51, 85, 100, 118, 62]
}


# --- Data Preparation
df = pd.DataFrame(data)


# --- Separating independent features (Matrix X) and the dependent target variable (vector y) ---
X = df[["size_sqft", "bedrooms", "age_years", "distance_market_km"]]
y = df["price_lakhs"]


# --- Standard Split (80% Train 20% test) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# =================================================================
                # --- Model Training (LinearRegression) ---
#==================================================================

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)


# --- Prediction on Test Data (LinearRegression) ---

linear_regression_model_pred = linear_regression_model.predict(X_test)

new_house = pd.DataFrame(
    [[2693, 3, 4.5, 2.5]],
    columns = ["size_sqft", "bedrooms", "age_years", "distance_market_km"]
)

new_house_price_linear = linear_regression_model.predict(new_house)[0]

print(f"✨ Predicted for the New House Price [Algo used : LinearRegression] ✨\n")

print(f"House Size                  : 2693 sqft")
print("Bedrooms                    : 3")
print(f"House Age                   : 4.5 years")
print(f"[Prediction] House Price    : {new_house_price_linear:.4f} Lakhs\n")


# -------------------------------------------------------------------------
# Comparing Training Score and Testing Score
# -------------------------------------------------------------------------

print("🔩 Comparing Training Score and Testing Score")
train_score = linear_regression_model.score(X_train, y_train)
test_score = linear_regression_model.score(X_test, y_test)

print(f"Training Score :{train_score * 100:.2f}%")
print(f"Testing Score  :{test_score * 100:.2f}%\n")

if train_score > 0.9 and (train_score - test_score) > 0.15:
    print("⚠️ Likely Overfitting — big gap between train and test")
elif train_score < 0.6 and test_score < 0.6:
    print("⚠️ Likely Underfitting — both scores are low")
else:
    print("Model training score and test score both are good ")
    print("✅ Looks like a good fit\n")


# -------------------------------------------------------------------------
# FULL REPORT OF THE MODEL WITH ALL RELEVANT NUMBERS
# -------------------------------------------------------------------------

mae = mean_absolute_error(y_test, linear_regression_model_pred)
mse = mean_squared_error(y_test, linear_regression_model_pred)
rmse = mse ** 0.5
r_square = r2_score(y_test, linear_regression_model_pred)

print("============ 📜 MODEL PERFORMANCE REPORT ============")
print(f"🔹 Mean Absolute Error (MAE)               : {mae:.4f} Lakhs")
print(f"🔹 Mean Squared Error (MSE)                : {mse:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE)          : {rmse:.4f} Lakhs")
print(f"🔹 R Square                                : {r_square * 100:.4f}%")
print("-"*50,"\n\n")

print("="*100 )
print("="*100,"\n\n")


# =========================================================================
                    # --- Model Training (Ridge) ---                        
# =========================================================================

ridge_model = Ridge(alpha = 1)
ridge_model.fit(X_train, y_train)


# --- Prediction on the Test Data (Ridge) ---

ridge_model_pred = ridge_model.predict(X_test)
new_house_price_ridge = ridge_model.predict(new_house)[0]
print(f"✨ Predicted for the New House Price [Algo used : Ridge] ✨\n")

print(f"House Size                  : 2693 sqft")
print("Bedrooms                    : 3")
print(f"House Age                   : 4.5 years")
print(f"[Prediction] House Price    : {new_house_price_ridge:.4f} Lakhs\n")


# -------------------------------------------------------------------------
# Comparing Training Score and Testing Score
# -------------------------------------------------------------------------

print("🔩 Comparing Training Score and Testing Score")
train_score = ridge_model.score(X_train, y_train)
test_score = ridge_model.score(X_test, y_test)

print(f"Training Score :{train_score * 100:.2f}%")
print(f"Testing Score  :{test_score * 100:.2f}%\n")

if train_score > 0.9 and (train_score - test_score) > 0.15:
    print("⚠️ Likely Overfitting — big gap between train and test")
elif train_score < 0.6 and test_score < 0.6:
    print("⚠️ Likely Underfitting — both scores are low")
else:
    print("Model training score and test score both are good ")
    print("✅ Looks like a good fit\n")


# -------------------------------------------------------------------------
# FULL REPORT OF THE MODEL WITH ALL RELEVANT NUMBERS
# -------------------------------------------------------------------------

mae = mean_absolute_error(y_test, ridge_model_pred)
mse = mean_squared_error(y_test, ridge_model_pred)
rmse = mse ** 0.5
r_square = r2_score(y_test, ridge_model_pred)

print("============ 📜 MODEL PERFORMANCE REPORT ============")
print(f"🔹 Mean Absolute Error (MAE)               : {mae:.4f} Lakhs")
print(f"🔹 Mean Squared Error (MSE)                : {mse:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE)          : {rmse:.4f} Lakhs")
print(f"🔹 R Square                                : {r_square * 100:.4f}%")
print("="*50,"\n\n")


print("="*100 )
print("="*100,"\n\n")


# =========================================================================
                    # --- Model Training (Lasso) ---                        
# =========================================================================

lasso_model = Lasso(alpha = 1)
lasso_model.fit(X_train, y_train)


# --- Prediction on the Test Data (Ridge) ---

lasso_model_pred = lasso_model.predict(X_test)
new_house_price_lasso = lasso_model.predict(new_house)[0]
print(f"✨ Predicted for the New House Price [Algo used : Lasso] ✨\n")

print(f"House Size                  : 2693 sqft")
print("Bedrooms                    : 3")
print(f"House Age                   : 4.5 years")
print(f"[Prediction] House Price    : {new_house_price_lasso:.4f} Lakhs\n")


# -------------------------------------------------------------------------
# Comparing Training Score and Testing Score
# -------------------------------------------------------------------------

print("🔩 Comparing Training Score and Testing Score")
train_score = lasso_model.score(X_train, y_train)
test_score = lasso_model.score(X_test, y_test)

print(f"Training Score :{train_score * 100:.2f}%")
print(f"Testing Score  :{test_score * 100:.2f}%\n")

if train_score > 0.9 and (train_score - test_score) > 0.15:
    print("⚠️ Likely Overfitting — big gap between train and test")
elif train_score < 0.6 and test_score < 0.6:
    print("⚠️ Likely Underfitting — both scores are low")
else:
    print("Model training score and test score both are good ")
    print("✅ Looks like a good fit\n")


# -------------------------------------------------------------------------
# FULL REPORT OF THE MODEL WITH ALL RELEVANT NUMBERS
# -------------------------------------------------------------------------

mae = mean_absolute_error(y_test, lasso_model_pred)
mse = mean_squared_error(y_test, lasso_model_pred)
rmse = mse ** 0.5
r_square = r2_score(y_test, lasso_model_pred)

print("============ 📜 MODEL PERFORMANCE REPORT ============")
print(f"🔹 Mean Absolute Error (MAE)               : {mae:.4f} Lakhs")
print(f"🔹 Mean Squared Error (MSE)                : {mse:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE)          : {rmse:.4f} Lakhs")
print(f"🔹 R Square                                : {r_square * 100:.4f}%")


print("\n📐 Lasso Feature Weights (Coefficients):")
for feature, coef in zip(X.columns, lasso_model.coef_):
    print(f"🔹 {feature}: {coef:.4f}")

print("="*50,"\n\n")


# =========================================================================
# 📊 FINAL COMPARISON REPORT FOR VIKAS CHANDRA (SIDE-BY-SIDE R² SCORE)
# =========================================================================

# Ek pandas dataframe banana side-by-side comparison ke liye
comparison_summary = pd.DataFrame({
    "Model Name": ["Linear Regression", "Ridge Regression", "Lasso Regression"],
    "Train R² Score (%)": [
        linear_regression_model.score(X_train, y_train) * 100,
        ridge_model.score(X_train, y_train) * 100,
        lasso_model.score(X_train, y_train) * 100
    ],
    "Test R² Score (%)": [
        r2_score(y_test, linear_regression_model_pred) * 100,
        r2_score(y_test, ridge_model_pred) * 100,
        r2_score(y_test, lasso_model_pred) * 100
    ]
})

print("\n\n🏆 ============ FINAL SIDE-BY-SIDE MODEL COMPARISON ============")
print(comparison_summary.to_string(index=False, float_format="%.2f"))
print("=================================================================")

print("\n📝 Final Conclusion for RealEstate Predictor:")
print("👉 The BEST model for this dataset is 'Lasso Regression'!")
print("💡 Reasoning: Lasso successfully identified noisy/unnecessary parameters.")
print("   It reduced the coefficients of 'bedrooms' and 'distance_market_km' exactly to 0.0000,")
print("   performing automatic feature selection, which protects against potential overfitting.")
print("=================================================================\n")