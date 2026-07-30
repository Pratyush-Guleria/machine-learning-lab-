import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- Step 1: Data Preparation ----
data = {
    "car_age_years": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 4, 6, 8, 1],
    "km_driven_k":   [15, 25, 40, 55, 70, 85, 95, 110, 125, 140, 20, 60, 90, 120, 10],
    "price_lakhs":   [8.5, 7.2, 6.0, 5.1, 4.3, 3.6, 3.0, 2.5, 2.1, 1.8, 7.8, 5.5, 3.4, 2.3, 8.8]
}

df = pd.DataFrame(data)

# Separating independent features (Matrix X) and the dependent target variable (vector y)
X = df[["car_age_years", "km_driven_k"]]
y = df["price_lakhs"]

# ---- Step 2: Standard Split (80% Train, 20% Test) ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# ---- Step 3: Model Training ----
model = LinearRegression()
model.fit(X_train, y_train)

# ---- Step 4: Predictions on Test Data ----
y_pred = model.predict(X_test)


# =========================================================================
# 📊 REQUIREMENT 3: FULL REPORT OF THE MODEL WITH ALL RELEVANT NUMBERS
# =========================================================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Standard way to calculate RMSE across all versions
r2 = r2_score(y_test, y_pred)

print("============ 📜 MODEL PERFORMANCE REPORT ============")
print(f"🔹 Mean Absolute Error (MAE)    : {mae:.4f} Lakhs")
print(f"🔹 Mean Squared Error (MSE)     : {mse:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE): {rmse:.4f} Lakhs")
print(f"🔹 Model Accuracy (R2 Score)    : {r2 * 100:.2f}%")
print("=====================================================")


# =========================================================================
# 🔮 REQUIREMENT 4: PREDICT PRICE FOR A NEW CAR (5 years old, 75k km)
# =========================================================================
# Creating a proper 2D DataFrame for the new car prediction to avoid warnings
new_car = pd.DataFrame([[5, 75]], columns=["car_age_years", "km_driven_k"])
new_car_price = model.predict(new_car)[0]

print(f"\n✨ Predicted Price for the new car (5 yrs old, 75k km): {new_car_price:.2f} Lakhs")


# =========================================================================
# 📝 REQUIREMENT 5: MODEL EQUATION / MATHEMATICAL FORMULA
# =========================================================================
m1 = model.coef_[0]         # Slope for car_age_years
m2 = model.coef_[1]         # Slope for km_driven_k
c = model.intercept_        # Intercept / Constant

print("\n📐 Mathematical Equation of the Model:")
print(f"Price = ({m1:.4f} * car_age_years) + ({m2:.4f} * km_driven_k) + {c:.4f}")
print("=====================================================")
