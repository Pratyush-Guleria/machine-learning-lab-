import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


## User Data
data = {
    "ride_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "pickup_date": ["2024-01-06", "2024-01-08", "2024-01-13", "2024-01-15", "2024-01-20",
                     "2024-01-22", "2024-01-27", "2024-02-03", "2024-02-05", "2024-02-10"],
    "distance_km": [5, 12, 3, 20, 8, 15, 4, 25, 6, 18],
    "driver_name": ["Ravi", "Suresh", "Amit", "Vikram", "Rahul",
                     "Sanjay", "Deepak", "Manoj", "Ajay", "Karan"],
    "fare": [120, 280, 90, 450, 190, 340, 100, 550, 150, 400]
}

df = pd.DataFrame(data)

## Drop useless columns
df = df.drop(columns = ["ride_id", "driver_name"])


## Converting from str -> int
df["pickup_date"] = pd.to_datetime(df["pickup_date"])


## Creating 4 new features

# Day of week 
df["day_of_week"] = df["pickup_date"].dt.dayofweek


# Day Name
df["day_name"] = df["pickup_date"].dt.day_name()


# Is weekend 
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)  # 5 = Saturday, 6 = Sunday 


# Fair per category
df["distace_category"] = pd.cut(
    df["distance_km"],
    bins = [0, 8, 16, float("inf")],
    labels = ["Short", "Medium", "Long"]
)


# --- Separating independent features (Matrix X) and the dependent target variable (vector y) ---
X = df[["distance_km", "day_of_week", "is_weekend"]]
y = df["fare"]


# --- Standard Split (80% train 20% test) --- 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# =================================================================
                # --- Model Training (LinearRegression) ---
#==================================================================
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)


# Prediction on the Testing Data (LinearRegression)
linear_regression_model_pred = linear_regression_model.predict(X_test)

# Prediction on the new data
new_customer = pd.DataFrame(
    [[10, 2, 0]],
    columns = ["distance_km", "day_of_week", "is_weekend"]
)

new_fare = linear_regression_model.predict(new_customer)[0]

print(f"✨ Predicted for the New Fare Price [Algo used : LinearRegression] ✨\n")

print("Distance in km            : 10km")
print("Day of week               : Wednesday")
print("Is Weekend                : False")
print(f"[Prediction] New Fare     : {new_fare:.2f}\n")


# -------------------------------------------------------------------------
# Comparing Training Score and Testing Score
# -------------------------------------------------------------------------

print("🔩 Comparing Training Score and Testing Score")
train_score = linear_regression_model.score(X_train, y_train)
test_score = linear_regression_model.score(X_test, y_test)

print(f"Training Score :{train_score * 100:.2f}%")
print(f"Testing Score  :{test_score * 100:.2f}%\n")


# -------------------------------------------------------------------------
# FULL REPORT OF THE MODEL WITH ALL RELEVANT NUMBERS
# -------------------------------------------------------------------------

mae = mean_absolute_error(y_test, linear_regression_model_pred)
mse = mean_squared_error(y_test, linear_regression_model_pred)
rmse = mse ** 0.5
r_square = r2_score(y_test, linear_regression_model_pred)

print("============ 📜 MODEL PERFORMANCE REPORT ============")
print(f"🔹 Mean Absolute Error (MAE)               : {mae:.4f} currency")
print(f"🔹 Mean Squared Error (MSE)                : {mse:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE)          : {rmse:.4f} currency")
print(f"🔹 R Square                                : {r_square * 100:.4f}%")
print("-"*50,"\n\n")
