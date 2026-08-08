import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression

# User Data 
data = {
    "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "age": [22, 35, 45, 28, 52, 31, 26, 48, 39, 55, 24, 42, 33, 29, 50],
    "monthly_visits": [2, 8, 15, 4, 20, 6, 3, 18, 10, 22, 2, 14, 7, 5, 19],
    "avg_purchase_amount": [500, 1200, 2000, 700, 2500, 1000, 600, 2200, 1500, 2800, 450, 1900, 1100, 800, 2400],
    "will_renew_subscription": [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1]
}

# Converting to Data Frame
df = pd.DataFrame(data)

# --- Separating independent features (Matrix X) and the dependent target variable (vector y) ---
X = df[["age", "monthly_visits"]]
y = df["avg_purchase_amount"]


# Standard split (80% train and 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) 


# ===================================================================
#               Training Model on LinearRegression
# ===================================================================

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

# Prediction on the testing data
linear_regression_model_pred = linear_regression_model.predict(X_test)

# =====================================================================
#               Check that is there overfitting or underfitting
# =====================================================================
train_score = linear_regression_model.score(X_train, y_train)
test_score  = linear_regression_model.score(X_test, y_test)

print("\n🔩 Comparing Training Score and Testing Score")

print(f"\nTraining Score :{train_score * 100:.2f}%")
print(f"Testing Score  :{test_score * 100:.2f}%\n")


# ================================================================
#                       Evaluation Report
# ================================================================
mae = mean_absolute_error(y_test, linear_regression_model_pred)
mse = mean_squared_error(y_test, linear_regression_model_pred)
rmse = mse ** 0.5         # Standard way to find RMSE
r_square = r2_score(y_test, linear_regression_model_pred)


print("============ 📜 MODEL PERFORMANCE REPORT ============")
print(f"🔹 Mean Absolute Error (MAE)               : {mae:.4f} spending")
print(f"🔹 Mean Squared Error (MSE)                : {mse:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE)          : {rmse:.4f} spending")
print(f"🔹 R Square                                : {r_square * 100:.4f}%")
print("="*50)


# ==================================================================
#                            New Customer
# ==================================================================

new_customer = pd.DataFrame(
    [[40, 12]],
    columns = ["age", "monthly_visits"]
)
new_customer_pred = linear_regression_model.predict(new_customer)[0]

print(f"\n✨ Predicted for the new customer (age :40, monthly visits :12): {new_customer_pred:.2f} rupees\n")



# --- Split features (X) and target variable (y) ---
X = df[["age", "monthly_visits", "avg_purchase_amount"]]
y = df["will_renew_subscription"]


# --- Standard Split (80% train and 20% split) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# ===================================================================
#                       Training Model on Logistic Regression
# ===================================================================
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)

# Prediction on the testing data 
logistic_regression_model_pred = logistic_regression_model.predict(X_test)



# ====================================================================
#                       Using confusion matrix 
# ====================================================================
logistic_regression_model_confusion_matrix = confusion_matrix(y_test, logistic_regression_model_pred)



# ====================================================================
#                       Visualizing confusion matrix
# ====================================================================

sns.heatmap(
    logistic_regression_model_confusion_matrix,
    annot=True, 
    cmap="Blues", 
    fmt="d",
    xticklabels=["Predicted No", "Predicted Yes"],
    yticklabels=["Actual No", "Actual Yes"]
)
plt.title("Confusion Matrix")
plt.show()
print("✅ Graph Created Successfully of Visualization confusion matrix\n")



# =======================================================================
#                       Classification Report
# =======================================================================

# Precision Score
logistic_regression_precision_score = precision_score(y_test, logistic_regression_model_pred)

# Recall Score
logistic_regression_recall_score = recall_score(y_test, logistic_regression_model_pred)

# F1 Score
logistic_regression_f1_score = f1_score(y_test, logistic_regression_model_pred)

# ROC-AUC Score
logistic_regression_roc_auc_score = roc_auc_score(y_test, logistic_regression_model.predict_proba(X_test)[:, 1])


print("============ 📜 Classification REPORT ============")
print(f"🔹 Precision Score      : {logistic_regression_precision_score*100:.2f}%")
print(f"🔹 Recall Score         : {logistic_regression_recall_score*100:.2f}%")
print(f"🔹 F1 Score             : {logistic_regression_f1_score:.2f}")
print(f"🔹 ROC-AUC Score        : {logistic_regression_roc_auc_score}")
print("="*50)


# =======================================================================
#                           New Customer 
# =======================================================================
new_customer = pd.DataFrame(
    [[30, 9, 950]],
    columns = ["age", "monthly_visits", "avg_purchase_amount"]
)

# New Customer Prediction
new_customer_pred = logistic_regression_model.predict(new_customer)[0]
new_customer_pred = "Yes" if new_customer_pred == 1 else "No"

# Probability of New Customer Prediction
new_customer_proba = logistic_regression_model.predict_proba(new_customer)[0]


print(f"\n✨ Probability for the new customer (age :30, monthly visits :9, avg purchases :950): {new_customer_proba}")
print(f"✨ Predicted for the new customer (age :30, monthly visits :9, avg purchases :950): {new_customer_pred}\n")