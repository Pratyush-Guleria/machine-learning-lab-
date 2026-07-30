# Machine Learning — Model Evaluation (Regression) 📏

> **Author:** Pratyush Guleria
> **GitHub:** Pratyush-Guleria
> **Topic:** Measuring how good a regression model actually is

---

## Table of Contents
1. [Why Evaluate a Model?](#1-why-evaluate-a-model)
2. [The Core Idea — Error](#2-the-core-idea--error)
3. [MAE — Mean Absolute Error](#3-mae--mean-absolute-error)
4. [MSE — Mean Squared Error](#4-mse--mean-squared-error)
5. [RMSE — Root Mean Squared Error](#5-rmse--root-mean-squared-error)
6. [R² Score](#6-r²-score)
7. [Which Metric to Use When](#7-which-metric-to-use-when)
8. [Complete Example — All Metrics Together](#8-complete-example--all-metrics-together)
9. [Quick Revision](#9-quick-revision)

---

## 1. Why Evaluate a Model?

After `model.fit()` and `model.predict()`, you have predictions — but you don't yet know if they're GOOD predictions. Model Evaluation answers one question:

> **"How far off were my predictions from the actual values?"**

This is not optional — without evaluation, you have no idea if your model is usable or garbage.

---

## 2. The Core Idea — Error

**Error** = the difference between what the model predicted and what actually happened.

```
Actual salary:    50000
Predicted salary:  48000
Error:              2000   (model was off by ₹2000)
```

Every metric below is just a different way of **summarizing all the errors** across your entire test set into ONE number.

---

## 3. MAE — Mean Absolute Error

**What it means:** On average, how far off are your predictions — in the SAME units as your target (₹, marks, etc).

**Formula (intuition):**
```
1. Find the error for every prediction (actual - predicted)
2. Make every error positive (remove negative signs — "absolute")
3. Take the average of all those errors
```

**Syntax:**
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
```

```python
from sklearn.metrics import mean_absolute_error

y_test = [50000, 60000, 45000, 70000]
y_pred = [48000, 62000, 44000, 68000]

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")
# MAE: 1500.0
# On average, predictions are off by ₹1500
```

> 💡 MAE is easy to explain to non-technical people: "Our model is off by ₹1500 on average."

---

## 4. MSE — Mean Squared Error

**What it means:** Similar to MAE, but SQUARES each error before averaging — this makes big mistakes count MUCH more than small ones.

**Formula (intuition):**
```
1. Find the error for every prediction
2. SQUARE each error (this removes negatives AND punishes big errors more)
3. Take the average
```

**Syntax:**
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
```

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")
# MSE: 3500000.0
```

> ⚠️ **Problem with MSE:** The units get squared too (₹ becomes ₹²) — this makes the number hard to interpret directly. That's why RMSE exists.

**Why squaring matters:** If your model is off by ₹1000 on 9 predictions but off by ₹10,000 on 1 prediction — MSE will heavily penalize that one big miss, while MAE treats all errors more equally.

---

## 5. RMSE — Root Mean Squared Error

**What it means:** MSE's square root — brings the units back to normal (₹, marks, etc), while still punishing big errors more than MAE does.

**Syntax:**
```python
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_test, y_pred)
```

```python
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse}")
# RMSE: 1870.8
```

> 💡 **RMSE is the most commonly used regression metric in practice** — it's interpretable (same units as target) AND sensitive to large errors (important because big mistakes are usually worse in real life).

### MAE vs RMSE — same data, different story

```python
# If errors are consistent (all around 1500):
# MAE ≈ RMSE

# If there's one huge outlier error:
# RMSE will be much bigger than MAE
# This difference itself tells you something about your model's mistakes!
```

---

## 6. R² Score

**What it means:** A percentage-like score (0 to 1) showing how much of the variation in your data the model actually explains — completely different concept from MAE/MSE/RMSE.

```
R² = 1.0   → Model perfectly explains the data (rare, often means overfitting)
R² = 0.85  → Model explains 85% of the pattern in your data — good
R² = 0.0   → Model is no better than just guessing the average every time
R² < 0.0   → Model is WORSE than just guessing the average (bad model)
```

**Syntax:**
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
```

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2:.2f}")
# R2 Score: 0.94
# The model explains 94% of the variation in salary
```

> 💡 R² is the metric people usually ask about first — "how good is your model?" — answered with R², not MAE/RMSE.

---

## 7. Which Metric to Use When

| Metric | Best for | Sensitive to outliers? |
|--------|----------|------------------------|
| MAE | Simple, easy explanation | No |
| MSE | Mathematical/optimization purposes | Yes (heavily) |
| RMSE | Most common — general use, interpretable | Yes |
| R² | Answering "how good is the model overall?" | Somewhat |

> 💡 **In practice:** Report RMSE and R² together in almost every regression project. MAE is a nice addition for simple explanations to non-technical stakeholders.

---

## 8. Complete Example — All Metrics Together

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# Data
data = {
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary":     [25000, 30000, 38000, 45000, 52000,
                   58000, 65000, 70000, 78000, 85000]
}
df = pd.DataFrame(data)

X = df[["experience"]]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate — ALL metrics together
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")
```

---

## 9. Quick Revision

| What you want | Code |
|--------------|------|
| Import all metrics | `from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score` |
| MAE | `mean_absolute_error(y_test, y_pred)` |
| MSE | `mean_squared_error(y_test, y_pred)` |
| RMSE | `root_mean_squared_error(y_test, y_pred)` |
| R² Score | `r2_score(y_test, y_pred)` |
| Low MAE/RMSE | Good — small errors |
| High R² (close to 1) | Good — model explains data well |

---

## ML Progress Tracker 📊

```
✅ ML Intro, Train/Test Split
✅ Feature Scaling & Encoding
✅ Linear Regression
✅ Model Evaluation (Regression) — you can now measure how good ANY regression model is
⏳ Overfitting & Underfitting — next
⏳ Ridge & Lasso Regression
⏳ Feature Engineering
⏳ Logistic Regression
... (13 more topics on the roadmap)
```

---

> 📌 **Final Tip for MLOps:**
> - Always report RMSE + R² together in project READMEs — this is the industry standard combo
> - A model with great R² on training data but bad R² on test data = overfitting (next topic!)
> - These metrics get logged into MLflow later — every experiment run gets tracked with these numbers
> - Low error metrics + high R² = confidently deploy; otherwise, back to feature engineering or a different algorithm 🚀

---

*Notes by Pratyush Guleria*
*GitHub: Pratyush-Guleria*