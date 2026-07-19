# Machine Learning — Linear Regression 📈

> **Author:** Pratyush Guleria | MLOps Learning Journey 🚀
> **GitHub:** Pratyush-Guleria
> **Topic:** Predicting numbers with a straight line

---

## Table of Contents
1. [What is Linear Regression?](#1-what-is-linear-regression)
2. [The Intuition — No Heavy Math](#2-the-intuition--no-heavy-math)
3. [Simple vs Multiple Linear Regression](#3-simple-vs-multiple-linear-regression)
4. [Linear Regression in Scikit-learn](#4-linear-regression-in-scikit-learn)
5. [Making Predictions](#5-making-predictions)
6. [Understanding the Model — Coefficients](#6-understanding-the-model--coefficients)
7. [Visualizing the Regression Line](#7-visualizing-the-regression-line)
8. [Multiple Linear Regression Example](#8-multiple-linear-regression-example)
9. [Quick Revision](#9-quick-revision)

---

## 1. What is Linear Regression?

**Linear Regression** predicts a number by fitting the best straight line through your data points.

**Use case:** Predicting house price, salary, exam score — anything that's a continuous NUMBER.

```
Salary
  │           •
  │        •
  │     •  ← the line tries to pass as close
  │  •     as possible to all points
  │•
  └──────────────── Years of Experience
```

---

## 2. The Intuition — No Heavy Math

You've seen this equation before in school:

```
y = mx + c
```

- `m` = slope (how steep the line is)
- `c` = intercept (where the line starts, when x=0)
- `x` = your input (e.g., years of experience)
- `y` = your prediction (e.g., salary)

**What Linear Regression does:** It looks at all your data points and finds the BEST values for `m` and `c` — the line that has the least total distance from all points.

> 💡 You don't calculate `m` and `c` yourself — Scikit-learn does this automatically when you call `.fit()`.

**In ML terms, this equation becomes:**
```
y = coefficient × X + intercept
```

Same thing, different names — `coefficient` = `m` (slope), `intercept` = `c`.

---

## 3. Simple vs Multiple Linear Regression

| | Simple | Multiple |
|--|--------|----------|
| Inputs (X) | 1 feature | 2+ features |
| Example | Salary from Experience only | Salary from Experience + Education + City |
| Equation | `y = m·x + c` | `y = m1·x1 + m2·x2 + ... + c` |

---

## 4. Linear Regression in Scikit-learn

**Syntax:**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary":     [25000, 30000, 38000, 45000, 52000,
                   58000, 65000, 70000, 78000, 85000]
}
df = pd.DataFrame(data)

X = df[["experience"]]   # Features — must be 2D (double brackets)
y = df["salary"]         # Target — 1D (single brackets)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)     # this is where the model "learns"
```

> 💡 `.fit()` is the moment the model calculates the best `m` and `c` values using your training data.

---

## 5. Making Predictions

**Syntax:**
```python
predictions = model.predict(X_test)
```

```python
# Predict salaries for the test set
predictions = model.predict(X_test)
print(predictions)

# Predict for a brand new value — someone with 12 years experience
new_experience = [[12]]   # must be 2D — a list of lists
predicted_salary = model.predict(new_experience)
print(predicted_salary)    # e.g. [95000.5]
```

---

## 6. Understanding the Model — Coefficients

**Syntax:**
```python
model.coef_        # the slope(s) — m
model.intercept_   # the intercept — c
```

```python
print("Slope (m):", model.coef_)
print("Intercept (c):", model.intercept_)

# Example output:
# Slope (m): [6650.5]
# Intercept (c): 18500.2

# This means: salary = 6650.5 × experience + 18500.2
```

> 💡 The slope tells you: for every 1 unit increase in `experience`, salary increases by ₹6650.5 (approximately).

---

## 7. Visualizing the Regression Line

```python
import matplotlib.pyplot as plt

plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")

plt.title("Experience vs Salary — Linear Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (₹)")
plt.legend()
plt.grid(True)
plt.show()
```

> 💡 This visual check is important — if the line doesn't fit the data well, Linear Regression might not be the right algorithm for this problem.

---

## 8. Multiple Linear Regression Example

Same code, just more columns in `X`.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "experience": [1, 2, 3, 4, 5, 6, 7, 8],
    "education_years": [12, 14, 16, 16, 18, 16, 18, 20],
    "salary": [25000, 32000, 42000, 45000, 58000, 52000, 68000, 82000]
}
df = pd.DataFrame(data)

X = df[["experience", "education_years"]]   # 2 features now
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.coef_)         # one coefficient per feature — [m1, m2]
print(model.intercept_)    # c

# Predict for someone with 5 years experience, 16 years education
prediction = model.predict([[5, 16]])
print(prediction)
```

---

## 9. Quick Revision

| What you want | Code |
|--------------|------|
| Import | `from sklearn.linear_model import LinearRegression` |
| Create model | `model = LinearRegression()` |
| Train model | `model.fit(X_train, y_train)` |
| Predict | `model.predict(X_test)` |
| Predict new value | `model.predict([[value]])` |
| Get slope | `model.coef_` |
| Get intercept | `model.intercept_` |
| Multiple features | `X = df[["col1", "col2"]]` |

---

## ML Progress Tracker 📊

```
✅ ML Intro — Supervised/Unsupervised, Regression/Classification
✅ Train/Test Split
✅ Linear Regression — first algorithm done!
⏳ Model Evaluation (for regression) — next, needed before moving on
⏳ Logistic Regression
⏳ Decision Tree
⏳ Random Forest
⏳ KNN
⏳ Feature Scaling
⏳ Cross Validation
```

> 💡 We'll cover **Model Evaluation** next — right after Linear Regression — because you need to know HOW GOOD your predictions are before learning more algorithms.

---

> 📌 **Final Tip for MLOps:**
> - Linear Regression is simple but still used in production for baseline models
> - `model.coef_` and `model.intercept_` — useful for explaining WHY the model predicts what it predicts (explainability)
> - Always visualize the regression line — a bad fit is obvious visually
> - This exact `.fit()` → `.predict()` pattern is used for EVERY Scikit-learn algorithm — you'll see it again and again 🚀

---

*Notes by Pratyush Guleria — MLOps Learning Journey 🚀*
*GitHub: Pratyush-Guleria*