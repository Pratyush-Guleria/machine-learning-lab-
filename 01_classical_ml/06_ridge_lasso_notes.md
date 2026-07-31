# Machine Learning — Ridge & Lasso Regression 🎯

> **Author:** Pratyush Guleria
> **GitHub:** Pratyush-Guleria
> **Topic:** Fixing overfitting with Regularization

---

## Table of Contents
1. [The Problem This Solves](#1-the-problem-this-solves)
2. [What is Regularization?](#2-what-is-regularization)
3. [Ridge Regression (L2)](#3-ridge-regression-l2)
4. [Lasso Regression (L1)](#4-lasso-regression-l1)
5. [Ridge vs Lasso — Key Difference](#5-ridge-vs-lasso--key-difference)
6. [The alpha Parameter](#6-the-alpha-parameter)
7. [Complete Comparison Example](#7-complete-comparison-example)
8. [Which One to Use When](#8-which-one-to-use-when)
9. [Quick Revision](#9-quick-revision)

---

## 1. The Problem This Solves

Remember from Overfitting: a model overfits when it relies **too heavily** on certain features, fitting the training data (including its noise) too closely.

Ridge and Lasso are both **Linear Regression, with one addition** — a built-in mechanism that **penalizes the model for having large coefficients**, forcing it to stay simpler and generalize better.

> 💡 Both are literally imported from the same `sklearn.linear_model` module as `LinearRegression` — same `.fit()`, `.predict()` pattern you already know.

---

## 2. What is Regularization?

**Regularization** = adding a penalty that discourages the model from assigning huge importance (large coefficients) to any single feature.

**Intuition (no heavy math):**

Normal Linear Regression's only goal: minimize prediction error.

Regularized Regression's goal: minimize prediction error **AND** keep coefficients small.

```
Normal Linear Regression:
  "Just fit the data as closely as possible" 
  → can lead to very large coefficients → overfitting

Ridge/Lasso:
  "Fit the data well, BUT also keep coefficients small"
  → simpler model → less overfitting → generalizes better
```

The `alpha` parameter (covered below) controls HOW MUCH this penalty matters.

---

## 3. Ridge Regression (L2)

**What it does:** Shrinks all coefficients toward zero, but rarely makes them EXACTLY zero. Every feature stays in the model, just with reduced influence.

**Syntax:**
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

data = {
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary":     [25000, 30000, 38000, 45000, 52000,
                   58000, 65000, 70000, 78000, 85000]
}
df = pd.DataFrame(data)

X = df[["experience"]]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

print(model.coef_)
print(model.score(X_test, y_test))
```

> 💡 **Use Ridge when:** you believe most/all of your features are actually useful, and you just want to prevent the model from over-relying on any one of them.

---

## 4. Lasso Regression (L1)

**What it does:** Also shrinks coefficients, but can push some **all the way to exactly zero** — effectively removing those features from the model entirely.

**Syntax:**
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0)
model.fit(X_train, y_train)
```

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0)
model.fit(X_train, y_train)

print(model.coef_)
# Some coefficients might be exactly 0.0 — those features got "removed"
```

> 💡 **Lasso does automatic feature selection** — if a feature isn't useful, its coefficient becomes exactly 0, and it's effectively ignored by the model.

---

## 5. Ridge vs Lasso — Key Difference

| | Ridge (L2) | Lasso (L1) |
|--|-----------|------------|
| Shrinks coefficients | Yes | Yes |
| Can make coefficients exactly 0? | No — only shrinks toward 0 | Yes — can eliminate features |
| Effectively does feature selection? | No | Yes |
| Best when | Most features seem useful | You suspect many features are irrelevant |

```
Ridge:  coefficients → [0.8, 0.3, 0.5, 0.2]  (all shrunk, none zero)
Lasso:  coefficients → [0.9, 0.0, 0.4, 0.0]  (some pushed to exactly zero)
```

---

## 6. The `alpha` Parameter

`alpha` controls HOW STRONG the penalty is — this is the main knob you tune.

```python
Ridge(alpha=0)     # alpha=0 means NO penalty — behaves exactly like normal LinearRegression
Ridge(alpha=1.0)   # moderate penalty (default)
Ridge(alpha=10.0)  # strong penalty — coefficients shrink a lot, model becomes simpler
Ridge(alpha=100.0) # very strong penalty — risk of UNDERFITTING now (too simple)
```

> 💡 **Too high alpha = underfitting. Too low alpha = back to overfitting. This is a tradeoff you tune using Cross Validation (coming up in the roadmap).**

---

## 7. Complete Comparison Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

data = {
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "certifications": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    "salary": [28000, 32000, 40000, 45000, 55000,
               60000, 70000, 75000, 85000, 92000]
}
df = pd.DataFrame(data)

X = df[["experience", "certifications"]]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train all three side by side
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Lasso (alpha=1.0)": Lasso(alpha=1.0)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"{name}")
    print(f"  Coefficients : {model.coef_}")
    print(f"  Train R²     : {train_score:.4f}")
    print(f"  Test R²      : {test_score:.4f}\n")
```

---

## 8. Which One to Use When

```
Start here: Try plain LinearRegression first, check for overfitting

If overfitting detected AND you believe all features matter:
    → Use Ridge

If overfitting detected AND you suspect some features are useless/noisy:
    → Use Lasso (it will zero them out automatically)

If unsure:
    → Try both, compare test scores, pick the better one
```

---

## 9. Quick Revision

| What you want | Code |
|--------------|------|
| Import Ridge | `from sklearn.linear_model import Ridge` |
| Import Lasso | `from sklearn.linear_model import Lasso` |
| Create with penalty strength | `Ridge(alpha=1.0)` / `Lasso(alpha=1.0)` |
| Train | `model.fit(X_train, y_train)` |
| Check coefficients | `model.coef_` |
| Ridge behavior | Shrinks all coefficients, none become 0 |
| Lasso behavior | Can shrink coefficients to exactly 0 (feature selection) |
| alpha=0 | No penalty — same as LinearRegression |
| Higher alpha | Simpler model, risk of underfitting if too high |

---

## ML Progress Tracker 📊

```
✅ ML Intro, Train/Test Split
✅ Feature Scaling & Encoding
✅ Linear Regression
✅ Model Evaluation (Regression)
✅ Overfitting & Underfitting
✅ Ridge & Lasso Regression — you now know how to fight overfitting directly
⏳ Feature Engineering — next
⏳ Logistic Regression
... (12 more topics on the roadmap)
```

---

> 📌 **Final Tip for MLOps:**
> - Ridge/Lasso are your first line of defense against overfitting before trying more complex fixes
> - Lasso's automatic feature selection is genuinely useful in real projects with many features — it tells you which ones actually matter
> - In production ML pipelines, `alpha` gets tuned automatically via Hyperparameter Tuning (upcoming topic) rather than guessed manually
> - Compare Linear vs Ridge vs Lasso test scores side by side — this comparison itself is a common interview question 🚀

---

*Notes by Pratyush Guleria*
*GitHub: Pratyush-Guleria*