# Machine Learning — Overfitting & Underfitting ⚖️

> **Author:** Pratyush Guleria
> **GitHub:** Pratyush-Guleria
> **Topic:** Understanding when a model has learned "too much" or "too little"

---

## Table of Contents
1. [The Core Problem](#1-the-core-problem)
2. [Underfitting — Model Learned Too Little](#2-underfitting--model-learned-too-little)
3. [Overfitting — Model Memorized Instead of Learning](#3-overfitting--model-memorized-instead-of-learning)
4. [The Sweet Spot — Good Fit](#4-the-sweet-spot--good-fit)
5. [How to Detect Overfitting — Train vs Test Score](#5-how-to-detect-overfitting--train-vs-test-score)
6. [Detecting It in Code](#6-detecting-it-in-code)
7. [How to Fix Underfitting](#7-how-to-fix-underfitting)
8. [How to Fix Overfitting](#8-how-to-fix-overfitting)
9. [Quick Revision](#9-quick-revision)

---

## 1. The Core Problem

The entire goal of ML is: **learn the general PATTERN in data, not memorize the specific examples.**

A model needs to work well on data it has **never seen before** (test data, real-world future data) — not just the data it trained on.

```
Studying for an exam analogy:

Underfitting  → You barely studied — fail both practice tests AND real exam
Good Fit      → You understood the concepts — do well on practice AND real exam
Overfitting   → You memorized the practice test answers word-for-word —
                ace the practice test, but fail the real exam (different questions)
```

---

## 2. Underfitting — Model Learned Too Little

The model is **too simple** to capture the pattern in the data. It performs badly on BOTH training and test data.

**Signs:**
- Low accuracy / high error on training data
- Low accuracy / high error on test data too
- Model basically didn't learn anything useful

**Common causes:**
- Model is too simple for the problem (e.g., using Linear Regression for a very complex, non-linear pattern)
- Not enough useful features
- Not trained long enough / not enough data

```
Example: Trying to fit a straight line through data that's clearly curved
Result: The line misses almost every point — bad on train AND test
```

---

## 3. Overfitting — Model Memorized Instead of Learning

The model is **too complex** — it learned the training data so precisely (including its noise/randomness) that it fails to generalize to new data.

**Signs:**
- Very high accuracy / very low error on training data
- Noticeably worse accuracy / higher error on test data
- Big GAP between train performance and test performance

**Common causes:**
- Model is too complex for the amount of data available
- Training for too long
- Too many features relative to the number of data points
- Not enough training data

```
Example: A model that perfectly predicts every single training example,
including weird one-off outliers, but then performs poorly on new data
because it learned the "noise" instead of the real pattern
```

> 💡 **Overfitting is the MORE common problem in practice** — it's what Ridge/Lasso Regression (next topic) are specifically designed to fix.

---

## 4. The Sweet Spot — Good Fit

```
Model Complexity →

Underfit          Good Fit          Overfit
(too simple)      (just right)      (too complex)
    │                  │                  │
Train: Bad         Train: Good        Train: Excellent
Test:  Bad         Test:  Good        Test:  Bad
```

> 💡 The goal is NOT the highest possible training accuracy — it's the best TEST accuracy, with training and test scores close to each other.

---

## 5. How to Detect Overfitting — Train vs Test Score

The core technique: **compare model performance on training data vs test data.**

| Scenario | Train Score | Test Score | Diagnosis |
|----------|-------------|-------------|-----------|
| Both low | 60% | 58% | **Underfitting** |
| Both high, close together | 92% | 89% | **Good Fit** ✅ |
| Train very high, test much lower | 99% | 65% | **Overfitting** |

---

## 6. Detecting It in Code

**Syntax:**
```python
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
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

X = df[["experience"]]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# .score() gives R² for regression models automatically
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R²: {train_score:.4f}")
print(f"Test R²:  {test_score:.4f}")

if train_score > 0.9 and (train_score - test_score) > 0.15:
    print("⚠️ Likely Overfitting — big gap between train and test")
elif train_score < 0.6 and test_score < 0.6:
    print("⚠️ Likely Underfitting — both scores are low")
else:
    print("✅ Looks like a good fit")
```

> 💡 `.score()` on a regression model automatically returns R² — same thing you calculated manually with `r2_score()` before, just a shortcut.

---

## 7. How to Fix Underfitting

```
1. Use a more complex/powerful model
   (e.g., switch from Linear Regression to Decision Tree/Random Forest)

2. Add more relevant features (Feature Engineering)

3. Reduce regularization if any is being applied
   (regularization intentionally simplifies models — too much causes underfitting)

4. Train for longer / provide more data
```

---

## 8. How to Fix Overfitting

```
1. Get more training data
   (harder for the model to memorize noise with more examples)

2. Simplify the model
   (fewer features, shallower Decision Tree, etc.)

3. Use Regularization — Ridge/Lasso Regression (next topic!)
   (mathematically discourages the model from relying too heavily on any one feature)

4. Use Cross Validation
   (tests the model more rigorously across different data splits)

5. Remove irrelevant/noisy features (Feature Selection)
```

---

## 9. Quick Revision

| Concept | One Line |
|---------|---------|
| Underfitting | Model too simple — bad on train AND test |
| Overfitting | Model too complex — great on train, bad on test |
| Good Fit | Train and test scores are both good AND close together |
| Detection | Compare `model.score(X_train, y_train)` vs `model.score(X_test, y_test)` |
| Fix Underfitting | More complex model, more features |
| Fix Overfitting | More data, simpler model, Regularization (Ridge/Lasso), Cross Validation |

---

## ML Progress Tracker 📊

```
✅ ML Intro, Train/Test Split
✅ Feature Scaling & Encoding
✅ Linear Regression
✅ Model Evaluation (Regression)
✅ Overfitting & Underfitting — you can now diagnose ANY model's health
⏳ Ridge & Lasso Regression — next (the fix for overfitting)
⏳ Feature Engineering
⏳ Logistic Regression
... (12 more topics on the roadmap)
```

---

> 📌 **Final Tip for MLOps:**
> - This train-vs-test comparison is the FIRST thing to check after training any model — before even looking at other metrics
> - A model that overfits in development will perform badly in production on real, unseen data — this is a critical production risk
> - MLflow (later in your roadmap) logs both train and test scores automatically for every experiment — this exact comparison
> - "My model has 99% accuracy" means nothing without also knowing the test accuracy 🚀

---

*Notes by Pratyush Guleria*
*GitHub: Pratyush-Guleria*