# Machine Learning — Logistic Regression 🎯

> **Author:** Pratyush Guleria
> **GitHub:** Pratyush-Guleria
> **Topic:** Your first Classification algorithm — predicting categories, not numbers

---

## Table of Contents
1. [Wait — Why "Regression" for Classification?](#1-wait--why-regression-for-classification)
2. [The Intuition — No Heavy Math](#2-the-intuition--no-heavy-math)
3. [Logistic Regression in Scikit-learn](#3-logistic-regression-in-scikit-learn)
4. [Predicting Classes vs Probabilities](#4-predicting-classes-vs-probabilities)
5. [The Decision Threshold](#5-the-decision-threshold)
6. [Multi-class Classification](#6-multi-class-classification)
7. [Complete Example](#7-complete-example)
8. [Logistic Regression vs Linear Regression](#8-logistic-regression-vs-linear-regression)
9. [Quick Revision](#9-quick-revision)

---

## 1. Wait — Why "Regression" for Classification?

Confusing name, but here's the deal: **Logistic Regression is a Classification algorithm**, despite the name. It's called "regression" because internally it uses a similar equation to Linear Regression — but the OUTPUT is squeezed into a probability (0 to 1), which is then used to decide a category.

```
Linear Regression:    predicts a NUMBER (salary, price)
Logistic Regression:  predicts a CATEGORY (pass/fail, spam/not spam)
                       — by first calculating a PROBABILITY
```

---

## 2. The Intuition — No Heavy Math

**Step 1:** Just like Linear Regression, it calculates a number using `y = mx + c` style math.

**Step 2:** That number gets passed through a special curve called the **Sigmoid function** — this squeezes ANY number into a range between 0 and 1.

```
Any number (-∞ to +∞) → Sigmoid function → Probability (0 to 1)

Example:
Raw calculation: 3.2  → Sigmoid → 0.96 (96% probability of "Yes")
Raw calculation: -1.5 → Sigmoid → 0.18 (18% probability of "Yes")
```

**Step 3:** If probability > 0.5, predict class "1" (Yes/Pass/Spam). If ≤ 0.5, predict class "0" (No/Fail/Not Spam).

> 💡 You don't calculate the Sigmoid yourself — Scikit-learn handles this internally. You just need to understand WHAT it's doing: converting a raw number into a "how confident am I" probability.

---

## 3. Logistic Regression in Scikit-learn

**Syntax:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    "hours_studied": [1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10],
    "passed":        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]   # 0=Fail, 1=Pass
}
df = pd.DataFrame(data)

X = df[["hours_studied"]]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)     # same .fit() pattern you already know!

predictions = model.predict(X_test)
print(predictions)   # array of 0s and 1s
```

> 💡 Notice the EXACT same pattern: `model = Algorithm()` → `model.fit()` → `model.predict()`. This is why Scikit-learn is so easy once you learn it once.

---

## 4. Predicting Classes vs Probabilities

This is unique to classification — you can get the FINAL answer, or the CONFIDENCE behind it.

**Syntax:**
```python
model.predict(X_test)            # gives the final class (0 or 1)
model.predict_proba(X_test)      # gives the probability for EACH class
```

```python
# Predict final class
prediction = model.predict([[6]])
print(prediction)   # [1] — predicts "Pass"

# Predict probability of each class
probability = model.predict_proba([[6]])
print(probability)
# [[0.15, 0.85]]
#   ↑     ↑
#  Fail   Pass
# 15% chance Fail, 85% chance Pass
```

> 💡 `predict_proba()` is extremely useful in real projects — "85% confident this will pass" is much more informative than just "Pass". Fraud detection, medical diagnosis, and risk scoring ALL rely on these probabilities, not just the final label.

---

## 5. The Decision Threshold

By default, Scikit-learn uses **0.5** as the cutoff — anything above 0.5 probability becomes class 1, anything below becomes class 0.

```
Probability of "Pass" = 0.85  →  0.85 > 0.5  →  Predicted: Pass (1)
Probability of "Pass" = 0.35  →  0.35 < 0.5  →  Predicted: Fail (0)
```

> 💡 **This threshold can be changed** for specific business needs. Example: for a disease detection model, you might want to flag anything above 0.3 probability as "at risk" — because missing a real case (false negative) is worse than a false alarm. This becomes important later with Imbalanced Datasets (upcoming topic).

---

## 6. Multi-class Classification

Logistic Regression isn't limited to just 2 classes — it works fine with 3+ categories too, using the same code.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    "petal_length": [1.4, 1.5, 4.5, 4.7, 5.5, 6.0, 1.3, 4.9, 5.8],
    "petal_width":  [0.2, 0.2, 1.5, 1.4, 2.1, 2.2, 0.3, 1.5, 2.3],
    "species":      ["setosa", "setosa", "versicolor", "versicolor",
                     "virginica", "virginica", "setosa", "versicolor", "virginica"]
}
df = pd.DataFrame(data)

X = df[["petal_length", "petal_width"]]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.predict(X_test))
print(model.predict_proba(X_test))   # now shows probability for EACH of the 3 classes
```

---

## 7. Complete Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    "hours_studied":  [1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9],
    "attendance_pct": [50, 55, 60, 65, 70, 75, 80, 85, 88, 90, 95, 98],
    "passed":         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[["hours_studied", "attendance_pct"]]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict for a new student
new_student = pd.DataFrame([[6, 82]], columns=["hours_studied", "attendance_pct"])
result = model.predict(new_student)[0]
confidence = model.predict_proba(new_student)[0]

print(f"Prediction: {'Pass' if result == 1 else 'Fail'}")
print(f"Confidence: {confidence[1]*100:.1f}% chance of passing")

# Check train/test scores — same overfitting check as before!
print(f"Train Accuracy: {model.score(X_train, y_train):.2f}")
print(f"Test Accuracy:  {model.score(X_test, y_test):.2f}")
```

> 💡 Note: `.score()` on a classification model returns **Accuracy**, not R² (that was only for regression). We'll cover Accuracy and other classification metrics properly in the next topic.

---

## 8. Logistic Regression vs Linear Regression

| | Linear Regression | Logistic Regression |
|--|--------------------|-----------------------|
| Predicts | A number | A category |
| Output range | Any number | Probability 0-1, then a class |
| `.score()` returns | R² | Accuracy |
| Use case | Salary, price, temperature | Pass/Fail, Spam/Not Spam, Disease/Healthy |
| Import from | `sklearn.linear_model` | `sklearn.linear_model` |

---

## 9. Quick Revision

| What you want | Code |
|--------------|------|
| Import | `from sklearn.linear_model import LogisticRegression` |
| Create & train | `model = LogisticRegression(); model.fit(X_train, y_train)` |
| Predict class | `model.predict(X_test)` |
| Predict probability | `model.predict_proba(X_test)` |
| Check accuracy | `model.score(X_test, y_test)` |
| Default threshold | 0.5 (probability above this → class 1) |

---

## ML Progress Tracker 📊

```
✅ ML Intro, Train/Test Split
✅ Feature Scaling & Encoding
✅ Linear Regression, Ridge & Lasso Regression
✅ Model Evaluation (Regression)
✅ Overfitting & Underfitting
✅ Feature Engineering
✅ Logistic Regression — your FIRST classification algorithm!
⏳ Model Evaluation (Classification) — next, needed to properly judge this model
⏳ Imbalanced Dataset Handling
⏳ Decision Tree → Random Forest → Gradient Boosting → XGBoost
⏳ KNN, Naive Bayes, SVM
⏳ Cross Validation, Hyperparameter Tuning, Feature Selection, Pipelines
⏳ K-Means Clustering, Save Model
```

---

> 📌 **Final Tip for MLOps:**
> - `predict_proba()` is used constantly in production — thresholds get tuned based on business needs (fraud detection, medical risk, etc.)
> - `.score()` alone (accuracy) can be misleading — you'll see why in the very next topic (Imbalanced Datasets)
> - Same `.fit()`/`.predict()` pattern you'll see for EVERY classification algorithm coming up (Decision Tree, Random Forest, KNN...)
> - Logistic Regression is often the FIRST model tried for any classification problem — simple, fast, and a good baseline 🚀

---

*Notes by Pratyush Guleria*
*GitHub: Pratyush-Guleria*