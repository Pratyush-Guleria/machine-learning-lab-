# Machine Learning — Imbalanced Dataset Handling ⚖️

> **Author:** Pratyush Guleria
> **GitHub:** Pratyush-Guleria
> **Topic:** Handling datasets where one class massively outnumbers another

---

## Table of Contents
1. [The Problem — Quick Recap](#1-the-problem--quick-recap)
2. [How to Detect Class Imbalance](#2-how-to-detect-class-imbalance)
3. [Solution 1 — class_weight Parameter](#3-solution-1--class_weight-parameter)
4. [Solution 2 — SMOTE (Oversampling)](#4-solution-2--smote-oversampling)
5. [Solution 3 — Undersampling](#5-solution-3--undersampling)
6. [Balanced Accuracy](#6-balanced-accuracy)
7. [Which Solution to Use When](#7-which-solution-to-use-when)
8. [Complete Example — Before and After](#8-complete-example--before-and-after)
9. [Quick Revision](#9-quick-revision)

---

## 1. The Problem — Quick Recap

You saw this in the previous topic: if 990 out of 1000 samples are "Normal" and only 10 are "Fraud", a model can get 99% accuracy by just predicting "Normal" every single time — while catching ZERO actual fraud cases.

```
This happens because the model is "lazy" — since 99% of training
examples are "Normal", predicting "Normal" always minimizes error
during training. The model never gets punished enough for missing
the rare class.
```

**This is called Class Imbalance**, and it's extremely common in real-world data: fraud detection, disease diagnosis, defect detection, churn prediction — the "interesting" class is almost always the rare one.

---

## 2. How to Detect Class Imbalance

**Syntax:**
```python
df["target_column"].value_counts()
```

```python
import pandas as pd

data = {
    "transaction_id": range(1, 21),
    "is_fraud": [0]*18 + [1]*2   # 18 normal, 2 fraud
}
df = pd.DataFrame(data)

print(df["is_fraud"].value_counts())
```
```
0    18
1     2
Name: is_fraud, dtype: int64
```

```python
# See it as percentages — often clearer
print(df["is_fraud"].value_counts(normalize=True) * 100)
```
```
0    90.0
1    10.0
```

> 💡 **Rule of thumb:** If one class is less than ~20-30% of the data, it's worth considering these techniques. Severe imbalance (like 99/1) almost always needs handling.

---

## 3. Solution 1 — `class_weight` Parameter

The simplest fix — tell the model "pay MORE attention to mistakes on the rare class" by weighting it higher during training. No new data is created.

**Syntax:**
```python
model = LogisticRegression(class_weight="balanced")
```

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = {
    "amount": [100, 150, 200, 5000, 120, 180, 90, 6000, 110, 160],
    "is_fraud": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

X = df[["amount"]]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Without class_weight — model likely ignores the rare class
model_normal = LogisticRegression()
model_normal.fit(X_train, y_train)

# WITH class_weight="balanced" — model pays more attention to the rare class
model_balanced = LogisticRegression(class_weight="balanced")
model_balanced.fit(X_train, y_train)

print("Normal model:")
print(classification_report(y_test, model_normal.predict(X_test)))

print("Balanced model:")
print(classification_report(y_test, model_balanced.predict(X_test)))
```

> 💡 **This is the EASIEST fix — try this FIRST.** Just one parameter change, works with Logistic Regression, Decision Tree, Random Forest, SVM — most Scikit-learn classifiers support `class_weight="balanced"`.

---

## 4. Solution 2 — SMOTE (Oversampling)

**SMOTE** (Synthetic Minority Oversampling Technique) creates NEW, synthetic examples of the minority (rare) class — rather than just duplicating existing ones, it generates realistic new data points based on existing minority samples.

**Installation:**
```bash
pip install imbalanced-learn
```

**Syntax:**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

data = {
    "amount": [100, 150, 200, 5000, 120, 180, 90, 6000, 110, 160, 130, 170],
    "hour": [10, 14, 9, 2, 11, 15, 13, 3, 12, 16, 10, 14],
    "is_fraud": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
}
df = pd.DataFrame(data)

X = df[["amount", "hour"]]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Before SMOTE:", y_train.value_counts().to_dict())

smote = SMOTE(random_state=42, k_neighbors=1)   # k_neighbors small for tiny demo data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE:", y_train_resampled.value_counts().to_dict())

# Now train on the BALANCED data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)
```

> ⚠️ **Critical rule:** Only apply SMOTE to the TRAINING data, NEVER to the test data. The test set must represent real-world conditions (imbalanced) to give you an honest evaluation.

> 💡 SMOTE is more powerful than `class_weight` but adds complexity and an extra library. Use it when `class_weight` alone doesn't give good enough results.

---

## 5. Solution 3 — Undersampling

The opposite approach — instead of creating more minority samples, you REMOVE some majority class samples to balance things out.

```python
import pandas as pd

data = {
    "amount": [100, 150, 200, 5000, 120, 180, 90, 6000, 110, 160],
    "is_fraud": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# Separate the classes
fraud = df[df["is_fraud"] == 1]
normal = df[df["is_fraud"] == 0]

# Randomly sample from majority class to match minority class size
normal_undersampled = normal.sample(n=len(fraud), random_state=42)

# Combine back together
balanced_df = pd.concat([fraud, normal_undersampled])
print(balanced_df["is_fraud"].value_counts())
```

> ⚠️ **Downside:** You're throwing away real data — this can hurt performance if you don't have much data to begin with. Generally, SMOTE (adding synthetic data) is preferred over undersampling (removing real data) unless you have a LOT of majority class data to spare.

---

## 6. Balanced Accuracy

Regular accuracy treats both classes equally in its calculation — which is exactly the problem. **Balanced Accuracy** averages the recall of EACH class, giving equal importance regardless of how many samples each class has.

**Syntax:**
```python
from sklearn.metrics import balanced_accuracy_score

bal_acc = balanced_accuracy_score(y_test, y_pred)
```

```python
from sklearn.metrics import balanced_accuracy_score, accuracy_score

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print(f"Balanced Accuracy: {bal_acc:.2f}")
# On imbalanced data, these two numbers can be VERY different
# Balanced Accuracy tells the true story
```

> 💡 On an imbalanced dataset, always report Balanced Accuracy (or F1 Score) alongside regular Accuracy — never Accuracy alone.

---

## 7. Which Solution to Use When

```
Start here: Check df["target"].value_counts() on EVERY new classification dataset

Mild imbalance (70/30, 80/20):
    → class_weight="balanced" is usually enough

Severe imbalance (95/5, 99/1):
    → class_weight="balanced" first, then try SMOTE if results are still poor

Very large dataset with severe imbalance:
    → Undersampling is viable (you have data to spare)

Always:
    → Use Precision, Recall, F1, Balanced Accuracy — NEVER just Accuracy
    → Only balance the TRAINING data — leave test data as-is
```

---

## 8. Complete Example — Before and After

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score

data = {
    "amount":  [100, 150, 200, 5000, 120, 180, 90, 6000, 110, 160, 5500, 95],
    "is_fraud": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
}
df = pd.DataFrame(data)

print("Class distribution:")
print(df["is_fraud"].value_counts())

X = df[["amount"]]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# WITHOUT handling imbalance
model_before = LogisticRegression()
model_before.fit(X_train, y_train)
print("\n--- BEFORE (no imbalance handling) ---")
print(classification_report(y_test, model_before.predict(X_test), zero_division=0))

# WITH class_weight
model_after = LogisticRegression(class_weight="balanced")
model_after.fit(X_train, y_train)
print("--- AFTER (class_weight='balanced') ---")
print(classification_report(y_test, model_after.predict(X_test), zero_division=0))
```

> 💡 `stratify=y` in `train_test_split` ensures BOTH train and test sets keep the same class proportions as the original data — very useful for imbalanced datasets.

---

## 9. Quick Revision

| What you want | Code |
|--------------|------|
| Check class balance | `df["target"].value_counts()` |
| Check as percentage | `df["target"].value_counts(normalize=True)` |
| Simplest fix | `LogisticRegression(class_weight="balanced")` |
| Create synthetic samples | `SMOTE().fit_resample(X_train, y_train)` |
| Remove majority samples | `df.sample(n=minority_count)` |
| Fair evaluation metric | `balanced_accuracy_score(y_test, y_pred)` |
| Keep class proportions in split | `train_test_split(..., stratify=y)` |

---

## ML Progress Tracker 📊

```
✅ ML Intro, Train/Test Split
✅ Feature Scaling & Encoding
✅ Linear, Ridge, Lasso Regression + Regression Evaluation
✅ Overfitting & Underfitting
✅ Feature Engineering
✅ Logistic Regression
✅ Model Evaluation (Classification)
✅ Imbalanced Dataset Handling — you now know how to handle real-world skewed data
⏳ Decision Tree — next (tree-based algorithms begin here)
⏳ Random Forest → Gradient Boosting → XGBoost
⏳ KNN, Naive Bayes, SVM
⏳ Cross Validation, Hyperparameter Tuning, Feature Selection, Pipelines
⏳ K-Means Clustering, Save Model
```

---

> 📌 **Final Tip for MLOps:**
> - `df["target"].value_counts()` should be a habit — run it on EVERY new classification dataset before doing anything else
> - `stratify=y` in train_test_split is a small change with big impact on imbalanced data
> - Fraud detection, medical diagnosis, defect detection — nearly all real-world classification problems in industry are imbalanced to some degree
> - In interviews, being asked "how would you handle imbalanced data" is extremely common — you now have a complete, structured answer 🚀

---

*Notes by Pratyush Guleria*
*GitHub: Pratyush-Guleria*