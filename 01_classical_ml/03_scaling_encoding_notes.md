# Machine Learning — Feature Scaling & Encoding 🔧

> **Author:** Pratyush Guleria | MLOps Learning Journey 🚀
> **GitHub:** Pratyush-Guleria
> **Topic:** Preparing data for ML models — scaling numbers and encoding text

---

## Table of Contents
1. [Why Preprocessing Matters](#1-why-preprocessing-matters)
2. [Feature Scaling — Why?](#2-feature-scaling--why)
3. [StandardScaler](#3-standardscaler)
4. [MinMaxScaler](#4-minmaxscaler)
5. [StandardScaler vs MinMaxScaler](#5-standardscaler-vs-minmaxscaler)
6. [Why We Fit on Train, Transform on Both](#6-why-we-fit-on-train-transform-on-both)
7. [Label Encoding](#7-label-encoding)
8. [One-Hot Encoding](#8-one-hot-encoding)
9. [Label Encoding vs One-Hot Encoding](#9-label-encoding-vs-one-hot-encoding)
10. [Quick Revision](#10-quick-revision)

---

## 1. Why Preprocessing Matters

ML models only understand **numbers** — and they work best when those numbers are on a similar scale. Real data has 2 problems:

1. **Different scales** — `salary` (30000-90000) vs `experience` (1-10) — huge scale difference confuses some models
2. **Text data** — `department` = "CSE", "ECE" — models can't use text directly

This topic fixes both problems.

---

## 2. Feature Scaling — Why?

**Problem:** If one feature is 0-10 and another is 0-100000, some algorithms think the bigger-number feature is more "important" — even if it's not.

**Example:**
```
experience: 1 to 10
salary:     25000 to 95000
```

Without scaling, `salary` dominates just because its numbers are bigger — not because it's actually more important.

**Solution:** Scale all features to a similar range before training.

> 💡 Not all algorithms need scaling. Linear Regression, Logistic Regression, KNN — YES need it. Decision Tree, Random Forest — NO, they don't care about scale.

---

## 3. StandardScaler

**StandardScaler** transforms data so it has **mean = 0** and **standard deviation = 1** — this is called "standardization" or "Z-score normalization".

**Formula (intuition only, no need to calculate manually):**
```
scaled_value = (value - mean) / standard_deviation
```

**Syntax:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    "experience": [1, 2, 3, 5, 8, 10],
    "salary": [25000, 32000, 40000, 58000, 78000, 95000]
}
df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

print(scaled_data)
# Both columns now have similar scale — roughly -2 to +2 range
```

---

## 4. MinMaxScaler

**MinMaxScaler** squeezes all values into a fixed range — usually 0 to 1.

**Formula (intuition only):**
```
scaled_value = (value - min) / (max - min)
```

**Syntax:**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame({
    "experience": [1, 2, 3, 5, 8, 10],
    "salary": [25000, 32000, 40000, 58000, 78000, 95000]
})

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

print(scaled_data)
# All values now between 0 and 1
```

---

## 5. StandardScaler vs MinMaxScaler

| | StandardScaler | MinMaxScaler |
|--|----------------|---------------|
| Range | No fixed range (~-3 to +3) | Fixed — 0 to 1 |
| Best for | Data with outliers | Data without extreme outliers |
| Common with | Linear/Logistic Regression | Neural Networks, image data |

> 💡 **When unsure, StandardScaler is the safer default** — it handles outliers better.

---

## 6. Why We Fit on Train, Transform on Both

> ⚠️ **Critical rule:** Fit the scaler ONLY on training data, then use it to transform both train and test.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

# fit_transform on TRAIN only — scaler LEARNS mean/std from training data
X_train_scaled = scaler.fit_transform(X_train)

# transform (NOT fit_transform) on TEST — uses the SAME mean/std from training
X_test_scaled = scaler.transform(X_test)
```

> 💡 **Why this matters:** If you fit the scaler on test data too, information from the "unseen" test set leaks into training — called "data leakage". This gives falsely good results that won't hold up in the real world.

---

## 7. Label Encoding

**Label Encoding** converts text categories into numbers — 0, 1, 2, 3...

**Use when:** Categories have a natural ORDER (e.g., "Low", "Medium", "High").

**Syntax:**
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["column_encoded"] = encoder.fit_transform(df["column"])
```

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({
    "name": ["Pratyush", "Rahul", "Amit"],
    "level": ["Junior", "Senior", "Mid"]
})

encoder = LabelEncoder()
df["level_encoded"] = encoder.fit_transform(df["level"])

print(df)
```
```
       name   level  level_encoded
0  Pratyush  Junior              1
1     Rahul  Senior              2
2      Amit     Mid              0
```

> ⚠️ **Problem with Label Encoding:** It assigns 0, 1, 2 — the model might think "2 is greater than 0" (Senior > Junior), which is only correct if there's a real order. For unordered categories like "CSE", "ECE", "IT" — this creates a FALSE relationship.

---

## 8. One-Hot Encoding

**One-Hot Encoding** creates a separate column for each category — with 1 (yes) or 0 (no). No false order created.

**Syntax:**
```python
pd.get_dummies(df, columns=["column_name"])
```

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Pratyush", "Rahul", "Amit", "Sneha"],
    "department": ["CSE", "ECE", "CSE", "IT"]
})

df_encoded = pd.get_dummies(df, columns=["department"])
print(df_encoded)
```
```
       name  department_CSE  department_ECE  department_IT
0  Pratyush            True           False          False
1     Rahul           False            True          False
2      Amit            True           False          False
3     Sneha           False           False           True
```

> 💡 Each department becomes its own True/False column — no false numeric relationship between categories.

### Using Scikit-learn's OneHotEncoder (alternative)

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[["department"]])
print(encoded)
```

> 💡 `pd.get_dummies()` is simpler and used more often in practice. `OneHotEncoder` is used inside Scikit-learn pipelines.

---

## 9. Label Encoding vs One-Hot Encoding

| | Label Encoding | One-Hot Encoding |
|--|-----------------|-------------------|
| Output | One column, numbers | Multiple True/False columns |
| Use when | Categories have order (Low/Med/High) | Categories have NO order (CSE/ECE/IT) |
| Risk | False order assumption | More columns created |
| Common for | Ordinal data | Nominal/categorical data |

> 💡 **Rule of thumb:** If you can't rank the categories (CSE isn't "more" than ECE), use One-Hot Encoding. If there's a clear order (Low < Medium < High), Label Encoding is fine.

---

## 10. Quick Revision

| What you want | Code |
|--------------|------|
| Standardize (mean=0, std=1) | `StandardScaler().fit_transform(X)` |
| Scale to 0-1 | `MinMaxScaler().fit_transform(X)` |
| Fit on train only | `scaler.fit_transform(X_train)` |
| Transform test | `scaler.transform(X_test)` |
| Label encode | `LabelEncoder().fit_transform(df["col"])` |
| One-hot encode | `pd.get_dummies(df, columns=["col"])` |

---

> 📌 **Final Tip for MLOps:**
> - `fit_transform` on train, `transform` on test — this rule prevents data leakage, one of the most common ML mistakes
> - `pd.get_dummies()` for categorical features before feeding to Linear/Logistic Regression
> - Decision Trees and Random Forests don't need scaling — but DO need encoding for text columns
> - Save your fitted scaler/encoder (using `joblib` or `pickle`) — you'll need the SAME one when the model goes to production 🚀

---

*Notes by Pratyush Guleria — MLOps Learning Journey 🚀*
*GitHub: Pratyush-Guleria*