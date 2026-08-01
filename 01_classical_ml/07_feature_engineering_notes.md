# Machine Learning — Feature Engineering 🛠️

> **Author:** Pratyush Guleria
> **GitHub:** Pratyush-Guleria
> **Topic:** Creating better features from raw data to improve model performance

---

## Table of Contents
1. [What is Feature Engineering?](#1-what-is-feature-engineering)
2. [Why It Matters More Than the Algorithm](#2-why-it-matters-more-than-the-algorithm)
3. [Creating New Features from Existing Ones](#3-creating-new-features-from-existing-ones)
4. [Extracting Features from Dates](#4-extracting-features-from-dates)
5. [Binning — Converting Numbers to Categories](#5-binning--converting-numbers-to-categories)
6. [Combining Features](#6-combining-features)
7. [Polynomial Features](#7-polynomial-features)
8. [Dropping Useless Features](#8-dropping-useless-features)
9. [Quick Revision](#9-quick-revision)

---

## 1. What is Feature Engineering?

**Feature Engineering** = using domain knowledge and creativity to create NEW columns from your existing data — columns that help the model find patterns more easily.

> 💡 This is not a single Scikit-learn function — it's mostly **Pandas skills you already have** (`apply()`, `map()`, arithmetic on columns), applied with the goal of helping the model learn better.

**Simple example:**
```python
# Raw data
"date_of_birth": "2005-03-15"

# Engineered feature (much more useful to a model)
"age": 20
```

---

## 2. Why It Matters More Than the Algorithm

There's a well-known saying in ML: **"Better data beats a better algorithm."**

```
Bad features + best algorithm  = mediocre results
Great features + simple algorithm = often BETTER results
```

A Linear Regression model with well-engineered features can outperform a complex Random Forest with raw, unprocessed features. This is why real-world data scientists spend MORE time on feature engineering than on choosing algorithms.

---

## 3. Creating New Features from Existing Ones

```python
import pandas as pd

data = {
    "income": [50000, 80000, 30000, 120000],
    "expenses": [30000, 40000, 25000, 60000]
}
df = pd.DataFrame(data)

# New feature: how much they save
df["savings"] = df["income"] - df["expenses"]

# New feature: savings as a ratio (often more useful than raw numbers)
df["savings_ratio"] = df["savings"] / df["income"]

print(df)
```
```
   income  expenses  savings  savings_ratio
0   50000     30000    20000           0.40
1   80000     40000    40000           0.50
2   30000     25000     5000           0.17
3  120000     60000    60000           0.50
```

> 💡 `savings_ratio` is often MORE predictive than raw `savings` because it accounts for scale — someone saving ₹20,000 out of ₹50,000 is different from someone saving ₹20,000 out of ₹500,000.

---

## 4. Extracting Features from Dates

Dates by themselves aren't very useful to ML models — but the PARTS of a date often are.

**Syntax:**
```python
df["date_column"] = pd.to_datetime(df["date_column"])
df["year"] = df["date_column"].dt.year
df["month"] = df["date_column"].dt.month
df["day_of_week"] = df["date_column"].dt.dayofweek
```

```python
import pandas as pd

data = {
    "purchase_date": ["2024-01-15", "2024-06-20", "2024-12-25", "2024-03-10"]
}
df = pd.DataFrame(data)

# Convert to actual datetime type first
df["purchase_date"] = pd.to_datetime(df["purchase_date"])

# Extract useful parts
df["year"] = df["purchase_date"].dt.year
df["month"] = df["purchase_date"].dt.month
df["day_of_week"] = df["purchase_date"].dt.dayofweek   # 0=Monday, 6=Sunday
df["is_weekend"] = df["day_of_week"].isin([5, 6])       # True if Sat/Sun

print(df)
```

> 💡 **MLOps use case:** "Is this transaction on a weekend?" or "What month was this sale?" are often much more predictive for a model than the raw date string.

---

## 5. Binning — Converting Numbers to Categories

**Binning** groups continuous numbers into categories — sometimes this helps models find patterns more easily than raw numbers.

**Syntax:**
```python
pd.cut(df["column"], bins=[...], labels=[...])
```

```python
import pandas as pd

data = {"age": [15, 22, 35, 45, 60, 70, 8]}
df = pd.DataFrame(data)

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 19, 35, 60, 100],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"]
)

print(df)
```
```
   age    age_group
0   15         Teen
1   22  Young Adult
2   35  Young Adult
3   45        Adult
4   60        Adult
5   70       Senior
6    8        Child
```

> 💡 After binning, you'd typically One-Hot Encode `age_group` (from your earlier notes) before feeding it to a model.

---

## 6. Combining Features

Sometimes two separate columns are more useful together than apart.

```python
import pandas as pd

data = {
    "length": [10, 15, 20],
    "width": [5, 8, 10]
}
df = pd.DataFrame(data)

# Combine into a more meaningful feature
df["area"] = df["length"] * df["width"]

print(df)
```

```python
# Real example — combining city and state into one location feature
data = {
    "city": ["Delhi", "Mumbai", "Shimla"],
    "state": ["Delhi", "Maharashtra", "Himachal Pradesh"]
}
df = pd.DataFrame(data)

df["location"] = df["city"] + ", " + df["state"]
print(df)
```

---

## 7. Polynomial Features

Sometimes the relationship between a feature and the target isn't a straight line — it's curved. **Polynomial features** create powers of existing features (squared, cubed) to help Linear models capture curves.

**Syntax:**
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[["x"]])

print(X_poly)
# [[1. 1.]
#  [2. 4.]     ← original x, and x²
#  [3. 9.]
#  [4. 16.]
#  [5. 25.]]
```

> 💡 This lets a simple Linear Regression model fit curved relationships — the model is still "linear" mathematically, but the added `x²` feature lets it bend.

---

## 8. Dropping Useless Features

Not all engineering is about ADDING features — sometimes it's about REMOVING ones that don't help (or actively hurt) the model.

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Pratyush", "Rahul"],       # names don't predict anything numerical
    "id": [101, 102],                     # IDs are usually meaningless for prediction
    "age": [17, 19],
    "salary": [45000, 38000]
})

# Drop columns that don't help the model
df_clean = df.drop(columns=["name", "id"])
print(df_clean)
```

> 💡 IDs, names, and other purely identifying columns almost never help a model — they should usually be dropped before training (but keep them aside if you need to identify rows later, e.g., for a final report).

---

## 9. Quick Revision

| What you want | Code |
|--------------|------|
| Create ratio feature | `df["ratio"] = df["a"] / df["b"]` |
| Convert to datetime | `pd.to_datetime(df["col"])` |
| Extract year/month | `df["col"].dt.year`, `.dt.month` |
| Extract day of week | `df["col"].dt.dayofweek` |
| Bin numbers into categories | `pd.cut(df["col"], bins=[...], labels=[...])` |
| Combine columns | `df["new"] = df["a"] + df["b"]` |
| Create polynomial features | `PolynomialFeatures(degree=2).fit_transform(X)` |
| Drop useless columns | `df.drop(columns=["col1", "col2"])` |

---

## ML Progress Tracker 📊

```
✅ ML Intro, Train/Test Split
✅ Feature Scaling & Encoding
✅ Linear Regression
✅ Model Evaluation (Regression)
✅ Overfitting & Underfitting
✅ Ridge & Lasso Regression
✅ Feature Engineering — you can now improve ANY dataset before training
⏳ Logistic Regression — next (first classification algorithm!)
⏳ Model Evaluation (Classification)
... (11 more topics on the roadmap)
```

---

> 📌 **Final Tip for MLOps:**
> - Feature Engineering is often where the REAL skill difference shows between junior and senior data scientists
> - Date-based features (day of week, month, is_weekend) are extremely common in real business datasets — sales, transactions, bookings
> - Always drop ID/name columns before training — but keep a copy of the original DataFrame with them for final reports
> - This is also where your domain knowledge matters — knowing the PROBLEM helps you invent the right features 🚀

---

*Notes by Pratyush Guleria*
*GitHub: Pratyush-Guleria*