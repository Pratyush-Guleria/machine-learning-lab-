# Machine Learning — Introduction & Train/Test Split 🤖

> **Author:** Pratyush Guleria 
> **GitHub:** Pratyush-Guleria
> **Topic:** ML fundamentals and preparing data for training

---

## Table of Contents
1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [Supervised vs Unsupervised Learning](#2-supervised-vs-unsupervised-learning)
3. [Regression vs Classification](#3-regression-vs-classification)
4. [What is Scikit-learn?](#4-what-is-scikit-learn)
5. [The ML Workflow](#5-the-ml-workflow)
6. [Train/Test Split](#6-traintest-split)
7. [Features (X) and Target (y)](#7-features-x-and-target-y)
8. [Quick Revision](#8-quick-revision)

---

## 1. What is Machine Learning?

**Machine Learning** is teaching a computer to find patterns in data and make predictions — without explicitly programming every rule.

**Traditional programming vs ML:**

```
Traditional Programming:
Rules + Data → Program → Output

Machine Learning:
Data + Output → Program → Rules (the model learns the rules itself)
```

**Simple example:**
- Traditional: You write `if hours_studied > 5: pass` (you define the rule)
- ML: You show the model 1000 students' hours studied AND whether they passed — the model figures out the pattern itself

---

## 2. Supervised vs Unsupervised Learning

### Supervised Learning

You give the model **input data + correct answers** — it learns to predict the answer for new data.

```
Input (Features)          Output (Label/Target)
Hours studied: 5     →    Pass
Hours studied: 1     →    Fail
Hours studied: 8     →    Pass
```

> The model learns: "more hours studied → more likely to pass" — then predicts for a NEW student.

**Examples:** Predicting house prices, spam detection, predicting if a student passes.

### Unsupervised Learning

You give the model **only input data — no answers**. It finds patterns/groups on its own.

```
Customer spending patterns (no labels given)
↓
Model groups them into: "High spenders", "Occasional buyers", "Bargain hunters"
```

**Examples:** Customer segmentation, grouping similar news articles.

| | Supervised | Unsupervised |
|--|-----------|---------------|
| Data has labels? | Yes | No |
| Goal | Predict a known outcome | Find hidden patterns |
| Example | Will this email be spam? | Group customers into segments |

> 💡 **Most beginner and internship-level ML work is Supervised Learning** — this is where we'll spend most of our time.

---

## 3. Regression vs Classification

Supervised Learning splits into two types based on WHAT you're predicting.

### Regression — predicting a NUMBER

```
Input: House size, location, rooms
Output: Price (₹45,00,000) — a continuous number
```

**Examples:** House price, salary prediction, temperature forecast.

### Classification — predicting a CATEGORY

```
Input: Email content
Output: "Spam" or "Not Spam" — a category
```

**Examples:** Pass/Fail, Spam/Not Spam, Disease/No Disease.

| | Regression | Classification |
|--|-----------|-----------------|
| Predicts | A number | A category/class |
| Example | Salary = ₹45,000 | Status = "Pass" |
| Common algorithms | Linear Regression | Logistic Regression, Decision Tree |

---

## 4. What is Scikit-learn?

**Scikit-learn** is the most popular Python library for classical Machine Learning — it has ready-made algorithms, no need to build them from scratch.

```python
from sklearn.linear_model import LinearRegression   # import an algorithm
```

> 💡 Scikit-learn handles the math internally — you focus on WHAT algorithm to use and HOW to prepare your data, not the formulas.

### Installation

```bash
pip install scikit-learn
```

---

## 5. The ML Workflow

Every ML project follows roughly the same steps:

```
1. Load Data           (Pandas — pd.read_csv)
2. Clean Data          (Pandas — handle missing values)
3. Explore Data        (Matplotlib/Seaborn — visualize)
4. Split Data          (Train/Test Split)
5. Choose Algorithm    (Scikit-learn)
6. Train Model         (model.fit())
7. Make Predictions    (model.predict())
8. Evaluate Model       (accuracy, error metrics)
```

> 💡 Notice steps 1-3 are exactly what you already know — NumPy, Pandas, Matplotlib, Seaborn were all preparation for this moment.

---

## 6. Train/Test Split

You NEVER train and test a model on the same data — the model needs to be tested on data it has never seen, to check if it truly "learned" or just "memorized".

**Analogy:** Studying from a textbook (training) vs taking a real exam with new questions (testing).

**Syntax:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
import pandas as pd
from sklearn.model_selection import train_test_split

data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]   # 0 = fail, 1 = pass
}
df = pd.DataFrame(data)

X = df[["hours_studied"]]   # Features (input)
y = df["passed"]            # Target (output)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)   # (8, 1) — 80% for training
print(X_test.shape)    # (2, 1) — 20% for testing
```

### Parameters explained

| Parameter | What it does |
|-----------|-------------|
| `test_size=0.2` | 20% of data goes to testing, 80% to training |
| `random_state=42` | Makes the split reproducible — same split every time you run it |

> 💡 `random_state=42` is exactly the same concept as `np.random.seed(42)` you learned in NumPy — makes results reproducible.

### Common split ratios

```python
test_size=0.2   # 80/20 split — most common
test_size=0.3   # 70/30 split — when you have less data
test_size=0.1   # 90/10 split — when you have lots of data
```

---

## 7. Features (X) and Target (y)

This naming convention is used in EVERY ML project — memorize this pattern.

| Symbol | Meaning | Also called |
|--------|---------|-------------|
| `X` | Input columns used to predict | Features, Independent variables |
| `y` | Column you want to predict | Target, Label, Dependent variable |

```python
import pandas as pd

df = pd.read_csv("students.csv")

# X = everything EXCEPT the target column
X = df[["hours_studied", "attendance", "previous_score"]]

# y = ONLY the target column
y = df["passed"]
```

> 💡 **Rule of thumb:** `X` is always a DataFrame (2D, even with one column — use double brackets `[[...]]`). `y` is always a Series (1D, single brackets `[...]`).

---

## 8. Quick Revision

| Concept | One Line |
|---------|---------|
| Machine Learning | Model learns patterns from data instead of explicit rules |
| Supervised | Data has labels — predict known outcome |
| Unsupervised | Data has no labels — find hidden patterns |
| Regression | Predict a number (price, salary) |
| Classification | Predict a category (pass/fail, spam/not spam) |
| Scikit-learn | Python library with ready ML algorithms |
| Train/Test Split | Split data so model is tested on unseen data |
| `X` | Features — input columns |
| `y` | Target — column to predict |
| `random_state` | Makes split reproducible (like `np.random.seed`) |

---

> 📌 **Final Tip for MLOps:**
> - Every single ML project starts with: load → clean → split → train → evaluate
> - `random_state=42` — use consistently across your entire project for reproducibility
> - Get comfortable with `X` and `y` naming — it's used everywhere in ML code, including job interviews
> - Regression vs Classification is the FIRST question to answer for any ML problem — it decides which algorithms you can use 🚀

---

*Notes by Pratyush Guleria
*GitHub: Pratyush-Guleria*