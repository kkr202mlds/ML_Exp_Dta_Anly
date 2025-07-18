# ML_Exp_Dta_Anly
Machine Learning IMDB Exploratory Data Analysis Transformer
* **Transformer models** (likely for text)
* **Logistic Regression (LR)**
* **K-Nearest Neighbors Classifier (KNC)**
* **Decision Tree Classifier (DTC)**

**📊 Exploratory Data Analysis (EDA)** is the process of analyzing and visualizing a dataset to:
* **Understand its structure**
* **Summarize its key characteristics**
* **Identify patterns, trends, and anomalies**
* **Detect missing values or outliers**
* **Generate hypotheses for further analysis**

### 🧰 Typical EDA Tasks Include:
* **Descriptive statistics** (`mean`, `median`, `std`, `min`, `max`, etc.)
* **Data type checks** and **missing value handling**
* **Univariate analysis** (e.g., distributions of a single feature)
* **Bivariate/multivariate analysis** (e.g., correlation between features)
* **Data visualizations** like:

  * Histograms
  * Boxplots
  * Count plots
  * Scatter plots
  * Heatmaps

### ✅ Why EDA is Important?
* Helps **identify data quality issues**
* Informs **feature engineering**
* Guides **model selection and tuning**

---

## 📁 IMDb Movies EDA + ML Models

---

### 1. **Load the Dataset**

```python
import pandas as pd

df = pd.read_csv('IMDb_movies.csv')  # Replace with your actual file
df.head()
```

---

### 2. **EDA (Exploratory Data Analysis)**

```python
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Example plots
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['avg_vote'], bins=10)
plt.title('Average Vote Distribution')
plt.show()

sns.countplot(x='genre', data=df)
plt.xticks(rotation=90)
plt.title('Genre Count')
plt.show()
```

---

### 3. **Preprocessing**

Let’s predict whether a movie is **good (avg\_vote ≥ 7)** or **bad (avg\_vote < 7)**.

```python
df = df[df['avg_vote'].notnull()]  # Drop rows with null target
df['label'] = (df['avg_vote'] >= 7).astype(int)  # 1 = Good, 0 = Bad
```

#### Text Preprocessing (e.g., `description`)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

df['description'] = df['description'].fillna('')  # Fill missing descriptions
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_text = vectorizer.fit_transform(df['description'])

y = df['label']
```

---

### 4. **Train-Test Split**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)
```

---

### 5. **Machine Learning Models**

#### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))
```

#### K-Nearest Neighbors Classifier

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("KNN:\n", classification_report(y_test, y_pred_knn))
```

#### Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)

print("Decision Tree:\n", classification_report(y_test, y_pred_dtc))
```

---

### 6. **Transformer Model (Optional for Advanced Text Analysis)**

You can use **Hugging Face Transformers** for a deeper model like BERT:

```python
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
print(classifier("This movie was a thrilling masterpiece!"))
```

Or fine-tune BERT for classification, but that’s a more advanced setup. Let me know if you want a walkthrough.

---

### ✅ Summary

| Model               | Precision | Recall | F1-Score | Notes                      |
| ------------------- | --------- | ------ | -------- | -------------------------- |
| Logistic Regression | ...       | ...    | ...      | Good baseline              |
| KNN                 | ...       | ...    | ...      | Sensitive to vectorization |
| DTC                 | ...       | ...    | ...      | May overfit                |
| BERT/Transformer    | ...       | ...    | ...      | Powerful but slow          |

---
