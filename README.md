# 📧 Spam Email Classification

<div align="center">

**A Machine Learning project for detecting Spam and Ham messages using TF-IDF and Logistic Regression.**

<br>

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas\&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn\&logoColor=white)
![TF-IDF](https://img.shields.io/badge/NLP-TF--IDF-green)

</div>

---

## 📌 Overview

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify messages into two categories:

* 🟢 **Ham** — Normal / legitimate messages
* 🔴 **Spam** — Unwanted or suspicious messages

### Machine Learning Pipeline

```text
Text Data
   ↓
Data Cleaning
   ↓
Label Encoding
   ↓
TF-IDF Feature Extraction
   ↓
Train/Test Split
   ↓
Logistic Regression
   ↓
Prediction
   ↓
Model Evaluation
```

---

## 📊 Dataset

The dataset contains **6,464 messages** with two columns:

| Column  | Description                      |
| ------- | -------------------------------- |
| `label` | Target category: `spam` or `ham` |
| `text`  | Message content                  |

### Example

| label | text                                   |
| ----- | -------------------------------------- |
| ham   | Hello, how are you?                    |
| spam  | Congratulations! You won a free prize! |

---

## 🛠️ Technologies Used

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 🤖 Scikit-learn
* 🧠 TF-IDF
* 📈 Matplotlib
* 📊 Seaborn
* 💾 Joblib
* 🌐 Flask *(optional for deployment)*

---

## 🔄 Machine Learning Workflow

### 1️⃣ Data Loading

```python
import pandas as pd

email = pd.read_csv("after data cleaning email.csv")
```

### 2️⃣ Data Cleaning

```python
email["text"] = email["text"].fillna("empty message")
```

### 3️⃣ Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded = encoder.fit_transform(email["label"])
```

### 4️⃣ TF-IDF Feature Engineering

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(email["text"])
```

### 5️⃣ Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    encoded,
    test_size=0.2,
    random_state=42,
    stratify=encoded
)
```

### 6️⃣ Model Training

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
```

### 7️⃣ Prediction

```python
predictions = model.predict(X_test)
```

Prediction probability:

```python
probabilities = model.predict_proba(X_test)
```

### 8️⃣ Model Evaluation

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
```

---

## 💾 Saving the Model

After training, save both the trained model and TF-IDF vectorizer:

```python
import joblib

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
```

### Saved Files

```text
spam_model.pkl
tfidf_vectorizer.pkl
```

This allows the trained model to be reused without training it again.

---

## 🧪 Example Prediction

```python
message = ["Congratulations! You have won a free prize"]

message_tfidf = vectorizer.transform(message)

prediction = model.predict(message_tfidf)

print(prediction)
```

---

## 🎯 Project Objectives

* ✅ Clean and preprocess text data
* ✅ Encode categorical labels
* ✅ Apply TF-IDF feature engineering
* ✅ Split data into training and testing sets
* ✅ Train a Logistic Regression classifier
* ✅ Predict Spam and Ham messages
* ✅ Calculate model accuracy
* ✅ Save the trained model
* ✅ Save the TF-IDF vectorizer
* ✅ Prepare the model for deployment

---

## 🧠 Concepts Demonstrated

* Data Cleaning
* Label Encoding
* Natural Language Processing
* TF-IDF
* Feature Engineering
* Train/Test Split
* Supervised Learning
* Logistic Regression
* Classification
* Model Training
* Prediction
* Prediction Probability
* Accuracy Score
* Model Saving with Joblib

---

## 🚀 Future Improvements

* 🌐 Build a Flask web application
* 📝 Add a message input form
* 📊 Display prediction probability
* 🧹 Improve text preprocessing
* 🤖 Compare different classification algorithms
* 📈 Add confusion matrix and classification report
* ☁️ Deploy the application online

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/rabia743/Email-Spam-detector-ML-model.git
cd Email-Spam-detector-ML-model
```

Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib
```

---

## 📌 Important

The **same TF-IDF vectorizer used during training must be reused during prediction**.

```text
Training:

Text
 ↓
TF-IDF Vectorizer
 ↓
Machine Learning Model
```

```text
Prediction:

New Text
 ↓
Same TF-IDF Vectorizer
 ↓
Saved Model
 ↓
Prediction
```

---

## 👩‍💻 Author

**Rabia Zafar**

Python Developer | Machine Learning Developer

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

**Built with Python, NLP & Machine Learning**

</div>
