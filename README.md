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
<div align="center"><!-- 🎬 ANIMATED EMAIL BANNER --><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&duration=2800&pause=800&color=00C2FF&center=true&vCenter=true&width=700&lines=%F0%9F%93%A7+Spam+Email+Detector;%F0%9F%A4%96+Powered+by+Machine+Learning;%F0%9F%A7%A0+TF-IDF+%2B+Logistic+Regression;%E2%9C%89%EF%B8%8F+Spam+or+Ham%3F+Let's+Find+Out!" alt="Typing SVG" />



<!-- 🎨 ANIMATED EMAIL SVG --><svg width="260" height="200" viewBox="0 0 260 200" xmlns="http://www.w3.org/2000/svg"> <defs> <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%"> <stop offset="0%" style="stop-color:#00C2FF;stop-opacity:1" /> <stop offset="100%" style="stop-color:#8A2BE2;stop-opacity:1" /> </linearGradient> <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="0%"> <stop offset="0%" style="stop-color:#ffffff;stop-opacity:1" /> <stop offset="100%" style="stop-color:#e0f7ff;stop-opacity:1" /> </linearGradient> <filter id="glow"> <feGaussianBlur stdDeviation="3" result="coloredBlur"/> <feMerge> <feMergeNode in="coloredBlur"/> <feMergeNode in="SourceGraphic"/> </feMerge> </filter> </defs> <!-- Floating Email Envelope --> <g filter="url(#glow)"> <g> <animateTransform attributeName="transform" type="translate" values="0,0; 0,-12; 0,0" dur="3s" repeatCount="indefinite"/> <!-- Envelope body --> <rect x="40" y="70" width="180" height="110" rx="10" fill="url(#grad2)" stroke="url(#grad1)" stroke-width="3"/> <!-- Envelope flap --> <polygon points="40,70 130,130 220,70" fill="url(#grad1)" opacity="0.9"> <animate attributeName="opacity" values="0.9;0.5;0.9" dur="2s" repeatCount="indefinite"/> </polygon> <!-- Letter sliding out --> <g> <animateTransform attributeName="transform" type="translate" values="0,0; 0,-25; 0,0" dur="3s" repeatCount="indefinite"/> <rect x="65" y="40" width="130" height="80" rx="5" fill="#ffffff" stroke="#00C2FF" stroke-width="2"/> <line x1="78" y1="60" x2="182" y2="60" stroke="#00C2FF" stroke-width="2" stroke-linecap="round"/> <line x1="78" y1="72" x2="160" y2="72" stroke="#8A2BE2" stroke-width="2" stroke-linecap="round"/> <line x1="78" y1="84" x2="175" y2="84" stroke="#00C2FF" stroke-width="2" stroke-linecap="round"/> <line x1="78" y1="96" x2="140" y2="96" stroke="#8A2BE2" stroke-width="2" stroke-linecap="round"/> </g> </g> </g> <!-- Spam indicator badge --> <g> <animateTransform attributeName="transform" type="scale" values="1;1.15;1" dur="1.5s" repeatCount="indefinite" additive="sum"/> <circle cx="220" cy="60" r="18" fill="#FF3B3B" opacity="0.95"/> <text x="220" y="66" font-size="18" text-anchor="middle" fill="white" font-weight="bold">!</text> </g> <!-- Ham indicator badge --> <g> <animateTransform attributeName="transform" type="scale" values="1;1.15;1" dur="1.5s" repeatCount="indefinite" additive="sum"/> <circle cx="40" cy="60" r="18" fill="#22C55E" opacity="0.95"/> <text x="40" y="66" font-size="16" text-anchor="middle" fill="white" font-weight="bold">✓</text> </g> </svg>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=3000&pause=1000&color=8A2BE2&center=true&vCenter=true&width=600&lines=%F0%9F%94%B4+Spam+Detected...;%F0%9F%9F%A2+Ham+Message...;%F0%9F%A4%96+AI+is+Analyzing...;%E2%9C%85+Classification+Complete!" alt="Status Animation" /></div>


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
