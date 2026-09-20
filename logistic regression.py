import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import load_npz
# independed value
X = load_npz("X_tfidf.npz")
# depended value
y = pd.read_csv("y.csv")["label"].values
# load the SAME vectorizer used during training
vectorizer = joblib.load("vectorizer.pkl")
# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = LogisticRegression(max_iter=1000)
# Train
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")
# New input
print("\nEnter a new email message:")
new_email = input("Email text: ")

# Convert new text to features
new_email_vector = vectorizer.transform([new_email])

# Prediction probability
probability = model.predict_proba(new_email_vector)
spam_probability = probability[0][1] * 100
not_spam_probability = probability[0][0] * 100

print("\nSpam Probability:", spam_probability, "%")
print("Not Spam Probability:", not_spam_probability, "%")

# Final result
result = model.predict(new_email_vector)
if result[0] == 1:
    print("Result: Spam")
else:
    print("Result: Not Spam")

# joblib.dump(model, "logistic_model.pkl")
# print("Model saved: logistic_model.pkl")