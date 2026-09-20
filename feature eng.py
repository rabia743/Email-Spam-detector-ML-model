import pandas as pd
# from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import joblib

email = pd.read_csv("after data cleaning email.csv")

# encoder
encoder = LabelEncoder()
encoded = encoder.fit_transform(email["label"])
print("Label:")
print(encoded)
# print(email.columns.tolist())
print(email[email['text'].isnull()])
email['text'] = email['text'].fillna('empty message')
# print(email[email['text'].isnull()])
# TF/IDF
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
X = vectorizer.fit_transform(email["text"])
# kin kin words ko features banaya
print(vectorizer.get_feature_names_out())

# save the data in numeric form
save_npz("X_tfidf.npz", X)
pd.DataFrame({"label": encoded}).to_csv("y.csv", index=False)
print("Saved numeric features and labels")

# save vectorizer and encoder 
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Saved numeric features and labels")