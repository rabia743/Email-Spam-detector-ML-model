import pandas as pd

email = pd.read_csv("email.csv")
print(email.head(5))
# print(email.to_string())

# data cleaning
print(email.isnull().sum())
# 2. Khaali columns drop karo (simple dropna)
email = email[["v1", "v2"]]
print(email.isnull().sum())
email = email.rename(columns={'v1': 'label', 'v2': 'text'})
# 4. Khaali rows drop karo
email = email.dropna()
print(email.duplicated().sum())
email.drop_duplicates(inplace=True)
print(email.duplicated().sum())
# CHECK ROWS AND COLUMNS
print(email.shape)
# CHECK DATA TYPES
print(email.dtypes)

# text cleaning
email["text"] = email["text"].str.strip().str.lower()
email["text"] = email["text"].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
email["text"] = email["text"].str.replace(r'\s+', ' ', regex=True).str.strip()

# email.to_csv("after data cleaning email.csv", index=False)
# print("Cleaned data saved successfully!")