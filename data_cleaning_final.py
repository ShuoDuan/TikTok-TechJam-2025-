import pandas as pd
import re

reviews = pd.read_csv(r"C:\Users\zhang\Desktop\review-Alabama_labeled.csv")
print("Reviews Info:")
print(reviews.info())
print(reviews.head())

reviews = reviews.drop_duplicates()

def clean_text(text):
    """Basic text cleaning: strip, collapse spaces."""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text

# Clean the text columns that exist in the Alabama dataset
for col in ["name", "text"]:
    if col in reviews.columns:
        reviews[col] = reviews[col].apply(clean_text)
reviews = reviews.dropna(subset=["text"])

if 'name' in reviews.columns:
    print(reviews['name'].value_counts())
print(reviews['rating'].unique())

reviews_cleaned = reviews.sort_values(by=['rating'], ascending=[True]).reset_index(drop=True)
print(reviews_cleaned.head())

reviews_cleaned.to_csv(r"C:\Users\zhang\Desktop\reviews_cleaned.csv", index=False)
print("\nCleaned data saved to 'C:\\Users\\zhang\\Desktop\\reviews_cleaned.csv'")

