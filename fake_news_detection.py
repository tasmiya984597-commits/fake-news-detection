import pandas as pd
import numpy as np
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load dataset
data = pd.read_csv("news.csv")

print(data.head())

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

data['text'] = data['text'].apply(clean_text)
X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

def predict_news(news):
    news = clean_text(news)
    news_vec = vectorizer.transform([news])
    result = model.predict(news_vec)
    
    if result[0] == 1:
        return "Real News"
    else:
        return "Fake News"

# Keep your sample news and run:
sample_news = """
BREAKING: Secret documents leaked from the deep state expose a massive cover-up by crooked 
politicians! Mainstream media is completely silent while patriotic whistleblowers blow the 
lid off the largest fraud scheme in American history. Watch the video before the corrupt elites 
censor it everywhere!
"""

print("Prediction:", predict_news(sample_news))