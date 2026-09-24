==================================================
PROJECT: Fake News Detection Using Machine Learning
==================================================

AUTHOR: Tasmiya
DATE: Jan 2026
PYTHON VERSION: 3.x

--------------------------------------------------
DESCRIPTION
--------------------------------------------------
An end-to-end Machine Learning project that classifies news 
articles as REAL or FAKE using Natural Language Processing (NLP) 
techniques and a Logistic Regression classifier.

The system processes textual data, extracts linguistic features 
using Term Frequency-Inverse Document Frequency (TF-IDF), and 
predicts the authenticity of given news content.

--------------------------------------------------
KEY FEATURES
--------------------------------------------------
- Text Preprocessing: Cleans raw article text for NLP modeling.
- Feature Extraction: Uses TF-IDF Vectorizer to extract numerical features.
- Classification: Employs Logistic Regression for binary prediction.
- Evaluation: Measures performance using Accuracy, Precision, Recall, 
  and Confusion Matrix.

--------------------------------------------------
TECH STACK & DEPENDENCIES
--------------------------------------------------
- Python 3.x
- pandas
- numpy
- scikit-learn

Installation command:
pip install pandas numpy scikit-learn

--------------------------------------------------
DATASET
--------------------------------------------------
- Dataset file: news.csv
- Contains labeled records with article titles, bodies, and labels (REAL / FAKE).
- Note: Make sure news.csv is placed in the project folder alongside the script.

--------------------------------------------------
HOW TO RUN
--------------------------------------------------
1. Clone the repository:
   git clone https://github.com/tasmiya984597/fake-news-detection.git

2. Open the directory:
   cd fake-news-detection

3. Ensure news.csv is present in this directory.

4. Run the model script:
   python fake_news_detection.py
