# final--year--project--Anu-Bhukya
final year project-Email phishing  detection via text and Header Features .
# **Phishing-Email Detection with Supervised Machine Learning** 

---

# Overview
This notebook implements a phishing email detection pipeline using various machine learning models. It processes a labeled dataset from Kaggle, applies text cleaning and TF-IDF transformation, and evaluates multiple classifiers including Logistic Regression, Naive Bayes, SVM, Random Forest, and Gradient Boosting.

---

# Dataset
- **Path/URL:**https://www.kaggle.com/datasets/subhajournal/phishingemails
- **Target column:** label1
- **Feature column(s):** Email Text
- **Feature count/types:** Not specified in notebook

---

# Features & Preprocessing
- Lowercasing, URL removal, punctuation stripping, digit removal
- Stopword removal and stemming using NLTK
- TF-IDF Vectorization


---

# Models
- LogisticRegression
- RandomForestClassifier
- LinearSVC
- MultinomialNB
- GradientBoostingClassifier


---

# Evaluation
- **Metrics:** accuracy_score, precision, recall, f1-score
- **Visualizations:** confusion matrix
- **Tuning:** None

---

# Environment & Requirements
- **Libraries:** LinearSVC, LogisticRegression, MultinomialNB, Pipeline, PorterStemmer, RandomForestClassifier, SelectKBest, TfidfVectorizer, WordCloud, chi2, confusion_matrix, matplotlib, nltk, os, re, stopwords, train_test_split, warnings
- **Install example:**
  ```bash
  pip install LinearSVC LogisticRegression MultinomialNB Pipeline PorterStemmer RandomForestClassifier SelectKBest TfidfVectorizer WordCloud chi2 confusion_matrix matplotlib nltk stopwords train_test_split
  ```
