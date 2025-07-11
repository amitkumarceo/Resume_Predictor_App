# 🧠 Resume Title Predictor Web App

A machine learning-powered web application that predicts job titles based on the content of resumes. Built with **Python**, **scikit-learn**, **TF-IDF vectorization**, **Logistic Regression**, and deployed using **Streamlit Cloud**.

---

## 🚀 Live Demo

🔗 [Click here to try the app](https://YOUR-DEPLOYED-STREAMLIT-URL)

---

## 📌 Project Overview

This project is designed to predict the most suitable **job title** based on the textual content of a resume. It processes resume text using **natural language processing (NLP)**, transforms it using **TF-IDF**, and predicts a job title using a **Logistic Regression** classifier.

---

## 🛠 Features

- 🔤 Text preprocessing (stopwords removal, punctuation stripping, HTML & URL cleaning)
- 🧹 Tokenization and normalization with `nltk`
- ✨ TF-IDF vectorization with bigrams
- 🤖 Multi-class classification using Logistic Regression
- 🎯 Label encoding and decoding of job titles
- 📊 Model evaluation (accuracy and classification report)
- 🌐 Interactive UI built with **Streamlit**
- ☁️ Deployed on **Streamlit Community Cloud**

---

## 📂 Dataset

- **Source**: [UpdatedResumeDataSet.csv](https://www.kaggle.com/datasets/saikiranrao/updated-resume-dataset)
- **Columns**:
  - `Resume`: Raw resume text
  - `Category`: Corresponding job title / category

---

## 🔍 Tech Stack

| Component        | Tool / Library       |
|------------------|----------------------|
| Programming Lang | Python               |
| Data Handling    | Pandas, NumPy        |
| NLP              | NLTK                 |
| Vectorization    | TfidfVectorizer      |
| Model            | Logistic Regression (scikit-learn) |
| Encoding         | LabelEncoder         |
| Visualization    | Matplotlib, Seaborn  |
| App UI           | Streamlit            |
| Deployment       | Streamlit Community Cloud |
| Model Storage    | joblib               |

---

## 🧪 Model Training Steps

1. **Data Preprocessing**
   - Lowercasing, removing HTML tags, URLs, punctuation, and stopwords
   - Tokenization using `nltk`

2. **Label Encoding**
   - Encoded job categories using `LabelEncoder`

3. **Text Vectorization**
   - Applied `TfidfVectorizer` (with bigrams and stopword removal)

4. **Model Training**
   - Trained a `LogisticRegression` classifier
   - Performed `GridSearchCV` for hyperparameter tuning

5. **Evaluation**
   - Accuracy score and `classification_report` printed
   - Confusion matrix plotted with Seaborn

6. **Model Saving**
   - Saved final model, vectorizer, and label encoder using `joblib`

---

## 📁 Files in This Repo

| File | Description |
|------|-------------|
| `app.py` | Streamlit app entry point |
| `job_model.pkl` | Trained logistic regression model |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer |
| `label_encoder.pkl` | LabelEncoder for job categories |
| `requirements.txt` | Python dependencies |
| `UpdatedResumeDataSet.csv` | Original dataset (optional) |
