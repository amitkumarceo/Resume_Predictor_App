#!/usr/bin/env python
# coding: utf-8

# ## STEP 0 - SETUP ENVIRONMENT

# #### Import NLTK and important datasets needed for TOKENIZATION

# In[1]:


import nltk
import sys
import subprocess

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


# #### IMPORT all libraries without error

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

print("All libraries imported successfully!!")


# ## PART 1 - Data Collection

# In[6]:


# load the csv file
df = pd.read_csv("UpdatedResumeDataSet.csv")

# Now will run basic dataset check
print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns)
print("\nMissing Values\n", df.isnull().sum())
print("\nUnique Categories:", df['Category'].nunique())
print("\nUnique List:\n", df['Category'].unique())


# ## PART 2 - Data Preprocessing (Text + Labels)

# In[7]:


import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[8]:


#set of english stopwords
stop_words = set(stopwords.words('english'))

# TEXT CLEANING function
def clean_text(text):
    text = text.lower() #lowercase
    text = re.sub(r'<[^>]+>', ' ', text) #remove html tags
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text) #remove links
    text = text.translate(str.maketrans('', '', string.punctuation)) #removes punct
    text = re.sub(r'\s+', ' ', text).strip() #removes whitespaces
    tokens = nltk.word_tokenize(text) #tokenizes text, very important for NLP
    tokens = [word for word in tokens if word not in stop_words] #loops and adds only words not stopwords
    return " ".join(tokens)


# Apply cleaning function to all resumes
df['Cleaned_Resume'] = df['Resume'].apply(clean_text)

# Show sample cleaned text
df[['Resume', 'Cleaned_Resume']].head(2)


# In[9]:


## ENCODE Job titles as Numbers for ML model
label_encoder = LabelEncoder()
df['Category_Code'] = label_encoder.fit_transform(df['Category'])

# Map encoded labels to their Category names
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"Label Mapping:\n", label_map)


# In[10]:


## SPLIT the data into training and testing sets

X = df['Cleaned_Resume']
y = df['Category_Code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ## Part 3 - FEATURE ENGINEERING(Text Vectorization)

# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
X_train_df.head()


# In[12]:


#get some features names
features = tfidf_vectorizer.get_feature_names_out()
print("Top 10 Feautres:", features[:10])


# ## PART 4 - MODEL BUILDING

# In[13]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize the model and train it
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)


# In[14]:


# Test the model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[15]:


# INSPECT Model Results - Job categories model struggled with

inaccurate_preds = pd.DataFrame({'True_Label': y_test, 'Predicted_Label': y_pred, 'Resume': X_test})
inaccurate_preds = inaccurate_preds[inaccurate_preds['True_Label'] != inaccurate_preds['Predicted_Label']]
print(inaccurate_preds.head())


# ## PART 5 - Hyperparameter Tuning to find best parameters

# In[16]:


from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2', 'l1'],
    'solver': ['liblinear', 'saga']
}

# Initialize the gridsearchcv with logistic regression model
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train_tfidf, y_train)

print("Best hyperparams:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)



# In[17]:


# Evaluating the best Model

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_tfidf)
accuracy_best = accuracy_score(y_test, y_pred_best)

print(f"Best model accuracy: {accuracy_best:.4f}")
print("\nBest model classification report:\n", classification_report(y_test, y_pred_best))


# ## PART 6 - Model Evaluation based on various Metrics

# In[18]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# confusion matrix for the best model
conf_matrix = confusion_matrix(y_test, y_pred_best)

#Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[19]:


# Print accuracy of the final model
print(f"Final Model Accuracy: {accuracy_best:.4f}")


# #### SAVE THE FINAL MODEL

# In[20]:


import joblib

joblib.dump(best_model, "job_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("model and vectorizer saved successfully!")


# ## PART 7 - Build the Web App for deployement

# In[27]:


import streamlit as st
import joblib

model = joblib.load("job_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit UI
st.title("Job title prediction from Resume")
st.write("Enter a resume text below and get the predicted job title.")

# Input box for entering resume text
resume_text = st.text_area("Resume Text")

# Prediction Button
if st.button("Predict"):
    if resume_text:
        resume_tfidf = tfidf_vectorizer.transform([resume_text])
        prediction = model.predict(resume_tfidf)
        predicted_title = label_encoder.inverse_transform(prediction)[0]
        st.write(f"Predicted Job Title best for you: {predicted_title}")
    else:
        st.write("Please enter text in the resume input box")



# In[ ]:




