import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO
import PyPDF2

# Initialize Streamlit app
st.title("Resume Classification and Interview Prediction")

# Load the dataset
@st.cache_data
def load_data():
    resume_data_set = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')
    return resume_data_set

resume_data_set = load_data()

# Function to clean resume text
def clean_resume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)  # Remove URLs
    resume_text = re.sub('RT|cc', ' ', resume_text)  # Remove RT and cc
    resume_text = re.sub('#\S+', '', resume_text)  # Remove hashtags
    resume_text = re.sub('@\S+', '  ', resume_text)  # Remove mentions
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)  # Remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)  # Remove non-ASCII characters
    resume_text = re.sub('\s+', ' ', resume_text)  # Remove extra whitespace
    return resume_text

resume_data_set['cleaned_resume'] = resume_data_set.Resume.apply(lambda x: clean_resume(x))

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Label Encoding the 'Category' column
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resume_data_set[i] = le.fit_transform(resume_data_set[i])

# Split the data into train and test sets
required_text = resume_data_set['cleaned_resume'].values
required_target = resume_data_set['Category'].values

tfidf_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
tfidf_vectorizer.fit(required_text)
word_features = tfidf_vectorizer.transform(required_text)

X_train, X_test, y_train, y_test = train_test_split(word_features, required_target, random_state=0, test_size=0.2)

# Model training using KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

# Displaying model performance
st.subheader("Model Performance")
st.write('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
st.write('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

# Uploading resume and selecting qualification
st.subheader("Predict Whether Your Resume Qualifies for an Interview")

uploaded_file = st.file_uploader("Upload your resume file", type=["pdf"])

# Job qualifications input
job_title = st.text_input("Job Title")
job_qualifications = st.text_area("Enter the job qualifications")
required_years_of_experience = st.number_input("Enter the required years of experience", min_value=0, step=1)

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

if st.button("Predict"):
    if uploaded_file is not None and job_qualifications:
        # Read and clean the uploaded resume file
        user_resume = read_pdf(uploaded_file)
        cleaned_user_resume = clean_resume(user_resume)

        # Transform the resume using the trained vectorizer
        user_resume_vector = tfidf_vectorizer.transform([cleaned_user_resume])

        # Predict the category
        prediction = clf.predict(user_resume_vector)
        predicted_category = le.inverse_transform(prediction)[0]

        # Display the prediction result
        st.write(f"Your resume is classified under the category: **{predicted_category}**")

        # Detailed qualification check
        if job_qualifications.lower() in cleaned_user_resume.lower():
            if "year" in cleaned_user_resume.lower():
                experience_match = False
                for word in cleaned_user_resume.lower().split():
                    if word.isdigit() and int(word) >= required_years_of_experience:
                        experience_match = True
                        break
                if experience_match:
                    st.success(f"Your resume is likely to qualify for the {job_title} role!")
                else:
                    st.warning(f"Your resume might not meet the {required_years_of_experience} years of experience requirement for the {job_title} role.")
            else:
                st.warning(f"Your resume does not clearly indicate your years of experience. Please update your resume to showcase your relevant experience.")
        else:
            st.warning(f"Your resume might not be suitable for the {job_title} role.")
            st.write("Here are some suggestions to improve your resume:")
            if "code" in cleaned_user_resume.lower() and "software" not in job_qualifications.lower():
                st.write("- Emphasize your programming and software development skills")
            if "data" in cleaned_user_resume.lower() and "data" not in job_qualifications.lower():
                st.write("- Highlight your data analysis and reporting skills")
            if "product" in cleaned_user_resume.lower() and "product" not in job_qualifications.lower():
                st.write("- Showcase your product management experience")
            if "market" in cleaned_user_resume.lower() and "market" not in job_qualifications.lower():
                st.write("- Emphasize your marketing and advertising skills")
            if "cyber" in cleaned_user_resume.lower() and "cyber" not in job_qualifications.lower():
                st.write("- Highlight your cybersecurity expertise")
            if "machine learning" in cleaned_user_resume.lower() and "machine learning" not in job_qualifications.lower():
                st.write("- Showcase your machine learning and AI skills")
            # Add more specific suggestions based on the job qualifications

        # Get user feedback on the prediction
        user_feedback = st.radio("Was the prediction accurate?", options=["Yes", "No"])

        if user_feedback == "No":
            st.write("Thank you for your feedback. We'll use this information to improve our model.")

            # Update the training data with the user's feedback
            if predicted_category != job_title:
                new_sample = {
                    "Resume": user_resume,
                    "cleaned_resume": cleaned_user_resume,
                    "Category": job_title
                }
                resume_data_set = resume_data_set.append(new_sample, ignore_index=True)

            # Retrain the model with the updated dataset
            required_text = resume_data_set['cleaned_resume'].values
            required_target = resume_data_set['Category'].values
            tfidf_vectorizer.fit(required_text)
            word_features = tfidf_vectorizer.transform(required_text)
            X_train, X_test, y_train, y_test = train_test_split(word_features, required_target, random_state=0, test_size=0.2)
            clf.fit(X_train, y_train)
    else:
        st.error("Please upload your resume file, enter the job qualifications, and required years of experience.")