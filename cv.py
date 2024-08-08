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
    resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')
    return resumeDataSet

resumeDataSet = load_data()

# Function to clean resume text
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # Remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # Remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # Remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # Remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # Remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)  # Remove non-ASCII characters
    resumeText = re.sub('\s+', ' ', resumeText)  # Remove extra whitespace
    return resumeText

resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Label Encoding the 'Category' column
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])

# Split the data into train and test sets
requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)

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

# Expanded qualifications list
qualifications = [
    "Data Science", "Software Developer", "Business Analyst", "Project Manager",
    "Product Manager", "Marketing Specialist", "HR Specialist", "Finance Analyst",
    "Operations Manager", "Graphic Designer", "Network Engineer", "DevOps Engineer",
    "Cybersecurity Specialist", "Machine Learning Engineer", "UI/UX Designer",
    "Cloud Architect", "AI Researcher", "Content Writer", "Sales Executive", "Other"
]

selected_qualification = st.selectbox("Select your desired qualification", qualifications)

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

if st.button("Predict"):
    if uploaded_file is not None:
        # Read and clean the uploaded resume file
        user_resume = read_pdf(uploaded_file)
        cleaned_user_resume = cleanResume(user_resume)

        # Transform the resume using the trained vectorizer
        user_resume_vector = word_vectorizer.transform([cleaned_user_resume])

        # Predict the category
        prediction = clf.predict(user_resume_vector)
        predicted_category = le.inverse_transform(prediction)[0]

        # Display the prediction result
        st.write(f"Your resume is classified under the category: **{predicted_category}**")

        # Simple qualification check
        if predicted_category == selected_qualification:
            st.success(f"Your resume is likely to qualify for a {selected_qualification} interview!")
        else:
            st.warning(f"Your resume might not be suitable for a {selected_qualification} role.")
    else:
        st.error("Please upload your resume file for prediction.")
