import streamlit as st
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Streamlit app
st.title("Job Qualification Checker and Recommender")

# Function to clean text
def clean_text(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub('\s+', ' ', text)
    return text.lower()

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to read PDF
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Sample job database (you can expand this or load from a file)
sample_jobs = {
    "Software Developer": "Python, Java, software development, algorithms, data structures",
    "Data Scientist": "Python, R, machine learning, statistics, data analysis",
    "Marketing Specialist": "digital marketing, social media, SEO, content creation, analytics",
    "Project Manager": "project management, agile, scrum, leadership, stakeholder management",
    "Financial Analyst": "financial modeling, Excel, accounting, budgeting, forecasting"
}

# Streamlit interface
st.subheader("Check if Your CV Qualifies for a Job and Get Recommendations")

# CV upload
uploaded_file = st.file_uploader("Upload your CV (PDF format)", type=["pdf"])

if uploaded_file is not None:
    # Read and clean the uploaded CV
    cv_text = read_pdf(uploaded_file)
    cleaned_cv = clean_text(cv_text)

    # Job Qualification Check
    st.subheader("Job Qualification Check")
    job_title = st.text_input("Job Title")
    job_qualifications = st.text_area("Enter the job qualifications (including experience requirements)")

    if st.button("Check Qualification"):
        if job_qualifications:
            # Clean job qualifications
            cleaned_job_qualifications = clean_text(job_qualifications)

            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([cleaned_cv, cleaned_job_qualifications])

            # Calculate similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            # Display results
            st.write(f"Similarity score: {similarity:.2f}")

            if similarity > 0.5:
                st.success(f"Congratulations! Your CV appears to qualify for the {job_title} position.")
            else:
                st.warning("Your CV may not fully match the job qualifications.")

            # Detailed breakdown
            st.subheader("Detailed Breakdown")
            key_qualifications = job_qualifications.lower().split('\n')
            for qual in key_qualifications:
                if qual.strip() in cleaned_cv:
                    st.write(f"✅ {qual.strip()}")
                else:
                    st.write(f"❌ {qual.strip()}")

            # Suggestions for improvement
            st.subheader("Suggestions for Improvement")
            missing_qualifications = [qual.strip() for qual in key_qualifications if qual.strip() not in cleaned_cv]
            if missing_qualifications:
                st.write("Consider adding or emphasizing these qualifications in your CV:")
                for qual in missing_qualifications:
                    st.write(f"- {qual}")
        else:
            st.error("Please enter the job qualifications.")

    # Job Recommendations
    st.subheader("Job Recommendations Based on Your CV")
    if st.button("Get Job Recommendations"):
        # Create TF-IDF vectors for CV and all jobs
        vectorizer = TfidfVectorizer(stop_words='english')
        job_descriptions = list(sample_jobs.values())
        tfidf_matrix = vectorizer.fit_transform([cleaned_cv] + job_descriptions)

        # Calculate similarities
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

        # Sort jobs by similarity
        sorted_jobs = sorted(zip(sample_jobs.keys(), similarities[0]), key=lambda x: x[1], reverse=True)

        # Display top 3 recommendations
        st.write("Top job recommendations for you:")
        for job, similarity in sorted_jobs[:3]:
            st.write(f"- {job} (Similarity: {similarity:.2f})")

    # Add a section for feedback
    # st.subheader("Feedback")
    # feedback = st.text_area("Please provide any feedback on the qualification check and job recommendations:")
    # if st.button("Submit Feedback"):
    #     st.success("Thank you for your feedback!")

else:
    st.error("Please upload your CV to proceed.")