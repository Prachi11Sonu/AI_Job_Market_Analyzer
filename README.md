# AI Job Market Analyzer

AI Job Market Analyzer is an interactive data analytics and machine learning dashboard designed to analyze job market trends, identify in-demand skills, and provide intelligent career insights based on job listing data. The project helps job seekers understand hiring trends, salary expectations, and skill requirements to make more informed career decisions.

The application processes job listing datasets and transforms them into an interactive dashboard where users can explore job trends, analyze market demand for different skills, predict salaries, and evaluate how well their resume matches available job opportunities.

---

## Project Features

### Market Overview Dashboard

* Displays total job postings, companies, and locations
* Shows top hiring companies and popular job roles
* Visualizes job distribution across different locations
* Calculates a city job market score based on:

  * number of jobs
  * average salary
  * number of hiring companies

### Skills Demand Analysis

* Identifies the most demanded skills from job listings
* Displays top skills using interactive charts
* Generates a **Skills Word Cloud** for quick visualization of popular technologies

### Salary Analytics

* Visualizes salary distribution across the dataset
* Shows the relationship between **experience and salary**
* Provides insights into salary trends in the job market

### Salary Prediction (Machine Learning)

* Predicts expected salary based on years of experience
* Provides a predicted salary range using a trained machine learning model

### AI Resume Analyzer

* Users can upload their resume in **PDF format**
* Extracts resume text using PDF parsing
* Detects technical skills present in the resume by comparing them with job dataset skills

### AI Resume Match Score

* Compares uploaded resume with job listings
* Uses **CountVectorizer and Cosine Similarity**
* Calculates a compatibility score showing how well the resume matches job requirements

### Job Recommendation System

* Users can input their skills
* The system calculates a **skill match score**
* Recommends the most relevant job roles from the dataset

### Skill Gap Analyzer

* Compares user skills with the most demanded skills in the market
* Identifies **skills the user should learn** to improve employability

### Career Path Predictor

* Suggests possible career growth paths such as:

  * Junior Role
  * Mid-Level Role
  * Senior Role
  * Advanced Technical Positions

### AI Job Market Chatbot

* A rule-based chatbot that answers questions about:

  * required skills for specific roles
  * highest paying cities
  * most demanded skills
  * job recommendations

---

## Tech Stack

Python, Streamlit, Pandas, Matplotlib, WordCloud, Scikit-learn (CountVectorizer & Cosine Similarity), pdfplumber, Machine Learning, Natural Language Processing (NLP), and Data Visualization.

---

## Project Structure

```
AI-JOB-MARKET-ANALYZER
│
├── app.py                 # Main Streamlit dashboard
├── predict.py             # Salary prediction model
├── jobs_processed.csv     # Processed job dataset
├── JobMarketAnalyzer.ipynb # Data preprocessing & analysis
├── requirements.txt       # Required libraries
└── README.md
```

---



Navigate to the project folder

```
cd AI-Job-Market-Analyzer
```

Install required dependencies

```
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit dashboard

```
streamlit run app.py
```

The application will open in your browser.

---

## Example Use Cases

* Job seekers can analyze which skills are currently in demand.
* Students can understand which technologies they should learn.
* Professionals can evaluate how well their resume matches available jobs.
* Users can explore salary trends and career opportunities across different locations.

---

## Future Improvements

* Integration with real-time job APIs
* Deep learning based resume analysis
* Advanced NLP job matching
* Skill demand forecasting
* Resume improvement suggestions

---

## Author

Developed as a **Data Science & Machine Learning portfolio project** to demonstrate skills in data analysis, machine learning, and interactive dashboard development.

---

## Project Links

GitHub Repository:
https://github.com/Prachi11Sonu/AI_Job_Market_Analyzer

Live Dashboard:
https://aijobanalyzer.streamlit.app/
