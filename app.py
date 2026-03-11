
import streamlit as st
import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from predict import predict_salary, predict_salary_range

# -------------------------------
# Page Config
# -------------------------------

st.set_page_config(
    page_title="AI Job Market Analyzer",
    layout="wide",
    page_icon="📊"
)

st.title("📊 AI Job Market Analyzer Dashboard")

st.markdown(
    "Analyze **job trends, skill demand, and salary predictions** using Machine Learning."
)

# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("jobs_processed.csv")

df["skills"] = df["skills"].apply(lambda x: ast.literal_eval(str(x)))

# -------------------------------
# Sidebar Filters
# -------------------------------

st.sidebar.header("Filters")

selected_location = st.sidebar.selectbox(
    "Select Location",
    ["All"] + sorted(df["location"].dropna().unique())
)

selected_company = st.sidebar.selectbox(
    "Select Company",
    ["All"] + sorted(df["company"].dropna().unique())
)

filtered_df = df.copy()

if selected_location != "All":
    filtered_df = filtered_df[filtered_df["location"] == selected_location]

if selected_company != "All":
    filtered_df = filtered_df[filtered_df["company"] == selected_company]

# -------------------------------
# Tabs
# -------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
"📊 Market Overview",
"📈 Skills Insights",
"💰 Salary Analytics",
"🤖 AI Tools",
"🎯 Career Tools",
"💬 AI Chatbot"
])

# -------------------------------
# Market Overview
# -------------------------------

with tab1:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Jobs", len(filtered_df))
    col2.metric("Companies", filtered_df["company"].nunique())
    col3.metric("Locations", filtered_df["location"].nunique())
    col4.metric("Avg Salary", f"${filtered_df['salary'].mean():,.0f}")

    st.subheader("Top Job Roles")
    st.bar_chart(filtered_df["title"].value_counts().head(10))

    st.subheader("Top Hiring Companies")
    st.bar_chart(filtered_df["company"].value_counts().head(10))

    st.subheader("Top Locations")
    st.bar_chart(filtered_df["location"].value_counts().head(10))

# -------------------------------
# Skills Insights
# -------------------------------

with tab2:

    st.subheader("Most Demanded Skills")

    skills = filtered_df["skills"].explode()

    skill_counts = Counter(skills)

    top_skills = pd.DataFrame(
        skill_counts.items(),
        columns=["skill","count"]
    ).sort_values(by="count",ascending=False).head(10)

    st.bar_chart(top_skills.set_index("skill"))

    st.subheader("☁️ Skills Word Cloud")

    skills_list = filtered_df["skills"].explode()
    skills_list = skills_list.dropna().astype(str)

    skills_text = " ".join(skills_list)

    wc = WordCloud(
        width=900,
        height=450,
        background_color="black"
    ).generate(skills_text)

    fig, ax = plt.subplots()

    ax.imshow(wc)
    ax.axis("off")

    st.pyplot(fig)

# -------------------------------
# Salary Analytics
# -------------------------------

with tab3:

    st.subheader("Experience vs Salary")

    st.scatter_chart(filtered_df[["months_experience","salary"]])

    st.subheader("Salary Distribution")

    fig, ax = plt.subplots()

    ax.hist(filtered_df["salary"], bins=25)

    mean_salary = filtered_df["salary"].mean()

    ax.axvline(mean_salary, color="red", linestyle="dashed")

    st.pyplot(fig)

# -------------------------------
# AI Tools
# -------------------------------

with tab4:

    st.subheader("Salary Prediction")

    exp = st.slider("Experience (Months)",0,120)

    if st.button("Predict Salary"):

        pred_salary = predict_salary(exp)

        lower, upper = predict_salary_range(exp)

        st.success(f"Predicted Salary: ${pred_salary:,.2f}")
        st.info(f"Range: ${lower:,.2f} - ${upper:,.2f}")

    st.subheader("Resume Analyzer")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    resume_text = ""

    if uploaded_file:

        with pdfplumber.open(uploaded_file) as pdf:

            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    resume_text += text

        st.success("Resume Uploaded")

        all_skills = df["skills"].explode().dropna().unique()

        detected_skills = []

        for skill in all_skills:

            if str(skill).lower() in resume_text.lower():
                detected_skills.append(skill)

        st.write("Detected Skills:")
        st.write(detected_skills)

        st.subheader("AI Resume Match Score")

        df["skills_text"] = df["skills"].apply(lambda x: " ".join(x))

        vectorizer = CountVectorizer()

        corpus = df["skills_text"].tolist() + [resume_text]

        vectors = vectorizer.fit_transform(corpus)

        similarity = cosine_similarity(vectors[-1], vectors[:-1])

        df["match_score"] = similarity[0]

        top_matches = df.sort_values(
            by="match_score",
            ascending=False
        ).head(5)

        top_matches["Match %"] = (top_matches["match_score"]*100).round(2)

        st.dataframe(
            top_matches[["title","company","location","Match %"]]
        )

# -------------------------------
# Career Tools
# -------------------------------

# with tab5:

#     st.subheader("Job Recommendation")

#     user_skills_input = st.text_input(
#         "Enter your skills",
#         "python, sql"
#     )

#     if st.button("Find Jobs"):

#         user_skills = [s.strip().lower() for s in user_skills_input.split(",")]

#         def match_score(job_skills):
#             return len(set(user_skills).intersection(set(job_skills)))

#         filtered_df["match_score"] = filtered_df["skills"].apply(match_score)

#         recommended_jobs = filtered_df.sort_values(
#             by="match_score",
#             ascending=False
#         ).head(5)

#         st.dataframe(
#             recommended_jobs[["title","company","location"]]
#         )

#         # Skill Gap Analyzer (added back)

#         st.subheader("Skills You Should Learn")

#         all_skills = df["skills"].explode()

#         top_market_skills = all_skills.value_counts().head(10).index.tolist()

#         missing_skills = list(set(top_market_skills) - set(user_skills))

#         st.write(missing_skills)

#         # Career Path Predictor

#         st.subheader("Career Path Suggestion")

#         career_paths = {

#             "data scientist":[
#                 "Junior Data Scientist",
#                 "Data Scientist",
#                 "Senior Data Scientist",
#                 "AI Engineer"
#             ],

#             "data analyst":[
#                 "Junior Data Analyst",
#                 "Data Analyst",
#                 "Senior Data Analyst",
#                 "Analytics Manager"
#             ]
#         }

#         best_role = recommended_jobs.iloc[0]["title"].lower()

#         if best_role in career_paths:

#             for role in career_paths[best_role]:

#                 st.info(role)







with tab5:

    st.subheader("Job Recommendation")

    user_skills_input = st.text_input(
        "Enter your skills",
        "python, sql"
    )

    if st.button("Find Jobs"):

        user_skills = [s.strip().lower() for s in user_skills_input.split(",")]

        # -----------------------
        # Job Matching
        # -----------------------

        def match_score(job_skills):

            job_skills = [str(s).lower() for s in job_skills]

            return len(set(user_skills).intersection(set(job_skills)))

        filtered_df["match_score"] = filtered_df["skills"].apply(match_score)

        recommended_jobs = filtered_df.sort_values(
            by="match_score",
            ascending=False
        ).head(5)

        st.subheader("Recommended Jobs")

        st.dataframe(
            recommended_jobs[["title","company","location"]]
        )

        # -----------------------
        # Skill Gap Analyzer
        # -----------------------

        st.subheader("Skills You Should Learn")

        all_market_skills = df["skills"].explode()

        top_market_skills = all_market_skills.value_counts().head(10).index.tolist()

        # missing_skills = list(set(top_market_skills) - set(user_skills))

        # st.write(missing_skills)
        missing_skills = list(set(top_market_skills) - set(user_skills))

        if missing_skills:

           for skill in missing_skills:

                st.warning(f"Learn: {skill}")

        else:

            st.success("Great! You already have the top market skills 🎉")

        # -----------------------
        # Career Path Predictor
        # -----------------------

        st.subheader("Career Path Suggestion")

        career_paths = {

            "data scientist":[
                "Junior Data Scientist",
                "Data Scientist",
                "Senior Data Scientist",
                "AI Engineer"
            ],

            "data analyst":[
                "Junior Data Analyst",
                "Data Analyst",
                "Senior Data Analyst",
                "Analytics Manager"
            ],

            "backend developer":[
                "Junior Backend Developer",
                "Backend Developer",
                "Senior Backend Developer",
                "Software Architect"
            ]

        }

        best_role = recommended_jobs.iloc[0]["title"].lower()

        for role in career_paths:

            if role in best_role:

                for step in career_paths[role]:

                    st.info(step)






# -------------------------------
# Chatbot
# -------------------------------

with tab6:

    st.subheader("AI Job Market Chatbot")

    question = st.text_input("Ask something about jobs")

    if question:

        q = question.lower()

        if ("skill" in q or "skills" in q) and "data scientist" in q:

            ds_skills = df[
                df["title"].str.contains("data scientist", case=False)
            ]["skills"].explode().value_counts().head(10)

            st.write("Top Skills Needed for Data Scientist:")

            st.dataframe(ds_skills)

        elif "highest salary" in q or "best city" in q:

            city_salary = df.groupby("location")["salary"].mean()

            top_city = city_salary.sort_values(ascending=False).head(5)

            st.write("Cities with Highest Salaries:")

            st.dataframe(top_city)

        elif "top skills" in q or "demanded skills" in q:

            skills = df["skills"].explode().value_counts().head(10)

            st.write("Most Demanded Skills:")

            st.dataframe(skills)

        elif "recommend" in q or "job for python" in q:

            python_jobs = df[
                df["skills"].apply(lambda x: "python" in x)
            ][["title","company","location"]].head(5)

            st.write("Jobs requiring Python:")

            st.dataframe(python_jobs)

        else:

            st.write("Try questions like:")
            st.write("- What skills are needed for Data Scientist?")
            st.write("- Which city has highest salary?")
            st.write("- Show top demanded skills")
            st.write("- Recommend jobs for Python")

