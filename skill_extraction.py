# import pandas as pd
# skills_list = [
#     "python","sql","machine learning","deep learning",
#     "aws","tensorflow","pandas","numpy","spark",
#     "tableau","power bi","excel"
# ]

# def extract_skills(text):
    
#     text = str(text).lower()
    
#     found = []
    
#     for skill in skills_list:
#         if skill in text:
#             found.append(skill)
            
#     return found

# df = pd.read_csv("cleaned_jobs.csv")

# df["skills"] = df["description"].apply(extract_skills)

# df.to_csv("jobs_with_skills.csv", index=False)

# print("Skill extraction completed")






import pandas as pd

# -------- Rule based skills list --------
skills_list = [
    "python","sql","machine learning","deep learning",
    "aws","tensorflow","pandas","numpy","spark",
    "tableau","power bi","excel"
]

def extract_skills_rule(text):

    text = str(text).lower()

    found = []

    for skill in skills_list:
        if skill in text:
            found.append(skill)

    return found


# -------- Deep learning skill extraction --------
from transformers import pipeline

nlp = pipeline("ner", model="dslim/bert-base-NER")

def extract_skills_bert(text):

    text = str(text)

    results = nlp(text)

    skills = []

    for r in results:
        if r["entity"].startswith("B"):
            skills.append(r["word"])

    return list(set(skills))


# -------- Load dataset --------
df = pd.read_csv("cleaned_jobs.csv")


# -------- Apply both methods --------
df["skills_rule"] = df["description"].apply(extract_skills_rule)

df["skills_bert"] = df["description"].apply(extract_skills_bert)


# -------- Save output --------
df.to_csv("jobs_with_skills.csv", index=False)

print("Skill extraction completed")