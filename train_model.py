import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("jobs_processed.csv")

X = df[["months_experience"]]
y = df["salary"]

model = LinearRegression()

model.fit(X, y)

joblib.dump(model, "salary_model.pkl")

print("Model trained and saved!")