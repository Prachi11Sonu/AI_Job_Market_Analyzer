import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("jobs_with_skills.csv")

df["sal_low"] = pd.to_numeric(df["sal_low"], errors="coerce")
df["sal_high"] = pd.to_numeric(df["sal_high"], errors="coerce")

df["salary"] = (df["sal_low"] + df["sal_high"]) / 2

df["months_experience"] = pd.to_numeric(df["months_experience"], errors="coerce")

df["salary"] = df["salary"].fillna(df["salary"].median())
df["months_experience"] = df["months_experience"].fillna(df["months_experience"].median())

# select features
features = ["months_experience","Seniority level","location"]

df_model = df[features + ["salary"]]

# convert categorical to numeric
df_model = pd.get_dummies(df_model, drop_first=True)

X = df_model.drop("salary", axis=1)
y = df_model["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)

error = mean_absolute_error(y_test, pred)

print("Model Error:", error)
