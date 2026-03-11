# import pandas as pd
# import joblib

# # Load trained model
# model = joblib.load("salary_model.pkl")

# def predict_salary(experience):

#     exp = pd.DataFrame({
#         "months_experience": [experience]
#     })

#     prediction = model.predict(exp)

#     return prediction[0]



# def predict_salary_range(experience):

#     pred = predict_salary(experience)

#     lower = pred * 0.9
#     upper = pred * 1.1

#     return lower, upper





import os
import pandas as pd
import joblib

# Ensure the path points to the folder where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "salary_model.pkl")

# Load trained model
model = joblib.load(model_path)

def predict_salary(experience):
    exp = pd.DataFrame({
        "months_experience": [experience]
    })
    prediction = model.predict(exp)
    return prediction[0]

def predict_salary_range(experience):
    pred = predict_salary(experience)
    lower = pred * 0.9
    upper = pred * 1.1
    return lower, upper
