import pandas as pd
import xgboost as xgb
import numpy as np

# Load the trained XGBoost model
# XGB = xgb.XGBClassifier()
# XGB.load_model('xgb_model.json')

XGB = xgb.Booster()
XGB.load_model('XGB.bin')

def predict_exited():
    """
    Take user input for 16 features, make predictions on the input data using the trained XGBoost model, and return the predicted probabilities of exited.

    Returns:
    prediction (array): Predicted probabilities of exited
    """
    # Take user input for 16 features
    CreditScore = float(input("Enter CreditScore: "))
    Age = float(input("Enter Age: "))
    Tenure = float(input("Enter Tenure: "))
    Balance = float(input("Enter Balance: "))
    NumOfProducts = float(input("Enter NumOfProducts: "))
    EstimatedSalary = float(input("Enter EstimatedSalary: "))
    BalanceSalaryRatio = float(input("Enter BalanceSalaryRatio: "))
    TenureByAge = float(input("Enter TenureByAge: "))
    CreditScoreGivenAge = float(input("Enter CreditScoreGivenAge: "))
    HasCrCard = float(input("Enter HasCrCard (0 or 1): "))
    IsActiveMember = float(input("Enter IsActiveMember (0 or 1): "))
    Geography_Spain = float(input("Enter Geography_Spain (0 or 1): "))
    Geography_France = float(input("Enter Geography_France (0 or 1): "))
    Geography_Germany = float(input("Enter Geography_Germany (0 or 1): "))
    Gender_Female = float(input("Enter Gender_Female (0 or 1): "))
    Gender_Male = float(input("Enter Gender_Male (0 or 1): "))

    # Create a DataFrame from the user input
    feature_values = [CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary, 
                      BalanceSalaryRatio, TenureByAge, CreditScoreGivenAge, HasCrCard, 
                      IsActiveMember, Geography_Spain, Geography_France, Geography_Germany, 
                      Gender_Female, Gender_Male]
    df_input = pd.DataFrame([feature_values], columns=[
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 
        'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreGivenAge', 'HasCrCard', 
        'IsActiveMember', 'Geography_Spain', 'Geography_France', 'Geography_Germany', 
        'Gender_Female', 'Gender_Male'
    ])

    # Make predictions using the trained XGBoost model
    # prediction = XGB.predict_proba(df_input)[:, 1]

    predictions = XGB.predict(xgb.DMatrix(df_input))
    prediction_probabilities = 1 / (1 + np.exp(-predictions))

    return prediction_probabilities

# Call the function to make a prediction
prediction = predict_exited()
print("Predicted probability of exited:", prediction)