import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_model():

    data = pd.read_csv(r"C:\Users\ssk12\OneDrive\Documents\GitHub\AI-Powered-Data-Visualization-Dashboard\dataset\chart_training_data.csv")

    X = data[
        [
            "num_columns",
            "num_categorical",
            "num_datetime"
        ]
    ]

    y = data["chart"]

    model = DecisionTreeClassifier()

    model.fit(X, y)

    joblib.dump(model, r"C:\Users\ssk12\OneDrive\Documents\GitHub\AI-Powered-Data-Visualization-Dashboard\dataset\chart_model.pkl")

def predict_chart(
    num_columns,
    num_categorical,
    num_datetime
):

    model = joblib.load(r"C:\Users\ssk12\OneDrive\Documents\GitHub\AI-Powered-Data-Visualization-Dashboard\dataset\chart_model.pkl")

    prediction = model.predict(
        [[
            num_columns,
            num_categorical,
            num_datetime
        ]]
    )

    return prediction[0]