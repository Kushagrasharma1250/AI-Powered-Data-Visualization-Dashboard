import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_model():

    data = pd.read_csv("chart_training_data.csv")

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

    joblib.dump(model, "chart_model.pkl")

def predict_chart(
    num_columns,
    num_categorical,
    num_datetime
):

    model = joblib.load("chart_model.pkl")

    prediction = model.predict(
        [[
            num_columns,
            num_categorical,
            num_datetime
        ]]
    )

    return prediction[0]