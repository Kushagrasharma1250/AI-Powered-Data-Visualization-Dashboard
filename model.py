import pandas as pd
from functools import lru_cache
from sklearn.tree import DecisionTreeClassifier
import joblib

MODEL_PATH = "chart_model.pkl"
FEATURE_COLUMNS = ["num_columns", "num_categorical", "num_datetime"]

def train_model():
    data = pd.read_csv("chart_training_data.csv")

    X = data[FEATURE_COLUMNS]
    y = data["chart"]

    model = DecisionTreeClassifier()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(MODEL_PATH)


def predict_chart(
    num_columns,
    num_categorical,
    num_datetime,
    x_type=None,
    y_type=None
):
    if x_type and y_type:
        if x_type == "Numerical" and y_type == "Numerical":
            return "Scatter Plot"
        if x_type == "Datetime" and y_type == "Numerical":
            return "Line Chart"
        if x_type == "Numerical" and y_type == "Datetime":
            return "Line Chart"
        if x_type == "Categorical" and y_type == "Numerical":
            return "Bar Chart"
        if x_type == "Numerical" and y_type == "Categorical":
            return "Bar Chart"
        if x_type == "Categorical" and y_type == "Categorical":
            return "Pie Chart"
        if x_type == "Datetime" and y_type == "Categorical":
            return "Bar Chart"
        if x_type == "Categorical" and y_type == "Datetime":
            return "Bar Chart"

    try:
        model = _load_model()
    except FileNotFoundError:
        train_model()
        model = _load_model()

    X = pd.DataFrame(
        [[
            num_columns,
            num_categorical,
            num_datetime
        ]],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(X)

    return prediction[0]