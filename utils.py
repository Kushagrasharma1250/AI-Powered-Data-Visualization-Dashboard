import pandas as pd
def detect_column_type(series):
    if pd.api.types.is_numeric_dtype(series):
        return 'Numerical'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'Datetime'
    elif pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return 'Categorical'
    else:
        return 'Other'

def analyze_dataset(df):
    analysis = {}
    for column in df.columns:
        analysis[column] = detect_column_type(df[column])
    return analysis