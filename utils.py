import pandas as pd
def detect_column_type(series):
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    elif pd.api.types.is_string_dtype(series):
        return 'categorical'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    else:        return 'other'

def analyze_dataset(df):
    analysis = {}
    for column in df.columns:
        analysis[column] = detect_column_type(df[column])
    return analysis