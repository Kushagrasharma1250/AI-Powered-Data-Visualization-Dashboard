import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils import analyze_dataset
from model import suggest_chart

st.title("AI-Powered Data Visualization Dashboard")

file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if file:

    if file.name.endswith("csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("Dataset Preview")
    st.dataframe(df)

    # Detect column types
    column_types = analyze_dataset(df)

    st.write("Detected Column Types")

    for col, dtype in column_types.items():
        st.write(col, ":", dtype)

    x_column = st.selectbox(
        "Select X-axis",
        df.columns
    )

    y_column = st.selectbox(
        "Select Y-axis",
        df.columns
    )

    x_type = column_types[x_column]
    y_type = column_types[y_column]

    chart = suggest_chart(
        x_type,
        y_type
    )

    st.write("Suggested Chart:", chart)

    # Generate chart
    fig, ax = plt.subplots()

    if chart == "Bar Chart":

        df.groupby(x_column)[y_column].sum().plot(
            kind="bar",
            ax=ax
        )

    elif chart == "Scatter Plot":

        ax.scatter(
            df[x_column],
            df[y_column]
        )

    elif chart == "Line Chart":

        ax.plot(
            df[x_column],
            df[y_column]
        )

    elif chart == "Histogram":

        ax.hist(
            df[x_column]
        )

    elif chart == "Pie Chart":

        df[x_column].value_counts().plot(
            kind="pie",
            ax=ax
        )

    st.pyplot(fig)