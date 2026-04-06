import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import analyze_dataset
from model import predict_chart

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Data Visualization Dashboard",
    layout="wide"
)

st.title("AI-Powered Data Visualization Dashboard")

st.write(
    "Upload a dataset and let AI automatically suggest the best visualization."
)

# -----------------------------
# File Upload
# -----------------------------

file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if file:

    # -----------------------------
    # Load Dataset
    # -----------------------------

    if file.name.endswith("csv"):
        df = pd.read_csv(file)

    else:
        df = pd.read_excel(file)

    st.subheader("Dataset Preview")

    st.dataframe(df)

    # -----------------------------
    # Detect Column Types
    # -----------------------------

    column_types = analyze_dataset(df)

    st.subheader("Detected Column Types")

    for col, dtype in column_types.items():

        st.write(
            col,
            ":",
            dtype
        )

    # -----------------------------
    # Sidebar Filters
    # -----------------------------

    st.sidebar.header("Data Filters")

    filter_column = st.sidebar.selectbox(
        "Select column to filter",
        df.columns
    )

    unique_values = df[
        filter_column
    ].dropna().unique()

    selected_value = st.sidebar.selectbox(
        "Select value",
        unique_values
    )

    filtered_df = df[
        df[filter_column] == selected_value
    ]

    st.subheader("Filtered Data")

    st.dataframe(filtered_df)

    # -----------------------------
    # Column Selection
    # -----------------------------

    st.subheader("Chart Generator")

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

    # -----------------------------
    # Count Column Types for ML
    # -----------------------------

    num_columns = 0
    num_categorical = 0
    num_datetime = 0

    for dtype in column_types.values():

        if dtype == "Numerical":
            num_columns += 1

        elif dtype == "Categorical":
            num_categorical += 1

        elif dtype == "Datetime":
            num_datetime += 1

    # -----------------------------
    # Predict Chart using ML
    # -----------------------------

    chart = predict_chart(
        num_columns,
        num_categorical,
        num_datetime,
        x_type,
        y_type
    )

    st.write(
        "AI Suggested Chart:",
        chart
    )

    # -----------------------------
    # Generate Chart
    # -----------------------------

    fig, ax = plt.subplots()

    try:

        if chart == "Bar Chart":

            df.groupby(
                x_column
            )[y_column].sum().plot(
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

    except Exception as e:

        st.error(
            "Error generating chart: "
            + str(e)
        )

    # -----------------------------
    # Download Chart
    # -----------------------------

    if st.button("Download Chart"):

        fig.savefig(
            "chart.png"
        )

        st.success(
            "Chart saved as chart.png"
        )

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------

    if st.button(
        "Show Correlation Heatmap"
    ):

        numeric_df = df.select_dtypes(
            include=["number"]
        )

        if not numeric_df.empty:

            fig2, ax2 = plt.subplots()

            sns.heatmap(
                numeric_df.corr(),
                annot=True,
                cmap="coolwarm",
                ax=ax2
            )

            st.pyplot(fig2)

        else:

            st.warning(
                "No numeric columns available."
            )

    # -----------------------------
    # Missing Values Detection
    # -----------------------------

    if st.button(
        "Check Missing Values"
    ):

        missing = df.isnull().sum()

        st.subheader(
            "Missing Values"
        )

        st.write(
            missing
        )

    # -----------------------------
    # Outlier Detection
    # -----------------------------

    if st.button(
        "Detect Outliers"
    ):

        numeric_df = df.select_dtypes(
            include=["number"]
        )

        for col in numeric_df.columns:

            Q1 = numeric_df[col].quantile(
                0.25
            )

            Q3 = numeric_df[col].quantile(
                0.75
            )

            IQR = Q3 - Q1

            outliers = numeric_df[
                (
                    numeric_df[col]
                    < Q1 - 1.5 * IQR
                )
                |
                (
                    numeric_df[col]
                    > Q3 + 1.5 * IQR
                )
            ]

            st.write(
                "Outliers in",
                col
            )

            st.dataframe(
                outliers
            )

    # -----------------------------
    # Dataset Summary
    # -----------------------------

    if st.button(
        "Show Dataset Summary"
    ):

        st.subheader(
            "Statistical Summary"
        )

        st.write(
            df.describe()
        )

else:

    st.info(
        "Please upload a dataset to begin."
    )