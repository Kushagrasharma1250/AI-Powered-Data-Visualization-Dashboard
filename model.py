def suggest_chart(x_type, y_type):

    if x_type == "Categorical" and y_type == "Numerical":
        return "Bar Chart"

    elif x_type == "Numerical" and y_type == "Numerical":
        return "Scatter Plot"

    elif x_type == "Datetime" and y_type == "Numerical":
        return "Line Chart"

    elif x_type == "Numerical":
        return "Histogram"

    else:
        return "Pie Chart"