import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def main():
    st.title("Data Exploration and Modeling App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        with st.tab("Home"):
            st.header("Home")
            # Load data and parse 'date' column as datetime
            data = pd.read_csv(uploaded_file, parse_dates=['date'])
    
            # Display basic information
            st.subheader("Data Information:")
            st.write("Shape:", data.shape)
            st.write("Total NAs:", data.isnull().sum().sum())
            st.write("Columns and Types:", data.dtypes)
    
            # Display summary table
            st.subheader("Summary:")
            st.write(data.describe())
    
            # Display NAs
            st.subheader("NAs:")
            st.write(data.isnull().sum())
    
            # Set date as index
            data.set_index('date', inplace=True)
    
            # Choose variable for Y-axis in plot
            y_variable = st.selectbox("Select variable for Y-axis:", data.columns)

        # Plot Tab
        with st.tab("Plot"):
            st.header("Plot")
            # Plot data
            st.subheader("Plot:")
            st.line_chart(data[y_variable])

        # Model Tab
        with st.tab("Model"):
            st.header("Model")

            # Add and remove variables by typing in a table
            st.subheader("Select Features for Linear Regression:")
            suggestions = data.columns.tolist()
            selected_variables = st.multiselect("Add variables:", suggestions, default=suggestions[:2])
    
            # Display selected variables table
            variables_table = pd.DataFrame({'Selected Variables': selected_variables})
            variables_table['Remove'] = [st.checkbox(f"Remove {variable}", key=f"remove_{variable}") for variable in selected_variables]
            st.table(variables_table)
    
            # Select target variable
            selected_kpi = st.selectbox("Select KPI for Linear Regression:", data.columns)
            # Check if enough variables are selected
            selected_variables = variables_table.loc[~variables_table['Remove'], 'Selected Variables'].tolist()
            # Check if enough variables are selected
            if selected_variables:
                # Perform linear regression
                X = sm.add_constant(data[selected_variables])  # Add constant term for intercept
                y = data[selected_kpi]
    
                model = sm.OLS(y, X).fit()
    
                # Display R-squared, Adjusted R-squared, Coefficients, t-statistics, and p-values
                st.subheader("Regression Statistics:")
                st.write(f"R-squared: {model.rsquared:.4f}")
                st.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    
                # Display coefficients, t-statistics, and p-values in a table
                coefficients_df = pd.DataFrame({
                    'Variable': ['Intercept'] + selected_variables,
                    'Coefficient': [model.params[0]] + model.params[1:].tolist(),
                    'T-Stat': model.tvalues.tolist(),
                    'P-Value': model.pvalues.tolist()
                })
                st.table(coefficients_df)
    
                # Display another plot with predicted values
                st.subheader("Linear Regression Prediction Plot:")
                predicted_values = model.predict(X)
                prediction_df = pd.DataFrame({'Actual': y, 'Predicted': predicted_values})
                st.line_chart(prediction_df)

if __name__ == "__main__":
    main()
