import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Center-align the title using Markdown
st.markdown("<h1 style='text-align: center;'>Car Prediction</h1>", unsafe_allow_html=True)

# Load the dataset
df = pd.read_csv('Car_sales.csv')
df.fillna(0, inplace=True)

# Select features (x) and target (y)
x = df[['Price_in_thousands', 'Horsepower', 'Wheelbase', 'Width', 'Length', 'Power_perf_factor']]
y = df['Sales_in_thousands']

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=101)

# Build and train the Linear Regression model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Create input fields for the user to enter car details
Price_in_thousands = st.number_input('Enter Price_in_thousands', min_value=1000)
Horsepower = st.number_input('Enter Horsepower of the car', min_value=0)
Wheelbase = st.number_input('Enter Wheelbase of the car', min_value=0)
Width = st.number_input('Enter Width', min_value=0)
Length = st.number_input('Enter Length', min_value=0)
Power_perf_factor = st.number_input('Enter Power_perf_factor', min_value=0)

# Predict car sales based on user inputs
input_data = [[Price_in_thousands, Horsepower, Wheelbase, Width, Length, Power_perf_factor]]
sale_price = model.predict(input_data)
if st.button("Predict Sales"):
    st.text(f'Predicted sales in thousands: {abs(sale_price[0]):.2f}')
    st.success("Successfully predicted the price")

# Calculate the mean absolute error
A = model.predict(xtest)
error = mean_absolute_error(ytest, A)
if st.button("Calculate MAE"):
    st.info(f'Mean Absolute Error: {error:.2f}')




