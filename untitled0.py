# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:03:10 2024

@author: 31655
"""

import streamlit as st
import pandas as pd
import pickle

# Load your pre-trained model
import joblib
model = joblib.load("D:/project/my_model.pkl") 
# Sample data (replace with actual data if needed)
data = pd.DataFrame({
    "year": [2014, 2014],
    "Quantity": [10000.0, 9000.0],
    "Market Price": [3.47, 3.47],
    "loyalty": [3324, 3534],
    "CME_DEC": [True, True],
    "CME_JUL": [False, False],
    "CME_MAR": [False, False],
    "CME_MAY": [False, False],
    "CME_SEP": [False, False],
})

# Title and Description
st.title("Predictive Model with CME Selection")
st.write("This app allows you to input numeric values and select a CME condition.")

# User Input for Features
st.write("### Enter Feature Values")
year = st.number_input("Year", min_value=2000, max_value=2100, value=2014, step=1)
quantity = st.number_input("Quantity", value=10000.0, step=100.0)
market_price = st.number_input("Market Price", value=3.47, step=0.01)
loyalty = st.number_input("Loyalty", value=3324, step=1)

# CME Selection
st.write("### Select CME")
cme_option = st.selectbox("CME Options", ["CME_DEC", "CME_JUL", "CME_MAR", "CME_MAY", "CME_SEP"])
cme_values = {col: col == cme_option for col in ["CME_DEC", "CME_JUL", "CME_MAR", "CME_MAY", "CME_SEP"]}

# Combine all inputs into a DataFrame
input_df = pd.DataFrame({
    "year": [year],
    "Quantity": [quantity],
    "Market Price": [market_price],
    "loyalty": [loyalty],
    **cme_values,  # Dynamically add CME values
})

# Display the input DataFrame
st.write("### User Inputs")
st.write(input_df)

# Predict button
if st.button("Predict"):
    # Make a prediction using the model
    prediction = model.predict(input_df)
    st.write("### Prediction Result")
    st.write(prediction)

# Optional: Visualization
st.write("### Visualization")
st.line_chart(input_df.drop(columns=list(cme_values.keys())).T)  # Example visualization without CME columns
