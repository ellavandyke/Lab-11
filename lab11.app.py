import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# Load the trained models and preprocessing pipelines
model_selected = keras.models.load_model('model_selected.h5')
model_all = keras.models.load_model('model_all.h5')
preprocessor_selected = joblib.load('preprocessor_selected.pkl')
preprocessor_all = joblib.load('preprocessor_all.pkl')

# Streamlit app title
st.title("Bridge Load Prediction App")

# Define the form for user input
st.sidebar.header("Bridge Data Input")
span = st.sidebar.number_input("Span (ft)", min_value=0, value=50)
deck_width = st.sidebar.number_input("Deck Width (ft)", min_value=0, value=20)
age = st.sidebar.number_input("Bridge Age (Years)", min_value=0, value=50)
num_lanes = st.sidebar.number_input("Number of Lanes", min_value=0, value=2)
material = st.sidebar.selectbox("Material", ["Concrete", "Steel", "Wood"])
condition = st.sidebar.selectbox("Condition Rating", ["Good", "Fair", "Poor"])

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'Span_ft': [span],
    'Deck_Width_ft': [deck_width],
    'Age_Years': [age],
    'Num_Lanes': [num_lanes],
    'Material': [material],
    'Condition_Rating': [condition]
})

# Process the input data using the preprocessor
input_data_selected = preprocessor_selected.transform(input_data)
input_data_all = preprocessor_all.transform(input_data)

# Predict the maximum load using both models
prediction_selected = model_selected.predict(input_data_selected)
prediction_all = model_all.predict(input_data_all)

# Display predictions
st.subheader("Predicted Maximum Load")
st.write(f"Using Selected Features: {prediction_selected[0][0]:.2f} Tons")
st.write(f"Using All Features: {prediction_all[0][0]:.2f} Tons")
