# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:45:14 2023

@author: Isha Bisht
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import sklearn
from streamlit_option_menu import option_menu

# Load the RandomForestClassifier model
loaded_model = joblib.load(open('random_forest_model1.2.2.joblib','rb'))


with st.sidebar:
    
    selected = option_menu('Quality Prediction System',
                          
                          ['Red Wine Quality Prediction'],
                          default_index=0)

if (selected == 'Red Wine Quality Prediction'):
    
    st.title('Red Wine Quality Prediction ')
    
   # User clicks the predict button
    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.text_input('Fixed Acidity')

    with col2:
        volatile_acidity = st.text_input('Volatile Acidity')

    with col3:
        citric_acid = st.text_input('Citric Acid')

    with col1:
        residual_sugar = st.text_input('Residual Sugar')

    with col2:
        chlorides = st.text_input('Chlorides')

    with col3:
        free_sulfur_dioxide = st.text_input('Free Sulfur Dioxide')

    with col1:
        total_sulfur_dioxide = st.text_input('Total Sulfur Dioxide')

    with col2:
        density = st.text_input('Density')

    with col3:
        pH = st.text_input('pH')

    with col1:
        sulphates = st.text_input('Sulphates')

    with col2:
        alcohol = st.text_input('Alcohol')

   
    # code for Prediction
    wine_quality_diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict Wine Quality'):
        # Create a numpy array from the entered values
        input_data = np.array([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            pH, sulphates, alcohol
        ]])

        # Ensure that the data types are compatible with the model
        input_data = input_data.astype(np.float64)

        # Use the loaded_model for prediction
        wine_quality_prediction = loaded_model.predict(input_data)

        # Assuming 1 is for good quality and 0 is for bad quality
        wine_quality_diagnosis = 'Good Quality Wine' if wine_quality_prediction[0] == 1 else 'Bad Quality Wine'

    st.success(wine_quality_diagnosis)