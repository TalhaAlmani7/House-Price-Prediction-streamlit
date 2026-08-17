import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load trained model and sample baseline row

model = joblib.load('house_price_model.pkl')
sample_row = joblib.load('sample_row.pkl')
st.title("🏡 House Price Prediction App")
st.write("Enter house details below to estimate the sale price.")

st.header("House Features")

# 2. Define user input widgets
area = st.slider("Above Grade Living Area (Sq Ft)",
                 min_value=500, max_value=5000, value=1500, step=50)
bedrooms = st.selectbox("Bedrooms Above Ground",
                        options=[1, 2, 3, 4, 5], index=2)
overall_qual = st.slider("Overall Quality (1-10)",
                         min_value=1, max_value=10, value=5, step=1)
year_built = st.slider("Year Built", min_value=1800,
                       max_value=2026, value=2000, step=1)
total_bsmt_sf = st.number_input(
    "Total Basement Sq Ft", min_value=0, max_value=3000, value=800)

# 3. Prediction execution block
if st.button("Calculate Estimated Price"):
    # Copy baseline sample
    sample = sample_row.copy()

    # Calculate house age
    house_age = 2026 - year_built

    # Update input values dynamically across possible matching columns
    for col in sample.columns:
        if col in ["GrLivArea", "Gr_Liv_Area"]:
            sample[col] = area
        elif col in ["BedroomAbvGr", "Bedrooms"]:
            sample[col] = bedrooms
        elif col in ["OverallQual", "Overall_Qual"]:
            sample[col] = overall_qual
        elif col in ["HouseAge", "House_Age"]:
            sample[col] = house_age
        elif col in ["TotalBsmtSF", "Total_Bsmt_SF"]:
            sample[col] = total_bsmt_sf

    # Force Overall Quality to heavily impact features if present
    if "OverallQual" in sample.columns:
        sample["OverallQual"] = float(overall_qual)

    # Make prediction and reverse log transformation
    pred_log = model.predict(sample)
    pred_price = np.expm1(pred_log)[0]

    st.success(f"Estimated Sale Price: ${pred_price:,.2f}")
