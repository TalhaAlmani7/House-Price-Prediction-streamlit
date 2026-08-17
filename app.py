import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Title
st.title("🏠 House Price Prediction App")
st.write("Enter house details below to estimate the sale price.")


# 2. Load Model and Preprocessed Sample Row
@st.cache_resource
def load_all():
    model = joblib.load("house_price_model.pkl")
    # Notebook se save ki hui preprocessed sample row load karein
    sample_row = joblib.load("sample_row.pkl")
    return model, sample_row


try:
    model, sample_row = load_all()
except Exception as e:
    st.error(f"Error loading model or sample row: {e}")

# 3. User Inputs
st.header("House Features")

area = st.slider(
    "Above Grade Living Area (Sq Ft)",
    min_value=500,
    max_value=5000,
    value=1500,
    step=50,
)
bedrooms = st.selectbox(
    "Bedrooms Above Ground", options=[1, 2, 3, 4, 5], index=2
)
overall_qual = st.slider("Overall Quality (1-10)",
                         min_value=1, max_value=10, value=5, step=1)
year_built = st.slider("Year built", min_value=1800,
                       max_value=2026, value=2000, step=1)
total_bsmt_sf = st.number_input(
    "Total Basement Sq Ft", min_value=0, max_value=3000, value=800)
# 4. Predict
if st.button("Calculate Estimated Price"):
    # Preprocessed sample ki copy lein
    sample = sample_row.copy()

    st.write(sample.columns.tolist())

    # User values update karein
    if "GrLivArea" in sample.columns:
        sample["GrLivArea"] = area
    if "BedroomAbvGr" in sample.columns:
        sample["BedroomAbvGr"] = bedrooms
    if "OverallQual" in sample.columns:
        sample["OverallQual"] = overall_qual
    if "YearBuilt" in sample.columns:
        sample["YearBuilt"] = year_built
    if "TotalBsmtSF" in sample.columns:
        sample["TotalBsmtSF"] = total_bsmt_sf

    # Prediction
    pred_log = model.predict(sample)
    pred_price = np.expm1(pred_log)[0]

    st.success(f"🏡 Estimated Sale Price: *${pred_price:,.2f}*")
