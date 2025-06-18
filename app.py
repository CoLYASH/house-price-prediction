#  streamlit run app.py
#  Local URL: http://localhost:8501
#   Network URL: http://10.20.86.244:8501

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("house_price_model_optimized.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit App Title
st.title("🏡 House Price Prediction")

st.write("## 🔮 Predict House Price")
# Numeric Inputs
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", min_value=500, max_value=5000, value=1500)
LotArea = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=50000, value=7500)
GarageCars = st.number_input("Garage Capacity (Cars)", min_value=0, max_value=5, value=2)
GarageArea = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=400)
YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000)
YearRemodAdd = st.number_input("Year Remodeled", min_value=1800, max_value=2024, value=2005)
BsmtFinSF1 = st.number_input("Finished Basement Area (sq ft)", min_value=0, max_value=3000, value=500)
TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=5000, value=1000)
BedroomAbvGr = st.number_input("Bedrooms Above Ground", min_value=1, max_value=10, value=3)
FullBath = st.number_input("Full Bathrooms", min_value=1, max_value=5, value=2)
HalfBath = st.number_input("Half Bathrooms", min_value=0, max_value=3, value=1)
TotRmsAbvGrd = st.number_input("Total Rooms Above Ground", min_value=2, max_value=15, value=7)
OverallQual = st.number_input("Overall Quality Rating (1-10)", min_value=1, max_value=10, value=5)
Fireplaces = st.number_input("Number of Fireplaces", min_value=0, max_value=5, value=1)

# Categorical Inputs
MSZoning = st.selectbox("Zoning Classification", ["RL", "RM", "C (all)", "FV", "RH"])
Neighborhood = st.selectbox("Neighborhood", ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel"])

# Predict Button
if st.button("Predict Price 💰"):
    # Creating DataFrame with Correct Feature Order
    input_data = pd.DataFrame(
        [[GrLivArea, LotArea, GarageCars, GarageArea, YearBuilt, YearRemodAdd, BsmtFinSF1, TotalBsmtSF,
          BedroomAbvGr, FullBath, HalfBath, TotRmsAbvGrd, OverallQual, Fireplaces, MSZoning, Neighborhood]],
        columns=['GrLivArea', 'LotArea', 'GarageCars', 'GarageArea', 'YearBuilt', 'YearRemodAdd',
                 'BsmtFinSF1', 'TotalBsmtSF', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotRmsAbvGrd',
                 'OverallQual', 'Fireplaces', 'MSZoning', 'Neighborhood'])

    # Predict
    predicted_price_log = model.predict(input_data)[0]
    predicted_price = np.expm1(predicted_price_log)  # Reverse Log Transform

    st.success(f"🏡 Estimated House Price: **${predicted_price:,.2f}**")

