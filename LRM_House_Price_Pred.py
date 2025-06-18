import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data
train_data = pd.read_csv(r'./data/train.csv')
test_data = pd.read_csv(r'./data/test.csv')

# Selecting Features (Added More Features)
num_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotRmsAbvGrd',
                'OverallQual', 'YearBuilt', 'GarageCars', 'TotalBsmtSF', 'LotArea',
                'YearRemodAdd', 'Fireplaces', 'GarageArea', 'BsmtFinSF1']

cat_features = ['MSZoning', 'Neighborhood']

# Target Variable (Log Transform)
y = np.log1p(train_data['SalePrice'])

# Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# Train-Test Split
X = train_data[num_features + cat_features]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Pipeline with XGBoost
xgb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42))
])

# Train Model
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_log = xgb_model.predict(X_val)
y_pred = np.expm1(y_pred_log)  # Reverse log transformation

# Compute Metrics
mean_abs_err = mean_absolute_error(np.expm1(y_val), y_pred)
rmse = np.sqrt(mean_squared_error(np.expm1(y_val), y_pred))
r2_scr = r2_score(np.expm1(y_val), y_pred)

# Display Performance Metrics
st.subheader("📊 Optimized Model Performance Metrics")
st.write(f"**Mean Absolute Error (MAE):** ${mean_abs_err:,.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
st.write(f"**R² Score:** {r2_scr:.3f}")

# Save the optimized model
import pickle
with open("house_price_model_optimized.pkl", "wb") as file:
    pickle.dump(xgb_model, file)

print("✅ Optimized Model trained and saved successfully!")

# streamlit run "C:\Users\Yash Dhoney\PycharmProjects\pythonProject\ML project\LRM_House_Price_Pred.py"
