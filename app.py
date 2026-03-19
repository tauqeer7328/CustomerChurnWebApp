import streamlit as st
import pandas as pd
import joblib
import sklearn
import numpy

# Load model and feature list
model = joblib.load("CustomerChurnPrediction.pkl")
feature_columns = joblib.load("customerchurn.pkl")

st.title("Customer Churn Prediction")

# -------- RAW INPUTS (ALL ORIGINAL COLUMNS) --------

CustomerID = st.number_input("CustomerID", value=1)
Tenure = st.number_input("Tenure", value=1)
CityTier = st.selectbox("CityTier", [1,2,3])
WarehouseToHome = st.number_input("WarehouseToHome", value=10)
HourSpendOnApp = st.number_input("HourSpendOnApp", value=2)
NumberOfDeviceRegistered = st.number_input("NumberOfDeviceRegistered", value=1)
SatisfactionScore = st.slider("SatisfactionScore",1,5,3)
NumberOfAddress = st.number_input("NumberOfAddress", value=1)
Complain = st.selectbox("Complain",[0,1])
OrderAmountHikeFromlastYear = st.number_input("OrderAmountHikeFromlastYear", value=10)
CouponUsed = st.number_input("CouponUsed", value=1)
OrderCount = st.number_input("OrderCount", value=1)
DaySinceLastOrder = st.number_input("DaySinceLastOrder", value=5)
CashbackAmount = st.number_input("CashbackAmount", value=5)

PreferredLoginDevice = st.selectbox("PreferredLoginDevice",
                                    ["Mobile Phone","Phone","Computer"])
PreferredPaymentMode = st.selectbox("PreferredPaymentMode",
                                    ["Debit Card","Credit Card","UPI","Cash on Delivery"])
Gender = st.selectbox("Gender",["Male","Female"])
PreferedOrderCat = st.selectbox("PreferedOrderCat",
                                ["Laptop & Accessory","Mobile","Fashion","Grocery"])
MaritalStatus = st.selectbox("MaritalStatus",
                             ["Single","Married","Divorced"])

# -------- Prediction --------

if st.button("Predict"):

    raw_data = pd.DataFrame({
        'CustomerID':[CustomerID],
        'Tenure':[Tenure],
        'CityTier':[CityTier],
        'WarehouseToHome':[WarehouseToHome],
        'HourSpendOnApp':[HourSpendOnApp],
        'NumberOfDeviceRegistered':[NumberOfDeviceRegistered],
        'SatisfactionScore':[SatisfactionScore],
        'NumberOfAddress':[NumberOfAddress],
        'Complain':[Complain],
        'OrderAmountHikeFromlastYear':[OrderAmountHikeFromlastYear],
        'CouponUsed':[CouponUsed],
        'OrderCount':[OrderCount],
        'DaySinceLastOrder':[DaySinceLastOrder],
        'CashbackAmount':[CashbackAmount],
        'PreferredLoginDevice':[PreferredLoginDevice],
        'PreferredPaymentMode':[PreferredPaymentMode],
        'Gender':[Gender],
        'PreferedOrderCat':[PreferedOrderCat],
        'MaritalStatus':[MaritalStatus]
    })

    # Apply one-hot encoding without drop_first to match training data encoding
    encoded = pd.get_dummies(raw_data, drop_first=False)

    # Align features with training columns; missing columns are filled with 0
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(encoded)
    prediction_proba = model.predict_proba(encoded)

    if prediction[0] == 1:
        st.error(f"Customer will Churn ❌ (Confidence: {prediction_proba[0][1]:.2%})")
    else:
        st.success(f"Customer will Stay ✅ (Confidence: {prediction_proba[0][0]:.2%})")