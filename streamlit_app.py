import streamlit as st
import joblib
import pandas as pd

# Load trained model + preprocessing
model = joblib.load("deployable/model.joblib")
preproc = joblib.load("deployable/preprocessing.joblib")

st.title("🏠 Flat Price Prediction in Prague")

# Sidebar inputs
st.sidebar.header("Enter Apartment Features")

size = st.sidebar.slider("Size (m²)", 20, 200, 50)
rooms = st.sidebar.slider("Rooms", 1, 6, 2)
district = st.sidebar.selectbox("District", ["Prague 1", "Prague 2", "Prague 3", "Other"])
year = st.sidebar.slider("Year Built", 1900, 2025, 2000)

# Convert to dataframe
input_data = pd.DataFrame([{
    "size": size,
    "rooms": rooms,
    "district": district,
    "year": year
}])

# Preprocess + predict
X_proc = preproc.transform(input_data)
prediction = model.predict(X_proc)[0]

st.subheader("💰 Predicted Price")
st.write(f"{prediction:,.0f} CZK")

