import streamlit as st
import numpy as np
import joblib
import pickle
import pandas as pd


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Car Price Predictor 🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------
# Banner Image (Background Feel)
# -------------------------------
st.image(
    "https://images.unsplash.com/photo-1503376780353-7e6692767b70",
    use_container_width=True
)


# -------------------------------
# Sidebar Panel
# -------------------------------
st.sidebar.title("🚗 Car Price Predictor")

st.sidebar.info(
"""
Enter car specifications  
Click Predict Price  
Get instant estimate
"""
)

st.sidebar.success("Model Used: Lasso Regression")


# -------------------------------
# Load Model & Scaler
# -------------------------------
model = pickle.load(open("lasso_model.pkl", "rb"))
scaler = joblib.load("scaler.pkl")

feature_names = scaler.feature_names_in_


# -------------------------------
# Main Header
# -------------------------------
st.header("🚗 Car Price Prediction Dashboard")
st.divider()


# -------------------------------
# Numeric Inputs Section
# -------------------------------
st.subheader("🔢 Enter Numeric Car Details")

col1, col2 = st.columns(2, gap="large")

with col1:
    symboling = st.number_input("Symboling", value=0)
    wheelbase = st.number_input("Wheel Base", value=100.0)
    carlength = st.number_input("Car Length", value=150.0)
    carwidth = st.number_input("Car Width", value=60.0)

with col2:
    curbweight = st.number_input("Curb Weight", value=2000.0)
    enginesize = st.number_input("Engine Size", value=120.0)
    horsepower = st.number_input("Horse Power", value=100.0)
    citympg = st.number_input("City MPG", value=25.0)


st.divider()


# -------------------------------
# Categorical Inputs Section
# -------------------------------
st.subheader("⚙️ Select Car Features")

col3, col4 = st.columns(2, gap="large")

with col3:

    carbody = st.selectbox(
        "Car Body",
        ["convertible", "hardtop", "hatchback", "sedan", "wagon"]
    )

    drivewheel = st.selectbox(
        "Drive Wheel",
        ["4wd", "fwd", "rwd"]
    )

    enginelocation = st.selectbox(
        "Engine Location",
        ["front", "rear"]
    )


with col4:

    enginetype = st.selectbox(
        "Engine Type",
        ["dohc", "dohcv", "l", "ohc", "ohcf", "ohcv", "rotor"]
    )

    cylindernumber = st.selectbox(
        "Cylinder Number",
        ["two", "three", "four", "five", "six", "eight", "twelve"]
    )


# -------------------------------
# Prepare Input Data
# -------------------------------
def prepare_input():

    input_data = dict.fromkeys(feature_names, 0)

    numeric_inputs = {
        'symboling': symboling,
        'wheelbase': wheelbase,
        'carlength': carlength,
        'carwidth': carwidth,
        'curbweight': curbweight,
        'enginesize': enginesize,
        'horsepower': horsepower,
        'citympg': citympg
    }

    for key in numeric_inputs:
        if key in input_data:
            input_data[key] = numeric_inputs[key]


    categorical_features = [

        f'carbody_{carbody}',
        f'drivewheel_{drivewheel}',
        f'enginelocation_{enginelocation}',
        f'enginetype_{enginetype}',
        f'cylindernumber_{cylindernumber}'
    ]


    for feature in categorical_features:
        if feature in input_data:
            input_data[feature] = 1


    return pd.DataFrame([input_data])


# -------------------------------
# Prediction Button
# -------------------------------
st.divider()

if st.button("🔮 Predict Price", use_container_width=True):

    with st.spinner("Predicting car price... 🚀"):

        try:

            data = prepare_input()
            scaled = scaler.transform(data)
            prediction = model.predict(scaled)

            price = round(prediction[0] * 84, 2)

            st.metric(
                label="💰 Estimated Car Price",
                value=f"₹ {price}"
            )

        except Exception as e:

            st.error(f"Error: {e}")