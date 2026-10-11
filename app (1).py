import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('liner.sav')

st.title("Delivery Delay Prediction App")
st.write("Enter the delivery details to predict if there will be a delay.")

# Input fields based on feature columns:
# ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
#  'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
#  'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
#  'Warehouse_Processing_Time']

delivery_distance = st.number_input("Delivery Distance", min_value=0.0, value=25.0)
traffic_congestion = st.slider("Traffic Congestion (1-5)", min_value=1, max_value=5, value=3)
weather_condition = st.slider("Weather Condition (1-3)", min_value=1, max_value=3, value=2)
delivery_slot = st.slider("Delivery Slot (1-3)", min_value=1, max_value=3, value=2)
driver_experience = st.number_input("Driver Experience (Years)", min_value=0, max_value=50, value=10)
num_stops = st.number_input("Number of Stops", min_value=0, max_value=20, value=5)
vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=5)
road_condition_score = st.slider("Road Condition Score (1-5)", min_value=1, max_value=5, value=3)
package_weight = st.number_input("Package Weight", min_value=0.0, value=15.0)
fuel_efficiency = st.number_input("Fuel Efficiency", min_value=0.0, value=12.0)
warehouse_processing_time = st.number_input("Warehouse Processing Time (mins)", min_value=0, value=60)

# Prepare input features
features = np.array([[
    delivery_distance, 
    traffic_congestion, 
    weather_condition,
    delivery_slot, 
    driver_experience, 
    num_stops, 
    vehicle_age,
    road_condition_score, 
    package_weight, 
    fuel_efficiency,
    warehouse_processing_time
]])

if st.button("Predict Delay"):
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)
    
    if prediction[0] == 1:
        st.error(f"Prediction: Delayed 🚨 (Probability: {probabilities[0][1]:.2%})")
    else:
        st.success(f"Prediction: On Time ✅ (Probability: {probabilities[0][0]:.2%})")
