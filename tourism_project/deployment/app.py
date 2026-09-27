import os
import streamlit as st
import pandas as pd
import joblib

# Load the model saved by the training script (located in the same directory)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase a new Wellness Tourism Package based on their profile and interaction details.
Enter the customer information below to generate a prediction.
""")

st.header("Customer Profile & Interaction Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=100000, value=20000)
    own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15)
    number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    preferred_property_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
    number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=30, value=2)
    num_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)

# Construct DataFrame matching the features expected by the trained pipeline
input_data = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "NumberOfPersonVisiting": num_person,
    "NumberOfFollowups": number_of_followups,
    "PreferredPropertyStar": preferred_property_star,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "MonthlyIncome": monthly_income,
    "TypeofContact": type_of_contact,
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": product_pitched,
    "MaritalStatus": marital_status,
    "Designation": designation
}])

if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"**Customer Likely to Purchase!** (Probability: {prediction_proba:.2%})")
    else:
        st.info(f"**Customer Unlikely to Purchase.** (Probability: {prediction_proba:.2%})")
