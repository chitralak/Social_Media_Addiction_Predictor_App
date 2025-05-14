import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
st.title("Social Media Addiction Predictor App")
st.write(" " "This app predicts whether a student is addicted to social media based on various factors like age, usage hours, and mental health score.Enter the details below to check your addiction score""")

#importing the trained model
model=joblib.load('social_media_addiction.pkl')
platform_encoder = joblib.load('platform_encoder.pkl')
platform_options = list(platform_encoder.classes_)
academic_encoder = joblib.load('academic_encoder.pkl')
academic_options = list(academic_encoder.classes_)
affect_encoder = joblib.load('affect_encoder.pkl')
affect_options = list(affect_encoder.classes_)

# get user input
age= st.number_input("Enter your Age")
academic_level=st.selectbox("Select your academic level",academic_options)
affect_academic=st.selectbox("Affects your academic performance",affect_options)
usage_hours = st.slider("How many hours do you spend on social media daily?", 0, 12, 4)
sleep_hours = st.slider("How many hours do you sleep per night?", 0, 12, 7)
platform = st.selectbox("Which social media platform do you use the most?",platform_options)

#show input for verification
st.write("### User Input:")
st.write("Age: {}".format(age))
st.write("Academic level :{}".format(academic_level))
st.write("affect_academic:{}".format(affect_academic))
st.write("Social Media Usage: {} hours/day".format(usage_hours))
st.write("Sleep Hours: {}".format(sleep_hours))
st.write("Most Used Platform: {}".format(platform))


platform_encoded = platform_encoder.transform([platform])[0]
academic_encoded = academic_encoder.transform([academic_level])[0]
affect_encoded = affect_encoder.transform([affect_academic])[0]

user_input=[[age,academic_encoded,affect_encoded,usage_hours,sleep_hours,platform_encoded]]
prediction=model.predict(user_input)
if prediction[0]==1:
    st.write("oh no ! You may be at risk for social media addiction!")
else:
    st.write("Good You're not  at risk for social media addiction!")


