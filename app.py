import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

X = df[["Height", "Weight", "Eye"]]
X = X.replace(["Brown", "Blue"], [1, 0])

y = df["Species"]

clf = LogisticRegression() 
clf.fit(X, y)


# Title
st.header("Cat & Dog Machine Learning App")

# Input bar 1
height = st.number_input("Enter Height")

# Input bar 2
weight = st.number_input("Enter Weight")

#Gender bar 3
status = st.radio("Select Gender: ", ('Male', 'Female'))
if (status == 'Male'):
	st.success("Male ")
else:
	st.success("Female ")


# Dropdown input
eyes = st.selectbox("Select Eye Colour", ("Blue", "Brown"))

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("clf.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[height, weight, eyes]], 
                     columns = ["Height", "Weight", "Eyes"])
    X = X.replace(["Brown", "Blue"], [1, 0])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    st.text(f"This instance is a {status} {prediction}")


