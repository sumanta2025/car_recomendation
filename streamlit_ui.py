import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed data and models
df = joblib.load("cars_dataset.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Car Recommendation System", layout="wide")
st.markdown("""
    <h1 style='color:black; text-align:center;'>Car Recommendation Model</h1>
""", unsafe_allow_html=True)

# User inputs
st.markdown("<h2 style='color:black;'>Enter Car Name</h2>", unsafe_allow_html=True)
name = st.text_input("")

st.markdown("<h2 style='color:black;'>Select Fuel Type</h2>", unsafe_allow_html=True)
fuel_type = st.selectbox("", df['Fuel_Type'].unique())

st.markdown("<h2 style='color:black;'>Enter Maximum Price (in INR)</h2>", unsafe_allow_html=True)
price = st.number_input("", min_value=0, step=100000)

st.markdown("<h2 style='color:black;'>Select Transmission Type</h2>", unsafe_allow_html=True)
transmission = st.selectbox("", df['Transmission'].unique())

# Recommend cars
recommend_button = st.markdown("""
    <style>
        div.stButton > button {
            background-color: white;
            color: black;
            font-size: 18px;
            font-weight: bold;
            border: 1px solid black;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

if st.button("Recommend"):
    if not name or not fuel_type or not transmission or price <= 0:
        st.error("Please fill in all fields with valid inputs.")
    else:
        # Combine user inputs into a single string
        user_input = f"{name} {fuel_type} {transmission}"

        # Transform user input using the TF-IDF vectorizer
        user_vector = tfidf.transform([user_input])

        # Compute similarity scores with the dataset
        similarity_scores = cosine_similarity(user_vector, tfidf.transform(df['Combined_Features']))
        df['Similarity'] = similarity_scores[0]

        # Filter cars based on price and sort by similarity
        filtered_df = df[df['price'].str.replace("INR ₹", "").str.replace(",", "").astype(float) <= price]
        top_10_recommendations = filtered_df.sort_values(by='Similarity', ascending=False).head(10)

        # Display recommendations
        for i, row in top_10_recommendations.iterrows():
            st.markdown(f"""
            <div style='background-color:white; padding:10px; margin-bottom:10px; border:1px solid black; border-radius:10px;'>
                <h4 style='color:black;'>Car {i+1}</h4>
                <p style='color:black; font-size:16px;'>Name: {row['Name']}</p>
                <p style='color:black; font-size:16px;'>Location: {row['Location']}</p>
                <p style='color:black; font-size:16px;'>Year: {row['Year']}</p>
                <p style='color:black; font-size:16px;'>Kilometers Driven: {row['Kilometers_Driven']}</p>
                <p style='color:black; font-size:16px;'>Fuel Type: {row['Fuel_Type']}</p>
                <p style='color:black; font-size:16px;'>Transmission: {row['Transmission']}</p>
                <p style='color:black; font-size:16px;'>Owner Type: {row['Owner_Type']}</p>
                <p style='color:black; font-size:16px;'>Mileage: {row['Mileage']}</p>
                <p style='color:black; font-size:16px;'>Engine: {row['Engine']}</p>
                <p style='color:black; font-size:16px;'>Seats: {row['Seats']}</p>
                <p style='color:black; font-size:16px;'>Price: {row['price']}</p>
            </div>
            """, unsafe_allow_html=True)
