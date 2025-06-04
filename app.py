import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned data
df = pd.read_csv("cleaned_data.csv")

# Load encoded data
encoded_df = pd.read_csv("encoded_data.csv")

# 🔧 Clean 'cost' column in original df
df['cost'] = df['cost'].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
df = df.dropna(subset=['cost'])

# 🔧 Ensure encoded_df is numeric and clean
encoded_df = encoded_df.apply(pd.to_numeric, errors='coerce')  # Convert all to numeric
encoded_df = encoded_df.dropna()  # Drop rows with NaNs from conversion

# Load encoder
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Streamlit UI
st.title("🍽️ Swiggy Restaurant Recommendation System")

city = st.selectbox("Select City", sorted(df['city'].dropna().unique()))
cuisine = st.selectbox("Select Cuisine", sorted(df['cuisine'].dropna().unique()))
rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5)
cost = st.slider("Maximum Cost", int(df['cost'].min()), int(df['cost'].max()), 500)

# Prepare input
input_data = pd.DataFrame([["Sample", city, cuisine]], columns=['name', 'city', 'cuisine'])
input_encoded = encoder.transform(input_data)

# Add numerical features
rating_count_dummy = 100  # Placeholder
input_vector = list(input_encoded[0]) + [rating, rating_count_dummy, cost]

# Match input_vector length to encoded_df.shape[1]
if len(input_vector) != encoded_df.shape[1]:
    st.error(f"⚠️ Mismatch: input vector length = {len(input_vector)} vs dataset = {encoded_df.shape[1]}")
else:
    if st.button("Get Recommendations"):
        similarities = cosine_similarity([input_vector], encoded_df)
        top_indices = similarities[0].argsort()[-5:][::-1]
        recommendations = df.iloc[top_indices]
        st.write("### Recommended Restaurants:")
        st.dataframe(recommendations[['name', 'city', 'cuisine', 'rating', 'cost']])
