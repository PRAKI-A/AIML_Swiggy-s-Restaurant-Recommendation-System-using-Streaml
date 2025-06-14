import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --- Configure page ---
st.set_page_config(page_title="Swiggy Recommender", layout="wide")

# --- Load data with caching for performance ---
@st.cache_data
def load_cleaned_data():
    return pd.read_csv("F:/Swiggy_recommendation/cleaned_data/cleaned_data.csv")

@st.cache_data
def load_encoded_data():
    return pd.read_csv("F:/Swiggy_recommendation/cleaned_data/encoded_data.csv")

@st.cache_data
def load_encoder():
    with open("encoder.pkl", "rb") as f:
        return pickle.load(f)

# Load data
cleaned_df = load_cleaned_data()
encoded_df = load_encoded_data()
encoder = load_encoder()


# --- App Header ---
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🍽️ Swiggy Restaurant Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Input Form ---
st.subheader("🔍 Filter Your Preferences")

col1, col2, col3, col4 = st.columns(4)
city = col1.selectbox("📍 City", sorted(cleaned_df['city'].dropna().unique()))

# Filter cuisine list based on selected city
city_df = cleaned_df[cleaned_df['city'] == city]
cuisine_options = sorted(city_df['cuisine'].dropna().unique())
cuisine = col2.selectbox("🍜 Cuisine", cuisine_options)

rating = col3.slider("⭐ Minimum Rating", 0.0, 5.0, 4.0, 0.1)
cost = col4.number_input("💰 Max Cost (₹)", min_value=50, max_value=2000, value=300)

# --- Recommendation Engine ---
if st.button("✨ Get Recommendations"):
    # Step 1: Filter cleaned_df logically first
    filtered_cleaned_df = cleaned_df[
        (cleaned_df['city'] == city) &
        (cleaned_df['cuisine'].str.contains(cuisine, case=False, na=False)) &
        (cleaned_df['rating'] >= rating) &
        (cleaned_df['cost'] <= cost)
    ].reset_index(drop=True)

    if filtered_cleaned_df.empty:
        st.warning("No restaurants match your exact preferences. Try adjusting your filters.")
    else:
        # Step 2: Match encoded_df rows using index mask
        matching_indices = filtered_cleaned_df.index
        filtered_encoded_df = encoded_df.loc[matching_indices].reset_index(drop=True)

        # Step 3: Create input vector
        user_input = {
            f"city_{city}": 1,
            f"cuisine_{cuisine}": 1,
            "rating": rating,
            "rating_count": 100,
            "cost": cost
        }

        input_vector = pd.DataFrame([user_input])
        for col in encoded_df.columns:
            if col not in input_vector.columns:
                input_vector[col] = 0
        input_vector = input_vector[encoded_df.columns]

        # Step 4: Compute similarity
        similarity = cosine_similarity(input_vector, filtered_encoded_df)[0]
        top_indices = similarity.argsort()[-5:][::-1]
        recommendations = filtered_cleaned_df.iloc[top_indices][['name', 'city', 'cuisine', 'rating', 'cost']]

        # Step 5: Show beautiful restaurant cards
        st.markdown("## 🏆 Top 5 Recommended Restaurants")

        for _, row in recommendations.iterrows():
            with st.container():
                st.markdown("""
                    <div style='
                        background-color: #fff8f0;
                        border: 2px solid #f0c674;
                        padding: 16px;
                        border-radius: 12px;
                        margin-bottom: 10px;
                    '>
                        <h4 style='color: #d45500; margin-bottom: 8px;'>🍽️ {name}</h4>
                        <p style='margin: 4px 0;'><strong>📍 City:</strong> {city}</p>
                        <p style='margin: 4px 0;'><strong>🍜 Cuisine:</strong> {cuisine}</p>
                        <p style='margin: 4px 0;'><strong>⭐ Rating:</strong> {rating:.1f}</p>
                        <p style='margin: 4px 0;'><strong>💰 Cost for Two:</strong> ₹{cost:.0f}</p>
                    </div>
                """.format(
                    name=row['name'],
                    city=row['city'],
                    cuisine=row['cuisine'],
                    rating=row['rating'],
                    cost=row['cost']
                ), unsafe_allow_html=True)

else:
    st.info("Select your preferences and click '✨ Get Recommendations' to see suggestions!")