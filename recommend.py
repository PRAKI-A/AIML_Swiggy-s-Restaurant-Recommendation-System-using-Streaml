import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load encoded data (features for recommendation)
encoded_df = pd.read_csv("F:/Swiggy_recommendation/cleaned_data/encoded_data.csv")

# Load cleaned data (to show details of recommended restaurants)
original_df = pd.read_csv("F:/Swiggy_recommendation/cleaned_data/cleaned_data.csv")

# Ensure indices match
assert encoded_df.shape[0] == original_df.shape[0], "Mismatch in row count!"

# Example user input (can be dynamic later)
user_input = {
    'city_Bangalore': 1,
    'cuisine_Chinese': 1,
    'rating': 4.0,
    'rating_count': 100,
    'cost': 250
}

# Convert input to DataFrame with all columns (missing = 0)
input_vector = pd.DataFrame([user_input])
for col in encoded_df.columns:
    if col not in input_vector.columns:
        input_vector[col] = 0
input_vector = input_vector[encoded_df.columns]

# Compute cosine similarity
similarities = cosine_similarity(input_vector, encoded_df)[0]

# Get top 5 indices
top_indices = similarities.argsort()[-5:][::-1]

# Show top recommendations
recommendations = original_df.iloc[top_indices]
print("\n✅ Top 5 Restaurant Recommendations:\n")
print(recommendations[['name', 'city', 'cuisine', 'rating', 'cost']])