import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load dataset
data_path = "cars dataset/car_data.csv"
df = pd.read_csv(data_path)

# Preprocessing
df.dropna(inplace=True)  # Drop rows with missing values

# Combine relevant features into a single string for TF-IDF
df['Combined_Features'] = df['Name'] + " " + df['Fuel_Type'] + " " + df['Transmission']

# Apply TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Combined_Features'])

# Save the TF-IDF model and dataset
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(df, "cars_dataset.pkl")
print("TF-IDF model and dataset saved successfully.")