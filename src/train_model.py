import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
from preprocess import preprocess_text

#This script is used to train the models (TF-IDF and nearest neighbors). It loads the data, applies preprocessing, fits the models, and saves them.

# Load data
data = pd.read_csv("people_wiki.csv")
data['text'] = data['text'].fillna('')

# Preprocess text
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Save cleaned data
data.to_csv("cleaned_wikipedia_data.csv", index=False)

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['cleaned_text'])

# Save the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Train Nearest Neighbor Model
model = NearestNeighbors(n_neighbors=5, metric='cosine')
model.fit(tfidf_matrix)

# Save the Nearest Neighbor model
with open("nearest_neighbor_model.pkl", "wb") as f:
    pickle.dump(model, f)