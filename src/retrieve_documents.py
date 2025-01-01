import pickle
import pandas as pd
from preprocess import preprocess_text

# Load trained models
with open("../tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("../nearest_neighbor_model.pkl", "rb") as f:
    model = pickle.load(f)

# Function to retrieve similar documents
def retrieve_similar_documents(query, tfidf, model, data, top_n=5):
    """
    Retrieves the names of individuals whose document texts are most similar to the query.
    
    Args:
        query (str): The search query.
        tfidf: The trained TF-IDF vectorizer.
        model: The trained Nearest Neighbors model.
        data (DataFrame): The dataset containing the 'name' and 'text' columns.
        top_n (int): Number of top similar documents to retrieve.
        
    Returns:
        List[Tuple]: A list of tuples containing names and snippets of the similar documents.
    """
    # Preprocess the query
    query = preprocess_text(query)

    # Transform the query into a TF-IDF vector
    query_tfidf = tfidf.transform([query])

    # Get the top-n similar documents
    distances, indices = model.kneighbors(query_tfidf, n_neighbors=top_n)

    # Extract names and snippets for the results
    results = []
    for idx in indices[0]:
        name = data.iloc[idx]['name']
        snippet = data.iloc[idx]['text'][:150]  # First 150 characters of the original text
        results.append((name, snippet))
    
    return results

# Main logic
if __name__ == "__main__":
    import sys

     # Check for query input
    if len(sys.argv) < 2:
        print("Error: Please provide a query.")
        print("Usage: python retrieve_documents.py <query>")
        sys.exit(1)

    # Get the query from command-line arguments
    query = sys.argv[1]

    # Load the dataset
    data = pd.read_csv("../cleaned_wikipedia_data.csv")

    # Retrieve the top similar names and snippets
    results = retrieve_similar_documents(query, tfidf, model, data)

    # Display the results
    print(f"Top people similar to the query '{query}':\n")
    for name, snippet in results:
        print(f"Name: {name}\nSnippet: {snippet}\n")