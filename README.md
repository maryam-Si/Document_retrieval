# Document-retrieval
### Project Overview

This project implements a Document Retrieval system that uses a nearest-neighbor algorithm to retrieve Wikipedia pages with similar contextual information. The goal is to build an effective information retrieval system that can find documents based on their semantic similarity to a given query.
----------------------------------------------------------
### Technologies Used

	-	Python: Main programming language.
	-	Scikit-learn: For implementing the nearest-neighbor algorithm and similarity measures.
	-	NLTK : Text preprocessing and natural language processing tasks.
	-	TF-IDF: Text vectorization technique to represent documents numerically.
	-	Cosine Similarity: Used to measure the similarity between documents.
	-	Pandas: For data handling and manipulation.

 --------------------------------------------------------------------------------------------
 ### Features
	-	Document Preprocessing: Tokenization, removal of stop words, stemming, and lemmatization.
	-	Similarity Computation: Cosine similarity measure between input query and Wikipedia documents.
	-	Efficient Retrieval: Retrieval of the most contextually relevant Wikipedia articles using the nearest-neighbor approach.
	-	Scalability: Capable of handling large-scale text data (Wikipedia dataset).
 --------------------------------------------------------------
 ### Installation
bash:
git clone https://github.com/maryam-Si/Document-Retrieval.git
cd Document-Retrieval

pip install -r requirements.txt

### Requirements
	- Python 3.x
	-	Scikit-learn
	-	NLTK
	-	spaCy
	-	Pandas
	-	Matplotlib (Optional, for visualizations)
-------------------------------------------------
 ## Usage

 **Preprocessing Data**: The first step is preprocessing the Wikipedia dataset, which involves cleaning and
 tokenizing the text. python preprocess_data.py
 **Train Model**:After training the model, you can use the nearest-neighbor algorithm to retrieve documents 
 similar to your query. python train_model.py 
 **Retrieve Documents**:After training the model, you can use the nearest-neighbor algorithm to retrieve 
 documents similar to your query.
 python retrieve_documents.py --query "Your search query"

 
