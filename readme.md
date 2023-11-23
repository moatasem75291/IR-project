# Search Engine Readme

This repository contains a simple search engine implemented in Python, consisting of two main files: `helper.py` and `searchEngine.py`. The search engine provides functionalities such as document tokenization and stemming, building a positional index model, executing phrase queries, calculating term frequency (TF), generating a weighted TF table, computing inverse document frequency (IDF), calculating TF-IDF, implementing cosine similarity, and testing the search engine.

## `helper.py`

### Import Libraries and Variables

The `helper.py` file begins with importing necessary libraries and defining some variables, including stop words, and an initial positional index.

### Read Documents and Tokenization & Stemming

The `read_and_tokenize_documents` function reads documents from a specified directory, tokenizes and stems them using NLTK, and returns a list of prepared documents.

### Positional Index Model

The `build_positional_index` function constructs a positional index model based on the given documents and their terms.

### Phrase Query

The `find_matching_positions` function takes a query and the positional index as input and returns documents that match the query.

### Term-Frequency (TF)

The file includes functions to flatten documents, calculate term frequency, and create a term frequency dataframe.

### Weighted TF Table

The `weighted_TF` function calculates the weighted term frequency, and `apply_weighted_term_freq_to_df` applies this weighting to a dataframe.

### Inverse Document Frequency (IDF)

Functions for calculating term frequency and document frequency, and subsequently IDF are included.

### TF-IDF

The `calculate_tf_idf_optimized` function computes the TF-IDF matrix.

### Cosine Similarity, Document Lengths, Normalization

Functions for calculating cosine similarity between a query and documents, determining document lengths, and normalizing term frequency-IDF are provided.

### Test the Search Engine

The file includes functions to create a query dataframe, calculate the product of query and normalized term frequency-IDF, calculate cosine similarity, identify related documents, and write results to a file.

## `searchEngine.py`

The structure of `searchEngine.py` mirrors that of `helper.py` but encapsulates the search engine functions. It includes the same functionalities for importing libraries, reading documents, building a positional index, executing phrase queries, calculating term frequency, creating a weighted TF table, computing IDF, calculating TF-IDF, implementing cosine similarity, and testing the search engine.

## Usage

To utilize the search engine, follow these steps:

1. Ensure you have the required libraries installed. You can install them using `pip install -r requirements.txt`.

2. Run `helper.py` to import the necessary functions and variables.

3. Run `searchEngine.py` to execute the search engine functionalities and test the system.

4. View the results in the console and check the generated text file.

Feel free to customize the code according to your specific requirements. Happy searching!
