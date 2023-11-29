from helper import *


# 1-Read, Tokenization, and Steeming.
docs_directory_path = "docs"
document_of_terms = read_and_tokenize_documents(docs_directory_path)
print(
    f"Terms after tokenization, stemming, and removing stop words: \n\n{document_of_terms}\n\n"
)


# 2-Positional index
positional_index = build_positional_index(document_of_terms)
print(f"Positiona Index: \n\n{positional_index}\n\n")


# 3-Term Frequancy-TF
term_freq_df = create_term_frequency_dataframe(document_of_terms)
print(f"Term Frequency Table: \n\n{term_freq_df}\n\n")


# 4-Weighted TF Table
weighted_term_freq_df = apply_weighted_term_freq_to_df(term_freq_df)
print(f"Weighted Term Frequency Table: \n\n{weighted_term_freq_df}\n\n")


# 5-Inverse Document Frequency Table-IDF
tfdf = calculate_term_frequency_document_frequency(term_freq_df)
print(f"Inverse Document Frequency Table: \n\n{tfdf}\n\n")


# 6-TF.IDF
tfidf = calculate_tf_idf_optimized(term_freq_df, tfdf)
print(f"TF.IDF Table: \n\n{tfidf}\n\n")


# 7-Document_Lengths
document_lengths = calculate_document_lengths(tfidf)
print(f"\n\nLength of Documents:\n\n{document_lengths}\n\n")


# 8-Normalization
normalized_term_freq_idf = normalize_term_freq_idf(tfidf, document_lengths)
print(f"TF.IDF Normalization: \n\n{normalized_term_freq_idf}\n\n")

# 9-Phrase Query
query = input("In the Phrase Query stage Search for: ")
related_docs_PQ_stage = find_matching_positions(query, positional_index)

# save the result for Phrase query
document_lengths = calculate_document_lengths(tfidf)
normalized_term_freq_idf = normalize_term_freq_idf(tfidf, document_lengths)
query_terms = tokenize_and_stem(query)
query_df = create_query_dataframe(query_terms, normalized_term_freq_idf, tfdf)
product_result = calculate_product(query_df, normalized_term_freq_idf)
similarity = calculate_cosine_similarity(product_result)

try:
    query_detailed = query_df.loc[query_terms]
except KeyError:
    print(f"No such query found in the database:{query_terms}")
    exit()

# Write results to a text file
results_content = f"""
Vector Space Model for Query:\n{query_detailed}\n\n
Product Sum:\n{(product_result.sum()).loc[related_docs_PQ_stage.split(", "),]}\n\n
Product (query * matched doc):\n{product_result.loc[query_terms, related_docs_PQ_stage.split(", ")]}\n\n
Similarity:\n{similarity.loc[related_docs_PQ_stage.split(", "),]}\n\n
Query Length:\n{math.sqrt(sum(query_df['idf'] ** 2))}\n"""
results_content += f"\n\nRelated Docs:\n{related_docs_PQ_stage}"

write_to_file("phrase_query_results.txt", results_content)

# 10-Final Result
q = input("Another search for each term: ")
query_terms = tokenize_and_stem(q)
query_df = create_query_dataframe(query_terms, normalized_term_freq_idf, tfdf)
product_result = calculate_product(query_df, normalized_term_freq_idf)
similarity = calculate_cosine_similarity(product_result)
related_docs = get_related_docs(query_df, positional_index)
try:
    query_detailed = query_df.loc[query_terms]
except KeyError:
    print(f"No such query found in the database:{query_terms}")
    exit()

# Write results to a text file
results_content = f"""
Vector Space Model for Query:\n{query_detailed}\n\n
Product Sum:\n{product_result.sum()}\n\n
Product (query * matched doc):\n{product_result}\n\n
Similarity:\n{similarity}\n\n
Query Length:\n{math.sqrt(sum(query_df['idf'] ** 2))}\n\n"""
results_content += (
    f"\n\nRelated Docs:\n{'doc_'+' ,doc_'.join(list(map(str,(related_docs))))}"
)

write_to_file("query_results.txt", results_content)
