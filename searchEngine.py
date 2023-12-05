from helper import *


# 1-Read, Tokenization, and Steeming.
docs_directory_path = "docs"
document_of_terms, stemmed_dict = read_and_tokenize_documents(docs_directory_path)
print(
    f"Terms after tokenization, stemming, and removing stop words: \n\n{document_of_terms}\n\n"
)


# 2-Positional index
positional_index = build_positional_index(document_of_terms)
print(f"Positional Index: \n\n{positional_index}\n\n")


# 3-Term Frequancy-TF
term_freq_df = create_term_frequency_dataframe(positional_index, stemmed_dict)
print(f"Term Frequency Table: \n\n{term_freq_df}\n\n")


# 4-Weighted TF Table
weighted_term_freq_df = apply_weighted_term_freq_to_df(term_freq_df)
print(f"Weighted Term Frequency Table: \n\n{weighted_term_freq_df}\n\n")


# 5-Inverse Document Frequency Table-IDF
tfdf = calculate_doc_frequency_doc_inv_frequency(term_freq_df)
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
end_search = ""
while end_search not in ["q", "Q"]:
    query = input("In the Phrase Query stage Search for: ")
    related_docs_PQ_stage = find_matching_positions(query, positional_index)
    query_terms = tokenize(query)
    if related_docs_PQ_stage:
        # save the result for Phrase query
        document_lengths = calculate_document_lengths(tfidf)
        normalized_term_freq_idf = normalize_term_freq_idf(tfidf, document_lengths)
        query_df = create_query_dataframe(query_terms, normalized_term_freq_idf, tfdf)
        product_result = calculate_product(query_df, normalized_term_freq_idf)
        similarity = calculate_cosine_similarity(product_result)
        try:
            query_detailed = query_df.loc[query_terms]
            # Write results to a text file
            results_content = f"""
Vector Space Model for Query:\n{query_detailed}\n\n
Product Sum:\n{(product_result.sum()).loc[related_docs_PQ_stage.split(", "),]}\n\n
Product (query * matched doc):\n{product_result.loc[query_terms, related_docs_PQ_stage.split(", ")]}\n\n
Similarity:\n{similarity.loc[related_docs_PQ_stage.split(", "),]}\n\n
Query Length:\n{math.sqrt(sum(query_df['idf'] ** 2))}\n\n"""
            results_content += f"\n\nRelated Docs:\n{related_docs_PQ_stage}"
            write_to_file("phrase_query_results.txt", results_content)

        except KeyError:
            print(f"No such query found in the database:{query_terms}\nTRry Again")
    else:
        results_content = f"No such query found in the database:{query_terms}"
        print(results_content)
    end_search = input("If you want to EXIT enter q/Q: ")
