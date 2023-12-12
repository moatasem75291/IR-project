from helper import *


# 1-Read, Tokenization, and Steeming.
docs_directory_path = "docs"
document_of_tokens, document_of_terms = read_and_tokenize_documents(docs_directory_path)
print(f"Tokens after tokenization: \n\n{document_of_tokens}\n\n")
print(f"Terms after stemming: \n\n{document_of_terms}\n\n")


# 2-Positional index
positional_index = build_positional_index(document_of_terms)
print(f"Positional Index: \n\n{positional_index}\n\n")


# 3-Term Frequancy-TF
term_freq_df = create_term_frequency_dataframe(document_of_tokens)
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
    query, boolean_operator = check_contains_boolean_logic(query)
    if boolean_operator == "and":
        query = query.split(" and ")
        returned_matches_doc = find_matching_positions(query, positional_index)
        if returned_matches_doc:
            returned_docs = and_boolean_query(returned_matches_doc)
            returned_docs = "doc_" + ", doc_".join(returned_docs)
            query = " ".join(query)
            build_output_content(query, returned_docs, tfidf, tfdf)

    elif boolean_operator == "or":
        query = query.split(" or ")
        returned_matches_doc = find_matching_positions(query, positional_index)
        if returned_matches_doc:
            returned_docs = or_boolean_query(returned_matches_doc)
            returned_docs = "doc_" + ", doc_".join(returned_docs)
            query = " ".join(query)
            build_output_content(query, returned_docs, tfidf, tfdf)
    elif boolean_operator == "not":
        query = query.split(" not ")
        returned_matches_doc = find_matching_positions(query, positional_index)
        print(returned_matches_doc)
        returned_docs = complement_boolean_query(
            returned_matches_doc[0], returned_matches_doc[1]
        )
        returned_docs = "doc_" + ", doc_".join(returned_docs)
        query = " ".join(query)
        build_output_content(query, returned_docs)

    else:
        returned_matches_doc = phrase_query_serach(query, positional_index)
        if returned_matches_doc:
            build_output_content(query, returned_matches_doc, tfidf, tfdf)

    end_search = input("If you want to EXIT enter q/Q: ")
