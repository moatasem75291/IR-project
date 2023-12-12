#####################################################################
#           IMPORT LIBRARIES AND SOME VARIABLES                     #
#####################################################################
import numpy as np
import os
from nltk.tokenize import word_tokenize
from natsort import natsorted
import math
import pandas as pd
from nltk.stem import PorterStemmer

positional_index = {}


#####################################################################
#        READ THE DOCUMENTS AND MAKE TOKENIZATION & STEMMING        #
#####################################################################
def read_and_tokenize_documents(directory_path):
    files_name = natsorted(os.listdir(directory_path))
    token_lists = []
    term_lists = []

    for file_name in files_name:
        with open(os.path.join(directory_path, file_name), "r") as f:
            document = f.read()
            tokens, terms = tokenize_and_stem(document)
            token_lists.append(tokens)
            term_lists.append(terms)

    return token_lists, term_lists


def tokenize_and_stem(doc):
    token_docs = word_tokenize(doc)
    prepared_tokens = [token.lower() for token in token_docs]
    stemmed_terms = [stem(token) for token in token_docs]
    return prepared_tokens, stemmed_terms


def stem(token):
    return PorterStemmer().stem(token.lower())


#####################################################################
#                      POSITIONAL INDEX MODEL                       #
#####################################################################
def build_positional_index(document_of_terms):
    for doc_id, document in enumerate(document_of_terms, start=1):
        for position, term in enumerate(document):
            if term not in positional_index:
                positional_index[term] = [0, {}]
            positional_index[term][0] += 1
            positional_index[term][1].setdefault(doc_id, []).append(position)
    return positional_index


#####################################################################
#                        TERM-FREQUANCY.TF                          #
#####################################################################
def create_term_frequency_dataframe(document_of_tokens):
    term_frequency = {}

    for doc_id, document in enumerate(document_of_tokens, start=1):
        for term in document:
            term_frequency.setdefault(term, {}).setdefault(doc_id, 0)
            term_frequency[term][doc_id] += 1

    df = pd.DataFrame.from_dict(term_frequency, orient="index").fillna(0)
    df.columns = [f"doc_{col}" for col in df.columns]

    return df


#####################################################################
#                         WEIGHTED TF TABLE                         #
#####################################################################
def weighted_TF(x):
    return np.log10(x) + 1 if x > 0 else 0


def apply_weighted_term_freq_to_df(term_freq_df):
    return term_freq_df.apply(lambda x: x.apply(weighted_TF))


#####################################################################
#                  INVERSE DOCUMENT FREQUANCY IDF                   #
#####################################################################
def calculate_doc_frequency_doc_inv_frequency(term_freq_df):
    number_of_docs = len(term_freq_df.columns)
    df = term_freq_df.sum(axis=1)
    inverse_doc_freq = np.log10(number_of_docs / df.astype(float))

    idf = pd.DataFrame(
        {"doc_freq": df, "inverse_doc_freq": inverse_doc_freq},
        index=term_freq_df.index,
    )
    return idf


#####################################################################
#                              TF.IDF                               #
#####################################################################
def calculate_tf_idf_optimized(term_freq_df, idf):
    term_freq_inverse_doc_freq = term_freq_df.mul(idf["inverse_doc_freq"], axis=0)
    return term_freq_inverse_doc_freq


#####################################################################
#                         COSINE SIMILARITY                         #
#                         DOCUMENT LENGTHS                          #
#                         NORMALIZATION                             #
#####################################################################
def cosine_similarity(query_vector, document_vectors):
    query_magnitude = np.linalg.norm(query_vector)
    document_magnitudes = np.linalg.norm(document_vectors, axis=1)

    dot_products = np.dot(document_vectors, query_vector)
    cosine_similarities = dot_products / (document_magnitudes * query_magnitude)

    return cosine_similarities


def get_query_vector(query, all_words):
    query_vector = np.zeros(len(all_words))
    query_terms = query.split()

    for term in query_terms:
        if term in all_words:
            query_vector[all_words.index(term)] += 1

    return query_vector


def calculate_document_lengths(term_freq_inve_doc_freq):
    return np.sqrt((term_freq_inve_doc_freq**2).sum(axis=0))


def normalize_term_freq_idf(term_freq_inve_doc_freq, document_lengths):
    return term_freq_inve_doc_freq.div(document_lengths, axis=1, level=1)


#####################################################################
#                     TEST THE SEARCH ENGINE                        #
#                     SAVE RESULT IN TXT FILE                       #
#####################################################################
def create_query_dataframe(
    query_terms, normalized_term_freq_idf, tfdf, positional_index
):
    query_df = pd.DataFrame(index=normalized_term_freq_idf.index)
    query_df["tf"] = [
        1 if term in query_terms else 0 for term in positional_index.keys()
    ]
    query_df["w_tf"] = query_df["tf"].apply(weighted_TF)
    query_df["idf"] = tfdf["inverse_doc_freq"] * query_df["w_tf"]
    query_df["tf_idf"] = query_df["w_tf"] * query_df["idf"]
    query_df["normalized"] = query_df["idf"] / np.sqrt((query_df["idf"] ** 2).sum())
    return query_df


def calculate_product(query_df, normalized_term_freq_idf):
    product = normalized_term_freq_idf.multiply(query_df["normalized"], axis=0)
    return product


def calculate_cosine_similarity(product_result):
    return product_result.sum()


def get_related_docs(query_df, positional_index):
    related_docs = set()

    for term in query_df[query_df["tf"] > 0].index:
        if term in positional_index:
            related_docs.update(positional_index[term][1].keys())

    return list(related_docs)


def write_to_file(file_path, content):
    with open(file_path, "w") as file:
        file.write(content)
    print("Your results are printed :)")


#####################################################################
#                            PHRASE QUERY                           #
#####################################################################
def find_matching_positions(queries, positional_index):
    final_result = []
    for query in queries:
        term_lists = [[] for _ in range(10)]
        _, query_terms = tokenize_and_stem(query)
        if query_terms:
            for term in query_terms:
                if term not in positional_index:
                    return False
                else:
                    for key, positions in positional_index[term][1].items():
                        term_lists[key - 1].extend(positions)

            matching_positions = [
                f"{pos}"
                for pos, positions in enumerate(term_lists, start=1)
                if len(positions) == len(query_terms)
            ]

            final_result.append(matching_positions)
        else:
            return False
    return final_result


def and_boolean_query(returned_matches_docs):
    if not returned_matches_docs or any(
        not doc_set for doc_set in returned_matches_docs
    ):
        print("No matched documents.")
        return None

    temp_result = set(returned_matches_docs[0])

    for doc_set in returned_matches_docs[1:]:
        temp_result = temp_result.intersection(set(doc_set))

    return list(temp_result)


def or_boolean_query(returned_matches_docs):
    if not returned_matches_docs or any(
        not doc_set for doc_set in returned_matches_docs
    ):
        print("No matched documents.")
        return None

    temp_result = set()

    for doc_set in returned_matches_docs:
        temp_result.update(doc_set)

    return list(temp_result)


def complement_boolean_query(first_list, second_list):
    full_set = set(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

    if not first_list:
        raise ("No first list provided.")

    if not second_list:
        raise ("No second list provided.")

    first_set = set(first_list)
    second_set = set(second_list)

    complement_set = first_set.difference(second_set)
    return list(complement_set)


def check_contains_boolean_logic(query):
    query_words = query.lower().split(" ")
    boolean_operator = None

    if "and" in query_words:
        boolean_operator = "and"
    elif "or" in query_words:
        boolean_operator = "or"
    elif "not" in query_words:
        boolean_operator = "not"
    query_words = " ".join(query_words)

    return query_words, boolean_operator


def phrase_query_serach(query, positional_index):
    term_lists = [[] for _ in range(10)]
    _, query_terms = tokenize_and_stem(query)
    if query_terms:
        for term in query_terms:
            if term not in positional_index:
                return False
            else:
                for key, positions in positional_index[term][1].items():
                    term_lists[key - 1].extend(positions)

        matching_positions = [
            f"doc_{pos}"
            for pos, positions in enumerate(term_lists, start=1)
            if len(positions) == len(query_terms)
        ]

        return ", ".join(matching_positions)
    else:
        return False


def build_output_content(query, related_docs_PQ_stage, tfidf, tfdf):
    query_tokens, query_terms = tokenize_and_stem(query)

    if related_docs_PQ_stage:
        # save the result for Phrase query
        document_lengths = calculate_document_lengths(tfidf)
        normalized_term_freq_idf = normalize_term_freq_idf(tfidf, document_lengths)
        query_df = create_query_dataframe(
            query_terms, normalized_term_freq_idf, tfdf, positional_index
        )
        product_result = calculate_product(query_df, normalized_term_freq_idf)
        similarity = calculate_cosine_similarity(product_result)

        try:
            query_detailed = query_df.loc[query_tokens]

            # Write results to a text file
            results_content = f"""
Vector Space Model for Query:\n{query_detailed}\n\n
Product Sum:\n{(product_result.sum()).loc[related_docs_PQ_stage.split(", "),]}\n\n
Product (query * matched doc):\n{product_result.loc[query_tokens, related_docs_PQ_stage.split(", ")]}\n\n
Similarity:\n{similarity.loc[related_docs_PQ_stage.split(", "),]}\n\n
Query Length:\n{math.sqrt(sum(query_df['idf'] ** 2))}\n\n"""
            results_content += f"\n\nRelated Docs:\n{related_docs_PQ_stage}"
            write_to_file("phrase_query_results.txt", results_content)

        except KeyError:
            print(f"No such query found in the database:{query_tokens}\nTry Again.\n")
    else:
        results_content = f"No such query found in the database:{query_terms}"
        print(results_content)
