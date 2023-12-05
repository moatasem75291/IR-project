#####################################################################
#           IMPORT LIBRARIES AND SOME VARIABLES                     #
#####################################################################
import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted
import math
import pandas as pd
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words("english")) - set(["in", "to", "where"])
positional_index = {}
stemmed_dict = {}


#####################################################################
#        READ THE DOCUMENTS AND MAKE TOKENIZATION & STEMMING        #
#####################################################################
def read_and_tokenize_documents(directory_path):
    files_name = natsorted(os.listdir(directory_path))
    document_of_terms = []

    for files in files_name:
        with open(os.path.join(directory_path, files), "r") as f:
            document = f.read()
        stemmed_doc = tokenize_and_stem(document, stemmed_dict)
        document_of_terms.append(stemmed_doc)

    return document_of_terms, stemmed_dict


def tokenize_and_stem(doc, stemmed_dict):
    token_docs = word_tokenize(doc)
    prepared_doc = [
        stem(token, stemmed_dict)
        for token in token_docs
        if token.lower() not in stop_words
    ]
    return prepared_doc


def tokenize(doc):
    token_docs = word_tokenize(doc)
    prepared_doc = [token for token in token_docs if token.lower() not in stop_words]
    return prepared_doc


def stem(token, stemmed_dict):
    stem_term = PorterStemmer().stem(token.lower())
    stemmed_dict[stem_term] = token.lower()
    return stem_term


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


def get_key_by_value(dictionary, target_value, default=None):
    """Get the key for a given value in a dictionary."""
    return next(
        (key for key, value in dictionary.items() if value == target_value), default
    )


#####################################################################
#                        TERM-FREQUANCY.TF                          #
#####################################################################
def create_term_frequency_dataframe(positional_index, stemmed_dict):
    term_frequency = {}
    for stem_term, (_, postings) in positional_index.items():
        term = stemmed_dict.get(stem_term, stem_term)
        term_frequency[term] = {}
        for doc_id, positions in postings.items():
            col_name = f"doc_{doc_id}"
            term_frequency[term][col_name] = len(positions)
    df = pd.DataFrame.from_dict(term_frequency, orient="index")
    df = df.fillna(0)

    return df


#####################################################################
#                         WEIGHTED TF TABLE                         #
#####################################################################
def weighted_TF(x):
    return math.log10(x) + 1 if x > 0 else 0


def apply_weighted_term_freq_to_df(df):
    return df.map(weighted_TF)


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
def create_query_dataframe(query_terms, normalized_term_freq_idf, tfdf):
    query_df = pd.DataFrame(index=normalized_term_freq_idf.index)
    query_df["tf"] = [1 if term in query_terms else 0 for term in query_df.index]
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


def find_matching_positions(query, positional_index):
    term_lists = [[] for _ in range(10)]
    query_terms = tokenize(query)
    if query_terms:
        for term in query_terms:
            if term not in list(stemmed_dict.values()):
                return False
            else:
                for key, positions in positional_index[
                    get_key_by_value(stemmed_dict, term)
                ][1].items():
                    term_lists[key - 1].extend(positions)

        matching_positions = [
            f"doc_{pos}"
            for pos, positions in enumerate(term_lists, start=1)
            if len(positions) == len(query_terms)
        ]

        return f'{", ".join(matching_positions)}'
    else:
        return False


# def find_matching_positions(query, positional_index):
#     term_lists = [[] for _ in range(10)]
#     query_terms = tokenize(query)
#     for term in query_terms:
#         if term not in list(stemmed_dict.values()):
#             return False
#         else:
#             for key, positions in positional_index[
#                 get_key_by_value(stemmed_dict, term)
#             ][1].items():
#                 term_lists[key - 1].extend(positions)

#     matching_positions = [
#         f"doc_{pos}"
#         for pos, positions in enumerate(term_lists, start=1)
#         if len(positions) == len(query_terms)
#     ]

#     return f'{", ".join(matching_positions)}'
