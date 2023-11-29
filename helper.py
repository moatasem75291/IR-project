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

stop_words = stopwords.words("english")
stop_words.remove("in")
stop_words.remove("to")
stop_words.remove("where")
positional_index = {}


#####################################################################
#        READ THE DOCUMENTS AND MAKE TOKENIZATION & STEMMING        #
#####################################################################
def read_and_tokenize_documents(directory_path):
    files_name = natsorted(os.listdir(directory_path))
    document_of_terms = []

    for files in files_name:
        with open(os.path.join(directory_path, files), "r") as f:
            document = f.read()
        document_of_terms.append(tokenize_and_stem(document))

    return document_of_terms


def tokenize_and_stem(doc):
    token_docs = word_tokenize(doc)
    prepared_doc = [terms for terms in token_docs if terms not in stop_words]
    return prepared_doc


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
def flatten_documents(documents):
    return [word for doc in documents for word in doc]


def calculate_term_frequency(doc, all_words):
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return list(words_found.values())


def create_term_frequency_dataframe(document_of_terms):
    all_words = flatten_documents(document_of_terms)
    term_freq_data = [
        calculate_term_frequency(doc, all_words) for doc in document_of_terms
    ]

    term_freq = pd.DataFrame(
        term_freq_data,
        index=["doc_" + str(i) for i in range(1, len(document_of_terms) + 1)],
        columns=positional_index.keys(),
    )
    return term_freq.T


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
def calculate_term_frequency_document_frequency(term_freq_df):
    term_freq = term_freq_df.sum(axis=1)
    inverse_doc_freq = np.log10(10 / term_freq.astype(float))

    idf = pd.DataFrame(
        {"term_freq": term_freq, "inverse_doc_freq": inverse_doc_freq},
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

    for term in tokenize_and_stem(query):
        if term not in positional_index:
            return "No matches!!"
        else:
            for key, positions in positional_index[term][1].items():
                term_lists[key - 1].extend(positions)

    matching_positions = [
        f"doc_{pos}"
        for pos, positions in enumerate(term_lists, start=1)
        if len(positions) == len(tokenize_and_stem(query))
    ]

    return f'{", ".join(matching_positions)}'
