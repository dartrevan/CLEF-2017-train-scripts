from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import norm
from pandas import read_csv
import numpy as np


def cosine_similarities(raw_texts, dictionary_path='dictionary/AmericanDictionary.csv'):
    dictionary = read_csv(dictionary_path, sep=';', error_bad_lines=False)
    documents_per_code = dictionary.groupby('Icd1')['DiagnosisText'].agg(lambda x: ' '.join(x))
    tfidf_mapper = TfidfVectorizer()
    dictionary_vectors = tfidf_mapper.fit_transform(documents_per_code)
    raw_texts_vectors = tfidf_mapper.transform(raw_texts)
    products = raw_texts_vectors.dot(dictionary_vectors.T)
    raw_norms = norm(raw_texts_vectors, axis=1)
    dictionary_norms = norm(dictionary_vectors, axis=1)
    raw_norms[raw_norms == 0] = 1.0
    dictionary_norms[dictionary_norms == 0] = 1.0
    d = products/np.expand_dims(raw_norms, axis=1)
    d /= np.expand_dims(dictionary_norms, axis=0)
    return d
