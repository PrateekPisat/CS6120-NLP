import os
from collections import Counter

import gensim.models.keyedvectors as word2vec
import pandas as pd
import spacy

from utils import file_opener

nlp = spacy.load("en_core_web_sm")


def get_google_vectors(pos_training_files, neg_training_files, test=False):
    cached_model_path = ".{sep}models{sep}model_google.csv".format(sep=os.sep)
    if os.path.exists(cached_model_path) and not test:
        df = pd.read_csv(cached_model_path)
    else:
        path = os.environ["model_path"]
        model = word2vec.KeyedVectors.load_word2vec_format(path, binary=True)
        data = list()

        # Create positive vectors
        for file in file_opener(pos_training_files):
            data.append(get_google_vector(file, model, 0, test))

        # Create positive vectors
        for file in file_opener(neg_training_files):
            data.append(get_google_vector(file, model, 1, test))
        # Shuffle data
        df = pd.DataFrame(data)
        # Dont save if we are testing.
        if not test:
            # Dump datagrame as csv
            df.to_csv(cached_model_path, index=False)
    # return results.
    return df


def get_google_vector(file, model, y, test):
    # Create vectors
    review_vector = Counter()
    tokens = [token.text for token in nlp(file.read())]
    for token in tokens:
        review_vector[token] = model[token] if token in model else model["UNK"]
    # Get total of each embedding
    result_array = sum(list(review_vector.values()))
    # Get average of total embeddings.
    result_array /= len(tokens)
    if test:
        return list(result_array)
    else:
        return list(result_array) + [y]
