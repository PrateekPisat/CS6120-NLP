import os

import gensim.models.keyedvectors as word2vec
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from utils import (
    MLE_model,
    get_repeated_unigrams,
    get_repeated_bigrams,
    get_max_sentence_similarity
)

path = os.environ["model_path"]
g_model = word2vec.KeyedVectors.load_word2vec_format(path, binary=True)


def get_non_redundancy_model(train_files, summarries, train_nonredundancy):
    # Train labels
    X = list()
    y = list()
    # generate features
    for file in train_files:
        text = summarries[file]
        n_repreated_unigrams = get_repeated_unigrams(text)
        n_repreated_bigrams = get_repeated_bigrams(text)
        max_sentence_similarity = get_max_sentence_similarity(text, g_model)
        X.append([n_repreated_unigrams, n_repreated_bigrams, max_sentence_similarity])
        y.append(train_nonredundancy[file])
    models = MLE_model(X, y)
    scores = [np.average(cross_val_score(model, X, y, cv=10)) for model in models]
    # return model
    return models[np.argmax(scores)]


def get_non_redundancy_score(model, test_files, summarries, test_nonredundancy):
    # Setup labels
    y_true = list()
    X = list()
    # generate features
    for file in test_files:
        text = summarries[file]
        n_repreated_unigrams = get_repeated_unigrams(text)
        n_repreated_bigrams = get_repeated_bigrams(text)
        readability_score = get_max_sentence_similarity(text, g_model)
        X.append([n_repreated_unigrams, n_repreated_bigrams, readability_score])
        y_true.append(test_nonredundancy[file])
    # make prediction
    y_pred = list(model.predict(np.array(X)))
    # return result
    MSE = mean_squared_error(y_true, y_pred)
    pearson_cor = pearsonr(y_true, y_pred)
    return (MSE, pearson_cor[0])
