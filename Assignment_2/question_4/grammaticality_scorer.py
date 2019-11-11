import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from utils import (
    MLE_model,
    get_continous_repated_unigrams,
    get_continous_reated_bigrams,
    get_readability_score
)


def get_grammaticallity_model(train_files, summarries, train_grammaticality):
    # Train labels
    X = list()
    y = list()
    # generate features
    for file in train_files:
        text = summarries[file]
        n_repreated_unigrams = get_continous_repated_unigrams(text)
        n_repreated_bigrams = get_continous_reated_bigrams(text)
        readability_score, grammar_score, n_datelines = get_readability_score(text)
        X.append(
            [
                n_repreated_unigrams,
                n_repreated_bigrams,
                readability_score,
                grammar_score,
                n_datelines
            ]
        )
        y.append(train_grammaticality[file])
    # build model
    X = np.array(X)
    y = np.array(y)
    models = MLE_model(X, y)
    scores = [np.average(cross_val_score(model, X, y, cv=10)) for model in models]
    # return model
    return models[np.argmax(scores)]


def get_grammaticallity_score(model, test_files, summarries, test_grammaticality):
    # Setup labels
    y_true = list()
    X = list()
    # generate features
    for file in test_files:
        text = summarries[file]
        n_repreated_unigrams = get_continous_repated_unigrams(text)
        n_repreated_bigrams = get_continous_reated_bigrams(text)
        readability_score, grammar_score, n_datelines = get_readability_score(text)
        X.append(
            [
                n_repreated_unigrams,
                n_repreated_bigrams,
                readability_score,
                grammar_score,
                n_datelines
            ]
        )
        y_true.append(test_grammaticality[file])
    # make prediction
    y_pred = list(model.predict(np.array(X)))
    # return result
    MSE = mean_squared_error(y_true, y_pred)
    pearson_cor = pearsonr(y_true, y_pred)
    return (MSE, pearson_cor[0])
