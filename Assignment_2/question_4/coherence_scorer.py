import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from utils import (
    MLE_model,
    get_repeated_noun_chunks,
    get_coreffered_entities,
)


def get_coherence_model(train_files, summarries, train_coherence):
    # Train labels
    X = list()
    y = list()
    # generate features
    for index, file in enumerate(train_files):
        print(index)
        text = summarries[file]
        n_repreated_noun_chunks = get_repeated_noun_chunks(text)
        n_coreffered_entities = get_coreffered_entities(text)
        X.append([n_repreated_noun_chunks, n_coreffered_entities])
        y.append(train_coherence[file])
    models = MLE_model(X, y)
    scores = [np.average(cross_val_score(model, X, y, cv=10)) for model in models]
    # return model
    return models[np.argmax(scores)]


def get_coherence_score(model, test_files, summarries, test_coherence):
    # Setup labels
    y_true = list()
    X = list()
    # generate features
    for file in test_files:
        text = summarries[file]
        n_repreated_noun_chunks = get_repeated_noun_chunks(text)
        n_coreffered_entities = get_coreffered_entities(text)
        X.append([n_repreated_noun_chunks, n_coreffered_entities])
        y_true.append(test_coherence[file])
    # make prediction
    y_pred = list(model.predict(np.array(X)))
    # return result
    MSE = mean_squared_error(y_true, y_pred)
    pearson_cor = pearsonr(y_true, y_pred)
    return (MSE, pearson_cor[0])
