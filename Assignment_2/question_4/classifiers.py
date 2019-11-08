import os
from collections import Counter

import gensim.models.keyedvectors as word2vec
import nltk
import numpy as np
import readability
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import spatial
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier

from read_data import read_summarries, read_test_data, read_train_data

nlp = spacy.load("en_core_web_sm")
path = os.environ["model_path"]
g_model = word2vec.KeyedVectors.load_word2vec_format(path, binary=True)


def get_gramaticallity_model(train_files, summarries, train_grammaticality):
    # Train labels
    X = list()
    y = list()
    # generate features
    for file in train_files:
        text = summarries[file]
        n_repreated_unigrams = get_repeated_unigrams(text)
        n_repreated_bigrams = get_repeated_bigrams(text)
        readability_score = get_readability_score(text)
        X.append([n_repreated_unigrams, n_repreated_bigrams, readability_score])
        y.append(train_grammaticality[file])
    # build model
    X = np.array(X)
    y = np.array(y)
    model = MLE_model(X, y)
    # return model
    return model


def get_non_redundancy_model(train_files, summarries, train_grammaticality):
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
        y.append(train_grammaticality[file])
    model = MLE_model(X, y)
    # return model
    return model


def get_gramaticallity_score(model, test_files, summarries, test_grammaticality):
    # Setup labels
    y_true = list()
    X = list()
    # generate features
    for file in test_files:
        text = summarries[file]
        n_repreated_unigrams = get_repeated_unigrams(text)
        n_repreated_bigrams = get_repeated_bigrams(text)
        readability_score = get_readability_score(text)
        X.append([n_repreated_unigrams, n_repreated_bigrams, readability_score])
        y_true.append(test_grammaticality[file])
    # make prediction
    y_pred = list(model.predict(np.array(X)))
    # return result
    MSE = mean_squared_error(y_true, y_pred)
    pearson_cor = pearsonr(y_true, y_pred)
    return (MSE, pearson_cor[0])


def get_non_redundancy_score(model, test_files, summarries, test_grammaticality):
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
        y_true.append(test_grammaticality[file])
    # make prediction
    y_pred = list(model.predict(np.array(X)))
    # return result
    MSE = mean_squared_error(y_true, y_pred)
    pearson_cor = pearsonr(y_true, y_pred)
    return (MSE, pearson_cor[0])


# Helpers


def get_repeated_unigrams(buffer):
    count = 0
    words = word_tokenize(buffer)
    unigram_counts = Counter(words)
    for words in unigram_counts:
        if unigram_counts[words] > 1:
            count += 1
    return count


def get_repeated_bigrams(buffer):
    count = 0
    words = word_tokenize(buffer)
    bigrams = nltk.bigrams(words)
    bigram_counts = Counter(bigrams)
    for words in bigram_counts:
        if bigram_counts[words] > 1:
            count += 1
    return count


def get_google_vector(sent, model):
    # Create vectors
    review_vector = Counter()
    tokens = [token.text for token in nlp(sent)]
    for token in tokens:
        review_vector[token] = model[token] if token in model else model["UNK"]
    # Get total of each embedding
    result_array = sum(list(review_vector.values()))
    # Get average of total embeddings.
    result_array /= len(tokens)
    return list(result_array)


def get_max_sentence_similarity(buffer, g_model):
    data = list()
    similarity = list()

    sents = sent_tokenize(buffer)
    for sent in sents:
        data.append(get_google_vector(sent, g_model))
    if len(data) == 1:
        return 1
    for vec_1 in data:
        for vec_2 in data:
            if vec_1 == vec_2:
                continue
            else:
                similarity.append(1 - spatial.distance.cosine(vec_1, vec_2))
    return np.max(similarity)


def get_readability_score(buffer):
    reading_ease = []
    sents = sent_tokenize(buffer)
    for sent in sents:
        try:
            results = readability.getmeasures(sent, lang='en')
        except ValueError:
            continue
        reading_ease += [results['readability grades']['FleschReadingEase']]
    return np.min(reading_ease)


def MLE_model(X, y):
    # build model
    X = np.array(X)
    y = np.array(y)
    model = MLPClassifier(
        solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(35, 10),
        random_state=1
    )
    model.fit(X, y)
    return model


if __name__ == "__main__":
    # Get Train Data
    train_coherence, train_grammaticality, train_nonredundancy = read_train_data()
    test_coherence, test_grammaticality, test_nonredundancy = read_test_data()
    train_files = list(train_coherence.keys())
    test_files = list(test_coherence.keys())
    summarries = read_summarries()
    # Train Classifiers
    # gramaticallity_model = get_gramaticallity_model(train_files, summarries, train_grammaticality)
    non_redundancy_model = get_non_redundancy_model(train_files, summarries, train_nonredundancy)
    # Test Classifiers
    # MSE, pearson_cor = get_gramaticallity_score(
    #     gramaticallity_model, test_files, summarries, test_grammaticality
    # )
    # print(MSE, pearson_cor)
    MSE, pearson_cor = get_non_redundancy_score(
        non_redundancy_model, test_files, summarries, test_nonredundancy
    )
    print(MSE, pearson_cor)
