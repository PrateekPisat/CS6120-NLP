import os
from random import randrange

import numpy as np
import spacy
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import TruncatedSVD

nlp = spacy.load("en_core_web_sm")


def cross_validation_split(dataset, folds=10, use_svd=False):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for _ in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(np.array(fold))
    return train_test_split(dataset_split, use_svd=use_svd)


def train_test_split(cross_valid_splits, use_svd=False):
    data = list()
    data_to_split = list(cross_valid_splits)
    for i in range(len(data_to_split)):
        test = np.array(data_to_split.pop(i))
        train = np.concatenate([*data_to_split])
        X_train, y_train = np.hsplit(train, [len(train[0]) - 1])
        X_test, y_test = np.hsplit(test, [len(test[0]) - 1])
        if use_svd:
            # Use SVD
            svd = TruncatedSVD(n_components=300)
            X_train = svd.fit_transform(X_train, y_train.ravel())
            X_test = svd.fit_transform(X_test, y_test.ravel())
        data.append((X_train, y_train, X_test, y_test))
        data_to_split = list(cross_valid_splits)
    return data


def get_corpus(pos_training_files, neg_training_files):
    corpus = []
    for file in file_opener(pos_training_files):
        corpus += ["".join(sent_tokenize(file.read()))]
    for file in file_opener(neg_training_files):
        corpus += ["".join(sent_tokenize(file.read()))]
    return corpus


def get_term_feequency(file, vocab):
    doc = nlp(file.read())
    for token in doc:
        if not token.is_stop and not token.pos_ == "PUNCT":
            vocab[token.text] += 1
    return vocab


def file_opener(files):
    for file in files:
        with open(file) as f:
            yield f


def get_training_files():
    pos_files = []
    neg_files = []
    pos_directory = ".{sep}{dir}{sep}pos{sep}".format(sep=os.sep, dir="train")
    neg_directory = ".{sep}{dir}{sep}neg{sep}".format(sep=os.sep, dir="train")

    for _, __, files in os.walk(pos_directory):
        for f in files:
            pos_files.append(pos_directory + f)

    for _, __, files in os.walk(neg_directory):
        for f in files:
            neg_files.append(neg_directory + f)

    return pos_files, neg_files


def get_test_files():
    test_files = []
    test_directory = ".{sep}tests{sep}".format(sep=os.sep)

    for _, __, files in os.walk(test_directory):
        for f in files:
            test_files.append(test_directory + f)
    return test_files
