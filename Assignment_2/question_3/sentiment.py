import os
from collections import Counter, defaultdict
from random import randrange

import gensim.models.keyedvectors as word2vec
import numpy as np
import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

nlp = spacy.load("en_core_web_sm")


def build_vocab(pos_training_files, neg_training_files, threshold=300):
    # Init
    vocab = defaultdict(lambda: 0)
    # Iterate over all positive reviews
    for file in file_opener(pos_training_files):
        vocab = get_term_feequency(file, vocab)
    # Iterateve over all negative reviews
    for file in file_opener(neg_training_files):
        vocab = get_term_feequency(file, vocab)
    # Only consider the top {$threshold} commonly occouring words
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    filtered_vocab = set(sorted_vocab[:threshold])
    # return results
    return filtered_vocab


def get_bow_vectors(pos_training_files, neg_training_files, test=False):
    path = ".{sep}models{sep}model_bow.csv".format(sep=os.sep)
    if os.path.exists(path) and not test:
        df_shuffled = pd.read_csv(path)
    else:
        # Get vocab
        vocab = build_vocab(pos_training_files, neg_training_files)
        # init
        data = list()
        vector = Counter()
        for word in vocab:
            vector[word] = 0
        # Create positive vectors
        for file in file_opener(pos_training_files):
            word_counts = Counter([token.text for token in nlp(file.read())])
            review_vector = Counter()
            for word in vocab:
                review_vector[word] = word_counts[word] + vector[word]
            data += [
                list(review_vector.values()) + [0] if not test else list(review_vector.values())
            ]
        # Create negative vectors
        for file in file_opener(neg_training_files):
            word_counts = Counter([token.text for token in nlp(file.read())])
            review_vector = Counter()
            for word in vocab:
                review_vector[word] = word_counts[word] + vector[word]
            data += [
                list(review_vector.values()) + [1] if not test else list(review_vector.values())
            ]
        # Shuffle data
        df = pd.DataFrame(data, columns=[*vocab, "Target"])
        df_shuffled = df.sample(frac=1)
        # Don't cache test vectors.
        if not test:
            # Dump datagrame as csv
            df_shuffled.to_csv(path, index=False)
    # return results.
    return df_shuffled


def get_google_vectors(pos_training_files, neg_training_files, test=False):
    cached_model_path = ".{sep}models{sep}model_google.csv".format(sep=os.sep)
    if os.path.exists(cached_model_path) and not test:
        df_shuffled = pd.read_csv(cached_model_path)
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
        df_shuffled = df.sample(frac=1)
        # Dont save if we are testing.
        if not test:
            # Dump datagrame as csv
            df_shuffled.to_csv(cached_model_path, index=False)
    # return results.
    return df_shuffled


def get_tf_idf_vectors(pos_training_files, neg_training_files):
    # Build corpus
    corpus = get_corpus(pos_training_files, neg_training_files)
    # setup vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    # get tragets
    y = [0 for file in pos_training_files] + [1 for file in neg_training_files]
    # Use SVD
    svd = TruncatedSVD(n_components=700)
    X = svd.fit_transform(X, y)
    # build dataframe
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    df = pd.concat([df_x, df_y], axis=1)
    return df


def get_MLE_model(
    pos_training_files,
    neg_training_files,
    vector_generator,
    test=False,
    report=False,
):
    # Get training data
    data = vector_generator(pos_training_files, neg_training_files, test=test)
    # Parameters:
    act_fn = ['relu', 'logistic']
    hidden_layer_sizes = [5, 10, 15, 20, 25, 30, 35, 40]
    learning_rate = ["constant", "invscaling", "adaptive"]
    # Get K-fold split(k=10)
    dataset = data.to_numpy()
    splits = cross_validation_split(dataset, folds=10)
    scores = list()
    models = list()
    for split in splits:
        X_train, y_train, X_test, y_test = train_test_split(split)
        for fn in act_fn:
            for rate in learning_rate:
                for size in hidden_layer_sizes:
                    # build model
                    model = MLPClassifier(
                        activation=fn,
                        learning_rate=rate,
                        solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(size, 10),
                        random_state=1
                    )
                    models.append(model.fit(X_train, y_train.ravel()))
                    scores.append(model.score(X_test, y_test.ravel()))

    # Report results
    if report:
        best_model = models[np.argmax(scores)]
        # Retrain on entire training set
        X_train, y_train = np.hsplit(dataset, [len(dataset[0]) - 1])
        best_model.fit(X_train, y_train.ravel())
        # Report to file
        file = ".{sep}reports{sep}Q3.txt".format(sep=os.sep)
        with open(file, "a") as f:
            f.write("\n\n")
            f.write("3.2 \n")
            # Get Accuracy
            f.write(
                "Score for the training data = {}".format(
                    best_model.score(X_train, y_train.ravel())
                )
            )

    return models[np.argmax(scores)]


# helpers


def test_best_model(test_model, test_files):
    # Load google word model.
    model_path = os.environ["model_path"]
    g_model = word2vec.KeyedVectors.load_word2vec_format(model_path, binary=True)
    for file in file_opener(test_files):
        review_vector = np.array(get_google_vector(file, g_model, -1, test=True))
        review_vector = review_vector.reshape(1, -1)
        predict_class = test_model.predict(review_vector)[0]
        if predict_class == 0:
            with open("./reports/pos.txt", "a") as f:
                f.write(file.name)
                f.write("\n")
        else:
            with open("./reports/neg.txt", "a") as f:
                f.write(file.name)
                f.write("\n")


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


def cross_validation_split(dataset, folds=10):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for _ in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(np.array(fold))
    return dataset_split


def get_corpus(pos_training_files, neg_training_files):
    corpus = []
    for file in file_opener(pos_training_files):
        corpus += ["".join(sent_tokenize(file.read()))]
    for file in file_opener(neg_training_files):
        corpus += ["".join(sent_tokenize(file.read()))]
    return corpus


def train_test_split(dataset, split=0.80):
    train = list()
    train_size = split * len(dataset)
    test = list(dataset)
    while len(train) < train_size:
        index = randrange(len(test))
        train.append(test.pop(index))
    train = np.array(train)
    test = np.array(test)
    X_train, y_train = np.hsplit(train, [len(train[0]) - 1])
    X_test, y_test = np.hsplit(test, [len(test[0]) - 1])
    return X_train, y_train, X_test, y_test


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
    files = []
    directory = ".{sep}tests{sep}".format(sep=os.sep)

    for _, __, files in os.walk(directory):
        for f in files:
            files.append(directory + f)
    return files


if __name__ == "__main__":
    pos, neg = get_training_files()
    # test_files = get_test_files()
    model = get_MLE_model(pos, neg, get_google_vectors)
    test_best_model(
        model, ["./tests/4941686.txt", "./tests/1862439071.txt"]
    )
