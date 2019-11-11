import os

import gensim.models.keyedvectors as word2vec
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

from bow_vectors import get_bow_vectors
from google_vectors import get_google_vector, get_google_vectors
from tf_idf_vetors import get_tf_idf_vectors
from utils import (
    cross_validation_split,
    file_opener,
    get_corpus,
    get_test_files,
    get_training_files,
)


def get_MLE_model(
    pos_training_files,
    neg_training_files,
    vector_generator,
    use_svd=False,
    test=False,
    report=False,
):
    # Get training data
    data = vector_generator(pos_training_files, neg_training_files, test=test)
    # Parameters:
    act_fn = ['relu', 'logistic']
    hidden_layer_sizes = [30, 35, 40]
    # Get K-fold split(k=10)
    dataset = data.to_numpy()
    splits = cross_validation_split(dataset, folds=10, use_svd=use_svd)
    scores = list()
    models = list()
    for fn in act_fn:
        for size in hidden_layer_sizes:
            # build model
            model = MLPClassifier(
                activation=fn,
                solver='lbfgs',
                alpha=1e-5,
                hidden_layer_sizes=(size, 10),
                random_state=1
            )
            score = 0
            for X_train, y_train, X_test, y_test in splits:
                model.fit(X_train, y_train.ravel())
                score += model.score(X_test, y_test.ravel())
                # import pdb; pdb.set_trace()
            models.append(model)
            scores.append(score / len(splits))

    # Report results
    best_model = models[np.argmax(scores)]
    # Retrain on entire training set
    X_train, y_train = np.hsplit(dataset, [len(dataset[0]) - 1])
    best_model.fit(X_train, y_train.ravel())
    score = best_model.score(X_train, y_train.ravel())
    print(score)
    if report:
        # Report to file
        file = ".{sep}reports{sep}Q3.txt".format(sep=os.sep)
        with open(file, "a") as f:
            f.write("\n\n")
            f.write("3.2 \n")
            # Get Accuracy
            f.write("Score for the training data = {}".format(score))

    return models[np.argmax(scores)]


def get_words_for_topics(
    pos_training_files,
    neg_training_files,
    n_topics=5,
    n_words=20,
    report=False
):
    topics_words_dict = dict()
    # Build corpus
    corpus = get_corpus(pos_training_files, neg_training_files)
    # setup vectorizer
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
    data = vectorizer.fit_transform(corpus)
    # Convert to numpy arrays.
    dataset = np.array([np.absolute(x) for x in data.toarray()])
    # split train and test data.
    X, y = np.hsplit(dataset, [len(dataset[0]) - 1])
    # set up SVD model.
    svd = TruncatedSVD(n_components=300)
    # transform dataset
    svd.fit_transform(X, y)
    # get vocab
    dictionary = vectorizer.get_feature_names()
    dictionary = dictionary[:-1]
    # get components
    components = svd.components_.transpose()
    # select the top n components
    top_n, _ = np.hsplit(components, [n_topics])
    # build datafram using top components
    topics = pd.DataFrame(top_n)
    topics["words"] = dictionary
    # Sort by each row and fetch words
    for i in range(n_topics):
        imp_words = list(topics.sort_values(i, ascending=False)["words"][:n_words])
        topics_words_dict[i] = imp_words
    # report to file
    if report:
        path_to_file = ".{sep}reports{sep}Q3.txt".format(sep=os.sep)
        with open(path_to_file, "a") as f:
            f.write("\n\n")
            for topic in topics_words_dict:
                f.write("Topic {}: {}\n".format(topic, str(topics_words_dict[topic])))

    return topics_words_dict


def test_best_model(test_model, test_files):
    # Load google word model.
    model_path = os.environ["model_path"]
    g_model = word2vec.KeyedVectors.load_word2vec_format(model_path, binary=True)
    # Clear files
    with open("./reports/pos.txt", "w") as f:
        f.truncate()
    with open("./reports/neg.txt", "w") as f:
        f.truncate()
    # classify reviews
    for file in file_opener(test_files):
        review_vector = np.array(get_google_vector(file, g_model, -1, test=True))
        review_vector = review_vector.reshape(1, -1)
        predict_class = test_model.predict(review_vector)[0]
        if predict_class == 0:
            with open("./reports/pos.txt", "a") as f:
                f.write(file.name.split(os.sep)[-1])
                f.write("\n")
        else:
            with open("./reports/neg.txt", "a") as f:
                f.write(file.name.split(os.sep)[-1])
                f.write("\n")


if __name__ == "__main__":
    pos, neg = get_training_files()
    test_files = get_test_files()
    # words = get_words_for_topics(pos, neg)
    # print(words)
    # model = get_MLE_model(pos, neg, get_google_vectors)
    model = get_MLE_model(pos, neg, get_bow_vectors)
    # model = get_MLE_model(pos, neg, get_tf_idf_vectors, use_svd=True)
    # test_best_model(model, test_files)
