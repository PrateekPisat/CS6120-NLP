import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import get_corpus


def get_tf_idf_vectors(pos_training_files, neg_training_files, test=False):
    #  Seach for cached model
    cached_model_path = ".{sep}models{sep}model_tf_idf.csv".format(sep=os.sep)
    if os.path.exists(cached_model_path) and not test:
        df = pd.read_csv(cached_model_path)
    else:
        df = get_tfidf_dataframe(pos_training_files, neg_training_files)
        # Dont save if we are testing.
        if not test:
            # Dump datagrame as csv
            df.to_csv(cached_model_path, index=False)
    return df


def get_tfidf_dataframe(pos_training_files, neg_training_files):
    # Build corpus
    corpus = get_corpus(pos_training_files, neg_training_files)
    # setup vectorizer
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    # Get vecotrs as numpy arrays.
    X = X.toarray()
    # get tragets
    y = [0 for file in pos_training_files] + [1 for file in neg_training_files]
    # build dataframe
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    df = pd.concat([df_x, df_y], axis=1)
    return df
