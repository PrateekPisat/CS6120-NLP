import os
from collections import Counter, defaultdict

import pandas as pd
import spacy

from utils import file_opener, get_term_feequency

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
        df = pd.read_csv(path)
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
        # Don't cache test vectors.
        if not test:
            # Dump datagrame as csv
            df.to_csv(path, index=False)
    # return results.
    return df
