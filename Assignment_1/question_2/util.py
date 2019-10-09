import os

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def report_to_file(vocab, header=None):
    with open('.' + os.sep + 'reports' + os.sep + 'featuresForWSD.txt', 'a') as f:
        f.write("{header}:\n\n".format(header=header))
        [f.write("{feature}\n".format(feature=feature)) for feature in vocab]


def _get_head_word_index(word_list, head_word, start_index):
    for index, word_tag in enumerate(word_list):
        word, _ = word_tag
        if head_word == word and index > start_index:
            return index


def get_top_features_for_instance(X, y, all_features):
    k_best = SelectKBest(score_func=chi2, k=10)
    # fit on train set
    k_best.fit(X, y)

    # model = LogisticRegression()
    # rfe = RFE(model, 10)
    # fit = rfe.fit(X, y)

    feature_indices = k_best.get_support(indices=True)

    return list(all_features[feature_indices])


def find_in_list_after_index(word_list, head_word, start_index):
    for index, word in enumerate(word_list):
        if word == head_word and index > start_index:
            return index