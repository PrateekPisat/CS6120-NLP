import os
import re
from collections import defaultdict

import numpy as np
from bs4 import BeautifulSoup
from nltk import pos_tag, bigrams, trigrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from util import (
    _get_head_word_index,
    get_top_features_for_instance,
    report_to_file,
    find_in_list_after_index
)

stop_words = set(stopwords.words('english'))


def get_vocab(tagged_vocab, report=False):
    vocab = set()
    for word, _ in tagged_vocab:
        vocab.add(word)

    if report:
        report_to_file(vocab, header="Bag of Words Features")

    return vocab


def get_tagged_vocab(files, report=False):
    vocab = set()
    for file in files:
        with open(file) as fin:
            content = fin.read()
        # clean the content
        content = re.sub('\n*', '', content)
        # parse content
        soup = BeautifulSoup(content, 'html.parser')
        # iterate through all context elements
        for context in soup.find_all('context'):
            # Tag the context.
            text = pos_tag(word_tokenize(context.get_text().strip()))
            # fetch <HEAD>[*]</HEAD>
            head_words = context.find_all("head")
            # Fetch index of head word.
            index = -1
            for head_word in head_words:
                index = _get_head_word_index(text, head_word.get_text(), start_index=index)
                window = text[index - 3: index] + text[index + 1: index + 4]
                # Add unique words to vocab
                for word_tag in window:
                    vocab.add(word_tag)

    if report:
        report_to_file(vocab, header="Collocation Features")
    return vocab


def get_ngram_vocab(files, report=False):
    vocab = set()
    for file in files:
        # read file
        with open(file) as fin:
            content = fin.read()
        # clean the content
        content = re.sub('\n*', '', content)
        # parse content
        soup = BeautifulSoup(content, 'html.parser')
        # iterate through all context elements
        for context in soup.find_all('context'):
            word_list = word_tokenize(context.get_text().strip())
            # fetch <HEAD>[*]</HEAD>
            head_words = context.find_all("head")
            index = -1
            for head_word in head_words:
                # Fetch index of head word.
                index = find_in_list_after_index(word_list, head_word.get_text(), index)
                # Build window of size +-3, then add bigrams and rigrams of the words in the window.
                head_prefix = word_list[index - 3: index]
                head_suffix = word_list[index + 1: index + 4]
                prefix = (list(bigrams(head_prefix)) + list(trigrams(head_prefix)))
                suffix = list(bigrams(head_suffix)) + list(trigrams(head_suffix))
                window = prefix + suffix
                # Add unique words to vocab
                for word in window:
                    vocab.add(word)

    if report:
        report_to_file(vocab, header="N-Gram Features")
    # return vocab
    return vocab


def get_bag_of_words_features(files, vocab):
    for file in files:
        feature_vectors = defaultdict(lambda: list())
        # read file
        with open(file) as fin:
            content = fin.read()
        # clean the content
        content = re.sub('\n*', '', content)
        # parse content
        soup = BeautifulSoup(content, 'html.parser')
        # iterate through all welt elements
        for welts in soup.find_all('welt'):
            for welt in welts:
                children = welt.findChildren()
                senses = []
                for child in children:
                    # Find all sense_ids.
                    if child.name == "ans":
                        senses += [child.attrs["senseid"]]
                    if child.name == "context":
                        # tokenize the sentence.
                        word_list = word_tokenize(child.get_text())
                        # Fetch the head word.
                        head_words = child.find_all("head")
                        pre_index = -1
                        for head_word in head_words:
                            # Find the head word in the sentence
                            index = find_in_list_after_index(
                                word_list, head_word.get_text(), pre_index
                            )
                            # Build a window of size +-3
                            window = word_list[index - 3: index] + word_list[index + 1: index + 4]
                            # count the occourences of a word in the windwow.
                            window_word_count = defaultdict(lambda: 0)
                            for word in window:
                                window_word_count[word] += 1
                            pre_index = index
                            # Init feature vetor.
                            feature_vector = [0 for word_tag in vocab]
                            # populate feature vector.
                            for index, word in enumerate(vocab):
                                if word in window:
                                    feature_vector[index] = window_word_count[word]
                            for sense in senses:
                                feature_vectors[sense] += [feature_vector]
        # return results
        return feature_vectors


def get_collocation_features(files, vocab):
    for file in files:
        feature_vectors = defaultdict(lambda: list())
        # read file
        with open(file) as fin:
            content = fin.read()
        # clean the content
        content = re.sub('\n*', '', content)
        # parse content
        soup = BeautifulSoup(content, 'html.parser')
        # iterate through all welt elements
        for welts in soup.find_all('welt'):
            for index, welt in enumerate(welts):
                children = welt.findChildren()
                senses = []
                for child in children:
                    if child.name == "ans":
                        senses += [child.attrs["senseid"]]
                    if child.name == "context":
                        # fetch head word.
                        pre_index = -1
                        head_words = child.find_all("head")
                        for head_word in head_words:
                            # pos tag the context.
                            word_list = pos_tag(word_tokenize(child.get_text()))
                            # Find the index of the head word
                            index = _get_head_word_index(word_list, head_word.get_text(), pre_index)
                            # Set up a window of size 3
                            window = word_list[index - 3: index] + word_list[index + 1: index + 4]
                            pre_index = index
                            window_word_count = defaultdict(lambda: 0)
                            # Count (word, tag) tuples in window.
                            for word_tag in window:
                                window_word_count[word_tag] += 1
                            # Init feature vetor.
                            feature_vector = [0 for word_tag in vocab]
                            # Populate feature vector.
                            for index, word_tag in enumerate(vocab):
                                if word_tag in window:
                                    feature_vector[index] = window_word_count[word_tag]
                            for sense in senses:
                                feature_vectors[sense] += [feature_vector]
        # return results.
        return feature_vectors


def get_ngram_features(files, vocab):
    for file in files:
        feature_vectors = defaultdict(lambda: list())
        # read file
        with open(file) as fin:
            content = fin.read()
        # clean the content
        content = re.sub('\n*', '', content)
        # parse content
        soup = BeautifulSoup(content, 'html.parser')
        # iterate through all welt elements
        for welts in soup.find_all('welt'):
            for index, welt in enumerate(welts):
                children = welt.findChildren()
                senses = []
                for child in children:
                    if child.name == "ans":
                        senses += [child.attrs["senseid"]]
                    if child.name == "context":
                        # fetch head word.
                        head_words = child.find_all("head")
                        # pos tag the context.
                        pre_index = -1
                        word_list = word_tokenize(child.get_text())
                        for head_word in head_words:
                            # Find the index of the head word
                            index = find_in_list_after_index(
                                word_list, head_word.get_text(), pre_index
                            )
                            # Set up a window of size 3 and setup bigrams for the window
                            head_prefix = word_list[index - 3: index]
                            head_suffix = word_list[index + 1: index + 4]
                            prefix = (list(bigrams(head_prefix)) + list(trigrams(head_prefix)))
                            suffix = list(bigrams(head_suffix)) + list(trigrams(head_suffix))
                            window = prefix + suffix
                            pre_index = index
                            window_word_count = defaultdict(lambda: 0)
                            for ngram in window:
                                window_word_count[ngram] += 1
                            # Init feature vetor.
                            feature_vector = [0 for ngram in vocab]
                            # Populate feature vector.
                            for index, ngram in enumerate(vocab):
                                if ngram in window:
                                    feature_vector[index] = window_word_count[ngram]
                            for sense in senses:
                                feature_vectors[sense] += [feature_vector]
        # return results.
        return feature_vectors


def get_all_top_features(bow_fvs, colloc_fvs, all_features, report=False):
    # Store top 10 features per instance in a dict.
    top_features_dict = dict()
    # build a combined vector for each instance
    for target in bow_fvs.keys():
        rows = [list(bow + colloc) for bow, colloc in zip(bow_fvs[target], colloc_fvs[target])]
        X = np.array(rows)
        y = np.array([target for _ in range(len(rows))])
        top_features_dict[target] = get_top_features_for_instance(X, y, all_features)

    if report:
        file = ".{sep}{dir}{sep}{file}".format(
            sep=os.sep, dir='reports', file='top_10_features.txt'
        )
        with open(file, 'a') as f:
            f.write("Top 10 Features per instance\n\n")
            for instance in top_features_dict:
                f.write("{}\n".format(instance))
                f.write("\t{}\n".format(str(top_features_dict[instance])))

    # Return Top features per instance.
    return top_features_dict


if __name__ == "__main__":
    # Get dataset
    files = [".{sep}{dir}{sep}{file}".format(sep=os.sep, dir='train', file='wsd_data.xml')]
    # Feature Extraction
    # Collocation features
    vocab_colloc = get_tagged_vocab(files)
    coll_vectors = get_collocation_features(files=files, vocab=vocab_colloc)

    # BOW Features.
    vocab_bow = get_vocab(vocab_colloc)
    bow_feature_vectors = get_bag_of_words_features(files, vocab=vocab_bow)

    # N-Gram features.
    # vocab = get_ngram_vocab(files)
    # ngram_vectors = get_ngram_features(files, vocab)

    # # Feature Selection
    all_features = np.array(list(vocab_bow) + list(vocab_colloc))
    top_features = get_all_top_features(bow_feature_vectors, coll_vectors, all_features, report=True)
