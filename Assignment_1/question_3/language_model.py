import logging
import math

from nltk.tokenize import sent_tokenize, word_tokenize

from models import (
    AddAlphaSmoothingModel,
    BigramModel,
    InterpolationModel,
    TrigramModel,
    UnigramModel,
)
from util import (
    _generate_ngrams,
    _get_vocab_size,
    get_all_lambdas,
    get_held_out_perplexity,
    unigram_count,
    bigram_count,
    trigram_count,
)


def get_add_alpha_smoothing_model(training_files, alpha=1):
    model = dict()
    bigram_counts = bigram_count(training_files)
    unigram_counts = unigram_count(training_files)
    vocab = set(unigram_counts.keys())
    V = _get_vocab_size(bigram_counts)

    for file in training_files:
        with open(file) as f:
            sents = sent_tokenize(f.read())
            for sent in sents:
                sent = ["<s>"] + word_tokenize(sent) + ["</s>"]
                bigrams = _generate_ngrams(sent, 2)
                for wi_1, wi in bigrams:
                    if wi_1 not in vocab:
                        wi_1 = "<UNK>"
                    if wi not in vocab:
                        wi = "<UNK>"
                    model[wi_1] = model.get(wi_1, dict())
                    model[wi_1][wi] = (bigram_counts[wi_1].get(wi, 0) + alpha) / (
                        unigram_counts.get(wi_1) + (alpha * V)
                    )
    return AddAlphaSmoothingModel(model=model, alpha=alpha, vocab=vocab)


def get_interpolation_model(training_set, held_out_set):
    trigram_model = get_trigram_model(training_set)
    bigram_model = get_bigram_model(training_set)
    unigram_model = get_unigram_model(training_set)
    smallest = math.inf

    lambdas = get_all_lambdas()
    for lambda1, lambda2, lambda3 in lambdas:
        per = get_held_out_perplexity(
            held_out_set, trigram_model, bigram_model, unigram_model,
            lambda1, lambda2, lambda3, trigram_model.vocab
        )
        logging.warning(
            "Lambda1 = {}, Lambda2 = {}, Lambda3 = {}, perplexity = {}".format(
                    lambda1, lambda2, lambda3, per
                )
            )
        if per < smallest:
            smallest = per
            best_lambdas = tuple([lambda1, lambda2, lambda3])

    logging.warning("Best Lambdas")
    logging.warning(
        "Lambda1 = {}, Lambda2 = {}, Lambda3 = {}, perplexity = {}".format(
                *best_lambdas, per
            )
        )
    return InterpolationModel(unigram_model, bigram_model, trigram_model, *best_lambdas)


def get_trigram_model(training_set):
    trigram_model = dict()
    word_counts = trigram_count(training_set)
    unigram_counts = unigram_count(training_set)
    vocab = set(unigram_counts.keys())

    for file in training_set:
        with open(file) as f:
            sents = sent_tokenize(f.read())
            for sent in sents:
                sent = ["<s>"] + ["<s>"] + word_tokenize(sent) + ["</s>"] + ["</s>"]
                trigrams = _generate_ngrams(sent, 3)
                for wi_2, wi_1, wi in trigrams:
                    if wi_1 not in vocab:
                        wi_1 = "<UNK>"
                    if wi_2 not in vocab:
                        wi_2 = "<UNK>"
                    if wi not in vocab:
                        wi = "<UNK>"
                    try:
                        total_count = float(sum(word_counts[(wi_2, wi_1)].values()))
                        trigram_model[(wi_2, wi_1)][wi] = (
                            word_counts[(wi_2, wi_1)].get(wi, 0) / total_count
                        )
                    except KeyError:
                        trigram_model[(wi_2, wi_1)] = trigram_model.get((wi_2, wi_1), dict())
                        trigram_model[(wi_2, wi_1)][wi] = 0

    return TrigramModel(trigram_model, vocab)


def get_bigram_model(training_set):
    bigram_model = dict()
    bigram_counts = bigram_count(training_set)
    unigram_counts = unigram_count(training_set)
    vocab = set(unigram_counts.keys())

    for file in training_set:
        with open(file) as f:
            sents = sent_tokenize(f.read())
            for sent in sents:
                sent = ["<s>"] + word_tokenize(sent) + ["</s>"]
                bigrams = _generate_ngrams(sent, 2)
                for wi_1, wi in bigrams:
                    if wi_1 not in vocab:
                        wi_1 = "<UNK>"
                    if wi not in vocab:
                        wi = "<UNK>"
                    bigram_model[wi_1] = bigram_model.get(wi_1, dict())
                    bigram_model[wi_1][wi] = (
                        (bigram_counts[wi_1].get(wi, 0)) / (unigram_counts.get(wi_1))
                    )
    return BigramModel(bigram_model, vocab)


def get_unigram_model(training_set):
    unigram_model = dict()
    word_counts = unigram_count(training_set)
    vocab = set(word_counts.keys())
    total_count = float(sum(word_counts.values()))

    for file in training_set:
        with open(file) as f:
            sents = sent_tokenize(f.read())
            for sent in sents:
                words = word_tokenize(sent)
                for wi in words:
                    if wi not in vocab:
                        wi = "<UNK>"
                    unigram_model[wi] = word_counts.get(wi, 0) / total_count
    return UnigramModel(unigram_model, vocab)
