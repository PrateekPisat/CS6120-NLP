import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize


def _get_bigram_count(words, unk_words, count_dict):
    bigrams = _generate_ngrams(words, 2)
    for wi_1, wi in bigrams:
        if wi_1 in unk_words:
            wi_1 = "<UNK>"
        if wi in unk_words:
            wi = "<UNK>"
        count_dict[wi_1] = count_dict.get(wi_1, dict())
        count_dict[wi_1][wi] = count_dict[wi_1].get(wi, 0) + 1
    return count_dict


def _get_trigram_count(words, unk_words, count_dict):
    trigrams = _generate_ngrams(words, 3)
    for wi_2, wi_1, wi in trigrams:
        if wi_1 in unk_words:
            wi_1 = "<UNK>"
        if wi_2 in unk_words:
            wi_2 = "<UNK>"
        if wi in unk_words:
            wi = "<UNK>"
        count_dict[(wi_1, wi_2)] = count_dict.get((wi_1, wi_2), dict())
        count_dict[(wi_1, wi_2)][wi] = count_dict[(wi_1, wi_2)].get(wi, 0) + 1
    return count_dict


def _generate_ngrams(words_list, n):
    ngrams_list = []
    for num in range(0, len(words_list) - (n - 1)):
        ngram = tuple(words_list[num: num + n])
        ngrams_list.append(ngram)
    return ngrams_list


def _get_vocab_size(count_dict):
    unique_words = set()
    for wi_1 in count_dict:
        for wi in list(count_dict[wi_1].keys()):
            unique_words.add(tuple([*wi_1, wi]))
    return len(unique_words)


def get_total_ngrams(count_dict):
    total_words = 0
    for wi_1 in count_dict:
        for wi in list(count_dict[wi_1].keys()):
            total_words += count_dict[wi_1][wi]
    return total_words


def get_total_ngram_count(buffer, ngram_count):
    count = 0
    sents = sent_tokenize(buffer)
    front_pad = []
    back_pad = []
    for _ in range(ngram_count - 1):
        front_pad += ["<s>"]
        back_pad += ["</s>"]

    for sent in sents:
        words = front_pad + word_tokenize(sent) + back_pad
        ngrams = _generate_ngrams(words, ngram_count)
        count += len(ngrams)

    return count


def get_unknown_words(files):
    """Return a set of words that occour less than 4 times."""
    count_dict = dict()

    for file in files:
        with open(file, "r") as f:
            sents = sent_tokenize(f.read())
            for sent in sents:
                words = word_tokenize(sent)
                for word in words:
                    count_dict[word] = count_dict.get(word, 0) + 1

    unk_words = set(dict(filter(lambda elem: elem[1] <= 3, count_dict.items())).keys())
    return unk_words


def unigram_count(files):
    """Return a dict of frequnency counts for unigrams."""
    count_dict = dict()
    unk_words = get_unknown_words(files)

    for file in files:
        with open(file, "r") as f:
            for sent in sent_tokenize(f.read()):
                for word in word_tokenize(sent):
                    if word in unk_words:
                        word = "<UNK>"
                    count_dict[word] = count_dict.get(word, 0) + 1
    return count_dict


def bigram_count(files):
    """Return a dict of frequency counts for all unique bigrams."""
    count_dict = dict()
    unk_words = get_unknown_words(files)

    for file in files:
        with open(file, "r") as f:
            unformatted_sents = sent_tokenize(f.read())
            sents = unformatted_sents
            for sent in sents:
                words = ["<s>"] + word_tokenize(sent) + ["</s>"]
                count_dict = _get_bigram_count(words, unk_words, count_dict)
    return count_dict


def trigram_count(files):
    """Return a dict of frequency counts for all unique bigrams."""
    count_dict = dict()
    unk_words = get_unknown_words(files)

    for file in files:
        with open(file, "r") as f:
            unformatted_sents = sent_tokenize(f.read())
            sents = unformatted_sents
            for sent in sents:
                words = ["<s>"] + ["<s>"] + word_tokenize(sent) + ["</s>"] + ["</s>"]
                count_dict = _get_trigram_count(words, unk_words, count_dict)
    return count_dict


def get_perplexity_for_file(sents, model, total_ngram_count):
    log_prob = 0
    smallest_prob = _get_smallest_prob(model.model)

    for sent in sents:
        words = ["<s>"] + word_tokenize(sent) + ["</s>"]
        bigrams = _generate_ngrams(words, 2)
        for wi_1, wi in bigrams:
            if wi not in model.vocab:
                wi = "<UNK>"
            if wi_1 not in model.vocab:
                wi_1 = "<UNK>"
            try:
                prob = model.model[wi_1][wi]
            except KeyError:
                prob = smallest_prob
            log_prob += np.log2(prob)

    perplexity = np.power(2, -(log_prob / total_ngram_count))
    return perplexity


def get_interpolation_perplexity_for_file(sents, model, total_ngram_count):
    log_prob = 0
    smallest_prob_trigram = _get_smallest_prob(model.trigram_model.model)
    smallest_prob_bigram = _get_smallest_prob(model.bigram_model.model)
    smallest_prob_unigram = _get_smallest_unigram_prob(model.unigram_model.model)

    for sent in sents:
        words = ["<s>"] + ["<s>"] + word_tokenize(sent) + ["</s>"] + ["</s>"]
        trigrams = _generate_ngrams(words, 3)
        for wi_2, wi_1, wi in trigrams:
            if wi not in model.vocab:
                wi = "<UNK>"
            if wi_1 not in model.vocab:
                wi_1 = "<UNK>"
            if wi_2 not in model.vocab:
                wi_2 = "<UNK>"

            # Dealing with OOV words
            try:
                trigram_prob = model.trigram_model.model[(wi_2, wi_1)][wi]
            except KeyError:
                trigram_prob = smallest_prob_trigram
            try:
                bigram_prob = model.bigram_model.model[wi_1][wi]
            except KeyError:
                bigram_prob = smallest_prob_bigram
            try:
                unigram_prob = model.unigram_model.model[wi]
            except KeyError:
                unigram_prob = smallest_prob_unigram

            prob = (
                model.lambda1 * trigram_prob + model.lambda2 * bigram_prob + model.lambda3 * unigram_prob
            )
            log_prob += np.log2(prob)
    perplexity = np.power(2, -(log_prob / total_ngram_count))
    return perplexity


def get_held_out_perplexity(
    held_out_set,
    trigram_model,
    bigram_model,
    unigram_model,
    lambda1,
    lambda2,
    lambda3,
    vocab,
):
    trigram_counts = trigram_count(held_out_set)
    smallest_prob_trigram = _get_smallest_prob(trigram_model.model)
    smallest_prob_bigram = _get_smallest_prob(bigram_model.model)
    smallest_prob_unigram = _get_smallest_unigram_prob(unigram_model.model)
    log_prob = 0

    for file in held_out_set:
        with open(file) as f:
            sents = sent_tokenize(f.read())
            for sent in sents:
                words = ["<s>"] + ["<s>"] + word_tokenize(sent) + ["<s>"] + ["<s>"]
                trigrams = _generate_ngrams(words, 3)
                for wi_2, wi_1, wi in trigrams:
                    if wi not in vocab:
                        wi = "<UNK>"
                    if wi_1 not in vocab:
                        wi_1 = "<UNK>"
                    if wi_2 not in vocab:
                        wi_2 = "<UNK>"
                    log_prob += get_log_prob(
                        trigram_counts, unigram_model, bigram_model, trigram_model,
                        wi, wi_1, wi_2, lambda1, lambda2, lambda3,
                        smallest_prob_trigram, smallest_prob_bigram, smallest_prob_unigram,
                    )

    return log_prob


def _get_smallest_prob(model):
    smallest_prob = 1
    for w1 in model:
        for w2 in model[w1]:
            if model[w1][w2] < smallest_prob:
                smallest_prob = model[w1][w2]
    return smallest_prob


def _get_smallest_unigram_prob(model):
    smallest_prob = 1
    for w1 in model:
        if model[w1] < smallest_prob:
            smallest_prob = model[w1]
    return smallest_prob


def get_log_prob(
    trigram_counts,
    unigram_model,
    bigram_model,
    trigram_model,
    wi,
    wi_1,
    wi_2,
    lambda1,
    lambda2,
    lambda3,
    smallest_prob_trigram,
    smallest_prob_bigram,
    smallest_prob_unigram,
):
    try:
        trigram_prob = trigram_model.model[(wi_2, wi_1)][wi]
    except KeyError:
        trigram_prob = smallest_prob_trigram
    try:
        bigram_prob = bigram_model.model[wi_1][wi]
    except KeyError:
        bigram_prob = smallest_prob_bigram
    try:
        unigram_prob = unigram_model.model[wi]
    except KeyError:
        unigram_prob = smallest_prob_unigram

    trigram_counts[(wi_2, wi_1)] = trigram_counts.get((wi_2, wi_1), dict())
    trigram_count = trigram_counts[(wi_2, wi_1)].get(wi, 0)
    return -(
        trigram_count * np.log2(
            lambda1 * trigram_prob + lambda2 * bigram_prob + lambda3 * unigram_prob
        )
    )


def get_all_lambdas():
    lambdas = list()
    for i in range(1, 11):
        lambda1 = i / 10
        for j in range(1, 11):
            lambda2 = j / 10
            for k in range(1, 11):
                lambda3 = k / 10
                if lambda1 + lambda2 + lambda3 != 1:
                    continue
                else:
                    lambdas.append(tuple([lambda1, lambda2, lambda3]))
    return lambdas
