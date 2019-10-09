import os
import random
import re

from nltk.tag import str2tuple
from nltk.tokenize import sent_tokenize

from models import HMMPosTagger
from print_counts import print_bigram_counts, print_unigram_counts


def get_training_data():
    training = list()
    dirc = '.{sep}{dir}{sep}'.format(sep=os.sep, dir='train')
    for _, __, files in os.walk(dirc):
        for file in files:
            training += [dirc+ os.sep + file]
    return training


def get_test_data():
    testing = list()
    dirc = '.{sep}{dir}{sep}'.format(sep=os.sep, dir='test')
    for _, __, files in os.walk(dirc):
        for file in files:
            testing += [dirc + file]
    return testing


def _generate_ngrams(words_list, n):
    ngrams_list = []
    for num in range(0, len(words_list) - (n - 1)):
        ngram = tuple(words_list[num: num + n])
        ngrams_list.append(ngram)
    return ngrams_list


def get_unknown_words(files):
    """Return a set of words that occour less than 6 times."""
    count_dict = dict()

    for file in files:
        with open(file, "r") as f:
            buffer = f.read()
            sents = sent_tokenize(buffer)
            for sent in sents:
                tagged_words = sent.strip().replace("\n", " ").replace("\t", " ")
                words_tags = [str2tuple(t) for t in tagged_words.split()]
                for word, _ in words_tags:
                    count_dict[word] = count_dict.get(word, 0) + 1

    unk_words = set(dict(filter(lambda elem: elem[1] <= 5, count_dict.items())).keys())
    return unk_words


def setup_counts(files, report=False):
    words_tag_counts = dict()
    unigram_counts = dict()
    bigram_counts = dict()
    states = list()
    unk_words = get_unknown_words(files)
    vocab = set()

    for file in files:
        with open(file, 'r') as f:
            buffer = f.read()
            sents = sent_tokenize(buffer)
            for sent in sents:
                tags_in_sents = list()
                tagged_words = sent.strip().replace("\n", " ").replace("\t", " ")
                words_tags = [("<START>", "<START>")] + [str2tuple(t) for t in tagged_words.split()] + [("<END>", "<END>")]
                for word, tag in words_tags:
                    if word in unk_words:
                        word = "<UNK>"
                    # Emmission Counts.
                    words_tag_counts[tag] = words_tag_counts.get(tag, dict())
                    words_tag_counts[tag][word] = words_tag_counts[tag].get(word, 0) + 1
                    # Unigram Counts
                    unigram_counts[tag] = unigram_counts.get(tag, 0) + 1
                    # Get states in sent to generate bigrams.
                    tags_in_sents += [tag]
                    # Add word to vocab
                    vocab.add(word)
                states += [tags_in_sents]
    for state in states:
        bigrams = _generate_ngrams(state, 2)
        for ti_1, ti in bigrams:
            bigram_counts[ti_1] = bigram_counts.get(ti_1, dict())
            bigram_counts[ti_1][ti] = bigram_counts[ti_1].get(ti, 0) + 1

    if report:
        file = '{file}'.format(file='tag_unigram_counts.txt')
        print_unigram_counts(unigram_counts, file=file)
        file = '{file}'.format(file='word_tag_counts.txt')
        print_bigram_counts(words_tag_counts, file=file, header="Word Tag Counts")
        file = '{file}'.format(file='tag_bigram_counts.txt')
        print_bigram_counts(bigram_counts, file=file, header="Tag Bigram Counts")

    return unigram_counts, bigram_counts, words_tag_counts, vocab


def setup_probabilities(unigram_counts, bigram_counts, words_tag_counts, vocab):
    initial_prob = dict()
    transition_prob = dict()
    emmission_prob = dict()

    total = sum(unigram_counts.values())
    for ti in unigram_counts:
        initial_prob[ti] = unigram_counts[ti] / total

    for ti_1 in unigram_counts:
        for ti in unigram_counts:
            transition_prob[ti_1] = transition_prob.get(ti_1, dict())
            try:
                transition_prob[ti_1][ti] = bigram_counts[ti_1].get(ti, 0) / unigram_counts[ti_1]
            except KeyError:
                transition_prob[ti_1][ti] = 0

    for ti in unigram_counts:
        for wi in vocab:
            emmission_prob[ti] = emmission_prob.get(ti, dict())
            try:
                emmission_prob[ti][wi] = words_tag_counts[ti].get(wi, 0) / unigram_counts[ti]
            except KeyError:
                emmission_prob[ti][wi] = 0

    return initial_prob, transition_prob, emmission_prob


def build_sentences(model, report=False):
    sents =  [build_sentence(model) for i in range(5)]
    if report:
        print_sentences(sents)
    return  sents


def build_sentence(model):
    next_token = "<START>"
    tag_sentence = ""
    while next_token != "<END>":
        population = list(model.transition_prob[next_token].keys())
        weights = list(model.transition_prob[next_token].values())
        next_token = random.choices(population, weights)[0]
        tag_sentence += "{} ".format(next_token)
    tag_sentence = tag_sentence.strip().rstrip("<END>")

    sentence = ""
    for tag in tag_sentence.split():
        population = list(model.emmission_prob[tag].keys())
        weights = list(model.emmission_prob[tag].values())
        next = random.choices(population, weights)[0]
        sentence += "{}/{} ".format(next, tag)

    sentence_prob = get_sentence_prob(sentence, model)

    return (sentence, sentence_prob)


def get_sentence_prob(sentence, model):
    prob = 1
    prev = "<START>"
    words_tags = sentence.split()
    for word_tag in words_tags:
        word, tag = word_tag.split("/")
        try:
            prob *= model.transition_prob[prev][tag] * model.emmission_prob[tag][word]
        except KeyError:
            prob *= model.transition_prob[prev][tag] * model.emmission_prob[tag]["<UNK>"]
        prev = tag
    return prob


def get_pos_tags_for(files, model):
    sentences_to_print = []
    regex = re.compile(r'< sentence ID = \d* >')
    for file in files:
        with open(file) as f:
            buffer = f.read().strip()
            sents = list(filter(None, buffer.split("< EOS >")))
            for sent in sents:
                filtered_sent = [
                    x for x in sent.strip().split("\n") if not re.findall(regex, x) and x != ''
                ]
                tags_for_sent, best_prob = veterbi(filtered_sent, model)
                tagged_sent = ""
                for word, tag in zip(filtered_sent, tags_for_sent):
                    tagged_sent += "{}/{}\n".format(word, tag)
                tagged_sent = "< sentenceID = {} >\n" + tagged_sent + "< EOS >\n"
                sentences_to_print.append(tagged_sent)

    with open("./Question_4.4.txt", 'w+') as wr:
        for index, sentence in enumerate(sentences_to_print):
            wr.truncate()
            wr.write(sentence.format(index + 1))


def veterbi(filtered_sent, model):
    v_matrix = [[0 for obs_t in filtered_sent] for state in model.pi.keys()]
    backpointer = [[0 for obs_t in filtered_sent] for state in model.pi.keys()]
    max_prob_for_s = dict()

    max_prob_for_s, backpointer, v_matrix = _init_step(filtered_sent, max_prob_for_s, backpointer, v_matrix, model)
    max_prob_for_s, backpointer, v_matrix = _recuresion_step(filtered_sent, max_prob_for_s, backpointer, v_matrix, model)
    best_back_path_pointer, best_path_prob = _termination_step(filtered_sent, max_prob_for_s, backpointer, v_matrix, model)

    # Buld back path
    current_col = len(filtered_sent) - 1
    best_back_path = [list(model.pi.keys())[best_back_path_pointer]]
    while current_col != 0:
        best_back_path_pointer = backpointer[best_back_path_pointer][current_col]
        best_back_path += [list(model.pi.keys())[best_back_path_pointer]]
        current_col -= 1

    return best_back_path[::-1], best_path_prob


def _init_step(filtered_sent, max_prob_for_s, backpointer, v_matrix, model):
    for index_s, state in enumerate(model.pi):
        obs_0 = filtered_sent[0]
        if obs_0 not in model.vocab:
            obs_0 = "<UNK>"
        v_matrix[index_s][0] = model.pi[state] * model.emmission_prob[state][obs_0]
        backpointer[index_s][0] = 0
    return max_prob_for_s, backpointer, v_matrix


def _recuresion_step(filtered_sent, max_prob_for_s, backpointer, v_matrix, model):
    for t in range(1, len(filtered_sent)):
        for index_s, state in enumerate(model.pi):
            obs_t = filtered_sent[t]
            if obs_t not in model.vocab:
                obs_t = "<UNK>"
            max = 0
            arg_max = -1
            for si_1, s_prime in enumerate(model.pi):
                prob_s_t = v_matrix[si_1][t-1] * model.transition_prob[s_prime].get(state, 0) * model.emmission_prob[state][obs_t]
                if prob_s_t > max:
                    max = prob_s_t
                    arg_max = si_1
            v_matrix[index_s][t] = max
            backpointer[index_s][t] = arg_max
    return max_prob_for_s, backpointer, v_matrix


def _termination_step(filtered_sent, max_prob_for_s, backpointer, v_matrix, model):
    T = len(filtered_sent) - 1
    max = 0
    for index, _ in enumerate(model.pi):
        if v_matrix[index][T] >= max:
            max = v_matrix[index][T]
            best_back_path_pointer = index

    return best_back_path_pointer, max


def _get_vocab_size(count_dict):
    unique_words = set()
    for wi_1 in count_dict:
        for wi in list(count_dict[wi_1].keys()):
            unique_words.add(tuple([*wi_1, wi]))
    return len(unique_words)


def print_sentences(sentences):
    file = '.{sep}{dir}{sep}{file}'.format(sep=os.sep, dir='reports', file='sentences.txt')
    with open(file, 'w') as f:
        f.truncate()
        for sentence, prob in sentences:
            f.write("{}\n".format(sentence))
            f.write("Prob: {prob}\n".format(prob=prob))


if __name__ == "__main__":
    training_data = get_training_data()
    test_files = get_test_data()
    unigram_counts, bigram_counts, words_tag_counts, vocab = setup_counts(training_data)
    initial_prob, transition_prob, emmission_prob = setup_probabilities(
        unigram_counts, bigram_counts, words_tag_counts, vocab
    )
    model = HMMPosTagger(initial_prob, transition_prob, emmission_prob, vocab)

    sentences = build_sentences(model)

    get_pos_tags_for(test_files, model)
