import json
import os
from collections import Counter

import language_check
import numpy as np
import neuralcoref
import nltk
import readability
import spacy
from nltk.corpus import stopwords
from scipy import spatial
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.neural_network import MLPClassifier

stop_words = set(stopwords.words('english'))
nlp = spacy.load("en")
neuralcoref.add_to_pipe(nlp)
tool = language_check.LanguageTool('en-US')


def read_train_data():
    coherence_map = dict()
    grammaticality_map = dict()
    nonredundancy_map = dict()
    # read data
    with open("./train/train_data.json", 'r') as fin:
        contents = json.load(fin)

    # iterate over the dictionary
    for k, v in contents.items():
        coherence_map[k] = int(v["coherence"])
        grammaticality_map[k] = int(v["grammaticality"])
        nonredundancy_map[k] = int(v["nonredundancy"])
    # results
    return (coherence_map, grammaticality_map, nonredundancy_map)


def read_test_data():
    coherence_map = dict()
    grammaticality_map = dict()
    nonredundancy_map = dict()
    # read data
    with open("./test/test_data.json", 'r') as fin:
        contents = json.load(fin)

    # iterate over the dictionary
    for k, v in contents.items():
        coherence_map[k] = int(v["coherence"])
        grammaticality_map[k] = int(v["grammaticality"])
        nonredundancy_map[k] = int(v["nonredundancy"])
    # results
    return (coherence_map, grammaticality_map, nonredundancy_map)


def read_summarries():
    summaries_map = dict()
    direc = "./summaries/"
    # read data
    for _, __, files in os.walk(direc):
        for file in files:
            with open(direc + file, "rb") as f:
                summaries_map[file] = f.read().decode("ISO-8859-1")
    # return results
    return summaries_map


def MLE_model(X, y):
    # build model
    X = np.array(X)
    y = np.array(y)
    models = []
    act_fn = ['relu', 'logistic']
    hidden_layer_sizes = [30, 35, 40]
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
            models.append(model.fit(X, y))

    return models


def get_repeated_unigrams(buffer):
    words = [word for word in word_tokenize(buffer) if word not in stop_words]
    unigram_counts = Counter(words)
    return unigram_counts.most_common(1)[0][1]


def get_repeated_bigrams(buffer):
    words = [word for word in word_tokenize(buffer) if word not in stop_words]
    bigrams = nltk.bigrams(words)
    bigram_counts = Counter(bigrams)
    return bigram_counts.most_common(1)[0][1]


def get_continous_repated_unigrams(buffer):
    count = 0
    words = word_tokenize(buffer)
    bigrams = nltk.bigrams(words)
    for w1, w2 in bigrams:
        if w1 == w2:
            count += 1
    return count


def get_continous_reated_bigrams(buffer):
    count = 0
    words = word_tokenize(buffer)
    bigrams = list(nltk.bigrams(words))
    for i in range(len(bigrams) - 1):
        if bigrams[i] == bigrams[i + 1]:
            count += 1
    return count


def get_google_vector(sent, model):
    # Create vectors
    review_vector = Counter()
    tokens = [token.text for token in nlp(sent)]
    for token in tokens:
        review_vector[token] = model[token] if token in model else model["UNK"]
    # Get total of each embedding
    result_array = sum(list(review_vector.values()))
    # Get average of total embeddings.
    result_array /= len(tokens)
    return list(result_array)


def get_repeated_noun_chunks(text):
    count = 0
    noun_chunks = [tuple([chunk.text]) for chunk in nlp(text).noun_chunks]
    for i in range(len(noun_chunks) - 1):
        if noun_chunks[i] == noun_chunks[i + 1]:
            count += 1
    return count


def get_coreffered_entities(text):
    doc = nlp(text)
    return len(doc._.coref_clusters)


def get_max_sentence_similarity(buffer, g_model):
    data = list()
    similarity = list()

    sents = sent_tokenize(buffer)
    for sent in sents:
        data.append(get_google_vector(sent, g_model))
    if len(data) == 1:
        return 1
    for vec_1 in data:
        for vec_2 in data:
            if vec_1 == vec_2:
                continue
            else:
                similarity.append(1 - spatial.distance.cosine(vec_1, vec_2))
    return np.max(similarity)


def get_readability_score(buffer):
    reading_ease = []
    grammaticality_score = len(tool.check(buffer))
    n_datelines = len([ent.text for ent in nlp(buffer).ents if ent.label_ == "DATE"])
    sents = sent_tokenize(buffer)
    for sent in sents:
        try:
            results = readability.getmeasures(sent, lang='en')
        except ValueError:
            continue
        reading_ease += [results['readability grades']['FleschReadingEase']]
    return (np.min(reading_ease), grammaticality_score, n_datelines)
