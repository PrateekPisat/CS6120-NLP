import os
from collections import defaultdict

import spacy


def build_doc(files, report=False):
    sent_count = 0
    verb_count = 0
    prep_dict = defaultdict(lambda: 0)
    entities = list()
    unique_entities = list()
    nlp = spacy.load("en_core_web_sm")
    for file in file_opener(files):
        doc = nlp(file.read())
        sent_count += get_total_sentences(doc)
        verb_count += get_verb_count(doc)
        prep_dict = get_prep_count(doc, prep_dict)
        temp_entities, temp_unique_entities = get_entities(doc)
        entities += temp_entities
        unique_entities += temp_unique_entities

    prep_count = sum(prep_dict.values())
    top_preps = list(sorted(prep_dict, key=prep_dict.get, reverse=True))[:3]

    if report:
        file = ".{sep}{dir}{sep}{file}".format(sep=os.sep, dir="reports", file="Q1.txt")
        with open(file, "w+") as f:
            f.truncate()
            f.write("****Q1 Counts****\n\n")
            f.write("sentence counts = {}\n\n".format(sent_count))
            f.write("Average Verbs = {}\n".format(verb_count / sent_count))
            f.write(
                (
                    "I am using the POS property of a given word provided by spaCy,"
                    " and checking if the values is `VERB`\n\n"
                )
            )
            f.write("preposition count = {}\n\n".format(prep_count))
            f.write("Top Prepositions = {}\n\n".format(top_preps))
            f.write("Entity count = {}\n\n".format(len(set(entities))))
            f.write("Unique Entity count = {}\n".format(len(set(unique_entities))))


def get_total_sentences(doc):
    """Return the total number of sentences for a diven doc."""
    sent_count = 0
    for _ in doc.sents:
        sent_count += 1
    return sent_count


def get_verb_count(doc):
    """Return total number of verbs for a given doc."""
    verbs = 0
    for token in doc:
        if token.head.pos_ == "VERB":
            verbs += 1
    return verbs


def get_prep_count(doc, prep_dict):
    """Return total number of verbs for a given doc."""
    for token in doc:
        if token.pos_ == "ADP":
            prep_dict[token.text] += 1
    return prep_dict


def get_entities(doc):
    """Return all the Named entities for a given doc."""
    unique_tags = set(
        ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]
    )

    entities = [ent.text for ent in doc.ents]
    unique_entities = [ent.text for ent in doc.ents if ent.label_ in unique_tags]

    return entities, unique_entities
# helpers


def file_opener(files):
    for file in files:
        with open(file) as f:
            yield f


def get_training_files():
    training = list()
    dir_name = ".{sep}{dir}{sep}".format(sep=os.sep, dir="train")
    for _, __, files in os.walk(dir_name):
        for file in files:
            training += [dir_name + file]
    return training


if __name__ == "__main__":
    training_files = get_training_files()
    build_doc(training_files, report=True)
