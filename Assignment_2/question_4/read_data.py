import json
import os


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
