import os

from nltk.tokenize import sent_tokenize

from language_model import get_add_alpha_smoothing_model, get_interpolation_model
from util import (
    get_perplexity_for_file,
    get_interpolation_perplexity_for_file,
    get_total_ngram_count,
)


def get_perplexity(model, test_files):
    """Calculate perplexity for `add_alpha_smoothing_model`."""
    perplex_dict = dict()
    for file in test_files:
        with open(file, "r") as f:
            buffer = f.read()
            total_ngram_count = get_total_ngram_count(buffer, 2)
            sents = sent_tokenize(buffer)
            perplexity = get_perplexity_for_file(sents, model, total_ngram_count)
        perplex_dict[file] = perplexity
    return perplex_dict


def get_perplexity_interpolation(model, test_files):
    """Calculate perplexity for `interpilation_model`."""
    perplex_dict = dict()
    for file in test_files:
        with open(file, "r") as f:
            buffer = f.read()
            total_ngram_count = get_total_ngram_count(buffer, 3)
            sents = sent_tokenize(buffer)
            perplexity = get_interpolation_perplexity_for_file(sents, model, total_ngram_count)
        perplex_dict[file] = perplexity
    return perplex_dict


if __name__ == "__main__":
    train_dir = ".{sep}{dir}{sep}".format(sep=os.sep, dir="train")
    test_dir = ".{sep}{dir}{sep}".format(sep=os.sep, dir="test")
    training_files = [
        "{train_dir}5d384641-37e5-4b1c-b4d6-0ee935141ecb.txt".format(train_dir=train_dir),
        "{train_dir}9ae9dae7-69a0-4116-9622-448f154bc269.txt".format(train_dir=train_dir),
        "{train_dir}14d44997-7510-4777-adf0-5e4dd387e0bf.txt".format(train_dir=train_dir),
        "{train_dir}0493e223-a0e2-4c6f-ade9-7172f35c18b1.txt".format(train_dir=train_dir),
        "{train_dir}521ce35b-288c-4b12-a7a8-70b836290f90.txt".format(train_dir=train_dir),
        "{train_dir}904393ad-fbc1-4512-8705-ce1c005c4915.txt".format(train_dir=train_dir),
        "{train_dir}30381986-3d6b-4227-9733-9483ead7343d.txt".format(train_dir=train_dir),
        "{train_dir}a894e49e-a0a6-4851-be23-4da89a52bb8e.txt".format(train_dir=train_dir),
        "{train_dir}da059d4f-19ac-4130-858a-f6241b56fe39.txt".format(train_dir=train_dir),
        "{train_dir}e0a13927-26e0-4ca8-b1a0-864b7604e566.txt".format(train_dir=train_dir),
    ]
    test_files = [
        "{test_dir}test01.txt".format(test_dir=test_dir),
        "{test_dir}test02.txt".format(test_dir=test_dir)
    ]
    model = get_add_alpha_smoothing_model(training_files, 0.1)
    di = get_perplexity(model, test_files)
    print("lambda = 0.1 " + str(di))
    model = get_add_alpha_smoothing_model(training_files, 0.3)
    di = get_perplexity(model, test_files)
    print("lambda = 0.3 " + str(di))
    model = get_interpolation_model(training_files[:-2], training_files[-2:])
    di = get_perplexity_interpolation(model, test_files)
    print("Interpolation " + str(di))
