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
    training_files = [
        "./train/5d384641-37e5-4b1c-b4d6-0ee935141ecb.txt",
        "./train/9ae9dae7-69a0-4116-9622-448f154bc269.txt",
        "./train/14d44997-7510-4777-adf0-5e4dd387e0bf.txt",
        "./train/0493e223-a0e2-4c6f-ade9-7172f35c18b1.txt",
        "./train/521ce35b-288c-4b12-a7a8-70b836290f90.txt",
        "./train/904393ad-fbc1-4512-8705-ce1c005c4915.txt",
        "./train/30381986-3d6b-4227-9733-9483ead7343d.txt",
        "./train/a894e49e-a0a6-4851-be23-4da89a52bb8e.txt",
        "./train/da059d4f-19ac-4130-858a-f6241b56fe39.txt",
        "./train/e0a13927-26e0-4ca8-b1a0-864b7604e566.txt",
    ]
    test_files = ["./test/test01.txt", "./test/test02.txt"]
    # model = get_add_alpha_smoothing_model(training_files, 0.1)
    # di = get_perplexity(model, test_files)
    # print(di)
    model = get_interpolation_model(training_files[:-2], training_files[-2:])
    di = get_perplexity_interpolation(model, test_files)
    print(di)
