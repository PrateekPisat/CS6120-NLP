import os

from util import (
    unigram_count,
    bigram_count,
    trigram_count,
)


def print_unigram_counts(counts):
    file = '.{sep}{dir}{sep}{file}'.format(sep=os.sep, dir='reports', file='ngramCounts.txt')
    with open(file, 'a') as f:
        f.write("Unigrm Counts\n\n")
        for word in counts:
            f.write("{word}: {count}\n".format(word=word, count=counts[word]))
        f.write("\n\n")


def print_bigram_counts(counts):
    file = '.{sep}{dir}{sep}{file}'.format(sep=os.sep, dir='reports', file='ngramCounts.txt')
    with open(file, 'a') as f:
        f.write("Bigrams Counts\n\n")
        for wi_1 in counts:
            f.write("{word}\n".format(word=wi_1))
            for wi in counts[wi_1]:
                f.write('\t{word}: {count}\n'.format(word=wi, count=counts[wi_1][wi]))
        f.write('\n\n')


def print_trigram_counts(counts):
    file = '.{sep}{dir}{sep}{file}'.format(sep=os.sep, dir='reports', file='ngramCounts.txt')
    with open(file, 'a') as f:
        f.write("Trigram Counts\n\n")
        for wi_2, wi_1 in counts:
            f.write("({wi_2}, {wi_1})\n".format(wi_2=wi_2, wi_1=wi_1))
            for wi in counts[(wi_2, wi_1)]:
                f.write('\t{word}: {count}\n'.format(word=wi, count=counts[(wi_2, wi_1)][wi]))
        f.write('\n\n')


if __name__ == '__main__':
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
    # Build Counts
    unigram_counts = unigram_count(training_files)
    bigram_counts = bigram_count(training_files)
    trigram_counts = trigram_count(training_files)
    # Print counts
    print_unigram_counts(unigram_counts)
    print_bigram_counts(bigram_counts)
    print_trigram_counts(trigram_counts)
