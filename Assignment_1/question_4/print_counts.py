import os

def print_unigram_counts(counts, file):
    file = '.{sep}{dir}{sep}{file}'.format(sep=os.sep, dir='reports', file=file)
    with open(file, 'a') as f:
        f.write("Unigrm Counts\n\n")
        for word in counts:
            f.write("{word}: {count}\n".format(word=word, count=counts[word]))
        f.write("\n\n")


def print_bigram_counts(counts, file, header=None):
    file = '.{sep}{dir}{sep}{file}'.format(sep=os.sep, dir='reports', file=file)
    with open(file, 'a') as f:
        f.write("{header}\n\n".format(header=header))
        for wi_1 in counts:
            f.write("{word}\n".format(word=wi_1))
            for wi in counts[wi_1]:
                f.write('\t{word}: {count}\n'.format(word=wi, count=counts[wi_1][wi]))
        f.write('\n\n')
