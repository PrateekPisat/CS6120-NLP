class UnigramModel:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab


class BigramModel:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab


class TrigramModel:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab


class AddAlphaSmoothingModel:
    def __init__(self, model, alpha, vocab):
        self.model = model
        self.alpha = alpha
        self.vocab = vocab


class InterpolationModel:
    def __init__(
        self, unigram_model, bigram_model, trigram_model, lambda1, lambda2, lambda3
    ):
        self.unigram_model = unigram_model
        self.bigram_model = bigram_model
        self.trigram_model = trigram_model
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.vocab = trigram_model.vocab
