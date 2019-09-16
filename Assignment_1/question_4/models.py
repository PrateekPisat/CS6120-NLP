class HMMPosTagger:
    def __init__(self, pi, transition_prob, emmission_prob, vocab):
        self.pi = pi
        self.transition_prob = transition_prob
        self.emmission_prob = emmission_prob
        self.vocab = vocab
