from .tokenizer import Tokenizer

class Vocabulary:
    def __init__(self, dataset):
        self.vocab = set()
        for question, _ in dataset:
            self.vocab.update(Tokenizer.tokenize(question))
        self.word2idx = {word: i for i, word in enumerate(sorted(self.vocab))}
        self.idx2word = {i: word for word, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def vectorize(self, text):
        vec = [0] * self.vocab_size
        for word in Tokenizer.tokenize(text):
            if word in self.word2idx:
                vec[self.word2idx[word]] = 1
        return vec
