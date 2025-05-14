from .vocabulary import Vocabulary
from .model import NanoChatModel

class NanoChat:
    def __init__(self, dataset):
        self.dataset = dataset
        self.questions = [q for q, _ in dataset]
        self.answers = [a for _, a in dataset]
        self.vocab = Vocabulary(dataset)
        self.model = NanoChatModel(input_size=self.vocab.vocab_size, output_size=len(self.answers))
        self.train()

    def train(self):
        X = [self.vocab.vectorize(q) for q in self.questions]
        y = list(range(len(self.answers)))
        self.model.train(X, y)

    def get_response(self, user_input):
        x = self.vocab.vectorize(user_input)
        out = self.model.forward(x)
        return self.answers[out.index(max(out))]

    def chat(self):
        print("🔹 NanoChat is ready! (type 'exit' to quit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = self.get_response(user_input)
            print("NanoChat:", response)
