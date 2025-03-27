from .base import Tokenizer


class NGramTokenizer(Tokenizer):
    def __init__(self, n: int = 2):
        self.n = n

    def tokenize(self, text: str) -> list:
        return [text[i : i + self.n] for i in range(len(text) - self.n + 1)]
