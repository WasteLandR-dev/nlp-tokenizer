from .base import Tokenizer


class SubwordTokenizer(Tokenizer):
    def __init__(self, max_subword_length: int = 4):
        self.max_subword_length = max_subword_length

    def tokenize(self, text: str) -> list:
        words = text.split()
        subwords = []
        for word in words:
            for i in range(0, len(word), self.max_subword_length):
                subwords.append(word[i : i + self.max_subword_length])
        return subwords
