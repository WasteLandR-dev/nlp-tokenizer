from .base import Tokenizer


class WhiteSpaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list:
        return text.split()
