from .base import Tokenizer

class CharacterTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list:
        return list(text)