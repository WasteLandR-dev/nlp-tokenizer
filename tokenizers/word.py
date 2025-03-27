import re
from .base import Tokenizer


class WordTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list:
        return re.findall(r"\b\w+\b", text)
