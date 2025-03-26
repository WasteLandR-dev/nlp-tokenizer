import re
from .base import Tokenizer

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: str = r'\w+'):
        self.pattern = re.compile(pattern)

    def tokenize(self, text: str) -> list:
        return self.pattern.findall(text)