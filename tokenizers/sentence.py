from .base import Tokenizer
import nltk  # TODO: Implement without nltk


class SentenceTokenizer(Tokenizer):
    def __init__(self):
        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            raise ImportError("NLTK is required. Use 'pip install nltk'")

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("stopwords", quiet=True)

        self.sent_tokenize = sent_tokenize

    def tokenize(self, text: str) -> list:
        return self.sent_tokenize(text)
