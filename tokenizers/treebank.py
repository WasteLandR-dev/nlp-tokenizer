from .base import Tokenizer
import nltk # TODO: Implement without nltk

class TreebankTokenizer(Tokenizer):
    def __init__(self):
        try:
            from nltk.tokenize import word_tokenize
        except ImportError:
            raise ImportError("NLTK is required. Use 'pip install nltk'")

        required_resources = [
            ('tokenizers/punkt', 'punkt'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('corpora/stopwords', 'stopwords'),
            ('tokenizers/punkt_tab', 'punkt_tab')
        ]
        
        for path, package in required_resources:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(package, quiet=True)

        self.word_tokenize = word_tokenize

    def tokenize(self, text: str) -> list:
        return self.word_tokenize(text)