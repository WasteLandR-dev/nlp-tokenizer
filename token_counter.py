from tokenizers import (
    WhiteSpaceTokenizer,
    WordTokenizer,
    SentenceTokenizer,
    CharacterTokenizer,
    NGramTokenizer,
    RegexTokenizer,
    TreebankTokenizer,
    SubwordTokenizer,
    BPETokenizer,
    WordPieceTokenizer,
)


class TokenCounter:
    TOKENIZERS = {
        "whitespace": WhiteSpaceTokenizer,
        "word": WordTokenizer,
        "sentence": SentenceTokenizer,
        "character": CharacterTokenizer,
        "ngram": NGramTokenizer,
        "regex": RegexTokenizer,
        "treebank": TreebankTokenizer,
        "subword": SubwordTokenizer,
        "bpe": BPETokenizer,
        "wordpiece": WordPieceTokenizer,
    }

    def __init__(self, method: str = "whitespace", **kwargs):
        tokenizer_cls = self.TOKENIZERS.get(method.lower())
        if not tokenizer_cls:
            raise ValueError(f"Invalid tokenizer method: {method}")

        self.tokenizer = tokenizer_cls(**kwargs)

    def count(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    def tokenize(self, text: str) -> list:
        """Return list of tokens"""
        return self.tokenizer.tokenize(text)

    def explain(self, text: str) -> dict:
        """
        Return dictionary with tokenization details:
        - tokens: list of tokens
        - counts: frequency of each token
        - tokenizer: name of used tokenizer
        """
        tokens = self.tokenize(text)
        return {
            "tokens": tokens,
            "counts": self._get_token_counts(tokens),
            "tokenizer": type(self.tokenizer).__name__,
            "token_count": len(tokens),
        }

    def _get_token_counts(self, tokens: list) -> dict:
        """Calculate frequency of each token"""
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return counts
