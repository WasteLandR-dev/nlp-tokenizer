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
