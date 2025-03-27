from .base import Tokenizer
from collections import defaultdict
import re


class WordPieceTokenizer(Tokenizer):
    def __init__(self, vocab_size: int = 1000, unknown_token: str = "[UNK]"):
        self.vocab_size = vocab_size
        self.unknown_token = unknown_token
        self.vocab = set()
        self.max_token_length = 0
        self.merge_history = set()
        self.word_cache = {}

    def train(self, corpus: list):
        word_counts = self._preprocess_corpus(corpus)
        self._initialize_vocab(word_counts)

        iteration = 0
        max_iterations = self.vocab_size * 2

        while len(self.vocab) < self.vocab_size and iteration < max_iterations:
            iteration += 1
            pair_scores = self._calculate_pair_scores(word_counts)

            if not pair_scores:
                break

            best_pair = max(pair_scores, key=pair_scores.get)
            if best_pair in self.merge_history:
                break

            self.merge_history.add(best_pair)
            self.vocab.add(best_pair)
            self._update_max_length(best_pair)
            word_counts = self._merge_pair_in_corpus(word_counts, best_pair)

    def _preprocess_corpus(self, corpus):
        word_counts = defaultdict(int)
        for text in corpus:
            text = text.lower()
            words = re.findall(r"\S+", text)
            for word in words:
                processed = f"^{word}$"
                word_counts[processed] += 1
        return word_counts

    def _initialize_vocab(self, word_counts):
        self.vocab = {self.unknown_token}
        char_counts = defaultdict(int)
        for word, count in word_counts.items():
            for char in word:
                if char not in {"^", "$"}:
                    char_counts[char] += count
                    self.vocab.add(char)
        self.max_token_length = max(len(c) for c in self.vocab)

    def _calculate_pair_scores(self, word_counts):
        pair_scores = defaultdict(int)
        for word, count in word_counts.items():
            tokens = self._split_word(word)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                joined = "".join(pair)
                pair_scores[joined] += count
        return pair_scores

    def _merge_pair_in_corpus(self, word_counts, pair):
        new_word_counts = defaultdict(int)
        pattern = re.compile(r"(?<!^)({})(?!$)".format(re.escape(pair)))

        for word, count in word_counts.items():
            new_word = pattern.sub(f" {pair} ", word)
            new_word = " ".join(new_word.split())
            if new_word != word:
                new_word_counts[new_word] += count
            else:
                new_word_counts[word] += count

        return new_word_counts

    def _update_max_length(self, token):
        self.max_token_length = max(self.max_token_length, len(token))

    def _split_word(self, word):
        if word in self.word_cache:
            return self.word_cache[word]

        tokens = []
        start = 0
        word_len = len(word)
        vocab = sorted(self.vocab, key=lambda x: -len(x))

        while start < word_len:
            end = min(word_len, start + self.max_token_length)
            found = False

            for token in vocab:
                token_len = len(token)
                if start + token_len > end:
                    continue

                if word.startswith(token, start, start + token_len):
                    tokens.append(token)
                    start += token_len
                    found = True
                    break

            if not found:
                tokens.append(self.unknown_token)
                start += 1

        self.word_cache[word] = tokens
        return tokens

    def tokenize(self, text: str) -> list:
        text = f"^{text.lower().strip()}$"
        tokens = self._split_word(text)
        return [t for t in tokens if t not in {"^", "$"}]
