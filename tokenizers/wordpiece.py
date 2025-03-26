from .base import Tokenizer
from collections import defaultdict
import re

class WordPieceTokenizer(Tokenizer):
    def __init__(self, vocab_size: int = 1000, unknown_token: str = "[UNK]"):
        self.vocab_size = vocab_size
        self.unknown_token = unknown_token
        self.vocab = set()
        self.max_token_length = 0

    def train(self, corpus: list):
        word_counts = defaultdict(int)
        for text in corpus:
            words = self._preprocess(text)
            for word in words:
                word_counts[word] += 1

        self.vocab = {self.unknown_token}
        char_counts = defaultdict(int)
        for word, count in word_counts.items():
            for char in word:
                char_counts[char] += count
                self.vocab.add(char)

        while len(self.vocab) < self.vocab_size:
            pair_scores = defaultdict(int)
            
            for word, count in word_counts.items():
                tokens = self._split_word(word)
                for i in range(len(tokens)-1):
                    pair = (tokens[i], tokens[i+1])
                    joined = "".join(pair)
                    pair_scores[joined] += count

            if not pair_scores:
                break

            best_pair = max(pair_scores, key=pair_scores.get)
            self.vocab.add(best_pair)
            
            new_word_counts = defaultdict(int)
            for word, count in word_counts.items():
                new_word = word.replace(best_pair, f" {best_pair} ")
                new_word = re.sub(r'\s+', ' ', new_word).strip()
                new_word_counts[new_word] += count
            word_counts = new_word_counts

        self.vocab = sorted(self.vocab, key=lambda x: -len(x))
        self.max_token_length = max(len(token) for token in self.vocab)

    def _preprocess(self, text: str) -> list:
        words = re.findall(r'\S+', text.lower())
        return [f'^{word}$' for word in words]

    def _split_word(self, word: str) -> list:
        tokens = []
        start = 0
        while start < len(word):
            end = min(len(word), start + self.max_token_length)
            found = False
            while end > start:
                substring = word[start:end]
                if substring in self.vocab:
                    tokens.append(substring)
                    start = end
                    found = True
                    break
                end -= 1
            if not found:
                tokens.append(self.unknown_token)
                start += 1
        return tokens

    def tokenize(self, text: str) -> list:
        words = self._preprocess(text)
        tokens = []
        for word in words:
            tokens.extend(self._split_word(word))
        return [token for token in tokens if token not in ['^', '$']]

    def _get_stats(self, word_counts):
        pairs = defaultdict(int)
        for word, count in word_counts.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] += count
        return pairs