from .base import Tokenizer
from collections import defaultdict
import re


class BPETokenizer(Tokenizer):
    def __init__(self, num_merges: int = 100):
        self.num_merges = num_merges
        self.vocab = defaultdict(int)
        self.merges = []
        self.merge_pairs = {}

    def train(self, corpus: list):
        word_counts = defaultdict(int)
        for text in corpus:
            words = self._preprocess(text)
            for word in words:
                word_counts[word] += 1

        vocab = defaultdict(int)
        for word, count in word_counts.items():
            chars = list(word) + ["</w>"]
            for char in chars:
                vocab[char] += count
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                vocab["".join(pair)] += count

        for _ in range(self.num_merges):
            pairs = self._get_pairs(word_counts)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            self.merge_pairs[best_pair] = True

            new_word_counts = defaultdict(int)
            for word, count in word_counts.items():
                new_word = self._merge_pair(word, best_pair)
                new_word_counts[new_word] += count
            word_counts = new_word_counts

        self.vocab = defaultdict(int)
        for word, count in word_counts.items():
            tokens = self._tokenize_word(word)
            for token in tokens:
                self.vocab[token] += count

    def _preprocess(self, text: str) -> list:
        words = re.findall(r"\S+", text.lower())
        return [word + "</w>" for word in words]

    def _get_pairs(self, word_counts):
        pairs = defaultdict(int)
        for word, count in word_counts.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += count
        return pairs

    def _merge_pair(self, word: str, pair: tuple) -> str:
        joined = "".join(pair)
        symbols = word.split()
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                symbols[i] = joined
                del symbols[i + 1]
            else:
                i += 1
        return " ".join(symbols)

    def _tokenize_word(self, word: str) -> list:
        symbols = list(word.replace("</w>", " </w>"))
        for pair in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    symbols[i] = pair[0] + pair[1]
                    del symbols[i + 1]
                else:
                    i += 1
        return [s.replace("</w", "</w>") for s in symbols if s]

    def tokenize(self, text: str) -> list:
        words = self._preprocess(text)
        tokens = []
        for word in words:
            tokens.extend(self._tokenize_word(word))
        return tokens
