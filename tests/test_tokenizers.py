import unittest
from ..token_counter import TokenCounter
from ..tokenizers import (
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


class TestBaseTokenizer(unittest.TestCase):
    def test_base_class_abstract(self):
        from ..tokenizers.base import Tokenizer

        with self.assertRaises(NotImplementedError):
            Tokenizer().tokenize("test")


class TestTokenizers(unittest.TestCase):
    def setUp(self):
        self.test_cases = [
            ("Simple text", "This is a test.", 4),
            ("Punctuation", "Hello, world!", 2),
            ("Contractions", "Don't stop believin'", 3),
            ("Empty string", "", 0),
            ("Special chars", "foo@example.com", 1),
        ]

    def test_whitespace_tokenizer(self):
        tokenizer = WhiteSpaceTokenizer()
        for name, text, _ in self.test_cases:
            with self.subTest(name=name):
                tokens = tokenizer.tokenize(text)
                self.assertIsInstance(tokens, list)
                self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_word_tokenizer(self):
        tokenizer = WordTokenizer()
        text = "Test: tokenizing with word_boundaries!"
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, ["Test", "tokenizing", "with", "word_boundaries"])

    def test_sentence_tokenizer(self):
        tokenizer = SentenceTokenizer()
        text = "First sentence. Second one! Third?"
        tokens = tokenizer.tokenize(text)
        self.assertEqual(len(tokens), 3)
        self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_character_tokenizer(self):
        tokenizer = CharacterTokenizer()
        text = "abc"
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, ["a", "b", "c"])

    def test_ngram_tokenizer(self):
        tokenizer = NGramTokenizer(n=3)
        text = "abcdef"
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, ["abc", "bcd", "cde", "def"])

    def test_regex_tokenizer(self):
        tokenizer = RegexTokenizer(r"\d+")
        text = "12 monkeys, 35 elephants"
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, ["12", "35"])

    def test_treebank_tokenizer(self):
        tokenizer = TreebankTokenizer()
        text = "Can't split contractions."
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, ["Ca", "n't", "split", "contractions", "."])

    def test_subword_tokenizer(self):
        tokenizer = SubwordTokenizer(max_subword_length=3)
        text = "university"
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, ["uni", "ver", "sit", "y"])


class TestTrainableTokenizers(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "This is a sample text for tokenization.",
            "Natural language processing needs good tokenizers!",
        ]
        self.test_text = "Quick sample tokenization"

    def test_bpe_tokenizer(self):
        tokenizer = BPETokenizer()
        tokenizer.train(self.corpus)
        tokens = tokenizer.tokenize(self.test_text)
        self.assertGreater(len(tokens), 3)
        self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_wordpiece_tokenizer(self):
        tokenizer = WordPieceTokenizer(vocab_size=50)
        tokenizer.train(self.corpus)
        tokens = tokenizer.tokenize(self.test_text)
        self.assertGreaterEqual(len(tokens), 2)
        self.assertTrue(all(isinstance(t, str) for t in tokens))


class TestTokenCounter(unittest.TestCase):
    def test_valid_tokenizers(self):
        methods = [
            "whitespace",
            "word",
            "sentence",
            "character",
            "ngram",
            "regex",
            "treebank",
            "subword",
        ]
        for method in methods:
            with self.subTest(method=method):
                counter = TokenCounter(method)
                count = counter.count("Test input")
                self.assertIsInstance(count, int)

    def test_invalid_tokenizer(self):
        with self.assertRaises(ValueError):
            TokenCounter("invalid_method")

    def test_custom_parameters(self):
        counter = TokenCounter("ngram", n=4)
        self.assertEqual(counter.tokenizer.n, 4)


if __name__ == "__main__":
    unittest.main()
