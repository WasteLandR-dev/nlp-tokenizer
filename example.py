from token_counter import TokenCounter

text = "Don't hesitate to ask questions. We're here to help!"

counter = TokenCounter('whitespace')
print("Whitespace tokens:", counter.count(text))  # 9

counter = TokenCounter('treebank')
print("Treebank tokens:", counter.count(text))  # 11

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a sample text for tokenization.",
    "Natural language processing needs good tokenizers!"
]
counter = TokenCounter('wordpiece', vocab_size=50)
counter.tokenizer.train(corpus)
print(f"WordPiece tokens: {counter.count(text)}")

counter = TokenCounter('bpe')
counter.tokenizer.train(corpus)
print("BPE tokens:", counter.count(text))