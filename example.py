from token_counter import TokenCounter

text = """Many words map to one token, but some don't: indivisible.

Unicode characters like emojis may be split into many tokens containing the underlying bytes: 🤚🏾

Sequences of characters commonly found next to each other may be grouped together: 1234567890"""

counter = TokenCounter("whitespace")
print("Whitespace tokens:", counter.count(text))

counter = TokenCounter("treebank")
print("Treebank tokens:", counter.count(text))

counter = TokenCounter("subword")
print("Subword tokens:", counter.tokenize(text))

corpus = [text]

counter = TokenCounter("wordpiece", vocab_size=50)
counter.tokenizer.train(corpus)
print(f"WordPiece tokens: {counter.count(text)}")

counter = TokenCounter("bpe")
counter.tokenizer.train(corpus)
print("BPE tokens:", counter.explain(text))
