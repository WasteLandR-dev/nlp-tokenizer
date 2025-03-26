# NLP Token Counter

A Python library for counting tokens using various tokenization methods. Provides simple and advanced tokenization strategies for natural language processing tasks.

## Features

- 10+ tokenization methods
- Simple unified API
- Customizable tokenizers
- No external dependencies for basic methods
- Easy extensibility

## Installation

1. Clone the repository:
```bash
git clone https://github.com/WasteLandR-dev/nlp-tokenizer.git
cd nlp_tokenizer
```

2. Install with pip:
```bash
pip install -e .
```

3. Install optional dependencies:
```bash
pip install nltk
```

## Usage

### Basic Example
```python
from token_counter import TokenCounter

text = "Don't hesitate to ask questions. We're here to help!"

counter = TokenCounter('whitespace')
print("Whitespace tokens:", counter.count(text))  # 9

counter = TokenCounter('treebank')
print("Treebank tokens:", counter.count(text))  # 11
```

### Advanced Tokenizers
```python
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a sample text for tokenization.",
    "Natural language processing needs good tokenizers!"
]

counter = TokenCounter('bpe')
counter.tokenizer.train(corpus)
print(f"BPE tokens: {counter.count(text)}")

counter = TokenCounter('wordpiece', vocab_size=500)
counter.tokenizer.train(corpus)
print(f"WordPiece tokens: {counter.count(text)}")
```

## Available Tokenizers

| Method              | Description                                  | Requires Training |
|---------------------|----------------------------------------------|-------------------|
| `whitespace`        | Splits on whitespace                         | No                |
| `word`              | Regex-based word boundaries                  | No                |
| `sentence`          | NLTK sentence tokenization                   | No                |
| `character`         | Individual characters                        | No                |
| `ngram`             | N-gram sequences (specify n with `n=3`)      | No                |
| `regex`             | Custom regex patterns                        | No                |
| `treebank`          | Penn Treebank tokenization                   | No                |
| `subword`           | Basic subword tokenization                   | No                |
| `bpe`               | Byte Pair Encoding                           | Yes               |
| `wordpiece`         | WordPiece tokenization                       | Yes               |

## Handling NLTK Resources

Some tokenizers require NLTK resources. First-time usage will prompt automatic downloads, but you can manually download all resources:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

## Contributing

1. Fork the repository
2. Create new tokenizers in `tokenizers/` following `base.py` interface
3. Add tests for new functionality
4. Submit a pull request

To install in development mode:
```bash
pip install -e .[dev]
```

## License

MIT License (see LICENSE file)

---

**Note**: For production use of BPE/WordPiece, consider using larger training corpora and optimizing the implementations.
```

This README includes:
1. Clear installation instructions
2. Usage examples for basic and advanced features
3. Table of available tokenizers
4. NLTK resource handling
5. Contribution guidelines
6. License information

You can customize the following sections as needed:
- Add your repository URL in the installation section
- Add specific examples for your use cases
- Include additional dependencies or requirements
- Add project-specific contribution guidelines
- Modify the license information if needed