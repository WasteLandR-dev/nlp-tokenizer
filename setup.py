from setuptools import setup, find_packages

setup(
    name="nlp_tokenizer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "nltk",
        "tokenizers",
    ],
)
