import unicodedata
from pathlib import Path
from typing import Generic, Protocol, Sequence, TypeVar, cast

import bm25s
from bm25s import BM25
from bm25s.hf import BM25HF
from tqdm.auto import tqdm
from typing_extensions import Self

from mutagrep.coderec.v3.symbol_mining import Symbol
from mutagrep.plan_search.domain_models import (RetrievedSymbol,
                                                SymbolRetrievalScoreType,
                                                SymbolRetriever)
from mutagrep.plan_search.typing_utils import implements


class TokenizerInterface(Protocol):
    def tokenize(self, tokenizables: Sequence[str]) -> list[list[str]]: ...


class CodeNGramTokenizer:
    def __init__(self, min_gram: int = 3, max_gram: int = 4):
        self.min_gram = min_gram
        self.max_gram = max_gram

    def tokenize_single(self, tokenizable: str) -> list[str]:
        return tokenize_code_ngram(tokenizable, self.min_gram, self.max_gram)

    def tokenize(
        self, tokenizables: Sequence[str], show_progress: bool = False
    ) -> list[list[str]]:
        if show_progress:
            iterator = tqdm(tokenizables)
        else:
            iterator = tokenizables
        return [self.tokenize_single(tokenizable) for tokenizable in iterator]


implements(TokenizerInterface)(CodeNGramTokenizer)


class Bm25SymbolRetriever:
    def __init__(
        self,
        text_retriever: BM25HF | BM25,
        symbol_sequence: Sequence[Symbol],
        tokenizer: TokenizerInterface,
    ):
        self.text_retriever = text_retriever
        self.symbol_sequence = symbol_sequence
        self.tokenizer = tokenizer

    def __call__(
        self, queries: Sequence[str], n_results: int = 5
    ) -> Sequence[RetrievedSymbol]:
        # Ignore any queries that are less than or equal to 2 characters, this causes
        # the BM25 retriever to fail.
        queries = [query for query in queries if len(query) > 2]
        query_tokens = self.tokenizer.tokenize(list(queries))
        query_tokens = cast(list[list[str]], query_tokens)
        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        results, scores = self.text_retriever.retrieve(query_tokens, k=n_results)
        retrieved_symbols: list[RetrievedSymbol] = []
        n_queries = len(queries)
        for i in range(n_queries):
            ids_for_query = results[i]
            scores_for_query = scores[i]
            symbols_for_query = [self.symbol_sequence[id] for id in ids_for_query]
            retrieved_symbols.extend(
                RetrievedSymbol(
                    symbol=symbol,
                    score=score,
                    score_type=SymbolRetrievalScoreType.SIMILARITY,
                )
                for symbol, score in zip(symbols_for_query, scores_for_query)
            )
        return retrieved_symbols

    @classmethod
    def build_from_symbol_sequence(cls, symbol_sequence: Sequence[Symbol]):
        tokenizer = CodeNGramTokenizer()
        corpus = [
            symbol.full_path
            for symbol in symbol_sequence
            if symbol.full_path is not None
        ]
        corpus_tokens = tokenizer.tokenize(corpus)
        text_retriever = BM25()
        text_retriever.index(corpus_tokens)
        return cls(text_retriever, symbol_sequence, tokenizer)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        symbol_sequence_path = path / "symbol_sequence.jsonl"
        text_retriever_path = path / "text_retriever"
        text_retriever_path.mkdir(parents=True, exist_ok=True)
        with open(symbol_sequence_path, "w") as f:
            for symbol in self.symbol_sequence:
                f.write(symbol.model_dump_json() + "\n")
        self.text_retriever.save(text_retriever_path)

    @classmethod
    def load(cls, path: Path) -> Self:
        with open(path / "symbol_sequence.jsonl", "r") as f:
            symbol_sequence = [Symbol.model_validate_json(line) for line in f]
        text_retriever = BM25.load(path / "text_retriever")
        tokenizer = CodeNGramTokenizer()
        return cls(text_retriever, symbol_sequence, tokenizer)


implements(SymbolRetriever)(Bm25SymbolRetriever)


def tokenize_code_ngram(
    code_snippet: str, min_gram: int = 2, max_gram: int = 4
) -> list[str]:
    """
    Tokenizes the input code snippet into n-grams based on the specified tokenizer settings.

    Parameters:
    - code_snippet (str): The code text to tokenize.
    - min_gram (int): The minimum length of n-grams.
    - max_gram (int): The maximum length of n-grams.

    Returns:
    - list[str]: A list of generated n-gram tokens.
    """

    def is_token_char(c):
        """Check if the character is a letter, digit, punctuation, or symbol."""
        category = unicodedata.category(c)
        return (
            category.startswith("L")
            or category.startswith("N")
            or category.startswith("P")
            or category.startswith("S")
        )

    tokens = []
    current_token = []

    # Iterate over each character and build tokens
    for char in code_snippet:
        if is_token_char(char):
            current_token.append(char)
        else:
            if current_token:
                tokens.append("".join(current_token))
                current_token = []
    # Append the last token if exists
    if current_token:
        tokens.append("".join(current_token))

    # Generate n-grams for each token
    ngram_tokens = []
    for token in tokens:
        token = token.lower()  # Apply lowercase filter
        token_length = len(token)
        for n in range(min_gram, max_gram + 1):
            if token_length >= n:
                for i in range(token_length - n + 1):
                    ngram = token[i : i + n]
                    ngram_tokens.append(ngram)

    return ngram_tokens
