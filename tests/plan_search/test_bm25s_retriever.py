from pathlib import Path

import pytest
from bm25s import BM25

from mutagrep.coderec.v3.symbol_mining import Symbol, SymbolCategory
from mutagrep.plan_search.symbol_retrievers.bm25_simple import (
    Bm25SymbolRetriever,
    CodeNGramTokenizer,
    tokenize_code_ngram,
)


@pytest.fixture
def symbol_corpus() -> list[Symbol]:
    symbol1 = Symbol(
        name="calculate_sum",
        docstring="Function to calculate the sum of two numbers.",
        code="def calculate_sum(a, b): return a + b",
        filename="math_operations.py",
        filepath="/path/to/math_operations.py",
        lineno=10,
        symbol_type=SymbolCategory.FUNCTION,
        full_path="math_module.calculate_sum",
    )

    symbol2 = Symbol(
        name="calculate_difference",
        docstring="Function to calculate the difference between two numbers.",
        code="def calculate_difference(a, b): return a - b",
        filename="math_operations.py",
        filepath="/path/to/math_operations.py",
        lineno=20,
        symbol_type=SymbolCategory.FUNCTION,
        full_path="math_module.calculate_difference",
    )

    symbol3 = Symbol(
        name="render_template",
        docstring="Class to render HTML templates.",
        code="class RenderTemplate: pass",
        filename="template_renderer.py",
        filepath="/path/to/template_renderer.py",
        lineno=5,
        symbol_type=SymbolCategory.CLASS,
        full_path="template_module.RenderTemplate",
    )

    return [symbol1, symbol2, symbol3]


def test_tokenize_code_ngram_3_4(symbol_corpus: list[Symbol]) -> None:
    assert symbol_corpus[0].code is not None
    tokenize_code_ngram(symbol_corpus[0].code)


def test_retrieving_single(symbol_corpus: list[Symbol]) -> None:
    text_retriever = BM25()
    corpus = [symbol.code for symbol in symbol_corpus if symbol.code is not None]
    tokenizer = CodeNGramTokenizer()
    corpus_tokens = tokenizer.tokenize(corpus)
    text_retriever.index(corpus_tokens)
    retriever = Bm25SymbolRetriever(text_retriever, symbol_corpus, tokenizer)
    results = retriever(["sum"], n_results=1)
    assert len(results) == 1
    assert results[0].symbol.name == "calculate_sum"


def test_retrieving_multiple(symbol_corpus: list[Symbol]) -> None:
    text_retriever = BM25()
    corpus = [symbol.code for symbol in symbol_corpus if symbol.code is not None]
    tokenizer = CodeNGramTokenizer()
    corpus_tokens = tokenizer.tokenize(corpus)
    text_retriever.index(corpus_tokens)
    retriever = Bm25SymbolRetriever(text_retriever, symbol_corpus, tokenizer)
    results = retriever(["calculate"], n_results=2)
    assert len(results) == 2
    expected_names = sorted(["calculate_sum", "calculate_difference"])
    assert sorted([result.symbol.name for result in results]) == expected_names


def test_build_from_symbol_sequence(symbol_corpus: list[Symbol]) -> None:
    retriever = Bm25SymbolRetriever.build_from_symbol_sequence(symbol_corpus)
    results = retriever(["sum"], n_results=1)
    assert len(results) == 1
    assert results[0].symbol.name == "calculate_sum"


def test_saving_and_loading(symbol_corpus: list[Symbol], tmp_path: Path) -> None:
    retriever = Bm25SymbolRetriever.build_from_symbol_sequence(symbol_corpus)
    path = tmp_path / "test_saving_and_loading"
    retriever.save(path)
    loaded_retriever = Bm25SymbolRetriever.load(path)
    results = loaded_retriever(["render"], n_results=1)
    assert len(results) == 1
    assert results[0].symbol.name == "render_template"
