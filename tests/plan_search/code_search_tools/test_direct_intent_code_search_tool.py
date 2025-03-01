from pathlib import Path

import pytest

from mutagrep.coderec.v3.symbol_mining import Symbol, SymbolCategory
from mutagrep.plan_search.code_search_tools.direct_intent_search import (
    DirectIntentSearchTool,
    NoDuplicatesDirectIntentSearchTool,
)
from mutagrep.plan_search.symbol_retrievers.bm25_simple import Bm25SymbolRetriever
from mutagrep.plan_search.symbol_retrievers.openai_vectorb import (
    OpenAiVectorSearchSymbolRetriever,
)
from mutagrep.vector_search import Embeddable


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

    return [symbol1, symbol2]


def test_direct_intent_search_bm25s(symbol_corpus: list[Symbol]) -> None:
    retriever = Bm25SymbolRetriever.build_from_symbol_sequence(symbol_corpus)
    search_tool = DirectIntentSearchTool(retriever, symbols_to_retrieve=2)

    output = search_tool("sum")
    assert output.satisfies_intention
    assert output.symbol_name == "calculate_sum"
    assert output.instrumentation is not None
    assert len(output.instrumentation.symbols_considered) == 2


def test_direct_intent_search_vectorb(
    symbol_corpus: list[Symbol],
    tmp_path: Path,
) -> None:
    retriever = OpenAiVectorSearchSymbolRetriever.instantiate_from_path(
        tmp_path / "vectorb_test_db",
    )
    for symbol in symbol_corpus:
        assert symbol.docstring is not None
        retriever.index_single(key=symbol.docstring, symbol=symbol)
    search_tool = DirectIntentSearchTool(retriever, symbols_to_retrieve=2)
    output = search_tool("I need to add two numbers")
    assert output.satisfies_intention
    assert output.symbol_name == "calculate_sum"
    assert output.instrumentation is not None
    assert len(output.instrumentation.symbols_considered) == 2


def test_direct_intent_search_vectorb_deduplication(
    symbol_corpus: list[Symbol],
    tmp_path: Path,
) -> None:
    retriever = OpenAiVectorSearchSymbolRetriever.instantiate_from_path(
        tmp_path / "vectorb_test_db",
    )

    symbol_a, symbol_b, *_ = symbol_corpus

    embeddable_a = Embeddable(
        key="I am addicted to chocolate",
        payload=symbol_a,
    )
    embeddable_b = Embeddable(
        key="My local horse is a good horse",
        payload=symbol_b,
    )

    # Now we repeat each embeddable 10 times
    embeddables = [embeddable_a] * 10 + [embeddable_b] * 10

    retriever.index(embeddables)

    search_tool = NoDuplicatesDirectIntentSearchTool(
        retriever,
        symbols_to_retrieve=2,
        overretrieve_factor=10,
    )

    output = search_tool("I am addicted to strawberries")
    assert output.instrumentation is not None
    assert len(output.instrumentation.symbols_considered) == 2
    assert output.instrumentation.symbols_considered[0].symbol == symbol_a
    assert output.instrumentation.symbols_considered[1].symbol == symbol_b
