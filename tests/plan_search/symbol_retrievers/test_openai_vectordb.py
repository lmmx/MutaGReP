from pathlib import Path

from mutagrep.coderec.v3.symbol_mining import Symbol, SymbolCategory
from mutagrep.plan_search.symbol_retrievers.openai_vectorb import (
    OpenAiVectorSearchSymbolRetriever,
)
from mutagrep.vector_search import Embeddable


def test_openai_vectorb(tmp_path: Path):
    retriever = OpenAiVectorSearchSymbolRetriever.instantiate_from_path(tmp_path)
    symbol_a = Symbol(
        name="a",
        full_path="a",
        code="a",
        filename="a",
        filepath="a",
        lineno=1,
        symbol_type=SymbolCategory.FUNCTION,
        docstring="a",
    )
    symbol_b = Symbol(
        name="b",
        full_path="b",
        code="b",
        filename="b",
        filepath="b",
        lineno=2,
        symbol_type=SymbolCategory.FUNCTION,
        docstring="b",
    )

    embeddables = [
        Embeddable(
            key="I enjoy eating chocolate.",
            payload=symbol_a,
        ),
        Embeddable(
            key="Concrete consists of cement, sand, and gravel.",
            payload=symbol_b,
        ),
    ]

    retriever.index(embeddables)

    assert retriever("I enjoy eating chocolate.", n_results=1)[0].symbol == symbol_a
    assert (
        retriever("Concrete consists of cement, sand, and gravel.", n_results=1)[
            0
        ].symbol
        == symbol_b
    )
