from mutagrep.plan_search.domain_models import (
    CodeSearchToolOutput,
    RetrievedSymbol,
    Symbol,
    CodeSearchInstrumentation,
    SymbolRetrievalScoreType,
)
from mutagrep.coderec.v3.symbol_mining import SymbolCategory


def test_code_search_tool_output_get_top_n_symbols_similarity_score() -> None:
    symbols = [
        RetrievedSymbol(
            symbol=Symbol(
                name="a",
                filepath="",
                lineno=10,
                docstring="",
                code="",
                filename="",
                symbol_type=SymbolCategory.FUNCTION,
                full_path="",
            ),
            score=0.0,
            score_type=SymbolRetrievalScoreType.SIMILARITY,
        ),
        RetrievedSymbol(
            symbol=Symbol(
                name="b",
                filepath="",
                lineno=10,
                docstring="",
                code="",
                filename="",
                symbol_type=SymbolCategory.FUNCTION,
                full_path="",
            ),
            score=1.0,
            score_type=SymbolRetrievalScoreType.SIMILARITY,
        ),
        RetrievedSymbol(
            symbol=Symbol(
                name="c",
                filepath="",
                lineno=10,
                docstring="",
                code="",
                filename="",
                symbol_type=SymbolCategory.FUNCTION,
                full_path="",
            ),
            score=0.5,
            score_type=SymbolRetrievalScoreType.SIMILARITY,
        ),
    ]

    code_search_tool_output = CodeSearchToolOutput(
        satisfies_intention=True,
        symbol_name="a",
        justification="",
        instrumentation=CodeSearchInstrumentation(
            symbols_considered=symbols,
            completion_tokens=10,
            prompt_tokens=10,
            total_tokens=10,
        ),
    )

    top_n_symbols = code_search_tool_output.get_top_n_symbols(3)
    assert len(top_n_symbols) == 3

    assert top_n_symbols[0].symbol.name == "b"
    assert top_n_symbols[1].symbol.name == "c"


def test_code_search_output_top_n_symbols_distance_score() -> None:
    symbols = [
        RetrievedSymbol(
            symbol=Symbol(
                name="a",
                filepath="",
                lineno=10,
                docstring="",
                code="",
                filename="",
                symbol_type=SymbolCategory.FUNCTION,
                full_path="",
            ),
            score=0.0,
            score_type=SymbolRetrievalScoreType.DISTANCE,
        ),
        RetrievedSymbol(
            symbol=Symbol(
                name="b",
                filepath="",
                lineno=10,
                docstring="",
                code="",
                filename="",
                symbol_type=SymbolCategory.FUNCTION,
                full_path="",
            ),
            score=1.0,
            score_type=SymbolRetrievalScoreType.DISTANCE,
        ),
        RetrievedSymbol(
            symbol=Symbol(
                name="c",
                filepath="",
                lineno=10,
                docstring="",
                code="",
                filename="",
                symbol_type=SymbolCategory.FUNCTION,
                full_path="",
            ),
            score=0.5,
            score_type=SymbolRetrievalScoreType.DISTANCE,
        ),
    ]

    code_search_tool_output = CodeSearchToolOutput(
        satisfies_intention=True,
        symbol_name="a",
        justification="",
        instrumentation=CodeSearchInstrumentation(
            symbols_considered=symbols,
            completion_tokens=10,
            prompt_tokens=10,
            total_tokens=10,
        ),
    )

    top_n_symbols = code_search_tool_output.get_top_n_symbols(3)
    assert len(top_n_symbols) == 3

    assert top_n_symbols[0].symbol.name == "a"
    assert top_n_symbols[1].symbol.name == "c"
    assert top_n_symbols[2].symbol.name == "b"
