import pytest

from mutagrep.coderec.v3.symbol_mining import Symbol, SymbolCategory
from mutagrep.plan_search.code_search_tools.openai_one_step import (
    OpenAiOneStepCodeSearchTool,
)
from mutagrep.plan_search.symbol_retrievers.bm25_simple import Bm25SymbolRetriever


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


def test_openai_one_step_search_tool_satisfiable_intent(
    symbol_corpus: list[Symbol],
) -> None:
    retriever = Bm25SymbolRetriever.build_from_symbol_sequence(symbol_corpus)
    search_tool = OpenAiOneStepCodeSearchTool(retriever, results_per_keyword=3)
    output = search_tool("I need to add two numbers")
    assert output.satisfies_intention
    assert output.symbol_name == "math_module.calculate_sum"
