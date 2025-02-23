from mutagrep.coderec.v3.symbol_mining import Symbol, SymbolCategory
from mutagrep.plan_search.components import GoalTest, PlanStep
from mutagrep.plan_search.domain_models import (
    CodeSearchInstrumentation,
    CodeSearchToolOutput,
    Node,
    Plan,
    RetrievedSymbol,
)
from mutagrep.plan_search.rankers.step_level_likert_llm_judge import (
    StepLevelLikertLlmJudge,
)


def test_likert_llm_judge_good_plan() -> None:
    # Create a good plan where symbols match the intention
    file_operations_symbols = [
        Symbol(
            name="read_file",
            symbol_type=SymbolCategory.FUNCTION,
            full_path="file_utils.read_file",
            docstring="Reads contents of a file",
            code="def read_file(path): ...",
            filename="file_utils.py",
            filepath="/src/utils/file_utils.py",
            lineno=1,
        ),
        Symbol(
            name="write_file",
            symbol_type=SymbolCategory.FUNCTION,
            full_path="file_utils.write_file",
            docstring="Writes content to a file",
            code="def write_file(path, content): ...",
            filename="file_utils.py",
            filepath="/src/utils/file_utils.py",
            lineno=10,
        ),
    ]

    plan_steps = [
        PlanStep(
            index=0,
            content="Read the input file",
            search_result=CodeSearchToolOutput(
                symbol_name="read_file",
                justification="This function reads file contents",
                satisfies_intention=True,
                instrumentation=CodeSearchInstrumentation(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                    symbols_considered=[
                        RetrievedSymbol(
                            symbol=file_operations_symbols[0],
                            score=1.0,
                        )
                    ],
                ),
            ),
        ),
        PlanStep(
            index=1,
            content="Write to the output file",
            search_result=CodeSearchToolOutput(
                symbol_name="write_file",
                justification="This function writes to files",
                satisfies_intention=True,
                instrumentation=CodeSearchInstrumentation(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                    symbols_considered=[
                        RetrievedSymbol(
                            symbol=file_operations_symbols[1],
                            score=1.0,
                        )
                    ],
                ),
            ),
        ),
    ]

    plan = Plan(
        user_query="Copy contents from input.txt to output.txt",
        steps=plan_steps,
        goal_test=GoalTest(satisfies_user_request=True, explanation=""),
    )

    node = Node(plan=plan)

    ranker = StepLevelLikertLlmJudge()
    score = ranker(node)
    assert score == 7.0


def test_likert_llm_judge_bad_plan() -> None:
    # Create a bad plan where symbols don't match the intention
    unrelated_symbols = [
        Symbol(
            name="calculate_tax",
            symbol_type=SymbolCategory.FUNCTION,
            full_path="tax_utils.calculate_tax",
            docstring="Calculates tax amount",
            code="def calculate_tax(amount): ...",
            filename="tax_utils.py",
            filepath="/src/utils/tax_utils.py",
            lineno=1,
        ),
        Symbol(
            name="format_currency",
            symbol_type=SymbolCategory.FUNCTION,
            full_path="tax_utils.format_currency",
            docstring="Formats amount as currency",
            code="def format_currency(amount): ...",
            filename="tax_utils.py",
            filepath="/src/utils/tax_utils.py",
            lineno=10,
        ),
    ]

    plan_steps = [
        PlanStep(
            index=0,
            content="Calculate tax amount",
            search_result=CodeSearchToolOutput(
                symbol_name="calculate_tax",
                justification="This function calculates tax",
                satisfies_intention=True,
                instrumentation=CodeSearchInstrumentation(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                    symbols_considered=[
                        RetrievedSymbol(
                            symbol=unrelated_symbols[0],
                            score=1.0,
                        )
                    ],
                ),
            ),
        ),
        PlanStep(
            index=1,
            content="Format the amount",
            search_result=CodeSearchToolOutput(
                symbol_name="format_currency",
                justification="This function formats currency",
                satisfies_intention=True,
                instrumentation=CodeSearchInstrumentation(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                    symbols_considered=[
                        RetrievedSymbol(
                            symbol=unrelated_symbols[1],
                            score=1.0,
                        )
                    ],
                ),
            ),
        ),
    ]

    plan = Plan(
        user_query="Copy contents from input.txt to output.txt",
        steps=plan_steps,
        goal_test=GoalTest(satisfies_user_request=True, explanation=""),
    )

    node = Node(plan=plan)

    ranker = StepLevelLikertLlmJudge()
    score = ranker(node)
    assert score < 7.0
