import pytest

from mutagrep.coderec.v3.symbol_mining import Symbol, SymbolCategory
from mutagrep.longcodearena.types import LongCodeArenaRecord
from mutagrep.plan_search.components import GoalTest, PlanStep
from mutagrep.plan_search.domain_models import (
    CodeSearchInstrumentation,
    CodeSearchToolOutput,
    Plan,
    RetrievedSymbol,
)
from mutagrep.plan_search.lca_judge import CodeVsCodeJudge, PlanVsPlanJudge


@pytest.fixture
def good_plan_bad_plan_record() -> tuple[LongCodeArenaRecord, Plan, Plan]:
    # Create a record for file operations task
    record = LongCodeArenaRecord(
        instruction="Read data from input.txt, process it, and write results to output.txt",
        clean_reference="""
def process_files():
    with open('input.txt', 'r') as f:
        data = f.read()

    processed = data.upper()  # Simple processing

    with open('output.txt', 'w') as f:
        f.write(processed)
        """,
        unique_apis=["open", "read", "write"],
        # Other fields can be None as they're not used in the judge
        repo_full_name="test/test",
        repo_name="test",
        repo_owner="test",
        reference="",
        path_to_reference_file="",
        path_to_examples_folder="",
        n_unique_apis=0,
        project_defined_elements=[],
        api_calls=[],
        internal_apis=[],
    )

    # Create symbols for the good plan
    file_operations_symbols = [
        Symbol(
            name="open",
            symbol_type=SymbolCategory.FUNCTION,
            full_path="builtins.open",
            docstring="Open file and return a stream",
            code="def open(file, mode='r', ...): ...",
            filename="builtins.py",
            filepath="/usr/lib/python3.8/builtins.py",
            lineno=1,
        ),
        Symbol(
            name="read",
            symbol_type=SymbolCategory.METHOD,
            full_path="_io.TextIOWrapper.read",
            docstring="Read and return a string from the stream",
            code="def read(self, size=-1): ...",
            filename="io.py",
            filepath="/usr/lib/python3.8/io.py",
            lineno=1,
        ),
        Symbol(
            name="write",
            symbol_type=SymbolCategory.METHOD,
            full_path="_io.TextIOWrapper.write",
            docstring="Write string to stream",
            code="def write(self, text): ...",
            filename="io.py",
            filepath="/usr/lib/python3.8/io.py",
            lineno=1,
        ),
    ]

    # Create a good plan that matches the reference implementation
    good_plan_steps = [
        PlanStep(
            index=0,
            content="Open and read the input file",
            search_result=CodeSearchToolOutput(
                symbol_name="open",
                justification="Used to open the input file for reading",
                satisfies_intention=True,
                instrumentation=CodeSearchInstrumentation(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                    symbols_considered=[
                        RetrievedSymbol(
                            symbol=file_operations_symbols[0],
                            score=1.0,
                        ),
                        RetrievedSymbol(
                            symbol=file_operations_symbols[1],
                            score=0.9,
                        ),
                    ],
                ),
            ),
        ),
        PlanStep(
            index=1,
            content="Write processed data to output file",
            search_result=CodeSearchToolOutput(
                symbol_name="write",
                justification="Used to write the processed data",
                satisfies_intention=True,
                instrumentation=CodeSearchInstrumentation(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                    symbols_considered=[
                        RetrievedSymbol(
                            symbol=file_operations_symbols[2],
                            score=1.0,
                        ),
                    ],
                ),
            ),
        ),
    ]

    # Create symbols for the bad plan (unrelated to file operations)
    math_operation_symbols = [
        Symbol(
            name="sqrt",
            symbol_type=SymbolCategory.FUNCTION,
            full_path="math.sqrt",
            docstring="Return the square root of x",
            code="def sqrt(x): ...",
            filename="math.py",
            filepath="/usr/lib/python3.8/math.py",
            lineno=1,
        ),
        Symbol(
            name="pow",
            symbol_type=SymbolCategory.FUNCTION,
            full_path="math.pow",
            docstring="Return x raised to the power y",
            code="def pow(x, y): ...",
            filename="math.py",
            filepath="/usr/lib/python3.8/math.py",
            lineno=1,
        ),
    ]

    # Create a bad plan with irrelevant operations
    bad_plan_steps = [
        PlanStep(
            index=0,
            content="Calculate square root",
            search_result=CodeSearchToolOutput(
                symbol_name="sqrt",
                justification="Calculate square root of a number",
                satisfies_intention=True,
                instrumentation=CodeSearchInstrumentation(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                    symbols_considered=[
                        RetrievedSymbol(
                            symbol=math_operation_symbols[0],
                            score=1.0,
                        ),
                    ],
                ),
            ),
        ),
        PlanStep(
            index=1,
            content="Calculate power",
            search_result=CodeSearchToolOutput(
                symbol_name="pow",
                justification="Raise number to a power",
                satisfies_intention=True,
                instrumentation=CodeSearchInstrumentation(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                    symbols_considered=[
                        RetrievedSymbol(
                            symbol=math_operation_symbols[1],
                            score=1.0,
                        ),
                    ],
                ),
            ),
        ),
    ]

    good_plan = Plan(
        user_query=record.instruction,
        steps=good_plan_steps,
        goal_test=GoalTest(satisfies_user_request=True, explanation=""),
    )

    bad_plan = Plan(
        user_query=record.instruction,
        steps=bad_plan_steps,
        goal_test=GoalTest(satisfies_user_request=True, explanation=""),
    )

    return record, good_plan, bad_plan


def test_plan_vs_plan_judge(
    good_plan_bad_plan_record: tuple[LongCodeArenaRecord, Plan, Plan],
) -> None:
    record, good_plan, bad_plan = good_plan_bad_plan_record
    judge = PlanVsPlanJudge(record)
    judgement = judge(good_plan.steps, bad_plan.steps)

    # The good plan should be chosen as the winner
    assert judgement.winner == "A", "Expected good plan (A) to win"
    assert judgement.win_rate > 0.5, "Expected good plan to have win rate > 0.5"


def test_code_vs_code_judge(
    good_plan_bad_plan_record: tuple[LongCodeArenaRecord, Plan, Plan],
) -> None:
    record, good_plan, bad_plan = good_plan_bad_plan_record
    judge = CodeVsCodeJudge(record)
    judgement = judge(good_plan.steps, bad_plan.steps)

    # The good plan should be chosen as the winner
    assert judgement.winner == "A", "Expected good plan (A) to win"
    assert judgement.win_rate > 0.5, "Expected good plan to have win rate > 0.5"
