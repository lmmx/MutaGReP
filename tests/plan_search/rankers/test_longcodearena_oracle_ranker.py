from mutagrep.coderec.v3.symbol_mining import Symbol, SymbolCategory
from mutagrep.plan_search.components import GoalTest, PlanStep
from mutagrep.plan_search.domain_models import (
    CodeSearchInstrumentation,
    CodeSearchToolOutput,
    Node,
    Plan,
    RetrievedSymbol,
)
from mutagrep.plan_search.lca_benchmark import load_longcode_arena_records
from mutagrep.plan_search.rankers.longcodearena_oracle import LongCodeArenaOracleRanker


def test_oracle_ranker_perfect_plan() -> None:
    record = load_longcode_arena_records()[0]

    # Turn the ground truth symbols into a plan.
    plan_steps: list[PlanStep] = []
    for idx, unique_api in enumerate(record.unique_apis):
        plan_steps.append(
            PlanStep(
                index=idx,
                content=unique_api,
                search_result=CodeSearchToolOutput(
                    symbol_name=unique_api,
                    justification="",
                    satisfies_intention=True,
                    instrumentation=CodeSearchInstrumentation(
                        completion_tokens=0,
                        prompt_tokens=0,
                        total_tokens=0,
                        symbols_considered=[
                            RetrievedSymbol(
                                symbol=Symbol(
                                    name=unique_api,
                                    symbol_type=SymbolCategory.FUNCTION,
                                    full_path=unique_api,
                                    docstring="",
                                    code="",
                                    filename="",
                                    filepath="",
                                    lineno=0,
                                ),
                                score=1.0,
                            ),
                        ],
                    ),
                ),
            ),
        )

    plan = Plan(
        user_query=record.instruction,
        steps=plan_steps,
        goal_test=GoalTest(satisfies_user_request=False, explanation=""),
    )

    # Create a node with the plan.
    node = Node(plan=plan)

    # Create the ranker.
    ranker = LongCodeArenaOracleRanker(record)

    # Rank the node.
    score = ranker(node)
    assert score == 1.0


def test_oracle_ranker_imperfect_plan() -> None:
    record = load_longcode_arena_records()[0]

    # Turn the ground truth symbols into a plan.
    plan_steps: list[PlanStep] = []
    for idx, unique_api in enumerate(record.unique_apis):
        plan_steps.append(
            PlanStep(
                index=idx,
                content=unique_api,
                search_result=CodeSearchToolOutput(
                    symbol_name=unique_api,
                    justification="",
                    satisfies_intention=True,
                    instrumentation=CodeSearchInstrumentation(
                        completion_tokens=0,
                        prompt_tokens=0,
                        total_tokens=0,
                        symbols_considered=[
                            RetrievedSymbol(
                                symbol=Symbol(
                                    name=unique_api,
                                    symbol_type=SymbolCategory.FUNCTION,
                                    full_path=unique_api,
                                    docstring="",
                                    code="",
                                    filename="",
                                    filepath="",
                                    lineno=0,
                                ),
                                score=1.0,
                            ),
                        ],
                    ),
                ),
            ),
        )

    # Take just half of the plan steps.
    plan_steps = plan_steps[: len(plan_steps) // 2]

    plan = Plan(
        user_query=record.instruction,
        steps=plan_steps,
        goal_test=GoalTest(satisfies_user_request=False, explanation=""),
    )

    # Create a node with the plan.
    node = Node(plan=plan)

    # Create the ranker.
    ranker = LongCodeArenaOracleRanker(record)

    # Rank the node.
    score = ranker(node)
    # Assert the score is between 0 and 1.
    assert 0 <= score <= 1
