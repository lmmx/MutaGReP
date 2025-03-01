from mutagrep.coderec.v3.symbol_mining import Symbol, SymbolCategory
from mutagrep.plan_search.components import GoalTest, PlanStep
from mutagrep.plan_search.domain_models import (
    CodeSearchInstrumentation,
    CodeSearchToolOutput,
    Node,
    Plan,
    RetrievedSymbol,
)
from mutagrep.plan_search.rankers.most_unique_symbols import MostUniqueSymbolsRanker


def test_most_unique_symbols_ranker() -> None:
    symbols = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

    # Turn the ground truth symbols into a plan.
    plan_steps: list[PlanStep] = []
    for idx, symbol_name in enumerate(symbols):
        plan_steps.append(
            PlanStep(
                index=idx,
                content=symbol_name,
                search_result=CodeSearchToolOutput(
                    symbol_name=symbol_name,
                    justification="",
                    satisfies_intention=True,
                    instrumentation=CodeSearchInstrumentation(
                        completion_tokens=0,
                        prompt_tokens=0,
                        total_tokens=0,
                        symbols_considered=[
                            RetrievedSymbol(
                                symbol=Symbol(
                                    name=symbol_name,
                                    symbol_type=SymbolCategory.FUNCTION,
                                    full_path=symbol_name,
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
        user_query="",
        steps=plan_steps,
        goal_test=GoalTest(satisfies_user_request=False, explanation=""),
    )

    # Create a node with the plan.
    node = Node(plan=plan)

    # Create the ranker.
    ranker = MostUniqueSymbolsRanker()

    # Rank the node.
    score = ranker(node)
    assert score == len(symbols)
