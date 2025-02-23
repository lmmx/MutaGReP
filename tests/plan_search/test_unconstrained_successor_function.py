from mutagrep.plan_search.successor_functions.xml_like_sampling_unconstrained import (
    UnconstrainedXmlOutputSuccessorFunction,
)
from mutagrep.plan_search.domain_models import (
    Node,
    Plan,
    CodeSearchToolOutput,
    CodeSearchInstrumentation,
)
from mutagrep.plan_search.components import PlanStep
from mutagrep.plan_search.mnms_search_tool import build_retriever_for_mnms
import rich
from mutagrep.plan_search.code_search_tools.direct_intent_search import (
    DirectIntentSearchTool,
)


def test_when_non_empty_plan() -> None:
    # Create a simple repo tree
    repo_tree = """
    |-- tool_api.py
    """

    retriever = build_retriever_for_mnms()
    search_tool = DirectIntentSearchTool(symbol_retriever=retriever)

    user_query = "I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment."

    starting_symbols = [_.symbol for _ in retriever([user_query])]

    successor_fn = UnconstrainedXmlOutputSuccessorFunction(
        search_tool=search_tool,
        starting_symbols=starting_symbols,
        repo_tree=repo_tree,
    )
    root = Node(
        plan=Plan(
            user_query="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
            steps=[
                PlanStep(
                    content="Analyze the sentiment of the letter.",
                    search_result=CodeSearchToolOutput(
                        symbol_name="text_classification",
                        satisfies_intention=True,
                        justification="The function `text_classification` can be used to analyze the sentiment of the letter.",
                        instrumentation=CodeSearchInstrumentation(
                            symbols_considered=list(
                                retriever(["Analyze the sentiment of the letter."])
                            ),
                            completion_tokens=100,
                            prompt_tokens=100,
                            total_tokens=200,
                        ),
                    ),
                    index=0,
                )
            ],
        )
    )
    successors = successor_fn(root)
    rich.print(successors)
    assert len(successors) >= 1
    # Verify that each successor has a valid plan
    for successor in successors:
        assert len(successor.plan.steps) > 0
        assert all(isinstance(step, PlanStep) for step in successor.plan.steps)


def test_when_empty_plan() -> None:
    # Create a simple repo tree
    repo_tree = """
    |-- tool_api.py
    """

    retriever = build_retriever_for_mnms()
    search_tool = DirectIntentSearchTool(symbol_retriever=retriever)

    user_query = "I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment."

    starting_symbols = [_.symbol for _ in retriever([user_query])]

    successor_fn = UnconstrainedXmlOutputSuccessorFunction(
        search_tool=search_tool,
        starting_symbols=starting_symbols,
        repo_tree=repo_tree,
    )
    root = Node(
        plan=Plan(
            user_query="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
            steps=[],
        )
    )
    successors = successor_fn(root)
    rich.print(successors)
    assert len(successors) >= 1
    # Verify that each successor has a valid plan
    for successor in successors:
        assert len(successor.plan.steps) > 0
        assert all(isinstance(step, PlanStep) for step in successor.plan.steps)


def test_sampling_multiple_successors() -> None:
    repo_tree = """
    |-- tool_api.py
    """

    retriever = build_retriever_for_mnms()
    search_tool = DirectIntentSearchTool(symbol_retriever=retriever)

    user_query = "I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment."

    starting_symbols = [_.symbol for _ in retriever([user_query])]

    beam_width = 3
    successor_fn = UnconstrainedXmlOutputSuccessorFunction(
        search_tool=search_tool,
        starting_symbols=starting_symbols,
        repo_tree=repo_tree,
        beam_width=beam_width,
    )
    root = Node(
        plan=Plan(
            user_query=user_query,
            steps=[
                PlanStep(
                    content="Analyze the sentiment of the letter.",
                    search_result=CodeSearchToolOutput(
                        symbol_name="text_classification",
                        satisfies_intention=True,
                        justification="The function `text_classification` can be used to analyze the sentiment of the letter.",
                        instrumentation=CodeSearchInstrumentation(
                            symbols_considered=list(
                                retriever(["Analyze the sentiment of the letter."])
                            ),
                            completion_tokens=100,
                            prompt_tokens=100,
                            total_tokens=200,
                        ),
                    ),
                    index=0,
                )
            ],
        )
    )
    successors = successor_fn(root)
    rich.print(successors)
    assert len(successors) == beam_width
