import pytest
import rich
from pydantic import ValidationError

from mutagrep.plan_search import mnms_benchmark
from mutagrep.plan_search.components import (
    GoalTestPlanSatisfiesUserRequest,
    LlmPlan,
    LlmPlanStep,
    PlanStep,
    SuccessorFunctionAddOrRemoveLastStep,
    SuccessorFunctionAddOrRemoveLastStepTextOnly,
    SuccessorFunctionMonotonicAddStep,
)
from mutagrep.plan_search.domain_models import CodeSearchToolOutput, Node, Plan
from mutagrep.plan_search.mnms_search_tool import MnmsSimpleCodeSearchTool
from mutagrep.plan_search.successor_functions import (
    plan_diff_successor_fn,
    xml_like,
    xml_like_sampling,
)


class TestXmlLikeSamplingBasedSuccessorFnMonotonic:
    @staticmethod
    def test_when_non_empty_plan() -> None:
        successor_fn = xml_like_sampling.XmlOutputSuccessorFunction(
            search_tool=MnmsSimpleCodeSearchTool(),
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
                        ),
                        index=0,
                    ),
                ],
            ),
        )
        successors = successor_fn(root)
        rich.print(successors)
        assert len(successors) >= 1

    @staticmethod
    def test_when_empty_plan() -> None:
        successor_fn = xml_like_sampling.XmlOutputSuccessorFunction(
            search_tool=MnmsSimpleCodeSearchTool(),
        )
        root = Node(
            plan=Plan(
                user_query="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
                steps=[],
            ),
        )
        successors = successor_fn(root)
        rich.print(successors)
        assert len(successors) == 1

    @staticmethod
    def test_sampling_multiple_successors() -> None:
        beam_width = 3
        successor_fn = xml_like_sampling.XmlOutputSuccessorFunction(
            search_tool=MnmsSimpleCodeSearchTool(),
            beam_width=beam_width,
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
                        ),
                        index=0,
                    ),
                ],
            ),
        )
        successors = successor_fn(root)
        rich.print(successors)
        assert len(successors) == beam_width


class TestXmlLikeSuccessorFnMonotonic:
    @staticmethod
    def test_when_non_empty_plan() -> None:
        successor_fn = xml_like.XmlOutputSuccessorFunction(
            allowed_actions=xml_like.MONOTONIC_ALLOWED_ACTIONS,
            search_tool=MnmsSimpleCodeSearchTool(),
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
                        ),
                        index=0,
                    ),
                ],
            ),
        )
        successors = successor_fn(root)
        rich.print(successors)

    @staticmethod
    def test_when_empty_plan() -> None:
        successor_fn = xml_like.XmlOutputSuccessorFunction(
            allowed_actions=xml_like.MONOTONIC_ALLOWED_ACTIONS,
            search_tool=MnmsSimpleCodeSearchTool(),
        )
        root = Node(
            plan=Plan(
                user_query="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
                steps=[],
            ),
        )
        successors = successor_fn(root)
        rich.print(successors)


class TestPlanDiffSuccessorFnAddOrRemoveLastStep:
    @staticmethod
    def test_when_non_empty_plan() -> None:
        successor_fn = plan_diff_successor_fn.AppendNewStepOrRemoveLastStep(
            search_tool=MnmsSimpleCodeSearchTool(),
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
                        ),
                        index=0,
                    ),
                ],
            ),
        )
        successors = successor_fn(root)
        rich.print(successors)

    @staticmethod
    def test_when_empty_plan() -> None:
        successor_fn = plan_diff_successor_fn.AppendNewStepOrRemoveLastStep(
            search_tool=MnmsSimpleCodeSearchTool(),
        )
        root = Node(
            plan=Plan(
                user_query="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
                steps=[],
            ),
        )
        successors = successor_fn(root)
        rich.print(successors)


class TestSuccessorFunctionAddOrRemoveLastStep:
    @staticmethod
    def test_when_non_empty_plan() -> None:
        successor_fn = SuccessorFunctionAddOrRemoveLastStep(
            search_tool=MnmsSimpleCodeSearchTool(),
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
                        ),
                        index=0,
                    ),
                ],
            ),
        )
        successors = successor_fn(root)
        rich.print(successors)

    @staticmethod
    def test_when_empty_plan() -> None:
        successor_fn = SuccessorFunctionAddOrRemoveLastStep(
            search_tool=MnmsSimpleCodeSearchTool(),
        )
        root = Node(
            plan=Plan(
                user_query="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
                steps=[],
            ),
        )
        successors = successor_fn(root)
        rich.print(successors)


class TestSuccessorFunctionAddOrRemoveLastStepTextOnly:
    @staticmethod
    def test_when_non_empty_plan() -> None:
        successor_fn = SuccessorFunctionAddOrRemoveLastStepTextOnly(
            search_tool=MnmsSimpleCodeSearchTool(),
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
                        ),
                        index=0,
                    ),
                ],
            ),
        )
        successors = successor_fn(root)
        rich.print(successors)

    @staticmethod
    def test_when_empty_plan() -> None:
        successor_fn = SuccessorFunctionAddOrRemoveLastStepTextOnly(
            search_tool=MnmsSimpleCodeSearchTool(),
        )
        root = Node(
            plan=Plan(
                user_query="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
                steps=[],
            ),
        )
        successors = successor_fn(root)
        rich.print(successors)


class TestSuccessorFunctionMonotonicAddStep:
    @staticmethod
    def test_when_non_empty_plan() -> None:
        successor_fn = SuccessorFunctionMonotonicAddStep(
            search_tool=MnmsSimpleCodeSearchTool(),
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
                        ),
                        index=0,
                    ),
                ],
            ),
        )
        successors = successor_fn(root)
        rich.print(successors)
        assert len(successors[0].plan.steps) == 2

    @staticmethod
    def test_when_empty_plan() -> None:
        successor_fn = SuccessorFunctionMonotonicAddStep(
            search_tool=MnmsSimpleCodeSearchTool(),
        )
        root = Node(
            plan=Plan(
                user_query="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
                steps=[],
            ),
        )

        successors = successor_fn(root)
        rich.print(successors)
        assert len(successors[0].plan.steps) == 1


class TestGoalTestPlanSatisfiesUserRequest:
    @staticmethod
    def test_when_plan_satisfies_user_request() -> None:
        goal_test_fn = GoalTestPlanSatisfiesUserRequest()
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
                        ),
                        index=0,
                    ),
                    PlanStep(
                        content="Create an image inspired by the sentiment.",
                        search_result=CodeSearchToolOutput(
                            symbol_name="image_generation",
                            satisfies_intention=True,
                            justification="The function `image_generation` can be used to create an image inspired by the sentiment.",
                        ),
                        index=1,
                    ),
                ],
            ),
        )

        goal_test = goal_test_fn(root)
        assert goal_test

    @staticmethod
    def test_when_plan_does_not_satisfy_user_request() -> None:
        goal_test_fn = GoalTestPlanSatisfiesUserRequest()
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
                        ),
                        index=0,
                    ),
                ],
            ),
        )

        goal_test = goal_test_fn(root)
        assert not goal_test


class TestLlmPlanDataModel:
    @staticmethod
    def test_when_edit_type_is_add_new_step_and_steps_is_empty() -> None:
        with pytest.raises(ValidationError):
            LlmPlan(
                steps=[],
                edit_type="add_new_step",
            )

    @staticmethod
    def test_when_edit_type_is_add_new_step_and_steps_is_not_empty() -> None:
        LlmPlan(
            steps=[
                LlmPlanStep(content="Analyze the sentiment of the letter.", index=0),
            ],
            edit_type="add_new_step",
        )

    @staticmethod
    def test_when_edit_type_is_remove_last_step_and_steps_is_empty() -> None:
        with pytest.raises(ValidationError):
            LlmPlan(
                steps=[],
                edit_type="remove_last_step",
            )


def test_scoring_perfect_plan() -> None:
    record = mnms_benchmark.MnmsRecord(
        id=0,
        user_request="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
        plan=[
            mnms_benchmark.MnmsPlanStep(id=0, name="text_classification", args=None),
            mnms_benchmark.MnmsPlanStep(id=1, name="image_generation", args=None),
        ],
        code="",
        alt_plans=[],
    )

    # Create a plan that is a perfect match for the record
    plan = Plan(
        user_query="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
        steps=[
            PlanStep(
                content="Analyze the sentiment of the letter.",
                index=0,
                search_result=CodeSearchToolOutput(
                    symbol_name="text_classification",
                    satisfies_intention=True,
                    justification="The function `text_classification` can be used to analyze the sentiment of the letter.",
                ),
            ),
            PlanStep(
                content="Create an image inspired by the sentiment.",
                index=1,
                search_result=CodeSearchToolOutput(
                    symbol_name="image_generation",
                    satisfies_intention=True,
                    justification="The function `image_generation` can be used to create an image inspired by the sentiment.",
                ),
            ),
        ],
    )

    scorable = [_.to_mnms_plan_step() for _ in plan.steps]

    score = mnms_benchmark.score_plan_for_record(record, scorable)

    assert score.precision == 1
    assert score.recall == 1
    assert score.f1 == 1
    assert score.length_error == 0
