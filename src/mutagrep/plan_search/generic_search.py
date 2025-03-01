from collections.abc import Callable, Sequence
from enum import Enum
from typing import Generic

from loguru import logger
from pydantic import BaseModel
from typing_extensions import assert_never

from mutagrep.plan_search.components import AlwaysReturnsVisitedFalse
from mutagrep.plan_search.domain_models import (
    GoalTestFunction,
    GoalTestT,
    HasBeenVisitedFunction,
    MetricT,
    Node,
    PlanStepT,
    ProblemRecordT,
    SearchContainer,
    SuccessorFunction,
)


class SearchState(Enum):
    QUEUE_EMPTY = "QUEUE_EMPTY"
    GOAL_FOUND = "GOAL_FOUND"
    STEP_COMPLETE = "STEP_COMPLETE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


class SearchResult(BaseModel, Generic[PlanStepT, GoalTestT]):
    search_state: SearchState
    nodes: list[Node[PlanStepT, GoalTestT]]


class PlanSearchForProblemOutput(
    BaseModel,
    Generic[PlanStepT, GoalTestT, MetricT, ProblemRecordT],
):
    search_result: SearchResult[PlanStepT, GoalTestT]
    metrics: list[tuple[MetricT, Node[PlanStepT, GoalTestT]]]
    problem_record: ProblemRecordT


class PlanSearcher(Generic[PlanStepT, GoalTestT]):
    def __init__(
        self,
        initial_state: Node[PlanStepT, GoalTestT],
        successor_fn: SuccessorFunction[PlanStepT],
        check_is_goal_state_fn: GoalTestFunction[PlanStepT, GoalTestT],
        container_factory: Callable[[], SearchContainer[Node[PlanStepT, GoalTestT]]],
        check_has_been_visited_fn: HasBeenVisitedFunction[
            PlanStepT,
            GoalTestT,
        ] = AlwaysReturnsVisitedFalse(),
        node_budget: int | None = None,
        beam_width: int | None = None,
        beam_depth: int | None = None,
    ):
        self.unvisited_nodes: SearchContainer[Node[PlanStepT, GoalTestT]] = (
            container_factory()
        )
        self.unvisited_nodes.append(initial_state)
        self.visited_nodes: list[Node[PlanStepT, GoalTestT]] = []
        self.level = initial_state.level
        self.successor_fn = successor_fn
        self.is_goal_state = check_is_goal_state_fn
        self.node_budget = node_budget
        self.beam_width = beam_width
        self.beam_depth = beam_depth

    @property
    def nodes(self) -> list[Node[PlanStepT, GoalTestT]]:
        """Returns every node created during the search.
        This includes nodes that are in the queue and nodes that have been visited.
        """
        return self.visited_nodes + list(self.unvisited_nodes)

    @staticmethod
    def _expand_node(
        node: Node[PlanStepT, GoalTestT],
        successor_fn: SuccessorFunction[PlanStepT],
        visited_nodes: list[Node[PlanStepT, GoalTestT]],
        unvisited_nodes: SearchContainer[Node[PlanStepT, GoalTestT]],
        beam_depth: int | None,
        node_budget: int | None,
        beam_width: int | None,
    ) -> Sequence[Node[PlanStepT, GoalTestT]]:
        if beam_depth is not None and node.level >= beam_depth:
            logger.info(
                f"Skipping expansion. beam_depth={beam_depth} >= node.level={node.level}",
            )
            return []

        successors = successor_fn(node)

        if beam_width is not None and len(successors) > beam_width:
            successors = successors[:beam_width]

        if node_budget is not None:
            budget_remaining = node_budget - len(visited_nodes) - len(unvisited_nodes)
            if budget_remaining <= 0:
                logger.info(
                    f"Skipping expansion. budget_used={len(visited_nodes) + len(unvisited_nodes)} >= node_budget={node_budget}",
                )
                return []
            successors = successors[:budget_remaining]

        return successors

    def expand_node(
        self,
        node: Node[PlanStepT, GoalTestT],
    ) -> Sequence[Node[PlanStepT, GoalTestT]]:
        return self._expand_node(
            node=node,
            successor_fn=self.successor_fn,
            visited_nodes=self.visited_nodes,
            unvisited_nodes=self.unvisited_nodes,
            beam_depth=self.beam_depth,
            node_budget=self.node_budget,
            beam_width=self.beam_width,
        )

    def step(self) -> SearchState:
        if self.node_budget is not None:
            budget_expended = len(self.visited_nodes) + len(self.unvisited_nodes)
            budget_remaining = self.node_budget - budget_expended
            if budget_remaining <= 0:
                logger.info(
                    f"Terminating search. Budget exceeded. budget_expended={len(self.visited_nodes)} + {len(self.unvisited_nodes)}"
                    f" >= node_budget={self.node_budget}",
                )
                return SearchState.BUDGET_EXCEEDED

        if self.unvisited_nodes:
            node = self.unvisited_nodes.peek_left()

            assert node is not None

            node = self.unvisited_nodes.popleft()
            self.visited_nodes.append(node)
            node.visited = True
            logger.info(
                f"Visiting node. node.level={node.level} plan_length={len(node.plan.steps)} queue_size={len(self.unvisited_nodes)}",
            )

            node.plan.goal_test = (goal_test := self.is_goal_state(node))

            # Check if the node is a goal state and if so, terminate the search.
            if goal_test:
                logger.info(f"Goal found at level {node.level}")
                return SearchState.GOAL_FOUND

            successors = self.expand_node(node)
            logger.info(
                f"Expanded node at level {node.level} into {len(successors)} successors",
            )

            for successor in successors:
                self.unvisited_nodes.append(successor)

            return SearchState.STEP_COMPLETE
        else:
            return SearchState.QUEUE_EMPTY

    def run(self) -> SearchResult[PlanStepT, GoalTestT]:
        logger.info(f"Initial state: {self.unvisited_nodes.peek_left()}")

        logger.info("Running search indefinitely until node expansion limit reached.")
        while True:
            search_state = self.step()
            match search_state:
                case SearchState.GOAL_FOUND:
                    break
                case SearchState.BUDGET_EXCEEDED:
                    break
                case SearchState.QUEUE_EMPTY:
                    break
                case SearchState.STEP_COMPLETE:
                    pass
                case _:
                    assert_never(search_state)
        return SearchResult(
            search_state=search_state,
            nodes=self.nodes,
        )
