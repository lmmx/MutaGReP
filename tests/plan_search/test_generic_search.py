from collections import defaultdict
from collections.abc import Sequence
from typing import cast

from mutagrep.plan_search.containers import (
    DequeSearchContainer,
    PriorityQueueSearchContainer,
    StackSearchContainer,
)
from mutagrep.plan_search.domain_models import (
    GoalTestT,
    Node,
    Plan,
    PlanStepT,
)
from mutagrep.plan_search.generic_search import (
    PlanSearcher,
    SearchState,
)


class StubGoalTest:
    def __call__(self, state: Node[PlanStepT, bool]) -> bool:
        return True


class StubSuccessorFunction:
    def __call__(
        self,
        state: Node[PlanStepT, GoalTestT],
    ) -> Sequence[Node[PlanStepT, GoalTestT]]:
        return []


class FanOutNSuccessorFunction:
    def __init__(self, n: int):
        self.n = n

    def __call__(
        self,
        state: Node[PlanStepT, GoalTestT],
    ) -> Sequence[Node[PlanStepT, GoalTestT]]:
        level = state.level + 1
        return [
            Node(
                plan=Plan(user_query=state.plan.user_query, steps=state.plan.steps),
                level=level,
                parent=state,
            )
            for _ in range(self.n)
        ]


class FindGoalAtLevelSuccessorFunction:
    def __init__(self, goal_level: int):
        self.goal_level = goal_level

    def __call__(self, state: Node[PlanStepT, bool]) -> bool:
        return state.level == self.goal_level


def test_bfs_planner():
    container = DequeSearchContainer[Node[int, bool]]()
    root = Node(plan=Plan(user_query="", steps=[]))
    planner = PlanSearcher(
        initial_state=root,
        successor_fn=StubSuccessorFunction(),
        check_is_goal_state_fn=StubGoalTest(),
        container_factory=lambda: container,
        check_has_been_visited_fn=lambda *args, **kwargs: False,
    )
    assert planner.step() == SearchState.GOAL_FOUND


def test_bfs_planner_finds_goal_at_level():
    container = DequeSearchContainer[Node[int, bool]]()
    steps = cast(list[int], [1])
    root = Node(plan=Plan(user_query="", steps=steps))
    planner = PlanSearcher(
        initial_state=root,
        successor_fn=FanOutNSuccessorFunction(n=2),
        check_is_goal_state_fn=FindGoalAtLevelSuccessorFunction(goal_level=4),
        container_factory=lambda: container,
        check_has_been_visited_fn=lambda *args, **kwargs: False,
    )
    steps_needed_to_find_goal = 2**4
    for _ in range(steps_needed_to_find_goal - 1):
        assert planner.step() == SearchState.STEP_COMPLETE

    assert planner.step() == SearchState.GOAL_FOUND


def test_node_budget_not_exceeded():
    container = DequeSearchContainer[Node[int, bool]]()
    root = Node(plan=Plan(user_query="", steps=[]), level=1)
    planner = PlanSearcher(
        initial_state=root,
        successor_fn=FanOutNSuccessorFunction(n=2),
        check_is_goal_state_fn=lambda *args, **kwargs: cast(bool, False),
        container_factory=lambda: container,
        check_has_been_visited_fn=lambda *args, **kwargs: False,
        node_budget=5,
    )
    result = planner.run()
    assert len(result.nodes) == 5


def test_dfs_planner():
    container = StackSearchContainer[Node[int, bool]]()
    root = Node(plan=Plan(user_query="", steps=[]))
    planner = PlanSearcher(
        initial_state=root,
        successor_fn=StubSuccessorFunction(),
        check_is_goal_state_fn=StubGoalTest(),
        container_factory=lambda: container,
        check_has_been_visited_fn=lambda *args, **kwargs: False,
    )
    assert planner.step() == SearchState.GOAL_FOUND


def test_dfs_planner_finds_goal_at_level():
    container = StackSearchContainer[Node[int, bool]]()
    steps = cast(list[int], [1])
    root = Node(plan=Plan(user_query="", steps=steps), level=0)
    planner = PlanSearcher(
        initial_state=root,
        successor_fn=FanOutNSuccessorFunction(n=2),
        check_is_goal_state_fn=FindGoalAtLevelSuccessorFunction(goal_level=4),
        container_factory=lambda: container,
        check_has_been_visited_fn=lambda *args, **kwargs: False,
    )

    # The first step should result in enqueuing 2 successors.
    # The successors will be at level 1.
    assert planner.step() == SearchState.STEP_COMPLETE
    assert len(container) == 2
    assert (next_node := container.peek_left()) is not None and next_node.level == 1

    # We will then expand the first successor, which will result in enqueuing 2 more successors.
    # The successors will be at level 2.
    assert planner.step() == SearchState.STEP_COMPLETE
    assert len(container) == 3
    assert (next_node := container.peek_left()) is not None and next_node.level == 2

    # We will then expand the second successor, which will result in enqueuing 2 more successors.
    # The successors will be at level 3.
    assert planner.step() == SearchState.STEP_COMPLETE
    assert len(container) == 4
    assert (next_node := container.peek_left()) is not None and next_node.level == 3

    # We will then expand the third successor, which will result in enqueuing 2 more successors.
    # The successors will be at level 4.
    assert planner.step() == SearchState.STEP_COMPLETE
    assert len(container) == 5
    assert (next_node := container.peek_left()) is not None and next_node.level == 4

    # We will then reach the fourth successor, which will be the goal.
    assert planner.step() == SearchState.GOAL_FOUND


def test_dfs_planner_expends_budget_within_beam_depth_with_beam_width_gt_1():
    """BFS and DFS behave differently when the beam width is > 1.
    With DFS, if the beam width is > 1 and a budget (max nodes to expand) is set,
    the budget will be spent entirely making one beam deeper.
    To ensure _search_ actually happens, we need to set a maximum beam depth.
    This test checks that the DFS search respects the maximum beam depth and
    expends the budget within the beam depth.
    """
    container = StackSearchContainer[Node[int, bool]]()
    root = Node(plan=Plan(user_query="", steps=[]))

    # If the search respects both the beam width and the budget, it will
    # create 2 beams and expand them to depth 4.
    beam_width = 2
    beam_depth = 4
    budget = beam_width * beam_depth

    planner = PlanSearcher(
        initial_state=root,
        successor_fn=FanOutNSuccessorFunction(n=beam_width),
        check_is_goal_state_fn=lambda *args, **kwargs: cast(bool, False),
        container_factory=lambda: container,
        check_has_been_visited_fn=lambda *args, **kwargs: False,
        beam_depth=beam_depth,
        beam_width=beam_width,
        node_budget=budget,
    )

    result = planner.run()

    # Check that we have expanded a number of nodes equal to the budget.
    assert len(result.nodes) == budget

    # Check that we have 2 nodes at each depth.
    for depth in range(beam_depth):
        if depth == 0:
            # The root node is not part of the beam.
            continue
        beam_width_at_depth = sum(1 for node in result.nodes if node.level == depth)
        assert beam_width_at_depth == beam_width


def test_dfs_planner_expends_budget_within_beam_depth_with_beam_width_1():
    container = StackSearchContainer[Node[int, bool]]()
    root = Node(plan=Plan(user_query="", steps=[]))

    beam_width = 1
    beam_depth = 4
    budget = beam_width * beam_depth

    planner = PlanSearcher(
        initial_state=root,
        successor_fn=FanOutNSuccessorFunction(n=beam_width),
        check_is_goal_state_fn=lambda *args, **kwargs: cast(bool, False),
        container_factory=lambda: container,
        check_has_been_visited_fn=lambda *args, **kwargs: False,
        beam_depth=beam_depth,
        beam_width=beam_width,
        node_budget=budget,
    )

    result = planner.run()

    # Check that we have expanded a number of nodes equal to the budget.
    assert len(result.nodes) == budget

    # Check that we have 1 node at each depth.
    for depth in range(beam_depth):
        if depth == 0:
            # The root node is not part of the beam.
            continue
        beam_width_at_depth = sum(1 for node in result.nodes if node.level == depth)
        assert beam_width_at_depth == beam_width


def test_dfs_planner_expends_budget_within_beam_depth_with_beam_width_1_and_overbudgeted():
    container = StackSearchContainer[Node[int, bool]]()
    root = Node(plan=Plan(user_query="", steps=[]))

    beam_width = 1
    beam_depth = 4
    # Set the budget to be greater than the number of nodes
    # that would be created by expanding the beam to the maximum depth.
    budget = beam_width * beam_depth * 2

    planner = PlanSearcher(
        initial_state=root,
        successor_fn=FanOutNSuccessorFunction(n=beam_width),
        check_is_goal_state_fn=lambda *args, **kwargs: cast(bool, False),
        container_factory=lambda: container,
        check_has_been_visited_fn=lambda *args, **kwargs: False,
        beam_depth=beam_depth,
        beam_width=beam_width,
        node_budget=budget,
    )

    result = planner.run()

    # Check that the plan search terminated with an empty queue.
    assert result.search_state == SearchState.QUEUE_EMPTY

    # Check that we have expanded a number of nodes
    # equal to beam_width * beam_depth + 1.
    # The +1 comes from the following sequence of events:
    # 1. The root node is expanded. visited=1, unvisited=1 depth=1
    # 2. The successors are expanded. visited=2, unvisited=1 depth=2
    # 3. The successors are expanded. visited=3, unvisited=1 depth=3
    # 4. The successors are expanded. visited=4, unvisited=1 depth=4
    # The node at depth 4 is not expanded because expanding it
    # would exceed the beam depth.
    assert len(result.nodes) == beam_width * beam_depth + 1

    # Check that we have 1 node at each depth.
    for depth in range(beam_depth):
        if depth == 0:
            # The root node is not part of the beam.
            continue
        beam_width_at_depth = sum(1 for node in result.nodes if node.level == depth)
        assert beam_width_at_depth == beam_width


def test_dfs_planner_expends_budget_within_beam_depth_with_beam_width_gt_1_and_overbudgeted():
    container = StackSearchContainer[Node[int, bool]]()
    root = Node(plan=Plan(user_query="", steps=[]))

    beam_width = 2
    beam_depth = 4
    # Set the budget to be greater than the number of nodes
    # that would be created by expanding the beam to the maximum depth.
    budget = beam_width * beam_depth * (beam_width + 2)

    planner = PlanSearcher(
        initial_state=root,
        successor_fn=FanOutNSuccessorFunction(n=beam_width),
        check_is_goal_state_fn=lambda *args, **kwargs: cast(bool, False),
        container_factory=lambda: container,
        check_has_been_visited_fn=lambda *args, **kwargs: False,
        beam_depth=beam_depth,
        beam_width=beam_width,
        node_budget=budget,
    )

    result = planner.run()

    # Check that the plan search terminated with an empty queue.
    assert result.search_state == SearchState.QUEUE_EMPTY

    level_to_nodes = defaultdict(list)
    for node in result.nodes:
        level_to_nodes[node.level].append(node)

    # Print the distribution of nodes at each level.
    for level, nodes in level_to_nodes.items():
        print(f"Level {level}: {len(nodes)} nodes")
        assert len(nodes) == beam_width**level


def test_best_first_planner():
    container = PriorityQueueSearchContainer[Node[int, bool]](
        priority_function=lambda node: node.level,
        max_heap=True,
    )
    root = Node(plan=Plan(user_query="", steps=[]))
    planner = PlanSearcher(
        initial_state=root,
        successor_fn=StubSuccessorFunction(),
        check_is_goal_state_fn=StubGoalTest(),
        container_factory=lambda: container,
        check_has_been_visited_fn=lambda *args, **kwargs: False,
    )
    assert planner.step() == SearchState.GOAL_FOUND


def test_best_first_planner_finds_goal_at_level():
    container = PriorityQueueSearchContainer[Node[int, bool]](
        priority_function=lambda node: node.level,
        max_heap=True,
    )
    root = Node(plan=Plan(user_query="", steps=[]))
    planner = PlanSearcher(
        initial_state=root,
        successor_fn=FanOutNSuccessorFunction(n=2),
        check_is_goal_state_fn=FindGoalAtLevelSuccessorFunction(goal_level=4),
        container_factory=lambda: container,
        check_has_been_visited_fn=lambda *args, **kwargs: False,
    )

    # It should find the goal in 4 steps.
    for _ in range(4):
        assert planner.step() == SearchState.STEP_COMPLETE

    assert planner.step() == SearchState.GOAL_FOUND


def test_best_first_planner_expends_budget_within_beam_depth_with_beam_width_gt_1_and_underbudgeted():
    container = PriorityQueueSearchContainer[Node[int, bool]](
        priority_function=lambda node: node.level,
        max_heap=True,
    )
    root = Node(plan=Plan(user_query="", steps=[]))

    beam_width = 6
    beam_depth = 12
    budget = 60

    planner = PlanSearcher(
        initial_state=root,
        successor_fn=FanOutNSuccessorFunction(n=beam_width),
        check_is_goal_state_fn=lambda *args, **kwargs: cast(bool, False),
        container_factory=lambda: container,
        check_has_been_visited_fn=lambda *args, **kwargs: False,
        beam_depth=beam_depth,
        beam_width=beam_width,
        node_budget=budget,
    )

    result = planner.run()

    # Check that the plan search terminated with an empty queue.
    assert result.search_state == SearchState.BUDGET_EXCEEDED

    level_to_nodes = defaultdict(list)
    for node in result.nodes:
        level_to_nodes[node.level].append(node)

    # Print the distribution of nodes at each level.
    for level, nodes in level_to_nodes.items():
        print(f"Level {level}: {len(nodes)} nodes")
        # assert len(nodes) == beam_width**level
