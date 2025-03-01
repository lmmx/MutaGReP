from typing import Generic

from mutagrep.plan_search.domain_models import Node

from .domain_models import GoalTestFunction, GoalTestT, PlanStepT
from .typing_utils import implements


class StubHasBeenVisitedFunction(Generic[PlanStepT]):
    """Stub implementation of the HasBeenVisitedFunction protocol."""

    def __call__(
        self,
        state: Node[PlanStepT, GoalTestT],
    ) -> bool:
        return False


implements(GoalTestFunction)(StubHasBeenVisitedFunction)
