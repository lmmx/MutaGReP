from typing import Generic

from mutagrep.plan_search.domain_models import GoalTestT, Node, RankingFunction
from mutagrep.plan_search.typing_utils import implements

from ..components import PlanStep
from ..lca_benchmark import (LongCodeArenaRecord,
                             score_plan_for_record_multisymbol)


class LongCodeArenaOracleRanker(Generic[GoalTestT]):
    def __init__(self, record: LongCodeArenaRecord):
        self.record = record

    def __call__(self, state: Node[PlanStep, GoalTestT]) -> float:
        scorable_plan = [step for step in state.plan.steps]
        score = score_plan_for_record_multisymbol(self.record, scorable_plan)
        return score.recall


implements(RankingFunction)(LongCodeArenaOracleRanker)
