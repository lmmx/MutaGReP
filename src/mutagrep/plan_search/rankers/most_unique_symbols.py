from typing import Generic

from mutagrep.plan_search.domain_models import GoalTestT, Node, RankingFunction
from mutagrep.plan_search.typing_utils import implements

from ..components import PlanStep


class MostUniqueSymbolsRanker(Generic[GoalTestT]):
    def __call__(self, state: Node[PlanStep, GoalTestT]) -> float:
        scorable_plan = [step for step in state.plan.steps]
        symbol_names: set[str] = set()
        for step in scorable_plan:
            if step.search_result.instrumentation is None:
                raise ValueError(
                    "Multi-symbol scoring relies on using the instrumentation object "
                    "to get the symbols considered. But it was none."
                )
            symbol_names.update(
                retrieved_symbol.symbol.full_path
                for retrieved_symbol in step.search_result.instrumentation.symbols_considered
            )
        return len(symbol_names)


implements(RankingFunction)(MostUniqueSymbolsRanker)
