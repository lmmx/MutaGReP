from collections.abc import Iterator, Sequence
from dataclasses import field
from enum import Enum
from typing import (
    Generic,
    Protocol,
    TypeVar,
)

from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import Self, assert_never
from ulid import ULID

from mutagrep.coderec.v3.symbol_mining import Symbol

PlanStepT = TypeVar("PlanStepT", bound=BaseModel | str | int)
GoalTestT = TypeVar("GoalTestT", bound=BaseModel | str | int | bool)
ProblemRecordT = TypeVar("ProblemRecordT", bound=BaseModel)
MetricT = TypeVar("MetricT", bound=BaseModel)
T = TypeVar("T")


class Plan(BaseModel, Generic[PlanStepT, GoalTestT]):
    """Class defining the interface for a plan."""

    user_query: str
    steps: list[PlanStepT]
    reasoning: str | None = None
    goal_test: GoalTestT | None = None


class Node(BaseModel, Generic[PlanStepT, GoalTestT]):
    """Class defining the interface for a node in the search tree."""

    plan: Plan[PlanStepT, GoalTestT]
    parent: Self | None = Field(default=None, repr=False)
    level: int = 0
    children: list[Self] = field(default_factory=list)
    visited: bool = False
    ulid: ULID = field(default_factory=ULID)

    def get_lineage(self) -> list[Self]:
        """Get the lineage of the node."""
        lineage = []
        current = self
        while current:
            lineage.append(current)
            current = current.parent
        return lineage


class SuccessorFunction(Protocol[PlanStepT]):
    """Protocol for successor functions."""

    def __call__(
        self,
        state: Node[PlanStepT, GoalTestT],
    ) -> Sequence[Node[PlanStepT, GoalTestT]]: ...


class GoalTestFunction(Protocol[PlanStepT, GoalTestT]):
    """Protocol for goal test functions."""

    def __call__(self, state: Node[PlanStepT, GoalTestT]) -> GoalTestT: ...


class RankingFunction(Protocol[PlanStepT, GoalTestT]):
    """Protocol for ranking functions."""

    def __call__(self, state: Node[PlanStepT, GoalTestT]) -> float: ...


class HasBeenVisitedFunction(Protocol[PlanStepT, GoalTestT]):
    """Protocol for goal test functions."""

    def __call__(
        self,
        state: Node[PlanStepT, GoalTestT],
        visited: Sequence[Node[PlanStepT, GoalTestT]],
    ) -> bool: ...


class SearchContainer(Protocol, Generic[T]):
    """Protocol for the search container used in BFS."""

    def append(self, item: T) -> None: ...

    def popleft(self) -> T: ...

    def __bool__(self) -> bool: ...

    def __len__(self) -> int: ...

    def peek_left(self) -> T | None: ...

    def __iter__(self) -> Iterator[T]: ...


class SymbolRetrievalScoreType(Enum):
    DISTANCE = "distance"
    SIMILARITY = "similarity"
    NONE = "none"


class RetrievedSymbol(BaseModel):
    symbol: Symbol
    score: float | None = None
    score_type: SymbolRetrievalScoreType = SymbolRetrievalScoreType.NONE


class CodeSearchInstrumentation(BaseModel):
    symbols_considered: list[RetrievedSymbol]
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class CodeSearchToolOutput(BaseModel):
    symbol_name: str | None
    justification: str | None
    satisfies_intention: bool
    instrumentation: CodeSearchInstrumentation | None = None

    def get_top_n_symbols(self, n: int) -> list[RetrievedSymbol]:
        """Get the top n symbols from the search."""
        if self.instrumentation is None:
            raise ValueError(
                "The search tool did not list which symbols were considered.",
            )

        if len(self.instrumentation.symbols_considered) == 0:
            raise ValueError("The list of symbols considered is empty.")

        if n > len(self.instrumentation.symbols_considered):
            logger.warning(
                f"The number of symbols to retrieve ({n}) is greater than the number"
                f" of symbols considered ({len(self.instrumentation.symbols_considered)})",
            )

        score_type = (
            first_symbol := self.instrumentation.symbols_considered[0]
        ).score_type
        # Assert that all symbols have the same score type.
        for symbol in self.instrumentation.symbols_considered:
            if symbol.score_type != score_type:
                raise ValueError(
                    "All symbols must have the same score type "
                    f"({first_symbol.symbol.name} {score_type} "
                    f" != {symbol.symbol.name} {symbol.score_type})",
                )

        # Assert that all symbols _have_ a score.
        for symbol in self.instrumentation.symbols_considered:
            if symbol.score is None:
                raise ValueError(f"The symbol {symbol.symbol.name} has no score.")

        match score_type:
            case SymbolRetrievalScoreType.DISTANCE:
                sort_direction = "ascending"
            case SymbolRetrievalScoreType.SIMILARITY:
                sort_direction = "descending"
            case SymbolRetrievalScoreType.NONE:
                raise ValueError(
                    f"The score type is {score_type} and cannot be ranked.",
                )
            case _:
                assert_never(score_type)

        def sort_key(symbol: RetrievedSymbol) -> float:
            if symbol.score is None:
                raise ValueError(f"The symbol {symbol.symbol.name} has no score.")
            return symbol.score

        return sorted(
            self.instrumentation.symbols_considered,
            key=sort_key,
            reverse=sort_direction == "descending",
        )[:n]


class CodeSearchTool(Protocol):
    def __call__(self, intention: str) -> CodeSearchToolOutput: ...


class SymbolRetriever(Protocol):
    def __call__(
        self,
        queries: Sequence[str],
        n_results: int = 5,
    ) -> Sequence[RetrievedSymbol]: ...
