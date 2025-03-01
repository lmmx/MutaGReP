import heapq
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Generic

from mutagrep.plan_search.domain_models import SearchContainer, T
from mutagrep.plan_search.typing_utils import implements


class DequeSearchContainer(Generic[T]):
    """A wrapper for deque that implements the SearchContainer protocol."""

    def __init__(self):
        self._deque: deque[T] = deque()

    def append(self, item: T) -> None:
        self._deque.append(item)

    def popleft(self) -> T:
        return self._deque.popleft()

    def __bool__(self) -> bool:
        return bool(self._deque)

    def __len__(self) -> int:
        return len(self._deque)

    def peek_left(self) -> T | None:
        return self._deque[0] if self._deque else None

    def __iter__(self) -> Iterator[T]:
        return iter(self._deque)


implements(SearchContainer)(DequeSearchContainer)


class StackSearchContainer(Generic[T]):
    """A wrapper for list that implements the SearchContainer protocol as a LIFO stack."""

    def __init__(self):
        self._stack: list[T] = []

    def append(self, item: T) -> None:
        self._stack.append(item)

    def popleft(self) -> T:
        return self._stack.pop()

    def __bool__(self) -> bool:
        return bool(self._stack)

    def __len__(self) -> int:
        return len(self._stack)

    def peek_left(self) -> T | None:
        return self._stack[-1] if self._stack else None

    def __iter__(self) -> Iterator[T]:
        return iter(self._stack)


implements(SearchContainer)(StackSearchContainer)


@dataclass(order=True)
class PrioritizedItem(Generic[T]):
    priority: float
    item: T = field(compare=False)


class PriorityQueueSearchContainer(Generic[T]):
    """A wrapper for heapq that implements the SearchContainer protocol.

    Args:
        priority_function: Function that returns a priority value for each item
        max_heap: If True, operates as a max heap. If False (default), operates as a min heap

    """

    def __init__(self, priority_function: Callable[[T], float], max_heap: bool = False):
        self._heap: list[PrioritizedItem[T]] = []
        self._priority_function = priority_function
        self._max_heap = max_heap

    def append(self, item: T) -> None:
        priority = self._priority_function(item)
        # Negate priority for max heap behavior
        if self._max_heap:
            priority = -priority
        heapq.heappush(self._heap, PrioritizedItem(priority, item))

    def popleft(self) -> T:
        if not self._heap:
            raise IndexError("pop from empty queue")
        return heapq.heappop(self._heap).item

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def peek_left(self) -> T | None:
        return self._heap[0].item if self._heap else None

    def __iter__(self) -> Iterator[T]:
        return (item.item for item in self._heap)


implements(SearchContainer)(PriorityQueueSearchContainer)
