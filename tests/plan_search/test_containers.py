import pytest

from mutagrep.plan_search.containers import (
    PriorityQueueSearchContainer,
    StackSearchContainer,
)
from mutagrep.plan_search.domain_models import (
    Node,
    Plan,
)


class TestPriorityQueueMinHeap:
    def test_priority_queue_basic_operations(self) -> None:
        # Priority function that prioritizes smaller numbers
        def priority_fn(x: int) -> float:
            return x

        queue = PriorityQueueSearchContainer[int](priority_fn)

        # Test empty queue
        assert not queue
        assert len(queue) == 0
        assert queue.peek_left() is None

        # Test adding items
        queue.append(3)
        queue.append(1)
        queue.append(4)
        queue.append(2)

        assert len(queue) == 4
        assert bool(queue)

        # Test peek
        assert queue.peek_left() == 1  # Smallest number should be first

        # Test popping items (should come out in sorted order)
        assert queue.popleft() == 1
        assert queue.popleft() == 2
        assert queue.popleft() == 3
        assert queue.popleft() == 4

        # Queue should be empty after all pops
        assert not queue
        assert len(queue) == 0

    def test_priority_queue_with_custom_priority(self) -> None:
        # Priority function that prioritizes strings by length
        def priority_fn(x: str) -> float:
            return len(x)

        queue = PriorityQueueSearchContainer[str](priority_fn)

        queue.append("hello")
        queue.append("hi")
        queue.append("greetings")
        queue.append("hey")

        # Should come out in order of string length
        assert queue.popleft() == "hi"
        assert queue.popleft() == "hey"
        assert queue.popleft() == "hello"
        assert queue.popleft() == "greetings"

    def test_priority_queue_empty_pop(self) -> None:
        queue = PriorityQueueSearchContainer[int](lambda x: x)

        with pytest.raises(IndexError):
            queue.popleft()

    def test_priority_queue_same_priority(self) -> None:
        # When items have the same priority, they should maintain FIFO order

        def priority_fn(x: str) -> float:
            return 1

        queue = PriorityQueueSearchContainer[str](priority_fn)

        queue.append("first")
        queue.append("second")
        queue.append("third")

        assert queue.popleft() == "first"
        assert queue.popleft() == "second"
        assert queue.popleft() == "third"


class TestPriorityQueueMaxHeap:
    def test_max_heap_basic_operations(self) -> None:
        # Priority function that prioritizes numbers
        def priority_fn(x: int) -> float:
            return x

        queue = PriorityQueueSearchContainer[int](priority_fn, max_heap=True)

        # Test empty queue
        assert not queue
        assert len(queue) == 0
        assert queue.peek_left() is None

        # Test adding items
        queue.append(3)
        queue.append(1)
        queue.append(4)
        queue.append(2)

        assert len(queue) == 4
        assert bool(queue)

        # Test peek
        assert queue.peek_left() == 4  # Largest number should be first

        # Test popping items (should come out in reverse sorted order)
        assert queue.popleft() == 4
        assert queue.popleft() == 3
        assert queue.popleft() == 2
        assert queue.popleft() == 1

        # Queue should be empty after all pops
        assert not queue
        assert len(queue) == 0

    def test_max_heap_with_custom_priority(self) -> None:
        # Priority function that prioritizes strings by length
        def priority_fn(x: str) -> float:
            return len(x)

        queue = PriorityQueueSearchContainer[str](priority_fn, max_heap=True)

        queue.append("hello")
        queue.append("hi")
        queue.append("greetings")
        queue.append("hey")

        # Should come out in order of decreasing string length
        assert queue.popleft() == "greetings"
        assert queue.popleft() == "hello"
        assert queue.popleft() == "hey"
        assert queue.popleft() == "hi"

    def test_max_heap_same_priority(self) -> None:
        # When items have the same priority, they should maintain FIFO order
        def priority_fn(x: str) -> float:
            return 1

        queue = PriorityQueueSearchContainer[str](priority_fn, max_heap=True)

        queue.append("first")
        queue.append("second")
        queue.append("third")

        assert queue.popleft() == "first"
        assert queue.popleft() == "second"
        assert queue.popleft() == "third"

    def test_interleaved_append_and_pop(self) -> None:
        queue = PriorityQueueSearchContainer[int](lambda x: x, max_heap=True)

        queue.append(1)
        queue.append(2)
        assert queue.peek_left() == 2
        assert queue.popleft() == 2
        queue.append(5)
        assert queue.peek_left() == 5
        assert queue.popleft() == 5

        queue.append(8)
        queue.append(9)
        queue.append(10)

        assert queue.peek_left() == 10
        assert queue.popleft() == 10
        assert queue.peek_left() == 9
        assert queue.popleft() == 9
        assert queue.peek_left() == 8
        assert queue.popleft() == 8
        assert queue.peek_left() == 1
        assert queue.popleft() == 1


def test_stack():
    container = StackSearchContainer[Node[int, bool]]()
    container.append(Node(plan=Plan(user_query="", steps=[]), level=1))
    container.append(Node(plan=Plan(user_query="", steps=[]), level=2))
    assert container.popleft().level == 2
