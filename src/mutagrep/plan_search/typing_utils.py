import functools
from collections.abc import Callable
from typing import TypeVar

P = TypeVar("P")
Q = TypeVar("Q")


def implements(protocol: type[P]) -> Callable[[type[P]], type[P]]:
    def decorator(cls: type[P]) -> type[P]:
        # The type checker will enforce that `cls` matches the `protocol` without casting.
        @functools.wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)

        # Returning the original class, which must be type-compatible with the protocol
        return cls

    return decorator
