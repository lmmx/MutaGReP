import functools
from typing import Callable, Type, TypeVar

P = TypeVar("P")
Q = TypeVar("Q")


def implements(protocol: Type[P]) -> Callable[[Type[P]], Type[P]]:
    def decorator(cls: Type[P]) -> Type[P]:
        # The type checker will enforce that `cls` matches the `protocol` without casting.
        @functools.wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)

        # Returning the original class, which must be type-compatible with the protocol
        return cls

    return decorator
