# SPDX-License-Identifier: Apache-2.0
from threading import RLock
from typing import Any, ClassVar, Dict, Generic, Type, TypeVar, cast

T = TypeVar("T")  # Type variable for the class using SingletonMeta


class SingletonMeta(type, Generic[T]):
    """A metaclass that implements the singleton pattern.

    This metaclass ensures that only one instance of a class is created.
    The instance is stored in a class-level dictionary and returned on subsequent calls.

    Attributes:
        __lock: A lock to ensure thread safety when creating instances
        _instances: A dictionary mapping class types to their singleton instances
    """

    _lock: ClassVar[RLock] = RLock()  # Using RLock to prevent deadlock in dependent singletons
    _instances: ClassVar[Dict[Type[Any], Any]] = {}

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        """Create or return the singleton instance of the class."""
        # Always prevent direct instantiation
        raise TypeError(f"{cls.__name__} cannot be instantiated directly. Use {cls.__name__}.instance() instead.")

    def instance(cls: Type[T]) -> T:
        """Return the singleton instance of the class.

        Args:
            cls: The class to create an instance of

        Returns:
            The singleton instance of the class
        """
        if cls not in SingletonMeta._instances:
            with SingletonMeta._lock:
                if cls not in SingletonMeta._instances:
                    # Create a new instance using type.__call__
                    instance = super().__call__()
                    SingletonMeta._instances[cls] = instance
        return cast(T, SingletonMeta._instances[cls])
