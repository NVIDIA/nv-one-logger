# SPDX-License-Identifier: Apache-2.0
"""Test for the singleton pattern."""

import threading

import pytest

from nv_one_logger.core.internal.singleton import SingletonMeta


class MySingleton(metaclass=SingletonMeta):
    """Test-only singleton."""

    def __init__(self) -> None:
        """Initialize the MySingleton."""
        self.value = 42


class AnotherSingleton(metaclass=SingletonMeta):
    """Another test-only singleton."""

    def __init__(self) -> None:
        """Initialize the AnotherSingleton."""
        # Simulate dependency: AnotherSingleton might depend on MySingleton
        self.my_singleton = MySingleton.instance()


class TestSingletonMeta:
    """Test class for SingletonMeta tests."""

    def test_singleton_instance_creation(self) -> None:
        """Test that only one instance is created and subsequent calls return the same instance."""
        # First call should create instance
        instance1 = MySingleton.instance()
        assert isinstance(instance1, MySingleton)
        assert instance1.value == 42

        # Second call should return same instance
        instance2 = MySingleton.instance()
        assert instance1 is instance2

    def test_direct_instantiation_prevention(self) -> None:
        """Test that direct instantiation is prevented."""
        with pytest.raises(TypeError, match="cannot be instantiated directly"):
            MySingleton()

    def test_thread_safety(self) -> None:
        """Test that the singleton is thread-safe."""

        def create_instances() -> None:
            AnotherSingleton.instance()
            MySingleton.instance()

        thread1 = threading.Thread(target=create_instances)
        thread2 = threading.Thread(target=create_instances)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()
        thread2.join()
