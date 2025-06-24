# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, cast

from typing_extensions import TypeAlias

from nv_one_logger.core.exceptions import OneLoggerError, assert_that

# It's best to keep this types as simple as possible to avoid difficulties with serialization
# at the point of sending the data to various backends. Backends such as wandb and OTEL only support primitive
# types and simple lists of primitives. So we need to ensure that the values we use here are
# easily serializable to primitive types.
# When adding new types, make sure to update the `to_json` and `from_json` methods and if needed,
# update various exporters to support the new type.
# Note: Event though the type hint here allows for any list of primitive values, we have validation
# code that expects all the elements of the list to be of the same type.
_PrimitiveValue: TypeAlias = Union[str, bool, int, float]
AttributeValue: TypeAlias = Union[_PrimitiveValue, List[_PrimitiveValue]]


@dataclass
class Attribute:
    """
    An attribute attached to a span or an event. Each attribute has a name and a single value.

    The timestamp of the attribute is considered to be the same as the timestamp of its parent (a span or an event).
    See https://opentelemetry.io/docs/specs/otel/common/#attribute for more information.
    """

    name: str
    value: AttributeValue

    def __post_init__(self) -> None:
        """Validate the attribute value."""
        if isinstance(self.value, list) and len(self.value) > 0:
            elelemt_type = type(self.value[0])
            if not all(isinstance(item, elelemt_type) for item in self.value):
                raise OneLoggerError(f"All elements of a list attribute must be of primitive type {type(self.value[0])} but got {type(self.value)}")


class Attributes(Dict[str, Attribute]):
    """A set of attributes for a single span or event."""

    def __init__(self, dictionary: Optional[Dict[str, AttributeValue]] = None) -> None:
        if dictionary:
            for name, value in dictionary.items():
                self.add(name, value)

    def add(self, name: str, value: AttributeValue) -> "Attributes":
        """Add a new metric to the set of attributes."""
        self.add_attribute(Attribute(name, value))
        return self

    def add_attribute(self, attribute: Attribute) -> Attribute:
        """Add a new metric to the set of attributes."""
        self[attribute.name] = attribute
        return self[attribute.name]

    def add_attributes(self, other_attributes: "Attributes") -> None:
        """Add a new metric to the set of attributes."""
        for attr in other_attributes.values():
            self.add_attribute(attr)

    def to_json(self) -> Dict[str, Any]:
        """Convert all attributes to a JSON-compatible dictionary that can be passed to json.dumps.

        Returns:
            dict: a JSON-compatible dictionary representation of all attributes.
        """
        return {attribute.name: attribute.value for attribute in self.values()}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Attributes":
        """Create a Attributes instance from a JSON-compatible dictionary.

        Args:
            data: Dictionary containing the attributes data

        Returns:
            Attributes: New attributes collection created from the data
        """
        attributes = cls()
        if not data:
            return attributes
        for name, value in data.items():
            attributes.add(name, value)

        return attributes

    @classmethod
    def merge(cls, attributes: Optional["Attributes"] = None, other: Optional["Attributes"] = None) -> "Attributes":
        """Merge two sets of attributes (see the note about the return type below).

        Returns: The merged attributes will be returned as an instance of the class on which merge() was called.
        That is,
           Attributes.merge(attr1, attr2) returns an instance of Attributes

           Assuming TrainingLoopAttributes is a subclass of Attributes,
           TrainingLoopAttributes.merge(attr1, attr2) returns an instance of TrainingLoopAttributes. This is useful when
           you have a subclass of Attributes and you want to merge attributes of that subclass with some new attributes without
           upcasting to the base Attributes class.

           If you truly want to preserve the type of the attributes, you are better off using the add_attributes() method instead.
        """
        merged = cls()
        if attributes:
            for attribute in attributes.values():
                merged.add_attribute(attribute)

        if other:
            for attribute in other.values():
                merged.add_attribute(attribute)
        return merged

    def get_bool_value(self, attribute_name: str) -> Optional[bool]:
        """Get the value of an attribute as a boolean. Works both for required and optional attributes.

        Args:
            attribute_name: The name of the boolean attribute to get the value of.

        Returns:
            The value of the attribute as a boolean, or None if the attribute is not present.

        Raises:
            OneLoggerError: If the attribute value is not a boolean.
        """
        if attribute_name not in self.keys():
            return None
        val = self[attribute_name].value
        assert_that(isinstance(val, bool), f"Attribute {attribute_name} must be a boolean. Got {val}.")
        return cast(bool, val)

    def get_int_value(self, attribute_name: str) -> Optional[int]:
        """Get the value of an attribute as an integer. Works both for required and optional attributes.

        Args:
            attribute_name: The name of the integer attribute to get the value of.

        Returns:
            The value of the attribute as an integer, or None if the attribute is not present.

        Raises:
            OneLoggerError: If the attribute value is not an integer.
        """
        if attribute_name not in self.keys():
            return None
        val = self[attribute_name].value
        assert_that(isinstance(val, int), f"Attribute {attribute_name} must be an integer. Got {val}.")
        return cast(int, val)

    def get_float_value(self, attribute_name: str) -> Optional[float]:
        """Get the value of an attribute as a float. Works both for required and optional attributes.

        Args:
            attribute_name: The name of the float attribute to get the value of.

        Returns:
            The value of the attribute as a float, or None if the attribute is not present.

        Raises:
            OneLoggerError: If the attribute value is not a float.
        """
        if attribute_name not in self.keys():
            return None
        val = self[attribute_name].value
        assert_that(isinstance(val, float), f"Attribute {attribute_name} must be a float. Got {val}.")
        return cast(float, val)

    def get_str_value(self, attribute_name: str) -> Optional[str]:
        """Get the value of an attribute as a string. Works both for required and optional attributes.

        Args:
            attribute_name: The name of the string attribute to get the value of.

        Returns:
            The value of the attribute as a string, or None if the attribute is not present.

        Raises:
            OneLoggerError: If the attribute value is not a string.
        """
        if attribute_name not in self.keys():
            return None
        val = self[attribute_name].value
        assert_that(isinstance(val, str), f"Attribute {attribute_name} must be a string. Got {val}.")
        return cast(str, val)
