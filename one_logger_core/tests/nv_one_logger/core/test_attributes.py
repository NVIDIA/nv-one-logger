# SPDX-License-Identifier: Apache-2.0
"""Tests for the Attributes and Attribute classes in one_logger.core."""

import pytest

from nv_one_logger.core.attributes import Attribute, Attributes
from nv_one_logger.core.exceptions import OneLoggerError


class TestAttributes:
    """Test class for Attributes and Attribute classes."""

    def test_attribute_creation(self) -> None:
        """Test that an Attribute can be created with all supported value types.

        Tests creation with:
        - Primitive types: str, bool, int, float
        - lists
        """

    # Test primitive types
    str_attr = Attribute(name="string", value="hello")
    bool_attr = Attribute(name="boolean", value=True)
    int_attr = Attribute(name="integer", value=42)
    float_attr = Attribute(name="float", value=3.14)

    assert str_attr.name == "string"
    assert str_attr.value == "hello"
    assert bool_attr.name == "boolean"
    assert bool_attr.value is True
    assert int_attr.name == "integer"
    assert int_attr.value == 42
    assert float_attr.name == "float"
    assert float_attr.value == 3.14

    # Test lists
    list_attr = Attribute(name="list", value=[1, 2, 3])

    assert list_attr.name == "list"
    assert list_attr.value == [1, 2, 3]

    def test_invalid_list_attribute_value(self) -> None:
        """Test that an error is raised if a list attribute value contains elements of different type."""
        with pytest.raises(OneLoggerError, match="All elements of a list attribute must be of primitive type"):
            Attribute(name="invalid_list", value=[1, "two", 3])

    def test_attributes_constructor(self) -> None:
        """Test that constructor initializes attributes from a dictionary."""
        attributes = Attributes({"foo": 42, "bar": "baz"})
        assert attributes["foo"].value == 42
        assert attributes["bar"].value == "baz"

    def test_add_attribute(self) -> None:
        """Test that add_attribute() adds a new attribute to the collection."""
        attributes = Attributes()
        attribute = Attribute(name="test", value=42)
        result = attributes.add_attribute(attribute)
        assert attributes["test"] == attribute
        assert result == attribute

    def test_add(self) -> None:
        """Test that add() creates and adds a new attribute to the collection."""
        attributes = Attributes()
        result = attributes.add("test", 42)
        assert result == attributes
        assert attributes["test"].name == "test"
        assert attributes["test"].value == 42

    def test_json_with_primitive_values(self) -> None:
        """Test that to_json() correctly serializes primitive values."""
        attributes = Attributes()
        attributes.add("string", "hello")
        attributes.add("int", 42)
        attributes.add("float", 3.14)
        attributes.add("bool", True)

        json_data = attributes.to_json()
        assert json_data["string"] == "hello"
        assert json_data["int"] == 42
        assert json_data["float"] == 3.14
        assert json_data["bool"] is True

        deserialized_attributes = Attributes.from_json(json_data)
        assert deserialized_attributes == attributes

    def test_json_with_list(self) -> None:
        """Test that to_json() correctly serializes list values."""
        attributes = Attributes()
        attributes.add("list", [1, 2, 3])

        json_data = attributes.to_json()
        assert json_data["list"] == [1, 2, 3]

        deserialized_attributes = Attributes.from_json(json_data)
        assert deserialized_attributes == attributes

    def test_merge_combines_attributes(self) -> None:
        """Test that merge() combines attributes from two collections."""
        attributes1 = Attributes()
        attributes1.add("a", 1)
        attributes1.add("b", 2)

        attributes2 = Attributes()
        attributes2.add("c", 3)
        attributes2.add("d", 4)

        merged = Attributes.merge(attributes1, attributes2)
        assert merged["a"].value == 1
        assert merged["b"].value == 2
        assert merged["c"].value == 3
        assert merged["d"].value == 4

    def test_merge_overwrites_duplicate_keys(self) -> None:
        """Test that merge() overwrites duplicate keys with values from the second collection."""
        attributes1 = Attributes()
        attributes1.add("a", 1)
        attributes1.add("b", 2)

        attributes2 = Attributes()
        attributes2.add("b", 3)
        attributes2.add("c", 4)

        merged = Attributes.merge(attributes1, attributes2)
        assert merged["a"].value == 1
        assert merged["b"].value == 3  # Overwritten by attributes2
        assert merged["c"].value == 4

    def test_get_bool_value(self) -> None:
        """Test that get_bool_value() correctly retrieves boolean values and handles errors."""
        attributes = Attributes()
        attributes.add("bool_attr", True)
        attributes.add("str_attr", "not_a_bool")

        # Test getting an existing boolean attribute
        assert attributes.get_bool_value("bool_attr") is True

        # Test getting a non-existent attribute
        assert attributes.get_bool_value("nonexistent") is None

        # Test getting a non-boolean attribute
        with pytest.raises(OneLoggerError, match="Attribute str_attr must be a boolean"):
            attributes.get_bool_value("str_attr")

    def test_get_int_value(self) -> None:
        """Test that get_int_value() correctly retrieves integer values and handles errors."""
        attributes = Attributes()
        attributes.add("int_attr", 42)
        attributes.add("str_attr", "not_an_int")

        # Test getting an existing integer attribute
        assert attributes.get_int_value("int_attr") == 42

        # Test getting a non-existent attribute
        assert attributes.get_int_value("nonexistent") is None

        # Test getting a non-integer attribute
        with pytest.raises(OneLoggerError, match="Attribute str_attr must be an integer"):
            attributes.get_int_value("str_attr")

    def test_get_float_value(self) -> None:
        """Test that get_float_value() correctly retrieves float values and handles errors."""
        attributes = Attributes()
        attributes.add("float_attr", 3.14)
        attributes.add("str_attr", "not_a_float")

        # Test getting an existing float attribute
        assert attributes.get_float_value("float_attr") == 3.14

        # Test getting a non-existent attribute
        assert attributes.get_float_value("nonexistent") is None

        # Test getting a non-float attribute
        with pytest.raises(OneLoggerError, match="Attribute str_attr must be a float"):
            attributes.get_float_value("str_attr")

    def test_get_str_value(self) -> None:
        """Test that get_str_value() correctly retrieves string values and handles errors."""
        attributes = Attributes()
        attributes.add("str_attr", "hello")
        attributes.add("int_attr", 42)

        # Test getting an existing string attribute
        assert attributes.get_str_value("str_attr") == "hello"

        # Test getting a non-existent attribute
        assert attributes.get_str_value("nonexistent") is None

        # Test getting a non-string attribute
        with pytest.raises(OneLoggerError, match="Attribute int_attr must be a string"):
            attributes.get_str_value("int_attr")

    def test_add_attributes_empty(self) -> None:
        """Test adding attributes from an empty Attributes object."""
        attrs1 = Attributes()
        attrs2 = Attributes()
        attrs1.add_attributes(attrs2)
        assert len(attrs1) == 0

    def test_add_attributes_single(self) -> None:
        """Test adding a single attribute from another Attributes object."""
        attrs1 = Attributes()
        attrs2 = Attributes({"test": 42})
        attrs1.add_attributes(attrs2)
        assert len(attrs1) == 1
        assert attrs1["test"].value == 42

    def test_add_attributes_multiple(self) -> None:
        """Test adding multiple attributes from another Attributes object."""
        attrs1 = Attributes({"existing": "value"})
        attrs2 = Attributes({"int_val": 42, "float_val": 3.14, "bool_val": True, "str_val": "test", "list_val": [1, 2, 3]})
        attrs1.add_attributes(attrs2)
        assert len(attrs1) == 6
        assert attrs1["existing"].value == "value"
        assert attrs1["int_val"].value == 42
        assert attrs1["float_val"].value == 3.14
        assert attrs1["bool_val"].value is True
        assert attrs1["str_val"].value == "test"
        assert attrs1["list_val"].value == [1, 2, 3]

    def test_add_attributes_overwrite(self) -> None:
        """Test that adding attributes overwrites existing attributes with the same name."""
        attrs1 = Attributes({"test": "original"})
        attrs2 = Attributes({"test": "new"})
        attrs1.add_attributes(attrs2)
        assert len(attrs1) == 1
        assert attrs1["test"].value == "new"

    def test_add_attributes_mixed_types(self) -> None:
        """Test adding attributes with mixed primitive types."""
        attrs1 = Attributes()
        attrs2 = Attributes({"string": "hello", "integer": 42, "float": 3.14, "boolean": True, "list_of_ints": [1, 2, 3], "list_of_strings": ["a", "b", "c"]})
        attrs1.add_attributes(attrs2)
        assert len(attrs1) == 6
        assert isinstance(attrs1["string"].value, str)
        assert isinstance(attrs1["integer"].value, int)
        assert isinstance(attrs1["float"].value, float)
        assert isinstance(attrs1["boolean"].value, bool)
        assert isinstance(attrs1["list_of_ints"].value, list)
        assert isinstance(attrs1["list_of_strings"].value, list)
