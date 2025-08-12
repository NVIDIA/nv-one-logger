# SPDX-License-Identifier: Apache-2.0
from nv_one_logger.training_telemetry.api.events import StandardTrainingJobEventName


class TestStandardTrainingJobEventName:
    """Test cases for StandardTrainingJobEventName enum."""

    def test_update_training_telemetry_config_event_name(self):
        """Test that UPDATE_TRAINING_TELEMETRY_CONFIG event name is correctly defined."""
        assert StandardTrainingJobEventName.UPDATE_TRAINING_TELEMETRY_CONFIG == "update_training_telemetry_config"

    def test_all_event_names_are_strings(self):
        """Test that all event names are strings."""
        for event_name in StandardTrainingJobEventName:
            assert isinstance(event_name.value, str)
            assert len(event_name.value) > 0

    def test_event_names_are_unique(self):
        """Test that all event names are unique."""
        event_names = [event_name.value for event_name in StandardTrainingJobEventName]
        assert len(event_names) == len(set(event_names)), "Event names should be unique"

    def test_event_names_follow_naming_convention(self):
        """Test that event names follow the snake_case naming convention."""
        for event_name in StandardTrainingJobEventName:
            # Check that the value is in snake_case
            assert event_name.value == event_name.value.lower()
            # Check that it contains only letters, numbers, and underscores
            assert event_name.value.replace("_", "").isalnum()

    def test_update_training_telemetry_config_event_name_consistency(self):
        """Test that UPDATE_TRAINING_TELEMETRY_CONFIG event name is consistent with other event names."""
        # Verify it follows the same pattern as other event names
        assert StandardTrainingJobEventName.UPDATE_TRAINING_TELEMETRY_CONFIG in StandardTrainingJobEventName
        assert isinstance(StandardTrainingJobEventName.UPDATE_TRAINING_TELEMETRY_CONFIG, StandardTrainingJobEventName)
