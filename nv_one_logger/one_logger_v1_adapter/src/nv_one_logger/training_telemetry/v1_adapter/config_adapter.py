"""Contains the V1ConfigAdapter class, which is used to adapt the v1 config to the v2 config."""

from typing import Any, Dict

from nv_one_logger.api.config import OneLoggerConfig, OneLoggerErrorHandlingStrategy
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.internal.utils import evaluate_value
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig


class ConfigAdapter:
    """This class is used to adapt the v1 config to the v2 config.

    This allows smooth transition from v1 to v2 of the one logger library without affecting the application code that uses the library.
    Using this adapter, you can start using the v2 implementation while still using the v1 config and API. Until we change the
    downstream consumers of the telemetry data (e.g., data infra that processes the metrics stored in wandb), you should use this adapter
    along with the v1-compatible wandb exporter (which adds compatibility on the output path of v2).
    """

    @staticmethod
    def convert_to_v2_config(v1_config: Dict[str, Any]) -> OneLoggerConfig:
        """Convert the v1 config to the v2 config. See class docstring for more details.

        Args:
            v1_config (Dict[str, Any]): The v1 config.

        Returns:
            OneLoggerConfig: The v2 config.
        """
        custom_metadata = evaluate_value(v1_config.get("metadata", {}))
        # NOTE: We deprecated "app_tag_run_version" for v2, but during the transition, we pass the "app_tag_run_version"
        # value provided by v1 users as custom metadata.
        custom_metadata["app_tag_run_version"] = v1_config.get("app_tag_run_version", "1.0.0")

        enable_for_current_rank = v1_config.get("enable_for_current_rank", False)

        # Create the training telemetry config
        training_telemetry_config = TrainingTelemetryConfig(
            perf_tag_or_fn=v1_config["app_tag"],
            global_batch_size_or_fn=v1_config["global_batch_size"],
            log_every_n_train_iterations=v1_config.get("log_every_n_train_iterations", 50),
            save_checkpoint_strategy=ConfigAdapter._convert_ckpt_strategy_to_enum(v1_config.get("save_checkpoint_strategy", "sync")),
            micro_batch_size_or_fn=v1_config.get("micro_batch_size", None),
            flops_per_sample_or_fn=v1_config.get("flops_per_sample", None),
            train_iterations_target_or_fn=v1_config.get("train_iterations_target", None),
            train_samples_target_or_fn=v1_config.get("train_samples_target", None),
            is_train_iterations_enabled_or_fn=v1_config["is_train_iterations_enabled"],
            is_validation_iterations_enabled_or_fn=v1_config["is_validation_iterations_enabled"],
            is_test_iterations_enabled_or_fn=v1_config.get("is_test_iterations_enabled", True),
            is_save_checkpoint_enabled_or_fn=v1_config.get("is_save_checkpoint_enabled", True),
            is_log_throughput_enabled_or_fn=v1_config.get("is_log_throughput_enabled", False),
            custom_metadata=custom_metadata,
        )

        # Create the base config with embedded telemetry config
        one_logger_config = OneLoggerConfig(
            application_name=v1_config["one_logger_project"],
            world_size_or_fn=v1_config["world_size"],
            session_tag_or_fn=v1_config["app_tag_run_name"],
            is_baseline_run_or_fn=v1_config.get("is_baseline_run", False),
            custom_metadata=custom_metadata,
            error_handling_strategy=(
                OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR
                if v1_config.get("quiet", False)
                else OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS
            ),
            enable_for_current_rank=enable_for_current_rank,
            telemetry_config=training_telemetry_config,
        )

        return one_logger_config

    @staticmethod
    def _convert_ckpt_strategy_to_enum(v1_strategy: str) -> CheckPointStrategy:
        """Convert checkpoint strategy string to enum value.

        Args:
            strategy (str): The checkpoint strategy string ("sync" or "async")

        Returns:
            str: The converted strategy string

        Raises:
            ValueError: If the strategy is not supported
        """
        if v1_strategy == "sync":
            return CheckPointStrategy.SYNC
        elif v1_strategy == "async":
            return CheckPointStrategy.ASYNC
        raise OneLoggerError(f"Unsupported checkpoint strategy: {v1_strategy}. Must be 'sync' or 'async'")
