"""Contains the V1ConfigAdapter class, which is used to adapt the v1 config to the v2 config."""

import uuid
from typing import Any, Dict, Tuple

from nv_one_logger.api.config import ApplicationType, OneLoggerErrorHandlingStrategy
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.internal.utils import evaluate_value
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.wandb.exporter.wandb_exporter import Config as WandBConfig


class ConfigAdapter:
    """This class is used to adapt the v1 config to the v2 config.

    This allows smooth transition from v1 to v2 of the one logger library without affecting the application code that uses the library.
    Using this adapter, you can start using the v2 implementation while still using the v1 config and API. Until we change the
    downstream consumers of the telemetry data (e.g., data infra that processes the metrics stored in wandb), you should use this adapter
    along with the v1-compatible wandb exporter (which adds compatibility on the output path of v2).
    """

    @staticmethod
    def convert_to_v2_config(v1_config: Dict[str, Any]) -> Tuple[TrainingTelemetryConfig, WandBConfig]:
        """Converts the v1 config to the v2 config. See class docstring for more details.

        Args:
            v1_config (Dict[str, Any]): The v1 config.

        Returns:
            TrainingTelemetryConfig: The v2 config.
        """
        custom_metadata = evaluate_value(v1_config.get("metadata", {}))
        # NOTE: We deprecated "app_tag_run_version" for v2, but during the transition, we pass the "app_tag_run_version"
        # value provided by v1 users as custom metadata.
        custom_metadata["app_tag_run_version"] = v1_config.get("app_tag_run_version", "1.0.0")

        enable_for_current_rank = v1_config.get("enable_for_current_rank", False)

        training_telemetry_config = TrainingTelemetryConfig(
            application_name=v1_config["one_logger_project"],
            perf_tag_or_fn=v1_config["app_tag"],
            session_tag_or_fn=v1_config["app_tag_run_name"],
            app_type_or_fn=ApplicationType.TRAINING,
            is_baseline_run_or_fn=v1_config["is_baseline_run"],
            custom_metadata=custom_metadata,
            error_handling_strategy=(
                OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR
                if v1_config.get("quiet", False)
                else OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS
            ),
            enable_one_logger=enable_for_current_rank,
            world_size_or_fn=v1_config["world_size"],
            global_batch_size_or_fn=v1_config["global_batch_size"],
            enable_for_current_rank=enable_for_current_rank,
            log_every_n_train_iterations=v1_config.get("log_every_n_train_iterations", 50),
            is_train_iterations_enabled_or_fn=v1_config["is_train_iterations_enabled"],
            is_validation_iterations_enabled_or_fn=v1_config["is_validation_iterations_enabled"],
            is_test_iterations_enabled_or_fn=v1_config.get("is_test_iterations_enabled", True),
            is_save_checkpoint_enabled_or_fn=v1_config.get("is_save_checkpoint_enabled", True),
            is_log_throughput_enabled_or_fn=v1_config.get("is_log_throughput_enabled", False),
            micro_batch_size_or_fn=v1_config.get("micro_batch_size", None),
            flops_per_sample_or_fn=v1_config.get("flops_per_sample", None),
            save_checkpoint_strategy=ConfigAdapter._convert_ckpt_strategy_to_enum(v1_config.get("save_checkpoint_strategy", "sync")),
            train_iterations_target_or_fn=v1_config.get("train_iterations_target", None),
            train_samples_target_or_fn=v1_config.get("train_samples_target", None),
        )
        training_telemetry_config.validate_config()

        wandb_config = WandBConfig(
            entity="hwinf_dcm",  # NOTE: should always be 'hwinf_dcm' for internal user.
            project=v1_config.get("one_logger_project", "e2e-tracking"),
            run_name=v1_config.get("one_logger_run_name", str(uuid.uuid4())),
        )

        return training_telemetry_config, wandb_config

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
