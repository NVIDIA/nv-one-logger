# SPDX-License-Identifier: Apache-2.0
# Lightning compatibility shim:
# This module bridges the transition from 'pytorch_lightning' to 'lightning.pytorch'.
# It first tries the new 'lightning.pytorch' and falls back to the legacy
# 'pytorch_lightning', re-exporting common symbols so callers can import from
# 'nv_one_logger.training_telemetry.integration._compat.lightning' without
# caring which Lightning package is installed.
try:
    # Prefer modern 'lightning.pytorch' when available
    import lightning.pytorch as ptl
    from lightning.pytorch import Callback, LightningModule, Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.utilities.types import STEP_OUTPUT

except ImportError:
    import pytorch_lightning as ptl
    from pytorch_lightning import Callback, LightningModule, Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.utilities.types import STEP_OUTPUT

__all__ = ["ptl", "Trainer", "Callback", "LightningModule", "STEP_OUTPUT", "ModelCheckpoint"]
