# Pytorch Lightning Integration
`one_logger_pytorch_lightning_integration` library provides an easy way to add telemetry to applications that use Pytorch Lightning for training.
The integration works by using [Lightning's callback ](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#callback) mechanism.
Given that the Lightning's callback API doesn't support everything we need, the _one_logger_pytorch_lightning_integration_ library supplements
the Lightning callback mechanism to ensure we support async checkpointing and some app lifecycle events not supported by Lightning callback API.

## Minmum Requirements

- `python` version `>= 3.9, < 3.14`.
- `torch` version `>= 2.8.0`.
- `pytorch-lightning` version `>=2.5.3`.

## Integrate nv-one-logger to PTL application via hook_trainer_cls

 `hook_trainer_cls` adds telemetry hooks to your [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html). 
 Using this method, several of the training events will be automatically captured. However, you still need to explicitly call the one logger API for other events. See [explicit vs implicit](#explicit-vs-implicit) section for more details.

```python
    TrainingTelemetryProvider.instance().with_base_config(config).with_exporter(exporter).configure_provider()
    ...
    HookedTrainer, nv_one_logger_callback = hook_trainer_cls(Trainer, TrainingTelemetryProvider.instance())
    # Instantiate using "HookedTrainer" passing it the same parameters you would pass to the regular Lightning Trainer.
    HookedTrainer = HookedTrainer(
        max_epochs=NUM_EPOCHS,
        limit_train_batches=NUM_TRAIN_BATCHES,
        limit_val_batches=NUM_VAL_BATCHES,
        devices=NUM_DEVICES,
        # No need to pass one logger callback. This will be added automatically. 
        # You can pass other callbacks that you need.
        callbacks=[...], 
        ....
    )

    # You can now use the "trainer" instance the same way you use a regular lightning trainer.
    ...
    # You can also use the returned callback from hook_trainer_clas (or get it via the "nv_one_logger_callback" propery of the trainer)
    # to invoke on_xxx methods that are not part of the Lightning Callback interface such as "on_model_init_start", on_model_init_end, 
    # on_dataloader_init_start, etc.   
    
    # Note that nv_one_logger_callback == trainer.nv_one_logger_callback
    
    nv_one_logger_callback.on_app_end()
```

## Explicit vs Implicit Telemetry Collection

As mentioned above, thanks to integration with the built-in 
[Lightning's callback mechanism](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#callback), when your code
calls `trainer.fit`, several training-related spans (e.g., training loop, training iterations, validation iterations, etc) 
are captured and reported to NV one logger training telemetry _implicitely_ (without the need for you to write any extra code).

However, several lifecycle events are not captured by the Lightning callback mechanism; therefore, if you are interested in collecting
telemetry on those events, you need to call the corresponding `TimeEventCallback` on_xxx methods _explicitely_.

The table below shows which spans are captured implicitely and which ones require an explicit call.

| Span                       | How to collect data?
|----------------------------|--------------------------------------------
| APPLICATION                | You need to explcitely call on_app_end. on_app_start is called automatically.
| TRAINING_LOOP              | Implict via trainer.fit()
| TRAINING_SINGLE_ITERATION  | Implict via trainer.fit()
| VALIDATION_LOOP            | Implict via trainer.fit()
| VALIDATION_SINGLE_ITERATION| Implict via trainer.fit()
| TESTING_LOOP               | Explicit: call on_testing_start and on_testing_end
| DATA_LOADER_INIT           | Explicit: call on_dataloader_init_end and on_dataloader_init_end
| MODEL_INIT                 | Explicit: call on_model_init_start and on_model_init_end
| OPTIMIZER_INIT             | Explicit: call on_optimizer_init_start and on_optimizer_init_end
| CHECKPOINT_LOAD            | Explicit: call on_load_checkpoint_start and on_load_checkpoint_end
| CHECKPOINT_SAVE_SYNC       | Implict. Works both for checkpoints saved automatically by trainer.fit() and checkpoints saved explicitly during training.
| CHECKPOINT_SAVE_ASYNC      | Implict. Works both for checkpoints saved automatically by trainer.fit() and checkpoints saved explicitly during training.


## Full Example

Below is a simple training application that shows how you can enable telemetry for Loghtning by adding a few lines of code.

```python
import os
import torch
from pytorch_lightning import LightningModule, Trainer
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from nv_one_logger.training_telemetry.integration.pytorch_lightning import hook_trainer_cls

class SimpleModel(LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        nv_one_logger_callback = trainer.nv_one_logger_callback        
        nv_one_logger_callback.on_optimizer_init_start()        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        nv_one_logger_callback.on_optimizer_init_end()
        
        return optimizer

def main():
    # 1. Configure OneLoggerTrainingTelemetryProvider
    base_config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_session",
        world_size_or_fn=5,
    )
    
    training_config = TrainingTelemetryConfig(
        world_size_or_fn=5,
        is_log_throughput_enabled_or_fn=True,
        flops_per_sample_or_fn=100,
        global_batch_size_or_fn=32,
        log_every_n_train_iterations=10,
        perf_tag_or_fn="test_perf",
    )
    exporter = FileExporter(file_path=Path("training_telemetry.json"))
    TrainingTelemetryProvider.instance().with_base_config(base_config).with_exporter(exporter).configure_provider()

    # 2. Create and hook the Trainer class
    HookedTrainer = hook_trainer_cls(Trainer, TrainingTelemetryProvider.instance())
    trainer = HookedTrainer(
        max_steps=train_iterations_target,
        devices=num_devices,
    )
    nv_one_logger_callback = trainer.nv_one_logger_callback

    # 3. Set training telemetry config after on_app_start is called
    TrainingTelemetryProvider.instance().set_training_telemetry_config(training_config)
    
    # 4. Initialize model with OneLogger hooks and timestamps
    nv_one_logger_callback.on_model_init_start()
    model = SimpleModel()
    nv_one_logger_callback.on_model_init_end()

    # 5. Load checkpoint if needed
    nv_one_logger_callback.on_load_checkpoint_start()
    if os.path.exists("pretrained.ckpt"):
        model = SimpleModel.load_from_checkpoint("pretrained.ckpt")
    nv_one_logger_callback.on_load_checkpoint_end()

    # 6. Create dummy dataset
    nv_one_logger_callback.on_dataloader_init_start()
    train_dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 10),
        torch.randn(1000, 1)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    nv_one_logger_callback.on_dataloader_init_end()
    
    # 7. Start training
    trainer.fit(model, train_loader)

    # 8. End application
    nv_one_logger_callback.on_app_end()

if __name__ == "__main__":
    main() 
```