from quantom_ips import register


register(
    id="LossTracker",
    entry_point="quantom_ips.algorithms.loss_tracker:LossTracker",
    group="algorithm",
)

register(
    id="PyTorchDataMover",
    entry_point="quantom_ips.algorithms.pytorch_data_mover:PyTorchDataMover",
    group="algorithm",
)

register(
    id="ModelCheckpointing",
    entry_point="quantom_ips.algorithms.model_checkpointing:ModelCheckpointing",
    group="algorithm",
)
