import os
from pathlib import Path
from .base_algorithm import Algorithm
import torch


class LossTracker(Algorithm):
    def __init__(self, logdir: str = "${hydra:runtime.output_dir}"):
        super().__init__()
        self.logdir = logdir
        self.losses = []

    def match(self, tag):
        return tag in ["before_fit", "after_batch", "after_fit"]

    def apply(self, opt, tag):
        # Ensure logging directory is saved before fitting
        if tag == "before_fit":
            os.makedirs(Path(self.logdir), exist_ok=True)

        # Keep a running list of losses coming from the optimizer
        if tag == "after_batch":
            self.losses.append(opt.losses)

        # After fitting, save losses list
        if tag == "after_fit":
            torch.save(self.losses, Path(self.logdir).joinpath("tracked_losses.pt"))
