import os
from pathlib import Path
from omegaconf import MISSING
from .base_algorithm import Algorithm
import time


class ModelCheckpointing(Algorithm):
    def __init__(
        self,
        logdir: str = "${hydra:runtime.output_dir}",
        frequency: int = 0,
        save_untrained: bool = True,
    ):
        super().__init__()
        self.logdir = logdir
        self.frequency = frequency
        self.save_untrained = save_untrained

    def match(self, tag):
        return tag in ["before_fit", "after_epoch", "after_fit"]

    def apply(self, opt, tag):
        # Ensure checkpoint directory is saved before fitting
        if tag == "before_fit":
            os.makedirs(Path(self.logdir).joinpath("checkpoints"), exist_ok=True)
            if self.save_untrained:
                self.save_opt(opt, 0)

        # Always save optimizer after fitting
        if tag == "after_fit":
            self.save_opt(opt, opt.current_epoch)

        # Save optimizer after every `frequency` epochs
        if tag == "after_epoch" and self.frequency > 0:
            if opt.current_epoch % self.frequency == 0:
                self.save_opt(opt, opt.current_epoch)

    def save_opt(self, opt, epoch):
        opt.save(
            Path(self.logdir).joinpath(f"checkpoints/{int(time.time())}_e{epoch}.pt")
        )
