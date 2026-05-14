import torch
import torch.nn as nn
from typing import Any, Optional
from omegaconf import MISSING

from quantom_ips import make
from quantom_ips.utils.torch_nn_registry import get_optimizer
from .base_optimizer import PyTorchOptimizer
from geomloss import SamplesLoss
import logging

logger = logging.getLogger(__name__)


class GeomlossOptimizer(PyTorchOptimizer):
    def __init__(
        self,
        generator: Any = MISSING,
        n_epochs: int = 1,
        gen_opt: str = "Adam",
        gen_lr: float = 1e-5,
        progress_bar: bool = False,
        noise_dim: int = 10,
        limit_batches: int = -1,
        limit_eval_batches: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_epochs=n_epochs,
            progress_bar=progress_bar,
            limit_batches=limit_batches,
            limit_eval_batches=limit_eval_batches,
        )

        self.gen_opt = gen_opt
        self.gen_lr = gen_lr
        self.noise_dim = noise_dim

        self.generator = make(generator)

        self.criterion = SamplesLoss(loss="energy")

        self.gen_optimizer = get_optimizer(
            self.gen_opt, self.generator.parameters(), self.gen_lr
        )

    def predict(self):
        # Returns a batch of solutions from the optimizer
        batch_size = self.targets.shape[0]
        device = self.targets.device
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size, self.noise_dim),
            device=device,
        )
        params = self.generator(noise)
        return params

    def update_model(self):
        # Updates the optimizer (and possibly the objective) based on a new observation
        fake_events = self.outputs
        real_events = self.targets

        self.generator.zero_grad()
        logger.debug(
            f"Updating with {fake_events.shape} and {real_events.shape} tensors"
        )
        loss = torch.mean(self.criterion(fake_events, real_events))
        loss.backward()
        self.gen_optimizer.step()

        losses = []
        return [loss.detach().cpu().item() for loss in losses]
