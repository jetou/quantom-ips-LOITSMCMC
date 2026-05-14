import mpi4py

mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
import logging
import os
import torch
import numpy as np
import time
import random
from pathlib import Path
from typing import Any, Optional
from omegaconf import MISSING
import matplotlib.pyplot as plt

from quantom_ips import make
from quantom_ips.utils.torch_nn_registry import get_optimizer
from .base_optimizer import BaseOptimizer


logger = logging.getLogger(__name__)


class DistributedGANOptimizer(BaseOptimizer):
    # Initialize:
    def __init__(
        self,
        discriminator: Any = MISSING,  # Can be set when registering or on command-line
        generator: Any = MISSING,
        gradient_transport: Any = MISSING,
        n_epochs: int = 5,
        limit_batches: int = -1,
        limit_eval_batches: Optional[int] = None,
        gen_opt: str = "Adam",
        gen_lr: float = 1e-5,
        disc_opt: str = "Adam",
        disc_lr: float = 2e-5,
        logdir: str = "${hydra:runtime.output_dir}",
        noise_dim: int = 10,
        label_noise: float = 0.0,
        loss_fn: Optional[str] = None,
        progress_bar: bool = False,
        batch_size: int = 0,
        gradient_scale: float = 1.0,
        outer_group_update_frequency: int = 5,
    ) -> None:
        super().__init__(
            n_epochs=n_epochs,
            progress_bar=progress_bar,
            limit_batches=limit_batches,
            limit_eval_batches=limit_eval_batches,
        )
        # Get the settings:
        self.n_epochs = n_epochs
        self.gen_opt = gen_opt
        self.gen_lr = gen_lr
        self.disc_opt = disc_opt
        self.disc_lr = disc_lr
        self.logdir = logdir
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.label_noise = label_noise
        self.gradient_scale = gradient_scale
        self.outer_group_update_frequency = outer_group_update_frequency

        # Setup the model and gradient transport:
        self.gradient_transport = make(gradient_transport)
        self.device = self.gradient_transport.device
        self.dtype = self.gradient_transport.dtype
        self.rank = self.gradient_transport.rank
        self.n_ranks = self.gradient_transport.n_ranks
        self.comm = self.gradient_transport.comm
        self.discriminator = make(discriminator).to(self.device, self.dtype)
        self.generator = make(generator).to(self.device, self.dtype)

        # Get the optimizers:
        self.gen_optimizer = get_optimizer(
            self.gen_opt, self.generator.parameters(), self.gen_lr
        )
        self.disc_optimizer = get_optimizer(
            self.disc_opt,
            self.discriminator.parameters(),
            self.disc_lr,
        )
        # Loss function:
        self.criterion = torch.nn.MSELoss()
        self.criterion_norm = 0.25

        if loss_fn is not None:
            if loss_fn.lower() == "bce":
                self.criterion = torch.nn.BCELoss()
                self.criterion_norm = -np.log(0.5)

        # Synchronize the generator across ranks:
        self.gradient_transport.sync_model(self.generator, self.gen_optimizer)

    # Define prediction:
    def predict(self):
        # Reset the generator optimizer:
        self.gen_optimizer.zero_grad()
        # We can infer the generator batch size from the target batch size, unless the
        # wishes to use a specific batch size:
        generator_batch_size = self.targets.shape[0]
        if self.batch_size > 0:
            generator_batch_size = self.batch_size

        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(generator_batch_size, self.noise_dim),
            device=self.device,
        )
        params = self.generator(noise)
        return params

    # Define the model update:
    def update_model(self):
        # Updates the optimizer (and possibly the objective) based on a new observation
        fake_events = self.outputs
        real_events = self.targets

        # Compute generator loss:
        y_fake = self.discriminator(fake_events)
        pos_label = torch.ones_like(y_fake) - torch.rand_like(y_fake) * self.label_noise
        gen_loss = self.criterion(y_fake, pos_label)
        # Get the gradients:
        gen_loss.backward()
        # Transport gradients:
        update_outer_group = False
        if (1 + self.current_epoch) % self.outer_group_update_frequency == 0:
            update_outer_group = True

        # Run update after first epoch:
        if self.current_epoch > 0:
            self.gradient_transport.forward(
                model=self.generator,
                use_outer_group_communication=update_outer_group,
                gradient_scale=self.gradient_scale,
            )
            self.gen_optimizer.step()

        # Update the discriminator:
        self.discriminator.zero_grad()
        y_fake = self.discriminator(fake_events.detach())
        y_real = self.discriminator(real_events.detach())
        pos_label = torch.ones_like(y_real) - torch.rand_like(y_real) * self.label_noise
        neg_label = (
            torch.zeros_like(y_real) + torch.rand_like(y_fake) * self.label_noise
        )
        disc_loss_fake = self.criterion(y_fake, neg_label)
        disc_loss_real = self.criterion(y_real, pos_label)

        disc_loss = disc_loss_fake + disc_loss_real
        if self.current_epoch > 0:
            disc_loss.backward()
            self.disc_optimizer.step()

        losses = [gen_loss, disc_loss_real, disc_loss_fake]
        return [loss.detach().cpu().item() / self.criterion_norm for loss in losses]
