import torch
import torch.nn as nn
from typing import Any, Optional
from omegaconf import MISSING

from quantom_ips import make
from quantom_ips.utils.torch_nn_registry import get_optimizer
from .base_optimizer import PyTorchOptimizer


class GANOptimizer(PyTorchOptimizer):
    def __init__(
        self,
        discriminator: Any = MISSING,  # Can be set when registering or on command-line
        generator: Any = MISSING,
        n_epochs: int = 1,
        gen_opt: str = "Adam",
        gen_lr: float = 1e-5,
        disc_opt: str = "Adam",
        disc_lr: float = 2e-5,
        logdir: str = "${hydra:runtime.output_dir}",
        progress_bar: bool = False,
        label_noise: float = 0.1,
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

        # self.n_epochs = n_epochs
        self.gen_opt = gen_opt
        self.gen_lr = gen_lr
        self.disc_opt = disc_opt
        self.disc_lr = disc_lr
        self.logdir = logdir
        # self.progress_bar = progress_bar
        self.label_noise = label_noise
        self.noise_dim = noise_dim

        self.discriminator = make(discriminator)
        self.generator = make(generator)

        self.device = self.generator.parameters().__next__().device
        self.dtype = self.generator.parameters().__next__().dtype
        self.criterion = nn.BCELoss()

        self.gen_optimizer = get_optimizer(
            self.gen_opt, self.generator.parameters(), self.gen_lr
        )
        self.disc_optimizer = get_optimizer(
            self.disc_opt,
            self.discriminator.parameters(),
            self.disc_lr,
        )

    def predict(self):
        # Reset the generator optimizer state:
        self.gen_optimizer.zero_grad()
        # Returns a batch of solutions from the optimizer
        batch_size = self.targets.shape[0]
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size, self.noise_dim),
            device=self.device,
        )
        params = self.generator(noise)
        return params

    def update_model(self):
        # Updates the optimizer (and possibly the objective) based on a new observation
        fake_events = self.outputs
        real_events = self.targets

        fake_labels = self.discriminator(fake_events)
        labels = torch.full_like(fake_labels, fill_value=1.0, device=self.device)
        labels -= self.label_noise * torch.rand(size=labels.shape, device=self.device)

        gen_loss = self.criterion(fake_labels, labels)
        gen_loss.backward()
        self.gen_optimizer.step()

        self.discriminator.zero_grad()
        fake_labels = self.discriminator(fake_events.detach())
        labels = torch.full_like(fake_labels, fill_value=0.0, device=self.device)
        labels += self.label_noise * torch.rand(size=labels.shape, device=self.device)
        disc_loss_fake = self.criterion(fake_labels, labels)

        real_labels = self.discriminator(real_events)
        labels = torch.full_like(real_labels, fill_value=1.0, device=self.device)
        labels -= self.label_noise * torch.rand(size=labels.shape, device=self.device)
        disc_loss_real = self.criterion(real_labels, labels)

        disc_loss = disc_loss_fake + disc_loss_real
        disc_loss.backward()
        self.disc_optimizer.step()

        losses = [gen_loss, disc_loss_real, disc_loss_fake]
        return [loss.detach().cpu().item() for loss in losses]
