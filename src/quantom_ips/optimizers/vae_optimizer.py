import logging
import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from pathlib import Path
from typing import Any
from omegaconf import MISSING
from geomloss import SamplesLoss

from quantom_ips import make
from quantom_ips.utils.torch_nn_registry import get_optimizer
from quantom_ips.utils.stateful_module import StatefulModule

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt


class VAEOptimizer(StatefulModule):
    def __init__(
        self,
        model: Any = MISSING,
        n_epochs: int = 1,
        opt: str = "Adam",
        lr: float = 1e-5,
        batch_size: int = 10,
        logdir: str = "${hydra:runtime.output_dir}",
        progress_bar: bool = False,
    ) -> None:
        super().__init__()

        self.n_epochs = n_epochs
        self.opt = opt
        self.lr = lr
        self.batch_size = batch_size
        self.logdir = logdir
        self.progress_bar = progress_bar

        self.vae = make(model)

    def train(self, env, data_parser):
        os.makedirs(Path(self.logdir), exist_ok=True)

        vae = self.vae
        dtype = vae.parameters().__next__().dtype
        device = vae.parameters().__next__().device

        optimizer = get_optimizer(self.opt, vae.parameters(), self.lr)

        criterion = SamplesLoss()

        n_epochs = self.n_epochs
        losses = []
        rewards = []

        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}")
            for real_events in tqdm(data_parser, disable=not self.progress_bar):
                real_events = real_events.to(dtype=dtype, device=device)
                vae.zero_grad()
                # Since our events are (batch, particles, events, dims),
                # we flatten batch and particles together
                n_batch, n_particles, n_events, n_dims = real_events.shape
                real_events = real_events.reshape(-1, n_events, n_dims)
                reco, latent, quantized = vae(real_events)
                obs, reward, terminated, truncated, info = env(reco, real_events)
                fake_events = obs["fake"]
                real_events = obs["real"]  # In case the env modified them in some way

                rewards.append(reward)

                fake_events = fake_events.reshape(-1, n_events, n_dims)
                loss = criterion(fake_events, real_events)
                loss.backward()
                optimizer.step()

                if epoch == n_epochs - 1:
                    print(quantized)

                losses.append(loss.detach().cpu().item())

        plt.imshow(reco.squeeze().detach().cpu().numpy())
        plt.savefig("testfig.png")

        plt.clf()
        real_events = real_events.detach().cpu().numpy()
        fake_events = fake_events.detach().cpu().numpy()
        fig, axs = plt.subplots(1, 2)
        axs[0].hist2d(real_events[0, :, 0], real_events[0, :, 1], bins=25)
        axs[1].hist2d(fake_events[0, :, 0], fake_events[0, :, 1], bins=25)
        plt.savefig("testhist.png")
        return losses
