from typing import Any

import numpy as np
import torch
from omegaconf import MISSING
from torch.utils.data import DataLoader, Dataset

from quantom_ips import make


class TMDDataLoader(DataLoader):
    def __init__(
        self, dataset: Any = MISSING, batch_size: int = 1, shuffle: bool = True
    ):
        dataset = make(dataset)
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


class TMDNumpyDataset(Dataset):
    def __init__(
        self,
        path: str = MISSING,
        n_particles: int = 1,
        n_samples: int = 1000,
        event_dim: int = 5,
    ):
        super().__init__()
        self.path = path
        self.n_particles = n_particles
        self.n_samples = n_samples
        self.event_dim = event_dim

        if not path.endswith(".npy"):
            raise ValueError("Unsupported file format. Use .npy")

        data = np.load(path)
        if data.ndim == 1:
            if data.shape[0] % event_dim != 0:
                raise ValueError(
                    f"Flat array length {data.shape[0]} is not divisible by D={event_dim}"
                )
            data = data.reshape(-1, event_dim)

        if not (data.ndim == 2 and data.shape[1] == event_dim):
            raise ValueError(
                f"Expected data as (N_events,{event_dim}) after reshape, got {data.shape}"
            )

        self.data = data.astype(np.float32)
        self.n_events_total = self.data.shape[0]
        self.sample_size = self.n_particles * self.n_samples

    def __len__(self):
        return max(1, self.n_events_total // self.sample_size)

    def __getitem__(self, idx):
        replace = self.n_events_total < self.sample_size
        inds = np.random.choice(
            self.n_events_total, size=self.sample_size, replace=replace
        )
        sample = self.data[inds]

        if self.n_particles > 1:
            sample = sample.reshape(self.n_particles, self.n_samples, self.event_dim)
        else:
            sample = sample.reshape(self.n_samples, self.event_dim)

        return torch.from_numpy(sample)
