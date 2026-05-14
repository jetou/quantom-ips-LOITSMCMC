import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any
from omegaconf import MISSING
from quantom_ips import make


class GaussianPickles(Dataset):
    """
    PyTorch Dataset that loads data from a .npy or .npz file.

    Args:
        path (str): Path to the .npy or .npz file.
    """

    def __init__(self, paths: tuple = MISSING, n_samples: int = 1000):
        super().__init__()
        self.paths = paths
        self.n_samples = n_samples

        # Load data into memory
        self.datasets = []
        self.coefficients = []
        for path in paths:

            if path.endswith(".pkl"):
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    self.datasets.append(data["data"])
                    self.coefficients.append(data["coefficients"])
            else:
                raise ValueError("Unsupported file format. Use .npy")

        # Create an n_datasets x n_coefficients matrix for the environment
        self.coefficients = torch.from_numpy(np.stack(self.coefficients))

        self.sample_size = self.n_samples
        self.max_samples = np.max([len(data) for data in self.datasets])

    def __len__(self):
        return self.max_samples // self.sample_size

    def __getitem__(self, idx):
        sample_list = []
        for data in self.datasets:
            inds = np.random.choice(data.shape[0], size=self.sample_size)
            sample = data[inds]
            sample = sample.reshape(1, self.n_samples, -1)
            sample_list.append(sample)

        sample = np.concatenate(sample_list, axis=0)

        # Convert to torch.Tensor
        if not isinstance(sample, torch.Tensor):
            sample = torch.from_numpy(np.array(sample))

        return sample, self.coefficients
