import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import MISSING


class NumpyDataset(Dataset):
    """
    PyTorch Dataset that loads data from a .npy or .npz file.

    Args:
        path (str): Path to the .npy or .npz file.
    """

    def __init__(self, path: str = MISSING):
        super().__init__()
        self.path = path

        # Load data into memory
        if path.endswith(".npy"):
            self.data = np.load(path)
        else:
            raise ValueError("Unsupported file format. Use .npy")

        self.data = torch.from_numpy(self.data)
        if self.data.ndim == 1:
            self.data = self.data.unsqueeze(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]
