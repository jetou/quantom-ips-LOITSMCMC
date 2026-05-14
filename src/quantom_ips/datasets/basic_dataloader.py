from torch.utils.data import DataLoader
from typing import Any
from omegaconf import MISSING
from quantom_ips import make


class BasicDataLoader(DataLoader):
    def __init__(
        self, dataset: Any = MISSING, batch_size: int = 1, shuffle: bool = True
    ):
        dataset = make(dataset)
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
