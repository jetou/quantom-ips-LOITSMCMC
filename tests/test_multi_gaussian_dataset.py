import pytest
import numpy as np
from torch.utils.data import DataLoader
from quantom_ips.datasets.multi_gaussian_dataset import MultiGaussianDataset


class TestMultiGaussianDataset:

    def test_dataset(self):
        n_datasets = 5
        n_events = 10000
        n_samples = 200

        # Create dataset:
        mg_dataset = MultiGaussianDataset(
            mu_x=tuple(np.random.uniform(0.5, 2.0, n_datasets)),
            sigma_x=tuple(np.random.uniform(0.05, 1.0, n_datasets)),
            mu_y=tuple(np.random.uniform(0.5, 2.0, n_datasets)),
            sigma_y=tuple(np.random.uniform(0.05, 1.0, n_datasets)),
            angle=tuple(np.random.randint(10, 90, n_datasets)),
            n_events=n_events,
            n_samples=n_samples,
        )
        # Make sure all dimensions are reasonable:
        # The shape here is: (n_datasets, n_events, 2)
        data = mg_dataset.data
        assert data.size()[0] == n_datasets
        assert data.size()[1] == n_events
        assert data.size()[2] == 2

        # Now check if dataset plays nicely with the torch dataloader:
        loader = DataLoader(mg_dataset, 1, True)
        for x in loader:
            # The shape in the data loader will be: (1, n_datasets, n_samples, 2)
            sample_size = x.size()[2]
            assert x.size()[1] == n_datasets
            assert sample_size == n_samples
            assert x.size()[3]
