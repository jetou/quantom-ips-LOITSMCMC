import numpy as np
import torch
from torch.utils.data import Dataset


class MultiGaussianDataset(Dataset):

    def __init__(
        self,
        mu_x: tuple = (1.0, 1.0),
        sigma_x: tuple = (0.2, 2.0),
        mu_y: tuple = (1.0, 1.0),
        sigma_y: tuple = (2.0, 0.2),
        angle: tuple = (0.0, 0.0),
        n_events: int = 1000,
        n_samples: int = 10,
        seed: int = 123,
    ):
        super().__init__()

        mu_x = torch.as_tensor(mu_x).view(-1, 1)
        sigma_x = torch.as_tensor(sigma_x).view(-1, 1)
        mu_y = torch.as_tensor(mu_y).view(-1, 1)
        sigma_y = torch.as_tensor(sigma_y).view(-1, 1)
        self.angle = angle
        self.n_events = n_events
        self.n_samples = n_samples

        # Register mu and sigma so sampling is a bit faster:
        self.mu_xy = torch.cat([mu_x, mu_y], 1)
        self.sigma_xy = torch.cat([sigma_x, sigma_y], 1)

        # Get the data:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = -1
        mps_rng_state = -1
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        if torch.backends.mps.is_available():
            mps_rng_state = torch.mps.get_rng_state()

        # Set seed so that we know the data is exactly the same on both ranks:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        self.data = self.create_data()

        # Undo the fixed seed:
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)
        if torch.backends.mps.is_available():
            torch.mps.set_rng_state(mps_rng_state)

    # Define rotation in 2D:
    def rotate(self, data, angle_deg, origin):
        if angle_deg != 0.0:
            # Convert angle from degrees to rad:
            angle_rad = torch.as_tensor(np.pi * angle_deg / 180.0)
            # Shift origin, if data is not centered at (0,0):
            shifted_data = data - origin
            new_x = (
                torch.cos(angle_rad) * shifted_data[:, 0]
                - torch.sin(angle_rad) * shifted_data[:, 1]
            )
            new_y = (
                torch.sin(angle_rad) * shifted_data[:, 0]
                + torch.cos(angle_rad) * shifted_data[:, 1]
            )
            return torch.cat([new_x[:, None], new_y[:, None]], 1) + origin
        return data

    # Create the data:
    def create_data(self):
        data = []
        for i in range(self.mu_xy.size()[0]):
            current_mu = self.mu_xy[i] * torch.ones(self.n_events, 2)
            current_sigma = self.sigma_xy[i] * torch.ones(self.n_events, 2)
            current_data = torch.normal(current_mu, current_sigma)
            data.append(self.rotate(current_data, self.angle[i], self.mu_xy[i]))
        return torch.stack(data)

    # Get len of the data set:
    def __len__(self):
        return self.n_events // self.n_samples

    # Get the dataset for each call / iteration:
    def __getitem__(self, idx):
        ridx = torch.randint(0, self.n_events, size=(self.n_samples,))
        return self.data[:, ridx, :]


class SimpleNGaussianDataset(MultiGaussianDataset):
    def __init__(
        self,
        mean: tuple = (3, 1),
        sigma: float = 0.5,
        stretch: float = 6.0,
        n_gaussians: int = 1,
        samples_per_batch: int = 1_000,
        samples_per_dataset: int = 10_000,
    ):
        self.mean = mean
        self.sigma = sigma
        self.stretch = stretch
        self.n_gaussians = n_gaussians
        mu_x = (mean[0],) * n_gaussians
        mu_y = (mean[1],) * n_gaussians
        sigma_x = (sigma * stretch,) * n_gaussians
        sigma_y = (sigma,) * n_gaussians

        if n_gaussians == 2:
            angle = tuple([-90.0, 0.0])
        else:
            angle_step = 360 / n_gaussians
            angle = tuple([i * angle_step for i in range(n_gaussians)])
        n_events = samples_per_dataset
        n_samples = samples_per_batch
        super().__init__(mu_x, sigma_x, mu_y, sigma_y, angle, n_events, n_samples)
