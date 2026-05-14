import torch
import pytest
from quantom_ips.envs.samplers.inverse_transform_sampler import InverseTransformSampler


def soft_gate(y, center, direction=1, width=0.1):
    return 0.5 * (1 + torch.tanh(direction * (y - center) / width))


class TestBITS:
    sampler = InverseTransformSampler()

    def test_oob(self):
        # ---------- Build the DHM target on [0,1]^2 with small cell sizes ----------
        nx, ny = 10_000, 1_000  # fine grid -> tiny cells
        R = 0.20
        x_shift = 0.10
        y_shift = 0.10
        sigma = 0.1
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0

        x = torch.linspace(x_min, x_max, nx)
        y = torch.linspace(y_min, y_max, ny)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        x1, y1 = 0.5 - x_shift, 0.5 + y_shift
        x2, y2 = 0.5 + x_shift, 0.5 - y_shift
        r1 = torch.sqrt((X - x1) ** 2 + (Y - y1) ** 2)
        r2 = torch.sqrt((X - x2) ** 2 + (Y - y2) ** 2)

        gate1 = soft_gate(Y, y1, +1)
        gate2 = soft_gate(Y, y2, -1)
        A = (
            torch.exp(-((r1 - R) ** 2) / (2 * sigma**2)) * gate1
            + torch.exp(-((r2 - R) ** 2) / (2 * sigma**2)) * gate2
        )
        # Number of samples to draw in each test
        N = 100_000
        grid_list = [x, y]
        print("Grid shape:", A.shape)
        print("Sample count N:", N)
        print("x_min, x_max:", x_min, x_max)
        print("y_min, y_max:", y_min, y_max)

        with torch.no_grad():
            samples = self.sampler.sample(A, grid_list, N)

        x, y = samples[:, 0], samples[:, 1]

        print("x_min, x_max:", x.min(), x.max())
        print("y_min, y_max:", y.min(), y.max())

        assert x.min() >= x_min
        assert y.min() >= y_min

        assert x.max() <= x_max
        assert y.max() <= y_max

    def test_zeros(self):
        grid_list = [torch.linspace(0, 1, 10)]
        A = torch.zeros((10))
        n = 100_000
        with torch.no_grad():
            samples = self.sampler.sample(A, grid_list, n)
        hist = torch.histogram(samples, 10).hist / n

        # Choosing a fairly loose threshold for 100_000 uniform samples
        # This could fail in very(!!) unlucky situations
        assert hist == pytest.approx(0.1, abs=5e-3)

    def test_single_nonzero_in_large_tensor(self):
        spaces = 10_000_000
        grid_list = [torch.linspace(0, 1, spaces)]
        A = torch.zeros((spaces))
        A[10294] = 1
        n = 100_000
        with torch.no_grad():
            samples = self.sampler.sample(A, grid_list, n)

        print(samples.min(), samples.max())
