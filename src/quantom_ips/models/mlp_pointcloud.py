import torch
import torch.nn as nn


class MLPPointCloud(nn.Module):
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        # An encoder for 2D point cloud data. It uses a shared MLP to process
        # each point, then a max-pooling operation to create a permutation-invariant
        # global feature vector.
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The input x is of shape (batch_size, num_points, 2)
        # 1. Apply the shared MLP to each point
        x = self.mlp(x)  # Shape: (batch_size, num_points, 256)

        # 2. Apply max-pooling along the point dimension to get a global feature vector
        x = torch.max(x, dim=1)[0]  # Shape: (batch_size, 256)

        # 3. Project to the latent dimension
        # The output range of tanh is [-1, 1], which is ideal for our quantizer.
        return torch.tanh(self.fc(x))
