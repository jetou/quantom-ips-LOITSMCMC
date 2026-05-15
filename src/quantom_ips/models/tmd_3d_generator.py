import torch
import torch.nn as nn
import torch.nn.functional as F
from quantom_ips.models.mlp import MLP
from omegaconf import OmegaConf, listconfig



class TMD3DGenerator(nn.Module):
    """
    Simple 3D TMD generator:
    noise (B, mlp_in) → MLP → (B, flat_dim) → reshape → (B, n_flav, nx, nQ2, nbt)
    """

    def __init__(
        self,
        mlp_in: int = 64,
        ff_dim: tuple = (256, 256, 256),
        nx: int = 10,
        nQ2: int = 10,
        nbt: int = 100,
        out_channels: int = 1,   # <-- ADD THIS (Hydra override target)
        n_flav: int | None = None,  # optional; lets you still pass n_flav explicitly
    ):
        super().__init__()

        # Hydra sometimes passes lists
        if isinstance(ff_dim, listconfig.ListConfig):
            ff_dim = OmegaConf.to_container(ff_dim)

        self.nx = nx
        self.nQ2 = nQ2
        self.nbt = nbt
        self.n_flav = n_flav
        self.input_dim = mlp_in

        flat_dim = n_flav * nx * nQ2 * nbt

        self.mlp = MLP(
            input_dim=mlp_in,
            output_dim=flat_dim,
            hidden_dims=ff_dim,
            out_activation=None,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, mlp_in)
        returns: (B, n_flav, nx, nQ2, nbt)
        """
        x = self.mlp(z)                # (B, flat_dim)
        x = F.softplus(x)
        B = x.shape[0]
        x = x.view(B, self.n_flav, self.nx, self.nQ2, self.nbt)
        # enforce positivity-ish
        #x = torch.sigmoid(x)
        return x
