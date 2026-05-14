import torch.nn as nn
import torch.nn.functional as F
from quantom_ips.models.mlp import MLP
from quantom_ips.models.layers.conv2dstack import ConvTranspose2dStack

from omegaconf import OmegaConf, listconfig


class Conv2DTGenerator(nn.Module):
    def __init__(
        self,
        mlp_in: int = 10,
        ff_dim: tuple = (100, 100),
        image_size: tuple = (10, 10),
        n_layers: int = 4,
    ):
        super().__init__()

        self.input_dim = mlp_in

        if isinstance(image_size, listconfig.ListConfig):
            image_size = OmegaConf.to_container(image_size)

        flat_image_dim = image_size[0] * image_size[1]
        self.mlp_model = MLP(
            input_dim=mlp_in,
            output_dim=flat_image_dim,
            hidden_dims=ff_dim,
            out_activation=None,
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(flat_image_dim, 1, 1))
        self.convT_stack = ConvTranspose2dStack(
            in_channels=flat_image_dim,
            out_channels=1,
            kernel_size=3,
            internal_channels=flat_image_dim,
            n_layers=n_layers,
            activation={"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
            out_activation="Sigmoid",
        )

        self.upsample = nn.Upsample(mode="bilinear", size=image_size)

    def forward(self, x):
        x = self.mlp_model(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.unflatten(x)
        x = self.convT_stack(x)
        x = self.upsample(x)
        return x
