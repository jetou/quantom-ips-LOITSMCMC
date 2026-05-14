import torch.nn as nn
from quantom_ips.utils.torch_nn_registry import get_activation, get_initializer
from typing import Optional


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dims: tuple = (256, 128),
        out_activation: Optional[str] = None,
        dropout_rate: Optional[float] = None,
        hidden_weight_initializer: Optional[str] = None,
        hidden_bias_initializer: Optional[str] = None,
        out_weight_initializer: Optional[str] = None,
        out_bias_initializer: Optional[str] = None,
    ):
        super().__init__()

        assert len(hidden_dims) >= 1, "hidden_dims must contain at least one dimension"
        first_layer = nn.Linear(input_dim, hidden_dims[0])
        if hidden_weight_initializer is not None:
            get_initializer(hidden_weight_initializer, first_layer.weight)
        if hidden_bias_initializer is not None:
            get_initializer(hidden_bias_initializer, first_layer.bias)
        layers = [first_layer, nn.LeakyReLU(0.2, inplace=True)]
        prev_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            current_layer = nn.Linear(prev_dim, h_dim)

            if hidden_weight_initializer is not None:
                get_initializer(hidden_weight_initializer, current_layer.weight)
            if hidden_bias_initializer is not None:
                get_initializer(hidden_bias_initializer, current_layer.bias)

            layers.append(current_layer)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        output_layer = nn.Linear(prev_dim, output_dim)
        if out_weight_initializer is not None:
            get_initializer(out_weight_initializer, output_layer.weight)
        if out_bias_initializer is not None:
            get_initializer(out_bias_initializer, output_layer.bias)
        layers.append(output_layer)
        if out_activation is not None:
            layers.append(get_activation(out_activation))
        self.mlp_model = nn.Sequential(*layers)

    def forward(self, input):
        return self.mlp_model(input)
