import torch.nn as nn
import torch


class ParameterModel(nn.Module):
    def __init__(
        self,
        output_dim: int = 1,
    ):
        super().__init__()

        self.params = nn.Parameter(torch.rand(output_dim), requires_grad=True)

    def forward(self, input):
        return self.params.unsqueeze(0).repeat(input.shape[0], 1)
