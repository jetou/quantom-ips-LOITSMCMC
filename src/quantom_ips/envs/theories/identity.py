import torch


class IdentityTheory:
    def __init__(
        self,
        grid_minimums: tuple = (0.0,),
        grid_maximums: tuple = (1.0,),
        average: bool = True,
    ):

        self.grid_minimums = grid_minimums
        self.grid_maximums = grid_maximums
        self.average = average

    def forward(self, data):
        grid = self.create_grid(data.shape[2:], data.dtype, data.device)
        loss = self.get_loss(data, grid)
        return self.transform(data, grid), grid, loss

    def get_loss(self, data, grid):
        return {}

    def transform(self, data, grid):
        if self.average:
            return data.mean(dim=0, keepdim=True)
        return data

    def create_grid(self, dims, dtype, device):
        if len(self.grid_maximums) == 1:
            grid_maximums = [self.grid_maximums[0] for _ in dims]
        else:
            grid_maximums = self.grid_maximums
        if len(self.grid_minimums) == 1:
            grid_minimums = [self.grid_minimums[0] for _ in dims]
        else:
            grid_minimums = self.grid_minimums

        grid = []
        for d, min, max in zip(dims, grid_minimums, grid_maximums):
            grid.append(torch.linspace(min, max, d, device=device, dtype=dtype))

        return grid
