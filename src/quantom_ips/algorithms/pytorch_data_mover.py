import os
from pathlib import Path
from .base_algorithm import Algorithm
import torch


class PyTorchDataMover(Algorithm):
    # Moves data from the dataloader to the correct dtype and device
    # !!! Assumes optimizer contains a dtype and device field !!!
    # Only handles inputs and targets which are tensors or lists/tuples of tensors
    def __init__(self):
        super().__init__()

    def match(self, tag):
        return tag in ["process_batch"]

    def apply(self, opt, tag):
        if tag == "process_batch":
            opt.inputs = self.update_tensor(opt.inputs, opt.dtype, opt.device)
            opt.targets = self.update_tensor(opt.targets, opt.dtype, opt.device)

    def update_tensor(self, tensor, dtype, device):
        if isinstance(tensor, (list, tuple)):
            tensor = [
                tensor_val.to(dtype=dtype, device=device) for tensor_val in tensor
            ]
        else:
            tensor = tensor.to(dtype=dtype, device=device)

        return tensor
