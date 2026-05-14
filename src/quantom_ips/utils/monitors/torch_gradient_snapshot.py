import torch
import numpy as np


class TorchGradientSnapshot:
    """
    This class is a simplified version of the torch_gradient_monitor, i.e. without the torch.metrics overhead which is convenient but also very resources intensive.
    This class simpy throws out the gradients of the model at a current stage.
    """

    # Initialize:
    # ******************************************
    def __init__(self, model, excluded_layer_idx=[-1]):
        self.model = model
        self.excluded_layer_idx = excluded_layer_idx  # --> Here, the user might add the indices of layers that shall not be considered
        # during the gradient analysis, e.g. a custom layer with weird parameters. Lets say the second and fourth hidden layer of a model shall be excluded,
        # then one would simply set: exclude_layer_idx = [2,4]

        # Run a small evaluation of the model:
        self.active_layer_idx = (
            []
        )  # --> Needed for later, when the results are visualized
        layer_counter = 0
        acc_layer_counter = 0
        # ++++++++++++++++++++++++++++++
        for name, weights in self.model.named_parameters():
            if (
                "bias" not in name and "weight" in name
            ):  # --> We are not interested in the bias values here
                layer_counter += 1

                if (
                    layer_counter not in self.excluded_layer_idx
                ):  # --> We do not count excluded layers
                    acc_layer_counter += 1

                    if (
                        weights.requires_grad
                    ):  # --> Look at active neurons / layers only
                        self.active_layer_idx.append(acc_layer_counter)
        # ++++++++++++++++++++++++++++++

        self.n_layers = acc_layer_counter
        self.n_active_layers = len(self.active_layer_idx)

    # ******************************************

    # Take snaphsot of gradients:
    # ******************************************
    def take_snapshot(self):
        # Set up gradients to be recorded:
        avg_grad = np.zeros(self.n_active_layers)
        min_grad = np.zeros(self.n_active_layers)
        max_grad = np.zeros(self.n_active_layers)

        i = 0
        # ++++++++++++++++++++++++
        for name, weights in self.model.named_parameters():
            if (weights.requires_grad) and ("bias" not in name) and ("weight" in name):

                current_gradient = weights.grad.abs().cpu().numpy()

                avg_grad[i] = np.mean(current_gradient)
                max_grad[i] = np.max(current_gradient)
                min_grad[i] = np.min(current_gradient)

                i += 1
        # ++++++++++++++++++++++++

        return {
            "average_gradients": avg_grad,
            "minimum_gradients": min_grad,
            "maximum_gradients": max_grad,
        }

    # ******************************************
