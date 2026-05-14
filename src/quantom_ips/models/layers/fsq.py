import torch
import torch.nn as nn


# === Straight-Through Estimator (STE) for Quantization ===
# This custom PyTorch function is essential for allowing gradients
# to flow backward through the non-differentiable rounding operation.
class StraightThroughEstimator(torch.autograd.Function):
    """
    Implements a Straight-Through Estimator for quantization.
    In the forward pass, it performs the rounding operation to get discrete values.
    In the backward pass, it simply passes the gradients straight through,
    ignoring the rounding operation to enable training of the encoder.
    """

    @staticmethod
    def forward(ctx, x, levels):
        # Quantize the input tensor `x`
        # 1. Clamp the input values to the range [-1, 1] for stable quantization.
        x_clamped = torch.clamp(x, -1.0, 1.0)

        # 2. Scale the clamped values from [-1, 1] to a range suitable for rounding,
        #    e.g., [0, num_levels - 1].
        x_scaled = (x_clamped + 1) / 2 * (levels - 1)

        # 3. Round the scaled values to the nearest integer to get the discrete codes.
        x_q = torch.round(x_scaled)

        # 4. De-quantize the integer codes back to the original value range [-1, 1].
        x_hat = (x_q / (levels - 1)) * 2 - 1

        return x_hat

    @staticmethod
    def backward(ctx, grad_output):
        # The core of STE: the gradient is passed directly through
        # to the preceding layer (the encoder), as if the rounding
        # operation did not happen.
        return grad_output, None


# === Finite Scalar Quantization (FSQ) Layer ===
class FSQuantization(nn.Module):
    """
    The main quantization layer for the VAE.
    It takes a continuous latent representation and quantizes it
    to a set of discrete values defined by `num_levels`.
    """

    def __init__(self, num_levels: int):
        super().__init__()
        if not isinstance(num_levels, int) or num_levels <= 1:
            raise ValueError("`num_levels` must be an integer greater than 1.")

        # Register `num_levels` as a non-trainable parameter.
        # This is a good practice to ensure it's part of the model's state.
        self.levels = nn.Parameter(
            torch.tensor(num_levels, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the quantization using the straight-through estimator.
        """
        # Call the custom autograd function to perform quantization with STE.
        quantized_x = StraightThroughEstimator.apply(x, self.levels)
        return quantized_x
