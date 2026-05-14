import torch
import torch.nn as nn
from .layers.fsq import FSQuantization

from typing import Any
from omegaconf import MISSING
from quantom_ips import make


# === The Complete FSQ-VAE Model ===
class FSQ_VAE(nn.Module):
    def __init__(
        self, encoder: Any = MISSING, decoder: Any = MISSING, num_levels: int = 3
    ):
        super().__init__()
        self.encoder = make(encoder)
        self.quantizer = FSQuantization(num_levels)
        self.decoder = make(decoder)

    def forward(self, x: torch.Tensor):
        # 1. Pass the input through the encoder to get
        #   a continuous latent representation.
        z_latent = self.encoder(x)

        # 2. Quantize the latent representation using our custom layer.
        z_quantized = self.quantizer(z_latent)

        # 3. Pass the quantized representation through
        #   the decoder to reconstruct the input.
        x_reconstructed = self.decoder(z_quantized)

        # We return both the reconstructed output and the latent representations
        # for potential loss calculations (e.g., reconstruction loss).
        return x_reconstructed, z_latent, z_quantized
