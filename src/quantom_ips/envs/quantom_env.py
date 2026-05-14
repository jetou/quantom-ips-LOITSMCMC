from typing import Any
from omegaconf import MISSING
from quantom_ips import make
from quantom_ips.algorithms.base_algorithm import Algorithm


class QuantomEnv(Algorithm):
    def __init__(self, theory: Any = MISSING, sampler: Any = MISSING):
        super().__init__()

        # Modules:
        self.theory = make(theory)
        self.sampler = make(sampler)

        # For saving info
        self.theory_losses = []

    def match(self, tag):
        return tag in ["before_update", "after_fit"]

    def apply(self, opt, tag):
        if tag == "before_update":
            params = opt.outputs
            real_samples = opt.targets
            probabilities, grid_axes, theory_loss = self.theory.forward(params)
            gen_samples = self.sampler.forward(
                probabilities, grid_axes, real_samples.shape[-2]
            )

            # Update optimizer outputs with generated samples
            opt.outputs = gen_samples

            # Store theory losses
            self.theory_losses.append(theory_loss.get("theory", None))
