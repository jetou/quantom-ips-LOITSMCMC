import logging
from typing import Any, Optional

from omegaconf import MISSING
from quantom_ips import make
from quantom_ips.utils.stateful_module import StatefulModule

logger = logging.getLogger(__name__)


class TMDQuantomEnv(StatefulModule):
    def __init__(
        self,
        theory: Any = MISSING,
        sampler: Any = MISSING,
        mode: str = "events",
        normalize_events: bool = True,
        sample_dir: Optional[str] = None,
    ):
        super().__init__()
        self.theory = make(theory)
        self.sampler = make(sampler) if sampler is not None else None
        if mode not in ["events", "xsec"]:
            raise ValueError("TMDQuantomEnv mode must be 'events' or 'xsec'")
        self.mode = mode
        self.normalize_events = normalize_events
        self.sample_dir = sample_dir

    def _normalize(self, x, eps=1e-8):
        dims = tuple(range(1, x.ndim))
        mu = x.mean(dim=dims, keepdim=True)
        sig = x.std(dim=dims, keepdim=True).clamp_min(eps)
        return (x - mu) / sig

    def _ensure_flavor_dim(self, x, n_flav: int):
        if x.ndim != 4:
            return x

        _, f, _, _ = x.shape
        if f == n_flav:
            return x
        if f == 1:
            return x.repeat(1, n_flav, 1, 1)
        return x

    def step(self, params, real_samples=None, n_samples: Optional[int] = None):
        if hasattr(self.theory, "e2") and hasattr(self.theory, "tar"):
            try:
                n_flav = int(self.theory.e2[self.theory.tar].numel())
                params = self._ensure_flavor_dim(params, n_flav=n_flav)
            except Exception:
                logger.debug("Could not infer SIDIS flavor dimension", exc_info=True)

        probabilities, grid_axes, theory_loss = self.theory.forward(params)
        info = {"grid_axes": grid_axes, "theory_loss": theory_loss}

        if self.mode == "xsec":
            obs = {"fake": self._normalize(probabilities)}

            if real_samples is not None:
                if real_samples.shape != probabilities.shape:
                    raise ValueError(
                        f"Real xsec shape {real_samples.shape} != fake xsec shape {probabilities.shape}"
                    )
                obs["real"] = self._normalize(real_samples)

            if (n_samples is not None) and (self.sampler is not None):
                info["fake_events"] = self.sampler.forward(
                    probabilities, grid_axes, n_samples
                )

            reward = theory_loss.get("theory", None)
            return obs, reward, True, False, info

        if real_samples is None:
            raise ValueError("In events mode you must pass real event samples.")
        if self.sampler is None:
            raise ValueError("In events mode you must provide a sampler.")

        batch_size = probabilities.shape[0]
        if real_samples.shape[0] == 1 and batch_size > 1:
            real_samples = real_samples.repeat(batch_size, 1, 1)

        if probabilities.shape[0] != real_samples.shape[0]:
            raise ValueError(
                f"Batch size mismatch: probs {probabilities.shape[0]} vs real {real_samples.shape[0]}"
            )

        gen_samples = self.sampler.forward(
            probabilities, grid_axes, real_samples.shape[-2]
        )
        info["fake_events_raw"] = gen_samples
        info["real_events_raw"] = real_samples

        if self.normalize_events:
            gen_out = (gen_samples - gen_samples.mean(dim=1, keepdim=True)) / (
                gen_samples.std(dim=1, keepdim=True).clamp_min(1e-8)
            )
            real_out = (real_samples - real_samples.mean(dim=1, keepdim=True)) / (
                real_samples.std(dim=1, keepdim=True).clamp_min(1e-8)
            )
        else:
            gen_out = gen_samples
            real_out = real_samples

        obs = {"fake": gen_out, "real": real_out}
        reward = theory_loss.get("theory", None)
        return obs, reward, True, False, info

    def forward(self, *args, **kwargs):
        return self.step(*args, **kwargs)
