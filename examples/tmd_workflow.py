import logging
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quantom_ips import list_modules, make
from quantom_ips.utils.torch_nn_registry import get_dtype

logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)
logger = logging.getLogger("tmd_workflow")

_orig_check_help = argparse.ArgumentParser._check_help


def _check_help_compat(self, action):
    if action.help is not None and not isinstance(action.help, str):
        return
    return _orig_check_help(self, action)


argparse.ArgumentParser._check_help = _check_help_compat


defaults = [
    {"env": "TMDQuantomEnv"},
    {"opt": "TMDGANCC"},
    {"dataloader": "tmd_events"},
    "_self_",
]


@dataclass
class TMDWorkflow:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    env: Any = MISSING
    opt: Any = MISSING
    dataloader: Any = MISSING
    device: Optional[str] = None
    dtype: str = "float32"
    output_dir: str = "${hydra:runtime.output_dir}"


cs = ConfigStore.instance()
cs.store(name="tmd_workflow", node=TMDWorkflow)


def get_device(device):
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@hydra.main(version_base=None, config_name="tmd_workflow")
def run(config) -> None:
    list_modules()

    dtype = get_dtype(config.dtype)
    device = get_device(config.device)
    output_dir = Path(config.output_dir)
    logger.info("Using device: %s", device)

    env = make(config.env).to(device=device, dtype=dtype)

    if getattr(env, "sample_dir", None) is None:
        data_path = Path(config.dataloader.dataset.path)
        env.sample_dir = str(data_path.parent)

    nx = getattr(env.theory, "nx", None)
    nbt = getattr(env.theory, "nbt", None)
    if nx is not None and nbt is not None:
        for key in ("generator", "model"):
            cfg = getattr(config.opt, key, None)
            if cfg is not None and "image_size" in cfg:
                cfg.image_size = [nx, nbt]

    if "discriminator" in config.opt:
        if getattr(config.opt, "train_on", "events") == "events":
            config.opt.discriminator.input_dim = getattr(
                config.dataloader.dataset, "event_dim", 5
            )

    opt = make(config.opt).to(device=device, dtype=dtype)
    opt.logdir = str(output_dir)
    dataloader = make(config.dataloader)

    results = opt.train(env, dataloader)

    output_dir.mkdir(parents=True, exist_ok=True)
    opt.save(output_dir / "optimizer.pt")
    env.save(output_dir / "environment.pt")
    torch.save(results, output_dir / "losses.pt")
    logger.info("Training complete.")


if __name__ == "__main__":
    run()
