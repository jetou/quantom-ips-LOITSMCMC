import logging
import hydra
import torch
from pathlib import Path

from quantom_ips import list_modules, make
from quantom_ips.utils.torch_nn_registry import get_dtype

logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)

logger = logging.getLogger("gan_training_workflow")


def get_device(device):
    if device is not None:
        return device

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@hydra.main(version_base=None, config_name=None, config_path="./conf")
def run(config) -> None:
    list_modules()

    opt = make(config.opt)
    if config.autodetect_device:
        dtype = get_dtype(config.dtype)
        device = get_device(config.device)
        logger.info(f"Using device: {device}")
        opt = opt.to(device=device, dtype=dtype)

    dataloader = make(config.dataloader)

    algorithms = [make(alg) for alg in config.algorithms.values()]

    opt.train(dataloader, algorithms)

    logger.info("Training complete.")


if __name__ == "__main__":
    run()
