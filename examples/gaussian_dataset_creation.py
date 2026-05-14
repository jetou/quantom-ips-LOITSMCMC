# make_dataset.py
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path


@dataclass
class DatasetConfig:
    # parameters (true values)
    parameters: tuple = (2.0, 1.0)

    # coefficients
    coefficients: tuple = (1.0, 1.0)

    # variance adjustment
    epsilon: float = 1.0

    # number of samples
    n_samples: int = 1000

    # random seed
    seed: int = 0

    # output file path
    out_path: str = "dataset.pkl"


cs = ConfigStore.instance()
cs.store(name="config", node=DatasetConfig)


@hydra.main(version_base=None, config_name="config")
def main(cfg: DatasetConfig):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    rng = np.random.default_rng(cfg.seed)

    # convert to numpy arrays for easy math
    params = np.array(cfg.parameters, dtype=float)
    coeffs = np.array(cfg.coefficients, dtype=float)

    # mean = dot product
    mean = float(np.dot(coeffs, params))
    var = 1.0 + cfg.epsilon
    if var <= 0:
        raise ValueError(f"Variance must be positive, got var={var}")

    # sample data
    data = rng.normal(loc=mean, scale=np.sqrt(var), size=cfg.n_samples)

    # pack into dictionary
    dataset = {
        "parameters": tuple(params.tolist()),
        "coefficients": tuple(coeffs.tolist()),
        "epsilon": float(cfg.epsilon),
        "mean": mean,
        "var": var,
        "n_samples": int(cfg.n_samples),
        "seed": int(cfg.seed),
        "data": data,
    }

    # save as pickle
    out_path = Path(to_absolute_path(cfg.out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved dataset to: {out_path}")


if __name__ == "__main__":
    main()
