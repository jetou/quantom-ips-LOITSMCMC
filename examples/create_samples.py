import sys, os
import logging
import hydra
import torch
import numpy as np
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import List, Any, Optional
from pathlib import Path
from hydra.core.config_store import ConfigStore

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quantom_ips import list_modules, make

logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)

logger = logging.getLogger("create_christina")

defaults = [
    {"env/theory": "IdentityTheory"},
    {"env/sampler":"LOITS2D"},
    "_self_",
]


@dataclass
class CreateSampleData:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    theory: Any = MISSING
    sampler: Any = MISSING
    device: Optional[str] = None
    dtype: str = "float32"
    output_dir: str = "${hydra:runtime.output_dir}"
    data_name: str = "wonderful_data"
    n_samples: int = 100000
    n_repeats: int = 10
    seed: int = 0


cs = ConfigStore.instance()
cs.store(name="create_christina", node=CreateSampleData)


def get_dtype(dtype_str):
    if "float32" in dtype_str:
        return torch.float32
    elif "float64" in dtype_str:
        return torch.float64
    else:
        raise ValueError("dtype must be either 'float32' or 'float64'")

def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device):
    if device is not None:
        return device

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@hydra.main(version_base=None, config_name="create_christina")
def run(config) -> None:
    #list_modules()

    #--number of independent flavors
    nf = 1

    dtype = get_dtype(config.dtype)
    device = get_device(config.device)
    output_dir = Path(config.output_dir)

    logger.info(f"Using device: {device}")
   
    # Load theory and sampler:
    theory = make(config.env.theory)
    sampler = make(config.env.sampler)

    # Create the density from which we sample events:
    logger.info(f"Create density on grid with size: {config.env.theory.nx,config.env.theory.nQ2}")
    #------------- theory has bt in its init
    x = theory.x
    bt = theory.bt
    # Get a grid:
    #---- NP_model_PDF has inside torch.meshgrid 
    density = theory.NP_model_PDF(bt, x,theory.bmax)

    #-- add x dependence to grid
    xshape = x**(-0.5) * (1-x)**3
    density = torch.einsum('x,xb->xb',xshape,density)
    # Expand density into 6 flavors
    density = density.unsqueeze(0).expand(nf,*density.shape)

    # Make sure that the density has the right shape: (batch_size x number of flavors x x_dim x y_dim)
    density = density[None,...]
    print('density shape:',density.shape)
  
    D = {}
    D['density'] = density.cpu().numpy()
    D['x']  = x.cpu().numpy()
    D['bt'] = bt.cpu().numpy()
     
    np.save(f"{output_dir}/density.npy",D)
    # Use theory:
    logger.info("Pass density through theory module")
    cross_sections, grid_axes, _ = theory.forward(density)

    # ------------------------------------------
    # Repeat sampling to estimate sampling noise
    # ------------------------------------------
    n_repeats = int(getattr(config, "n_repeats", 1))
    base_seed = int(getattr(config, "seed", 0))
    
    logger.info(f"Generate {config.n_samples} events, repeated {n_repeats} times (seed base={base_seed}).")
    
    events_reps = []
    
    for r in range(n_repeats):
        seed_all(base_seed + r)
    
        events = sampler.forward(cross_sections, grid_axes, config.n_samples)
    
        # Events may be (batch x targets x samples x dims) or (batch x samples x dims).
        if events.ndim == 4:
            events = events.flatten(0, 1)
        elif events.ndim != 3:
            raise ValueError(f"Expected sampled events with 3 or 4 dims, got {events.shape}")
        events = events.reshape(-1, events.shape[-1])
    
        # (optional) normalization stays off, as you currently have it
    
        events_reps.append(events.detach().cpu().numpy())
    
    events_reps = np.stack(events_reps, axis=0)  # (n_repeats, n_events, 5)
    
    # Save stacked repeats (recommended)
    np.save(f"{output_dir}/{config.data_name}_reps.npy", events_reps)
    
    # Also save a "default single" for backward compatibility
    np.save(f"{output_dir}/{config.data_name}.npy", events_reps[0])
    
    logger.info(f"Saved repeats to {output_dir}/{config.data_name}_reps.npy")
    logger.info("Data creation complete.")


    # Use sampler to generate 100k events:
    # logger.info(f"Generate {config.n_samples} events.")
    # print('cross section shape:',cross_sections.shape)
    # events = sampler.forward(cross_sections,grid_axes,config.n_samples) 

    
    G = {}
    G['x']   = grid_axes[0].cpu().numpy()
    G['Q2']  = grid_axes[1].cpu().numpy()
    G['z']   = grid_axes[2].cpu().numpy()
    G['qt']  = grid_axes[3].cpu().numpy()
    G['phi'] = grid_axes[4].cpu().numpy()

    np.save(f"{output_dir}/cross_section.npy",cross_sections.detach().cpu().numpy())
    np.save(f"{output_dir}/grid.npy",G)


if __name__ == "__main__":
    run()
