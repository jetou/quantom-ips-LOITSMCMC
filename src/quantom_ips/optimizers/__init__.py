from quantom_ips import register

register(
    id="GAN",
    entry_point="quantom_ips.optimizers.gan_optimizer:GANOptimizer",
    group="opt",
)

register(
    id="Geomloss",
    entry_point="quantom_ips.optimizers.geomloss_optimizer:GeomlossOptimizer",
    group="opt",
    defaults=[
        {"/model@generator": "MLP"},
        "_self_",
    ],
)

register(
    id="distGAN",
    entry_point=(
        "quantom_ips.optimizers.distributed_gan_optimizer:DistributedGANOptimizer"
    ),
    group="opt",
)

register(
    id="VAE",
    entry_point="quantom_ips.optimizers.vae_optimizer:VAEOptimizer",
    group="opt",
    defaults=[
        {"/model": "FSQ_VAE"},
        "_self_",
    ],
)
