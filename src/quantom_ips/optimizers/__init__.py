from quantom_ips import register

register(
    id="GAN",
    entry_point="quantom_ips.optimizers.gan_optimizer:GANOptimizer",
    group="opt",
)

register(
    id="TMDGAN",
    entry_point="quantom_ips.optimizers.tmd_gan_optimizer:GANOptimizer",
    group="opt",
    kwargs=dict(noise_dim=128),
    defaults=[
        {"/model@discriminator": "MLPDiscriminatorSigOut"},
        {"/model@generator": "TMDConv2DTGenerator"},
        "_self_",
    ],
)

register(
    id="TMDGANCC",
    entry_point="quantom_ips.optimizers.tmd_gan_optimizer_cc:GANOptimizer",
    group="opt",
    kwargs=dict(noise_dim=128),
    defaults=[
        {"/model@discriminator": "MLPDiscriminatorSigOut"},
        {"/model@generator": "TMDConv2DTGenerator"},
        "_self_",
    ],
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
