from quantom_ips import register


register(
    id="basic",
    entry_point="quantom_ips.datasets.basic_dataloader:BasicDataLoader",
    group="dataloader",
)

register(
    id="numpy",
    entry_point="quantom_ips.datasets.basic_dataloader:BasicDataLoader",
    group="dataloader",
    defaults=[{"dataset": "numpy"}, "_self_"],
)

register(
    id="numpy",
    entry_point="quantom_ips.datasets.numpy_dataset:NumpyDataset",
    group="dataloader/dataset",
)

register(
    id="mgaussian",
    entry_point="quantom_ips.datasets.multi_gaussian_dataset:MultiGaussianDataset",
    group="dataloader/dataset",
)

register(
    id="simplegaussian",
    entry_point="quantom_ips.datasets.multi_gaussian_dataset:SimpleNGaussianDataset",
    group="dataloader/dataset",
)

register(
    id="mgaussian_base",
    entry_point="quantom_ips.datasets.basic_dataloader:BasicDataLoader",
    group="dataloader",
    defaults=[{"dataset": "mgaussian"}, "_self_"],
)

register(
    id="gaussian_pickles",
    entry_point="quantom_ips.datasets.gaussian_pickles:GaussianPickles",
    group="dataloader/dataset",
)

register(
    id="infinite_gaussian",  # Infinite MultiVariate Gaussian"
    entry_point=(
        "quantom_ips.datasets.infinite_multivariate_gaussian:InfiniteGaussianDataset"
    ),
    group="dataloader/dataset",
)
