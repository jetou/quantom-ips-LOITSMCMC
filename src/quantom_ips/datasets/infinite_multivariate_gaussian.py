import torch
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal


class InfiniteGaussianDataset(Dataset):
    """
    A PyTorch Dataset that generates a batch of 'n_samples' from a 'd_dim'
    Multivariate Gaussian Distribution on every access.

    This version is restricted to a diagonal covariance matrix, meaning
    all dimensions are statistically independent.

    This setup mimics a dataset with "infinite statistics" as the data is
    never stored but always newly generated, refreshing the statistics
    with every batch/item retrieved.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        d_dim: int = 2,
        mu: tuple = (0, 1),
        variances: tuple = (0.2, 0.8),
        num_batches: int = 1000,
    ):
        """
        Initializes the dataset parameters.

        Args:
            n_samples (int): The number of samples to generate per batch (i.e., the DataLoader's effective batch size).
            d_dim (int): The dimensionality of the Gaussian distribution.
            mu (tuple): The mean vector (d_dim). Can be a list, tuple, or torch.Tensor.
            variances (tuple): The variance vector (d_dim). Must contain positive values
                                           for the diagonal of the covariance matrix.
            num_batches (int): The number of times the DataLoader will call __getitem__
                                before the "epoch" ends. This is an arbitrary large number
                                to facilitate the DataLoader loop.
        """
        super().__init__()
        self.n_samples = n_samples
        self.d_dim = d_dim
        self.num_batches = num_batches

        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.variances = torch.tensor(variances, dtype=torch.float32)

        # Validation checks
        if self.mu.shape != (d_dim,):
            raise ValueError(
                f"Mean vector (mu) must have shape ({d_dim},), but got {self.mu.shape}"
            )

        # Check for 1D shape (variances vector)
        if self.variances.shape != (d_dim,):
            raise ValueError(
                f"Variances vector must have shape ({d_dim},), but got {self.variances.shape}. For diagonal covariance, please provide a vector of positive variances."
            )

        # Check for positive variances
        if (self.variances <= 0).any():
            raise ValueError("All variance values must be strictly positive.")

        # 3. Construct the diagonal covariance matrix (sigma)
        self.sigma = torch.diag(self.variances)

        # Initialize the MultivariateNormal distribution object
        # Note: Jitter is no longer strictly necessary since all variances are positive,
        # but it provides stability in case of edge floating point numbers.
        jitter = 1e-6 * torch.eye(d_dim)
        self.distribution = MultivariateNormal(
            loc=self.mu, covariance_matrix=self.sigma + jitter
        )

    def __len__(self) -> int:
        """
        Returns an arbitrary length for the DataLoader to iterate over.
        Since the data is generated on demand, this represents the number of batches
        in one "virtual epoch".
        """
        return self.num_batches

    def __getitem__(self, idx: int):
        """
        Generates a fresh batch of 'n_samples' from the Gaussian distribution.
        The index 'idx' is ignored, making every item unique.

        Returns:
            Tuple[List, torch.Tensor]: The required tuple (empty_list, data_batch).
        """
        # Generate n_samples points from the multivariate Gaussian distribution
        # The resulting tensor shape is (n_samples, d_dim)
        data_batch = self.distribution.sample(sample_shape=(self.n_samples,))

        # Return the required tuple: (empty_list, data_batch)
        return ([], data_batch)
