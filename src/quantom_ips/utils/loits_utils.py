import torch
from torch import cuda, mps
import numpy as np

"""
Define a set of helper functions that will be used in bith, 1D and 2D LOITS.
"""


# Empty cache --> Helps taking care of the memory footprint in pytorch
def empty_cache(device):
    if device.lower() == "mps":
        mps.empty_cache()
    if device.lower() == "cuda":
        cuda.empty_cache()


# Calculate the norm: Integral(xsec(x) dx), with: x = bins
# which is approximated by the trapezoid rule:
def calculate_norm(bins, xsec):
    return torch.trapezoid(xsec, bins)


# Calculate the inverse from the given CDF, by performing a linear interpolation from
# the binned CDF from theory to the corresponding bin (in any given observables)


def linear_interpolation(u, cdf, bin):
    """
    This function is directly taken from Nobuo Satos original code
    The core function: bin(u) = m*u + b
    """

    # Determine m = (bin[i+1] - bin[i]) / (cdf[i+1] - cdf[i])
    m = (bin[1:] - bin[:-1]) / (cdf[1:] - cdf[:-1] + 1e-5)
    # b = -m*cdf[i] + bin[i]
    b = bin[:-1] - (m * cdf[:-1])
    # Make sure that we get the proper indices that obey: u >= cdf
    indicies = torch.sum(torch.ge(u[:, None], cdf[None, :]), 1) - 1
    # They should lie between 0 and n_points -1
    indicies = torch.clamp(indicies, 0, m.size()[0] - 1)
    # And return the interpolation:
    return m[indicies] * u + b[indicies]


# Compute the weight matrix:
# This piece of code was suggested by Steven Goldenberg --> This calculation is
# done on CPU --> We want to use the GPU for something else
def calculate_weight_tensor(n):
    return n > torch.arange(torch.max(n), device="cpu")


# Compute grid indices so that we are able to vectorize the sampler
# The following (very efficient) lines were written by Steven Goldenberg:
def calculate_grid_indices(size_dim_0, size_dim_1):
    return np.mgrid[0:size_dim_0, 0:size_dim_1].reshape(2, -1).T
