import numpy as np
import logging

"""
Simple function to compute the sum of absolute differences
"""


# Compute the log sad score:
def compute_logSAD(img_true, img_pred):
    abs_diff = np.abs(img_true - img_pred)

    return np.log(np.sum(abs_diff) + 1e-7)


# Get the log sad score from events:
def get_logSAD_from_events(real, pred, n_bins):
    # 2D case:
    if real.shape[1] == 2 and n_bins:
        real_corr, _, _ = np.histogram2d(real[:, 0], real[:, 1], n_bins)
        pred_corr, _, _ = np.histogram2d(pred[:, 0], pred[:, 1], n_bins)
        return compute_logSAD(real_corr, pred_corr)

    elif real.shape[1] == 1 and n_bins:
        real_counts, _ = np.histogram(real, n_bins)
        pred_counts, _ = np.histogram(pred, n_bins)
        return compute_logSAD(real_counts, pred_counts)

    else:
        logging.warning("The log[SAD] score can only be computed for 1D or 2D data")
        return -1.0
