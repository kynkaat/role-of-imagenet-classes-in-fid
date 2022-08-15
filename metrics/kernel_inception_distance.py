# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Kernel Inception Distance (KID). Adapted from Karras et al.:
   https://github.com/NVlabs/stylegan2-ada/blob/main/metrics/kernel_inception_distance.py"""

import numpy as np


def compute_kid(real_features: np.ndarray,
                gen_features: np.ndarray,
                num_subsets: int = 100,
                max_subset_size: int = 1000) -> float:
    """Computes Kernel Inception Distance."""
    assert real_features.ndim == 2 and gen_features.ndim == 2
    assert real_features.shape[0] == gen_features.shape[0]
    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return t / num_subsets / m
