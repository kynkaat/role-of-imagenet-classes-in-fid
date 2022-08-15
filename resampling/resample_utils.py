"""Utility functions used in resampling."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, \
                   Tuple

from metrics.frechet_inception_distance import compute_fid
from metrics.kernel_inception_distance import compute_kid


def plot_top1_distribution(fname: str,
                           real_probs: np.ndarray,
                           gen_probs: np.ndarray,
                           imagenet_labels: dict,
                           real_legend: str = 'Real',
                           gen_legend: str = 'Generated',
                           top_n: int = 25,
                           figsize: Tuple[int, int] = (9, 7),
                           axis_fontsize: int = 36,
                           label_fontsize: int = 20,
                           legend_fontsize: int = 28,
                           bar_width: float = 0.4,
                           dpi: int = 300) -> None:
    """Plots a Top-1 class distribution."""
    assert gen_probs.shape[0] == real_probs.shape[0]

    # Real and generated Top-1 class predictions.
    real_classes = np.argmax(real_probs, axis=1)
    gen_classes = np.argmax(gen_probs, axis=1)
    idxs_reals, real_counts = np.unique(real_classes, return_counts=True)
    idxs_gen, gen_counts = np.unique(gen_classes, return_counts=True)

    y_real: List[int] = []  # Real class counts.
    y_gen: List[int] = []   # Generated class counts.
    for idx in idxs_reals:
        y_real.append(real_counts[idxs_reals == idx][0])
        y_gen.append(gen_counts[idxs_gen == idx][0] if idx in idxs_gen else 0)

    # Order by number of reals in each class.
    ordering = np.argsort(y_real)[::-1]
    y_real = np.asarray(y_real)[ordering][:top_n]
    y_gen = np.asarray(y_gen)[ordering][:top_n]
    real_labels = [imagenet_labels[idx].split(',')[0].lower().capitalize() \
        for idx in idxs_reals[ordering]][:top_n]

    _, ax = plt.subplots(figsize=figsize)
    x = np.arange(0, top_n * 2, 2, dtype=int)
    plt.bar(x - bar_width,    
            y_real,
            tick_label=real_labels,
            color='tab:blue',
            label=real_legend)
    plt.bar(x + bar_width,    
            y_gen,
            tick_label=real_labels,
            color='tab:orange',
            label=gen_legend)
    ax.tick_params(axis='x',
                   labelrotation=90,
                   labelsize=label_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylabel('Number of images', fontsize=axis_fontsize)
    plt.xlim([-1.0, x.max() + 2])
    plt.legend(loc='best', fontsize=legend_fontsize)
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(fname, dpi=dpi)
    plt.close('all')


def binarize_features(feature_mode: str,
                      real_logits: np.ndarray,
                      gen_logits: np.ndarray,
                      num_dims: int,
                      max_classes: int = 1008) -> Tuple[np.ndarray, np.ndarray]:
    """Applies binarization to features."""
    assert real_logits.shape[1] == gen_logits.shape[1]
    assert real_logits.shape[1] == max_classes

    if feature_mode == 'top_n':  # Use Top-N classes.
        real_idxs = np.argsort(real_logits, axis=1)[:, ::-1][:, :num_dims]
        gen_idxs = np.argsort(gen_logits, axis=1)[:, ::-1][:, :num_dims]
    elif feature_mode == 'middle_n':  # Use Middle-N sorted classes.
        mid = real_logits.shape[1] // 2
        if num_dims % 2 == 0:
            shift = num_dims // 2
            real_idxs = np.argsort(real_logits, axis=1)[:, ::-1][:, (mid - shift):(mid + shift)]
            gen_idxs = np.argsort(gen_logits, axis=1)[:, ::-1][:, (mid - shift):(mid + shift)]
        else:
            shift = (num_dims - 1) // 2
            real_idxs = np.argsort(real_logits, axis=1)[:, ::-1][:, (mid - shift):(mid + shift + 1)]
            gen_idxs = np.argsort(gen_logits, axis=1)[:, ::-1][:, (mid - shift):(mid + shift + 1)]
    else:
        NotImplementedError(f'Unknown feature mode: {feature_mode}')

    # Set selected indices to one and others to zero.
    real_features = np.zeros_like(real_logits)
    gen_features = np.zeros_like(gen_logits)
    real_features[np.arange(real_features.shape[0], dtype=int)[:, None], real_idxs] = 1.0
    gen_features[np.arange(gen_features.shape[0], dtype=int)[:, None], gen_idxs] = 1.0

    # Consider only "non-zero" columns.
    col_mask_reals = np.logical_not(np.asarray([np.all(col == 0.0) for col in real_features.T]))
    col_mask_gen = np.logical_not(np.asarray([np.all(col == 0.0) for col in gen_features.T]))
    col_mask = np.logical_and(col_mask_reals, col_mask_gen)
    real_features = real_features[:, col_mask]
    gen_features = gen_features[:, col_mask]

    return real_features, gen_features


def evaluate_metric(real_features: np.ndarray,
                    gen_features: np.ndarray,
                    w_log: np.ndarray,
                    num_repeats: int = 3,
                    metric_name: str = 'fid') -> np.ndarray:
    """Computes metrics by resampling features according to weights."""
    assert metric_name in ['fid', 'kid']
    # Calculate "discrete" metric num_repeats times.
    metric_vals = np.zeros(num_repeats, dtype=np.float32)
    metric_fn = compute_fid if metric_name == 'fid' else compute_kid
    for i in range(num_repeats):
        idxs = sample_indices_with_weights(w_log=w_log, num_items=real_features.shape[0])
        metric_vals[i] = metric_fn(real_features=real_features,
                                   gen_features=gen_features[idxs])
    return metric_vals


def sample_indices_with_weights(w_log: np.ndarray,
                                num_items: int) -> np.ndarray:
    """Samples random indices according to weights."""
    assert len(w_log.shape) == 1
    # Convert log-parameterized weights to sampling probabilities.
    probs = np.exp(w_log) / np.exp(w_log).sum()
    return np.random.choice(w_log.shape[0], size=num_items, replace=True, p=probs)


def compute_weighted_mean_and_cov(gen_features: torch.Tensor,
                                  w_log: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes weighted mean and covariance."""
    # Weighted mean.
    mean_weighted = (1.0 / torch.exp(w_log).sum()) * \
        torch.matmul(torch.exp(w_log).t(), gen_features)  # Shape: (1 x D).

    # Weighted covariance matrix.
    cov_weighted = (1.0 / (torch.exp(w_log).sum() - 1.0)) * \
        torch.matmul((gen_features - mean_weighted).t(), torch.exp(w_log) * (gen_features - mean_weighted))  # Shape: (D x D).
    return mean_weighted, cov_weighted


def wasserstein2_loss(mean_reals: torch.Tensor,
                      mean_gen: torch.Tensor,
                      cov_reals: torch.Tensor,
                      cov_gen: torch.Tensor,
                      eps: float = 1e-12) -> torch.Tensor:
    """Computes 2-Wasserstein distance."""
    mean_term = torch.sum(torch.square(mean_reals - mean_gen.squeeze(0)))
    eigenvalues, _ = torch.eig(torch.matmul(cov_gen, cov_reals), eigenvectors=True)  # Eigenvalues shape: (D, 2) (real and imaginary parts).
    cov_term = torch.trace(cov_reals) + torch.trace(cov_gen) - 2 * torch.sum(torch.sqrt(eigenvalues[:, 0] + eps))
    return mean_term + cov_term
