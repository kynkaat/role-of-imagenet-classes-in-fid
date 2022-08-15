"""Utility functions used in generating FID sensitivity heatmaps."""

import cv2
import numpy as np
import PIL.Image
import torch
from typing import Any

from resampling.resample_utils import wasserstein2_loss


class ActivationsAndGradients:
    """Class to obtain intermediate activations and gradients.
       Adapted from: https://github.com/jacobgil/pytorch-grad-cam"""

    def __init__(self,
                 network: Any,
                 network_kwargs: dict,
                 target_layer_name: str = 'mixed_10') -> None:
        self.network = network
        self.network_kwargs = network_kwargs
        self.gradients: List[np.ndarray] = []
        self.activations: List[np.ndarray] = []

        target_layer = getattr(network.layers, target_layer_name)
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self,
                        module: Any,
                        input: Any,
                        output: Any) -> None:
        """Saves forward pass activations."""
        activation = output
        self.activations.append(activation.detach().cpu().numpy())

    def save_gradient(self,
                      module: Any,
                      grad_input: Any,
                      grad_output: Any) -> None:
        """Saves backward pass gradients."""
        # Gradients are computed in reverse order.
        grad = grad_output[0]
        self.gradients = [grad.detach().cpu().numpy()] + self.gradients  # Prepend current gradients.

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Resets hooked activations and gradients and calls model forward pass."""
        self.gradients = []
        self.activations = []
        return self.network(x, **self.network_kwargs)


def zero_one_scaling(image: np.ndarray) -> np.ndarray:
    """Scales an image to range [0, 1]."""
    if np.all(image == 0):
        return image
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min())


def show_sensitivity_on_image(image: np.ndarray,
                              sensitivity_map: np.ndarray,
                              colormap: int = cv2.COLORMAP_PARULA,
                              heatmap_weight: float = 2.0) -> np.ndarray:
    """Overlay the sensitivity map on the image."""
    # Convert sensitivity map to a heatmap.
    heatmap = cv2.applyColorMap(sensitivity_map, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    # Overlay original RGB image and heatmap with specified weights.
    scaled_image = zero_one_scaling(image=image)
    overlay = heatmap_weight * heatmap.transpose(2, 0, 1) + scaled_image
    overlay = zero_one_scaling(image=overlay)

    return np.clip(overlay * 255, 0.0, 255.0).astype(np.uint8)


def compute_sensivity_map(activations: np.ndarray,
                          gradients: np.ndarray) -> np.ndarray:
    """Computes sensitivity map from given activations and gradients."""
    weights = np.mean(gradients**2, axis=(2, 3), keepdims=True)
    return (weights * activations).sum(axis=1)


def compute_sensitivity_heatmap(gen_image: torch.Tensor,
                                acts_and_gradients: ActivationsAndGradients,
                                mean_reals: torch.Tensor,
                                cov_reals: torch.Tensor,
                                mean_gen: torch.Tensor,
                                cov_gen: torch.Tensor,
                                num_images: int) -> PIL.Image.Image:
    """Computes sensitivity heatmap for a generated image."""
    #----------------------------------------------------------------------------
    # Compute features of the visualized image.
    features = acts_and_gradients(gen_image)

    #----------------------------------------------------------------------------
    # Compute updated mean and covariance, when reference image is added to set of gen. images.
    mean = ((num_images - 1) / num_images) * mean_gen + (1 / num_images) * features
    cov = ((num_images - 2) / (num_images - 1)) * cov_gen + \
        (1 / num_images) * torch.mm((features - mean_gen).T, (features - mean_gen))

    loss = wasserstein2_loss(mean_reals=mean_reals,
                             mean_gen=mean,
                             cov_reals=cov_reals,
                             cov_gen=cov)
    loss.backward()

    #----------------------------------------------------------------------------
    # Get activations and gradients from the target layer by accessing hooks.
    activations = acts_and_gradients.activations[-1]
    gradients = acts_and_gradients.gradients[-1]

    # Calculate sensitivity map with the hooked activations and gradients.
    sensitivity_map = compute_sensivity_map(activations=activations,
                                            gradients=gradients)

    #----------------------------------------------------------------------------
    # Visualize sensitivity map on top of the input image.
    _, _, h, w = gen_image.shape
    gen_image_np = gen_image.detach().cpu().numpy()[0]
    sensitivity_map = zero_one_scaling(image=sensitivity_map)
    sensitivity_map = np.clip((sensitivity_map * 255.0).astype(np.uint8), 0.0, 255.0)
    sensitivity_map = np.array(PIL.Image.fromarray(sensitivity_map[0]).resize((w, h), resample=PIL.Image.LANCZOS).convert('L'))  # Scale to original image size.

    overlay_image = show_sensitivity_on_image(image=gen_image_np,
                                              sensitivity_map=sensitivity_map)
    im = PIL.Image.fromarray(overlay_image.transpose(1, 2, 0))

    return im
