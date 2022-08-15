"""Visualize regions of a generated image to which is the most sensitive to."""

import numpy as np
import os
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, \
                   Optional

from dataset import get_dataloader
from fid_heatmaps.heatmap_utils import ActivationsAndGradients, \
                                       compute_sensitivity_heatmap
from utils import bar_chart, \
                  create_grid, \
                  generate_images, \
                  load_feature_network, \
                  load_imagenet_labels, \
                  load_stylegan2


def visualize_heatmaps(results_dir: str,
                       zip_path: str,
                       pkl_path: str,
                       num_images: int = 50000,
                       batch_size: int = 32,
                       resolution: int = 256,
                       truncation_psi: float = 1.0,
                       noise_mode: str = 'const',
                       save_classifications: bool = False,
                       top_n: int = 10,
                       gen_grid: bool = False,
                       num_rows: int = 5,
                       num_cols: int = 8,
                       image_seeds: Optional[List[int]] = None,
                       random_seed: int = 0) -> None:
    """Visualizes to which regions in the images FID is the most sensitive to."""
    if gen_grid and image_seeds is not None:
        assert len(image_seeds) >= num_rows * num_cols, \
            f'At least {num_rows * num_cols} seed are required to generate a {num_rows}x{num_cols} image grid.'
    device = torch.device('cuda:0')
    imagenet_labels = load_imagenet_labels('./data/imagenet-2012-labels.yml')

    #----------------------------------------------------------------------------
    # Load StyleGAN2 pickle.
    G_ema = load_stylegan2(pkl_path=pkl_path).to(device)

    #----------------------------------------------------------------------------
    # Load Inception-V3.
    inception_v3 = load_feature_network(network_name='inception_v3_tf').to(device)

    #----------------------------------------------------------------------------
    # Compute features for reals.
    print('Computing real features...')
    dataloader = get_dataloader(zip_path=zip_path,
                                resolution=resolution,
                                batch_size=batch_size,
                                num_images=num_images)
    real_features: List[np.ndarray] = []

    for images, _ in dataloader:
        with torch.no_grad():
            real_features.append(inception_v3(images.to(device), return_features=True).cpu().numpy())
    real_features = np.concatenate(real_features, axis=0)

    #----------------------------------------------------------------------------
    # Generate set of images and compute their features.
    print('Computing generated features...')
    latents = np.random.RandomState(0).randn(num_images - 1, G_ema.z_dim)  # Fixed seed.
    gen_features = np.zeros([num_images - 1, inception_v3.feature_dim], dtype=np.float32)

    for begin in range(0, gen_features.shape[0], batch_size):
        end = min(begin + batch_size, gen_features.shape[0])
        gen_images = generate_images(G_ema=G_ema,
                                     z=torch.from_numpy(latents[begin:end]).to(device),
                                     truncation_psi=truncation_psi,
                                     noise_mode=noise_mode)

        with torch.no_grad():
            gen_features[begin:end] = inception_v3(gen_images, return_features=True).cpu().numpy()

    #----------------------------------------------------------------------------
    # Compute feature statistics.
    print('Computing feature statistics...')
    mean_reals = torch.from_numpy(np.mean(real_features, axis=0)).to(device)
    cov_reals = torch.from_numpy(np.cov(real_features, rowvar=False)).to(device)
    mean_gen = torch.from_numpy(np.mean(gen_features, axis=0)).to(device)
    cov_gen = torch.from_numpy(np.cov(gen_features, rowvar=False)).to(device)

    #----------------------------------------------------------------------------
    # Register forward and backward hooks to get activations and gradients, respectively.
    acts_and_gradients = ActivationsAndGradients(network=inception_v3,
                                                 network_kwargs=dict(return_features=True))

    #----------------------------------------------------------------------------
    # Visualize FID sensitivity heatmaps.
    if image_seeds is None:
        num_vis_images = num_rows * num_cols
        image_seeds = random_seed + np.arange(num_vis_images, dtype=int)
        grid_images: List[np.ndarray] = []
        top1_labels: List[str] = []

    print('Visualizing heatmaps...')
    for seed in tqdm(image_seeds):
        rnd = np.random.RandomState(seed)
        #----------------------------------------------------------------------------
        # Generate image that is visualized.
        vis_latent = torch.from_numpy(rnd.randn(1, G_ema.z_dim)).to(device)
        gen_image = generate_images(G_ema=G_ema,
                                    z=vis_latent,
                                    truncation_psi=truncation_psi,
                                    noise_mode=noise_mode)

        #----------------------------------------------------------------------------
        # Compute and visualize a sensitivity map.
        overlay_heatmap = compute_sensitivity_heatmap(gen_image=gen_image,
                                                      acts_and_gradients=acts_and_gradients,
                                                      mean_reals=mean_reals,
                                                      cov_reals=cov_reals,
                                                      mean_gen=mean_gen,
                                                      cov_gen=cov_gen,
                                                      num_images=num_images)

        #----------------------------------------------------------------------------
        # Compute classification probabilities.
        gen_probs = F.softmax(inception_v3(gen_image, return_features=False), dim=1).detach().cpu().numpy()[0]
        top1_class = np.argmax(gen_probs)
        label_str = imagenet_labels[top1_class].split(',')[0]

        if save_classifications:
            top_n_idxs = np.argsort(gen_probs)[::-1][:top_n]
            bar_chart(x=np.arange(top_n, dtype=int),
                      y=gen_probs[top_n_idxs],
                      ylabel='Probability',
                      xlabels=[imagenet_labels[c_idx].split(',')[0].capitalize() for c_idx in top_n_idxs],
                      path=os.path.join(results_dir, f'gen_probs_{seed}.png'))
            np.save(os.path.join(results_dir, f'gen_probs_{seed}.npy'), gen_probs)

        #----------------------------------------------------------------------------
        # Save heatmap.
        if not gen_grid:  # Save gen. image and heatmap separately.
            gen_image = gen_image.cpu().numpy()
            gen_im = PIL.Image.fromarray(gen_image[0].transpose(1, 2, 0))
            gen_im.save(os.path.join(results_dir, f'gen_image_seed{seed}.png'))
            overlay_heatmap.save(os.path.join(results_dir, f'sensitivity_heatmap_seed{seed}.png'))
        else:
            grid_images.append(np.array(overlay_heatmap).transpose(2, 0, 1))
            top1_labels.append(label_str)

    #----------------------------------------------------------------------------
    # Create a grid of overlay heatmaps.
    if gen_grid:
        bg = create_grid(images=grid_images,
                         labels=top1_labels,
                         num_rows=num_rows,
                         num_cols=num_cols)
        bg.save(os.path.join(results_dir, f'sensitivity_grid-seed{random_seed}-grid.png'))
