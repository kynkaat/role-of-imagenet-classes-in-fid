"""Probing the perceptual null space of FID by resampling features."""

import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from dataset import get_dataloader
from metrics.frechet_inception_distance import compute_fid
from resampling.resample_utils import binarize_features, \
                                      compute_weighted_mean_and_cov, \
                                      evaluate_metric, \
                                      plot_top1_distribution, \
                                      sample_indices_with_weights, \
                                      wasserstein2_loss
from utils import generate_images, \
                  gen_image_grid, \
                  load_feature_network, \
                  load_imagenet_labels, \
                  load_stylegan2, \
                  remove_visualizations, \
                  weight_grids, \
                  write_jsonl, \
                  write_tensorboard_scalars


def resample_fid(out_path: str,
                 zip_path: str,
                 pkl_path: str,
                 num_real_images: int = 50000,
                 num_gen_images: int = 250000,
                 batch_size: int = 32,
                 resolution: int = 256,
                 truncation_psi: float = 1.0,
                 noise_mode: str = 'const',
                 learning_rate: float = 10.0,
                 num_iterations: int = 200000,
                 log_freq: int = 500,
                 num_repeats: int = 3,
                 feature_mode: str = 'pre_logits',
                 num_dims: Optional[int] = None,
                 num_rows: int = 9,
                 num_cols: int = 16,
                 weight_percentile: float = 0.1,
                 use_cuda: bool = True,
                 random_seed: int = 0) -> None:
    """Directly optimize FID by resampling features."""
    assert feature_mode in ['pre_logits', 'logits', 'top_n', 'middle_n']
    if feature_mode in ['top_n', 'middle_n']:
        assert num_dims is not None, 'Number of dimensions (--num_dims) in Top-N/Middle-N can not be None.'
    rnd = np.random.RandomState(random_seed)
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    labels = load_imagenet_labels(path='./data/imagenet-2012-labels.yml')

    #----------------------------------------------------------------------------
    # Load StyleGAN2 and pre-trained networks.
    G_ema = load_stylegan2(pkl_path=pkl_path).to(device)
    inception_v3 = load_feature_network(network_name='inception_v3_tf').to(device)
    clip = load_feature_network(network_name='clip').to(device)

    if feature_mode in ['logits', 'top_n', 'middle_n']:
        feature_kwargs = dict(return_features=False)
        resample_dim = inception_v3.logits_dim
    else:  # Pre-logits resampling.
        feature_kwargs = dict(return_features=True)
        resample_dim = inception_v3.feature_dim
    metrics_kwargs = dict(return_features=True)

    #----------------------------------------------------------------------------
    # Compute class distribution for reals.
    print('Reading images and computing features...')

    # Create dataloader instance.
    dataloader = get_dataloader(zip_path=zip_path,
                                resolution=resolution,
                                batch_size=batch_size,
                                num_images=num_real_images)

    real_features = []
    real_probs = []
    real_metrics_features = []
    real_clip_features = []

    for images, _ in dataloader:
        # Compute features and probabilities.
        images = images.to(device)
        with torch.no_grad():
            real_features.append(inception_v3(images.to(device), **feature_kwargs).cpu().numpy())
            real_probs.append(F.softmax(inception_v3(images.to(device), return_features=False), dim=1).cpu().numpy())
            real_metrics_features.append(inception_v3(images.to(device), **metrics_kwargs).cpu().numpy())
            real_clip_features.append(clip(images.to(device)).cpu().numpy())

    real_features = np.concatenate(real_features, axis=0)
    real_probs = np.concatenate(real_probs, axis=0)
    real_metrics_features = np.concatenate(real_metrics_features, axis=0)
    real_clip_features = np.concatenate(real_clip_features, axis=0)
    print(f"{'Real features shape:':<28} {real_features.shape}")
    print(f"{'Real metrics features shape:':<28} {real_metrics_features.shape}")
    print(f"{'Real CLIP features shape:':<28} {real_clip_features.shape}\n")

    #----------------------------------------------------------------------------
    # Generate images and compute feature and class distribution.
    print('Generating images and computing features...')
    latents = rnd.randn(num_gen_images, G_ema.z_dim).astype(np.float32)
    gen_features = np.zeros([num_gen_images, resample_dim], dtype=np.float32)
    gen_metrics_features = np.zeros([num_gen_images, inception_v3.feature_dim], dtype=np.float32)
    gen_probs = np.zeros([num_gen_images, inception_v3.logits_dim], dtype=np.float32)
    gen_clip_features =  np.zeros([num_gen_images, clip.feature_dim], dtype=np.float32)

    for begin in range(0, num_gen_images, batch_size):
        end = min(begin + batch_size, num_gen_images)
        gen_images = generate_images(G_ema=G_ema,
                                     z=torch.from_numpy(latents[begin:end]).to(device),
                                     truncation_psi=truncation_psi,
                                     noise_mode=noise_mode)

        with torch.no_grad():
            gen_features[begin:end] = inception_v3(gen_images, **feature_kwargs).cpu().numpy()
            gen_probs[begin:end] = F.softmax(inception_v3(gen_images, return_features=False), dim=1).cpu().numpy()
            gen_metrics_features[begin:end] = inception_v3(gen_images, **metrics_kwargs).cpu().numpy()
            gen_clip_features[begin:end] = clip(gen_images).cpu().numpy()

    print(f"{'Gen. features shape':<28} {gen_features.shape}")
    print(f"{'Gen. metrics features shape:':<28} {gen_metrics_features.shape}")
    print(f"{'Gen. CLIP features shape:':<28} {gen_clip_features.shape}\n")

    # Compute random sampled FID and CLIP-FID.
    rnd_idxs = np.random.choice(num_gen_images, size=num_real_images, replace=False)
    fid_rnd = compute_fid(real_features=real_metrics_features,
                          gen_features=gen_metrics_features[rnd_idxs])
    clip_fid_rnd = compute_fid(real_features=real_clip_features,
                               gen_features=gen_clip_features[rnd_idxs])

    # Visualize random top-1 class distribution and image grid.
    save_fname = os.path.join(out_path, f'rnd_class_dist-fid_{fid_rnd:0.2f}-fid_clip_{clip_fid_rnd:0.2f}.png')
    plot_top1_distribution(fname=save_fname,
                           real_probs=real_probs,
                           gen_probs=gen_probs[rnd_idxs],
                           imagenet_labels=labels)

    num_grid_images = num_rows * num_cols
    bg_rnd = gen_image_grid(G_ema=G_ema,
                            latents=torch.from_numpy(latents[rnd_idxs][:num_grid_images]).to(device),
                            num_rows=num_rows,
                            num_cols=num_cols)
    bg_rnd.save(os.path.join(out_path, f'rnd_grid-fid_{fid_rnd:0.2f}-fid_clip_{clip_fid_rnd:0.2f}.png'))

    #----------------------------------------------------------------------------
    # Build features based on logits.
    if feature_mode in ['top_n', 'middle_n'] and num_dims is not None:  # Apply binarization.
        print('Binarizing real and gen. feature vectors...')
        real_features, gen_features = binarize_features(feature_mode=feature_mode,
                                                        real_logits=real_features,
                                                        gen_logits=gen_features,
                                                        num_dims=num_dims)

    #----------------------------------------------------------------------------
    # Compute real statistics.
    print('Computing real statistics...')
    mean_reals = torch.from_numpy(np.mean(real_features.astype(np.float64), axis=0)).to(device)
    cov_reals = torch.from_numpy(np.cov(real_features.astype(np.float64), rowvar=False)).to(device)

    #----------------------------------------------------------------------------
    # Initialize log-parameterized per-image weights.
    gen_features_torch = torch.from_numpy(gen_features).type(torch.float64).to(device)  # Shape: (N x D).
    w_log = torch.zeros([num_gen_images, 1], dtype=torch.float64, device=device, requires_grad=True)

    # Create an optimizer instance.
    optimizer = optim.SGD([w_log], lr=learning_rate)

    #----------------------------------------------------------------------------
    # Run optimization loop.
    start = time.time()
    best_fid = np.inf
    clip_fid = np.inf
    writer = SummaryWriter(log_dir=out_path)

    print('Optimizing FID directly...')
    print(f'Resampled feature shapes: real = {real_features.shape}, gen. = {gen_features.shape}\n')
    for it in range(num_iterations):
        optimizer.zero_grad()

        # Compute weighted mean and covariance with the current weights.
        weighted_mean_gen, weighted_cov_gen = \
            compute_weighted_mean_and_cov(gen_features=gen_features_torch, w_log=w_log)

        # Compute 2-Wasserstein/Fréchet Distance between real statistics
        # and statistics weighted of generated features.
        loss = wasserstein2_loss(mean_reals=mean_reals,
                                 mean_gen=weighted_mean_gen,
                                 cov_reals=cov_reals,
                                 cov_gen=weighted_cov_gen)

        # Backward pass and step optimizer.
        loss.backward()
        optimizer.step()

        if it == 0 or (it + 1) % log_freq == 0:
            cur_loss = loss.detach().cpu().numpy()
            w_cur = w_log.detach().cpu().numpy().ravel()

            # Compute metrics by resampling according to weights.
            fids = evaluate_metric(real_features=real_metrics_features,
                                   gen_features=gen_metrics_features,
                                   w_log=w_cur,
                                   num_repeats=num_repeats,
                                   metric_name='fid')
            clip_fids = evaluate_metric(real_features=real_clip_features,
                                        gen_features=gen_clip_features,
                                        w_log=w_cur,
                                        num_repeats=num_repeats,
                                        metric_name='fid')
            kids = evaluate_metric(real_features=real_metrics_features,
                                   gen_features=gen_metrics_features,
                                   w_log=w_cur,
                                   num_repeats=num_repeats,
                                   metric_name='kid')

            if fids.min() < best_fid:
                best_fid = fids.min()
                clip_fid = clip_fids.min()

                # Remove old visualizations.
                remove_visualizations(patterns=['sampling_weights-*.npy',
                                                'resampled_class_dist-*.png',
                                                'resampled_grid-*.png',
                                                'small_weight_grid.png',
                                                'large_weight_grid.png'],
                                      path=out_path)

                # Save weight vector.
                fname = f'sampling_weights-fid_{best_fid:0.2f}-num_gen_{num_gen_images}-seed_{random_seed}-it_{it + 1}.npy'
                np.save(os.path.join(out_path, fname), w_cur)

                # Visualize top-1 class distribution.
                resample_idxs = sample_indices_with_weights(w_log=w_cur,
                                                            num_items=num_real_images)
                save_fname = os.path.join(out_path, f'resampled_class_dist-fid_{best_fid:0.2f}-fid_clip_{clip_fid:0.2f}.png')
                plot_top1_distribution(fname=save_fname,
                                       real_probs=real_probs,
                                       gen_probs=gen_probs[resample_idxs],
                                       imagenet_labels=labels)

                # Save sample grid.
                bg_resampled = gen_image_grid(G_ema=G_ema,
                                              latents=torch.from_numpy(latents[resample_idxs][:num_grid_images]).to(device),
                                              num_rows=num_rows,
                                              num_cols=num_cols,
                                              batch_size=batch_size,
                                              noise_mode=noise_mode,
                                              truncation_psi=truncation_psi)
                bg_resampled.save(os.path.join(out_path, f'resampled_grid-fid_{best_fid:0.2f}-fid_clip_{clip_fid:0.2f}.png'))

                # Save small and large weight images.
                bg_small_weight, bg_large_weight = weight_grids(G_ema=G_ema,
                                                                latents=latents,
                                                                w_log=w_cur,
                                                                weight_percentile=weight_percentile,
                                                                num_rows=num_rows,
                                                                num_cols=num_cols,
                                                                batch_size=batch_size,
                                                                device=device,
                                                                noise_mode=noise_mode,
                                                                truncation_psi=truncation_psi)
                bg_small_weight.save(os.path.join(out_path, f'small_weight_grid.png'))
                bg_large_weight.save(os.path.join(out_path, f'large_weight_grid.png'))

            # Gather scalars for logging and save them to JSONL and Tensorboard.
            stats_path = os.path.join(out_path, 'optimization_stats.jsonl')
            stats = dict(it=it + 1,
                         loss=float(cur_loss),
                         fid=float(fids.min()),
                         fid_clip=float(clip_fids.min()),
                         kid_x1000=float(kids.min() * 1000.0),
                         w_min=float(w_cur.min()),
                         w_median=float(np.median(w_cur)),
                         w_max=float(w_cur.max()),
                         max_mem=float(torch.cuda.max_memory_allocated(device=device) * 1e-9))
            write_jsonl(path=stats_path,
                        data={**stats,
                              **dict(snapshot=os.path.basename(pkl_path).split('.')[0],
                                     total_gen_images=num_gen_images,
                                     random_seed=random_seed)})
            write_tensorboard_scalars(writer=writer,
                                      data=stats)

            # Report progress.
            elapsed_time = time.time() - start
            print(f'It. {it + 1}/{num_iterations}, loss = {cur_loss:0.5f}')
            print(f'FID = {fids.min():0.2f}')
            print(f'CLIP-FID = {clip_fids.min():0.2f}')
            print(f'Elapsed time: {elapsed_time:0.2f}s\n')
            start = time.time()
