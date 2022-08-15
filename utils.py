"""Miscellaneous utility functions."""

import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import PIL.Image
from PIL import ImageDraw
from PIL import ImageFont
import re
import requests
import time
import torch
import tqdm
from typing import Any, \
                   Dict, \
                   List, \
                   Optional, \
                   Tuple
import yaml

# Configure plot styles.
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'

_URL_TO_PKL_NAME = {'https://drive.google.com/uc?id=1j3pS3bdTXIYL56kpcpdMrrvPJVT90IY0': 'clip-vit_b32.pkl',
                    'https://drive.google.com/uc?id=119HvnQ5nwHl0_vUTEFWQNk4bwYjoXTrC': 'ffhq-fid50k_5.30-snapshot-022608.pkl',
                    'https://drive.google.com/uc?id=1yDD9iqw3YYbkn2d7N8ciYu81widI-uEL': 'inception_v3-tf-2015-12-05.pkl'}


def bar_chart(x: np.ndarray,
              y: np.ndarray,
              ylabel: str,
              xlabels: str,
              path: str,
              figsize: Tuple[float, float] = (10.0, 7.0),
              bar_width: float = 0.8,
              ylim: List[float] = [0.0, 1.0],
              label_fontsize: float = 28.0,
              tick_fontsize: float = 16.0,
              tick_rotation: float = 90.0,
              dpi: int = 300) -> None:
    """Creates a bar chart."""
    _, ax = plt.subplots(figsize=figsize)
    plt.bar(x=x, height=y, width=bar_width)
    plt.xticks(ticks=x, labels=xlabels)
    plt.yticks(fontsize=tick_fontsize)
    ax.tick_params(axis='x', labelrotation=tick_rotation, labelsize=tick_fontsize)
    plt.ylim(ylim)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.tight_layout()

    _, ext = os.path.basename(path).split('.')
    if ext == 'png':
        plt.savefig(path, dpi=dpi)
    else:
        plt.savefig(path)
    plt.close('all')


def create_grid(images: List[np.ndarray],
                num_rows: int,
                num_cols: int,
                labels: Optional[List[str]] = None,
                label_loc: Tuple[int, int] = (0, 0),
                fontsize: int = 32,
                font_path: str = './data/times-new-roman.ttf') -> PIL.Image:
    """Creates an image grid."""
    _, h, w = images[0].shape
    if labels is not None:
        assert len(images) == len(labels)
        font = ImageFont.truetype(font_path, fontsize)

    bg = PIL.Image.new('RGB', size=(num_cols * w, num_rows * h))

    for i in range(num_rows):
        for j in range(num_cols):
            im = PIL.Image.fromarray(images.pop(0).transpose(1, 2, 0))

            if labels is not None:
                draw = ImageDraw.Draw(im)
                draw.text(label_loc, f'{labels.pop(0)}'.capitalize(), font=font)

            bg.paste(im, box=(j * w, i * h))
    return bg


def create_results_dir(results_root: str,
                       description: str) -> str:
    """Creates a new results directory."""
    os.makedirs(results_root, exist_ok=True)
    run_id = get_run_id(results_root=results_root)
    results_dir = os.path.join(results_root, f"{run_id}-{description}")
    os.makedirs(results_dir)
    return results_dir


def download_pickle(url: str,
                    pkl_name: str,
                    pickle_dir: str = './.pickles/',
                    num_attempts: int = 10,
                    chunk_size: int = 512 * 1024,  # 512 KB.
                    retry_delay: int = 2) -> str:
    """Downloads network pickle file from an URL."""
    os.makedirs(pickle_dir, exist_ok=True)

    def _is_successful(response: requests.models.Response) -> bool:
        return response.status_code == 200

    # Download file from Google Drive URL.
    network_path = os.path.join(pickle_dir, pkl_name)
    if not os.path.exists(network_path):
        print(f'Downloading network pickle ({pkl_name})...')
        for attempts_left in reversed(range(num_attempts)):
            try:
                with requests.Session() as session:
                    with session.get(f'{url}&confirm=t', stream=True) as response:
                        assert _is_successful(response), \
                            f'Downloading network pickle ({pkl_name}) from URL {url} failed.'

                        # Save network pickle.
                        with open(network_path, 'wb') as f:
                            total = response.headers['Content-Length']
                            total = int(total)
                            pbar = tqdm.tqdm(total=total, unit="B", unit_scale=True)
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                f.write(chunk)
                                pbar.update(len(chunk))
                        break
            except KeyboardInterrupt:
                raise
            except:
                print(f'Failed. Retrying in {retry_delay}s (attempts left {attempts_left})...')
                time.sleep(retry_delay)
    else:
        print(f'Downloading {pkl_name} skipped; already exists in {pickle_dir}')

    return network_path


def gen_image_grid(G_ema: Any,
                   latents: torch.Tensor,
                   num_rows: int,
                   num_cols: int,
                   batch_size: int = 32,
                   noise_mode: str = 'const',
                   truncation_psi: float = 1.0) -> PIL.Image:
    """Creates a grid of generated images."""
    assert len(latents.shape) == 2
    num_images = latents.shape[0]

    gen_images: List[torch.Tensor] = []
    for begin in range(0, num_images, batch_size):
        end = min(begin + batch_size, num_images)
        gen_images.append(generate_images(G_ema=G_ema,
                                          z=latents[begin:end],
                                          noise_mode=noise_mode,
                                          truncation_psi=truncation_psi))
    gen_images = torch.cat(gen_images, dim=0)
    gen_images = [image for image in gen_images.cpu().numpy()]
    bg = create_grid(images=gen_images,
                     num_rows=num_rows,
                     num_cols=num_cols)

    return bg


def generate_images(G_ema: Any,
                    z: torch.Tensor,
                    noise_mode: str = 'const',
                    truncation_psi: float = 1.0) -> torch.Tensor:
    """Generates images with a given generator and z-latents."""
    assert len(z.shape) == 2
    label = torch.zeros([z.shape[0], G_ema.c_dim]).to(z.device)
    gen_images = G_ema(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    gen_images = (gen_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return gen_images


def get_run_id(results_root: str) -> str:
    """Locates the latest run id and increases it by one."""
    run_dirs = glob.glob(results_root + '/*')
    run_dirs = [run_dir.replace('\\','/') for run_dir in run_dirs]

    run_ids = []
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        # Extract run id from path.
        # Paths are always in fixed format: /path/to/run_dir/run_id-run_description.
        if re.search("[0-9]{5}-", run_dir) is not None:
            run_id = int(run_dir.split('/')[-1].split('-')[0])
            run_ids.append(run_id)

    # Determine new run id.
    if not run_ids:
        return ''.zfill(5)

    new_run_id = str(max(run_ids) + 1)
    new_run_id = new_run_id.zfill(5)  # Pad with zeros s.t. the length of id is 5.

    return new_run_id


def is_drive_url(s: str) -> bool:
    """Checks if string is a Google Drive URL."""
    return 'https://drive.google.com' in s


def load_feature_network(network_name: str) -> Any:
    """Loads a pre-trained feature network."""
    _network_urls = {#'clip': 'https://drive.google.com/uc?id=1VF0xYAfGEPH0bhNYLFS_yTEoVT2rkFFG',
                     'clip': 'https://drive.google.com/uc?id=1j3pS3bdTXIYL56kpcpdMrrvPJVT90IY0',
                     'inception_v3_tf': 'https://drive.google.com/uc?id=1yDD9iqw3YYbkn2d7N8ciYu81widI-uEL'}
    assert network_name in _network_urls.keys(), \
        f"Unknown feature network name {network_name}."
    url = _network_urls[network_name]
    network_path = download_pickle(url=url,
                                   pkl_name=_URL_TO_PKL_NAME[url])

    with open(network_path, 'rb') as f:
        network = pickle.load(f)
    return network


def load_imagenet_labels(path: str) -> Dict[int, str]:
    """Loads ImageNet labels (class_idx -> name) from a given path."""
    with open(path, 'r') as f:
        label_dict = yaml.load(f, Loader=yaml.FullLoader)
    return label_dict


def load_stylegan2(pkl_path: str) -> Any:
    """Loads a StyleGAN2 network pickle."""
    # Download if necessary.
    if is_drive_url(pkl_path):
        pkl_path = download_pickle(url=pkl_path,
                                   pkl_name=_URL_TO_PKL_NAME[pkl_path])

    with open(pkl_path, 'rb') as f:
        G_ema = pickle.load(f)['G_ema']
    return G_ema


def remove_existing_file(pattern: str,
                         path: str) -> None:
    """Removes file that matches a pattern."""
    files = glob.glob(path + f'/{pattern}')
    if len(files) == 1:
        os.remove(files[0])


def remove_visualizations(patterns: List[str],
                        path) -> None:
    """Removes existing visualizations."""
    for pattern in patterns:
        remove_existing_file(pattern=pattern,
                             path=path)


def weight_grids(G_ema: Any,
                latents: np.ndarray,
                w_log: np.ndarray,
                weight_percentile: float,
                num_rows: int,
                num_cols: int,
                device: torch.device,
                batch_size: int = 32,
                noise_mode: str = 'const',
                truncation_psi: float = 1.0) -> PIL.Image:
    """Creates a grid of small and large weight generated images."""
    small_weight_idxs = np.argsort(w_log)[:int(weight_percentile * w_log.shape[0])]
    large_weight_idxs = np.argsort(w_log)[::-1][:int(weight_percentile * w_log.shape[0])]
    selected_small_idxs = np.random.choice(small_weight_idxs,
                                           size=num_rows * num_cols,
                                           replace=False)
    selected_large_idxs = np.random.choice(large_weight_idxs,
                                           size=num_rows * num_cols,
                                           replace=False)

    bg_small_weight = gen_image_grid(G_ema=G_ema,
                                     latents=torch.from_numpy(latents[selected_small_idxs]).to(device),
                                     num_rows=num_rows,
                                     num_cols=num_cols,
                                     batch_size=batch_size,
                                     noise_mode=noise_mode,
                                     truncation_psi=truncation_psi)
    bg_large_weight = gen_image_grid(G_ema=G_ema,
                                     latents=torch.from_numpy(latents[selected_large_idxs]).to(device),
                                     num_rows=num_rows,
                                     num_cols=num_cols,
                                     batch_size=batch_size,
                                     noise_mode=noise_mode,
                                     truncation_psi=truncation_psi)

    return bg_small_weight, bg_large_weight


def write_jsonl(path: str,
                data: dict) -> None:
    """Writes to JSONL file."""
    with open(path, 'a') as f:
        f.write(json.dumps(data) + '\n')


def write_tensorboard_scalars(writer: Any,
                              data: dict) -> None:
    """Writes scalar data to Tensorboard."""
    _to_tb_name = {'fid': 'Metrics/FID',
                   'fid_clip': 'Metrics/FID_CLIP',
                   'kid_x1000': 'Metrics/KID_x1000',
                   'loss': 'Misc/Loss',
                   'w_min': 'Misc/w_min',
                   'w_median': 'Misc/w_median',
                   'w_max': 'Misc/w_max',
                   'max_mem': 'Misc/GPU_mem'}
    it = data.pop('it')
    for key, value in data.items():
        writer.add_scalar(_to_tb_name[key], value, it)
