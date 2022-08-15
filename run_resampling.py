"""Script to run resampling experiments."""

import click
import os

from resampling.resample import resample_fid
from utils import create_results_dir


@click.command()
@click.option('--zip_path', required=True, type=str, help='Path to zip dataset.')
@click.option('--network_pkl', required=True, type=str, help='Path to network pickle.')
@click.option('--resolution', default=256, type=int, help='Resolution of zip dataset images.')
@click.option('--num_real_images', default=50000, type=int, help='Number of real images in resampling.')
@click.option('--num_gen_images', default=250000, type=int, help='Number of gen. iamges in resampling.')
@click.option('--lr', default=10.0, type=float, help='Gradient descent (GD) learning rate.')
@click.option('--num_iterations', default=100000, type=int, help='Number of GD steps to run.')
@click.option('--feature_mode', default='pre_logits', type=str, help='Type of features that are resampled.')
@click.option('--num_dims', default=None, type=int, help='Number of dimensions in Top-N/Middle-N.')
def main(zip_path: str,
         network_pkl: str,
         resolution: int,
         num_real_images: int,
         num_gen_images: int,
         lr: float,
         num_iterations: int,
         feature_mode: str,
         num_dims: int) -> None:
    """Runs FID resampling.

    Examples:

    \b
    Run 'pre-logits' resampling:
    python run_resampling.py --zip_path=data_path \\
        --network_pkl=https://drive.google.com/uc?id=119HvnQ5nwHl0_vUTEFWQNk4bwYjoXTrC \\
        --feature_mode=pre_logits

    \b
    Run 'logits' resampling:
    python run_resampling.py --zip_path=data_path \\
        --network_pkl=https://drive.google.com/uc?id=119HvnQ5nwHl0_vUTEFWQNk4bwYjoXTrC \\
        --feature_mode=logits
    \b
    Run 'Top-N' resampling:
    python run_resampling.py --zip_path=data_path \\
        --network_pkl=https://drive.google.com/uc?id=119HvnQ5nwHl0_vUTEFWQNk4bwYjoXTrC \\
        --feature_mode=top_n --num_dims=1

    """
    # Create a new results directory.
    desc  = f"{os.path.basename(zip_path).split('.')[0]}"
    desc += f"-{feature_mode}"
    desc += f"-num_dims_{num_dims}" if num_dims is not None else ""
    desc += f"-num_reals_{num_real_images // 1000}k"
    desc += f"-num_gen_{num_gen_images // 1000}k"
    desc += f"-lr_{lr}"
    desc += f"-num_it_{num_iterations}"

    results_dir = create_results_dir(results_root='./results/resampling/',
                                     description=desc)

    # Run resampling.
    resample_fid(out_path=results_dir,
                 zip_path=zip_path,
                 pkl_path=network_pkl,
                 resolution=resolution,
                 num_real_images=num_real_images,
                 num_gen_images=num_gen_images,
                 learning_rate=lr,
                 num_iterations=num_iterations,
                 feature_mode=feature_mode,
                 num_dims=num_dims)


if __name__ == "__main__":
    main()
