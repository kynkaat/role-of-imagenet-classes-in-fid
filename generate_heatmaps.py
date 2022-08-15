"""Script to generate FID sensitivity heatmaps."""

import ast
import click
import os

from fid_heatmaps.sensitivity_heatmaps import visualize_heatmaps
from utils import create_results_dir


@click.command()
@click.option('--zip_path', required=True, type=str, help='Path to zip dataset.')
@click.option('--network_pkl', required=True, type=str, help='Path to network pickle.')
@click.option('--resolution', default=256, type=int, help='Resolution of zip dataset images.')
@click.option('--seeds', default=None, type=str, help='Generate images with specific seeds.')
@click.option('--save_classifications', default=False, type=bool, help='Flag to save classifications.')
@click.option('--gen_grid', default=False, type=bool, help='Create grid of heatmaps.')
@click.option('--num_rows', default=5, type=int, help='Number of rows in the grid.')
@click.option('--num_cols', default=8, type=int, help='Number of columns in the grid.')
@click.option('--random_seed', default=100, type=int, help='Generate images starting from this seed.')
def main(zip_path: str,
         network_pkl: str,
         resolution: int,
         seeds: list,
         save_classifications: bool,
         gen_grid: bool,
         num_rows: int,
         num_cols: int,
         random_seed: int) -> None:
    """Generates sensitivity heatmaps.

    Examples:

    \b
    Generate FFHQ heatmaps from Fig. 3 top.
    python generate_heatmaps.py --zip_path=data_path \\
        --network_pkl=https://drive.google.com/uc?id=119HvnQ5nwHl0_vUTEFWQNk4bwYjoXTrC \\
        --seeds=[107,540,386,780,544,879]

    \b
    Generate FFHQ heatmap grid from Fig. 11a.
    python generate_heatmaps.py --zip_path=data_path \\
        --network_pkl=https://drive.google.com/uc?id=119HvnQ5nwHl0_vUTEFWQNk4bwYjoXTrC \\
        --gen_grid=1 --num_rows=5 --num_cols=8 --random_seed=100

    """
    # Create a new results directory.
    desc  = f"{os.path.basename(zip_path).split('.')[0]}"
    desc += f"-fid_heatmaps"
    desc += f"-seeds_{seeds.replace(',', '_')[1:-1]}" if seeds is not None else f"-seed_{random_seed}"
    results_dir = create_results_dir(results_root='./results/fid-heatmaps',
                                     description=desc)

    # Visualize FID sensitivity heatmaps.
    seeds = ast.literal_eval(seeds) if seeds is not None else seeds
    visualize_heatmaps(results_dir=results_dir,
                       zip_path=zip_path,
                       pkl_path=network_pkl,
                       resolution=resolution,
                       image_seeds=seeds,
                       save_classifications=save_classifications,
                       gen_grid=gen_grid,
                       num_rows=num_rows,
                       num_cols=num_cols,
                       random_seed=random_seed)


if __name__ == "__main__":
    main()
