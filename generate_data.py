import numpy as np
from scipy import sparse

from percolation_dataset import PercolationDataset, GroundTruthFeatures, Node, Seeds

n_datasets = 1
dataset_size = 2000000
embedding_dimension = 100

for seed in range(n_datasets):
    print(f"Generating dataset {seed}...")
    dataset = PercolationDataset(mode="distribution",seeds=Seeds(graph=seed, embed=seed + 10000, value=seed + 20000))

    points, latents, X, y = dataset.construct_embed(size=dataset_size, d=embedding_dimension)
    ground_truth_features = GroundTruthFeatures(points, latents)
    gt_latent_features = ground_truth_features.get_latent_features(use_values=True)
    gt_summary_features = ground_truth_features.get_summary_features()
    np.savez_compressed(
        f"percolation_dataset_size{dataset_size}_dim{embedding_dimension}_seed{seed}.npz",
        X=X,
        y=y,
        **gt_summary_features,
    )
    sparse.save_npz(
        f"percolation_dataset_size{dataset_size}_dim{embedding_dimension}_seed{seed}_gt_features.npz",
        gt_latent_features,
    )
