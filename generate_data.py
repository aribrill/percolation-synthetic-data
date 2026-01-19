import numpy as np
from scipy import sparse

from percolation_dataset import PercolationDataset, GroundTruthFeatures

n_datasets = 10
dataset_size = 100000
embedding_dimension = 128

for seed in range(n_datasets):
    print(f"Generating dataset {seed}...")
    dataset = PercolationDataset(graph_seed=seed, embed_seed=seed + 10000, value_seed=seed + 20000)
    points, latents, X, y = dataset.construct_embed(size=dataset_size, d=embedding_dimension)
    ground_truth_features = GroundTruthFeatures(points, latents)
    gt_features = ground_truth_features.get_features()
    np.savez_compressed(
        f"percolation_dataset_size{dataset_size}_dim{embedding_dimension}_seed{seed}.npz",
        X=X,
        y=y,
    )
    sparse.save_npz(
        f"percolation_dataset_size{dataset_size}_dim{embedding_dimension}_seed{seed}_gt_features.npz",
        gt_features,
    )