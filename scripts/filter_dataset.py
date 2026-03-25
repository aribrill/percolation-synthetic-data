import sys
import numpy as np
from pathlib import Path
from scipy import sparse
from typing import List, Optional

sys.path.insert(0, str(Path("../").resolve()))
from percolation_dataset import PercolationDataset, GroundTruthFeatures, Node

# Set input paths
data_folder_path = Path("../")
data_path = data_folder_path / "percolation_dataset_size2000000_dim100_seed0.npz"
feature_data_path = data_folder_path / "percolation_dataset_size2000000_dim100_seed0_gt_features.npz"
meta_data_path = data_folder_path / "percolation_dataset_size2000000_dim100_seed0_gt_metadata.npz"
base_name = data_path.stem

# Dataset generation parameters (must match the original generation)
dataset_size = 2000000
embedding_dimension = 100
seed = 0

# Set filter parameters
min_size = 250
max_size = None

# Load data
with np.load(data_path) as res:
    X, y = res['X'], res['y']

features = sparse.load_npz(feature_data_path)
meta_res = np.load(meta_data_path, allow_pickle=True)

# Filter 
if min_size is not None and max_size is not None:
    mask = (np.array(meta_res['cluster_size']) > min_size) & (np.array(meta_res['cluster_size']) < max_size)
    suffix = f"_filtered_cluster_size{min_size}to{max_size}"
elif min_size is not None:
    mask = np.array(meta_res['cluster_size']) > min_size
    suffix = f"_filtered_cluster_size_min{min_size}"
elif max_size is not None:
    mask = np.array(meta_res['cluster_size']) < max_size
    suffix = f"_filtered_cluster_size_max{max_size}"
else:
    mask = np.ones(len(X), dtype=bool)
    suffix = ""

X_filtered = X[mask]
y_filtered = y[mask]
features_filtered = features[mask, :]
metadata_filtered = {key: np.array(meta_res[key])[mask] for key in meta_res}
# Remove all-zero columns after row filtering
col_mask = np.array(features_filtered.sum(axis=0)).flatten() > 0
features_filtered = features_filtered[:, col_mask]


# Store filtered dataset
np.savez_compressed(
    data_folder_path / f"{base_name}{suffix}.npz",
    X=X_filtered, y=y_filtered
)
sparse.save_npz(
    str(data_folder_path / f"{base_name}_gt_features{suffix}.npz"),
    features_filtered
)
np.savez_compressed(
    data_folder_path / f"{base_name}_gt_metadata{suffix}.npz",
    **metadata_filtered
)




print(f"Filtered dataset saved to {data_folder_path}")
print(f"Original size: {len(X)} points")
print(f"Filtered size: {len(X_filtered)} points ({mask.sum() / len(X) * 100:.1f}% retained)")
print(f"Number of clusters: {len(np.unique(metadata_filtered['cluster_id']))}")


