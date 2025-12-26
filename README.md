# percolation-synthetic-data
Generates statistically self-similar synthetic datasets based on a percolation cluster model.

## About

[in progress]

## Derivation

[in progress]

## Usage

A synthetic dataset is generated using `PercolationDataset.construct_embed()`. Ground-truth latent features for each sample can be generated as a sparse matrix using `GroundTruthFeatures.get_features()`. Because the dataset generation is stochastic, it is recommended to train and test models using multiple datasets generated with different random seeds. An example script to generate multiple datasets is provided in [generate_data.py](generate_data.py).

Datasets generated in the format of [generate_data.py](generate_data.py) can be loaded with:

```python
import numpy as np
from scipy import sparse

res = np.load("percolation_dataset_size<SIZE>_dim<DIM>_seed<SEED>.npz")
X, y = res['X'], res['y']
features = sparse.load_npz("percolation_dataset_size<SIZE>_dim<DIM>_seed<SEED>_gt_features.npz")
```

## Caveats

[in progress]

## More information

This project is led by Ari Brill. Contact information is on my [website](https://www.aribrill.com/).ercolationDataset.construct_embed()

This research program is part of the [renormalization research group](https://www.lesswrong.com/posts/74wSgnCKPHAuqExe7/renormalization-roadmap) at [Principles of Intelligence](https://princint.ai/).

For more information on the percolation cluster model of data structure, see the papers:

[Brill (2024), Neural Scaling Laws Rooted in the Data Distribution](https://arxiv.org/abs/2412.07942)

[Brill (2025), Representation Learning on a Random Lattice](https://arxiv.org/abs/2504.20197)
