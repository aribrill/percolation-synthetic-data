# percolation-synthetic-data
Generates statistically self-similar synthetic datasets based on a percolation cluster model.

## About

[in progress]

## Algorithm

This repository implements an algorithm to simulate a data distribution modeled as a critical percolation cluster distribution on a large high-dimensional lattice, using an explicitly hierarchical approach. The algorithm consists of two stages. First, in the generation stage, a set of percolation clusters is generated iteratively. Each iteration represents a single "fine-graining" step in which a single data point is decomposed into two related data points. The generation stage produces a set of undirected, treelike graphs representing the clusters, and a forest of binary latent features that denote each point's membership in a cluster or subcluster. Each point has an associated value that is a function of its latent subcluster membership features. Second, in the embedding stage, the graphs are embedded into a vector space following a branching random walk.

In the generation stage, each iteration follows one of two alternatives. With probability `create_prob`, a new cluster with one point is created. Otherwise, an existing point is selected at random and removed, becoming a latent feature. This parent is replaced by two new child points connected to each other by a new edge. Call these points $a$ and $b$. The child points $a$ and $b$ are assigned values as a stochastic function of the parent's value. Each former neighbor of the parent is then connected to either $a$ with probability `split_prob`, or to $b$ with probability `1 - split_prob`. The parameter values that yield the correct cluster structure can be shown to be `create_prob = 1/3` and `split_prob = 0.2096414`. The derivations of these values and full details on the algorithm will be presented in a forthcoming publication.

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
