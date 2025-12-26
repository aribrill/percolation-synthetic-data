# percolation-synthetic-data
Generates statistically self-similar synthetic datasets based on a percolation cluster model.

## About

Ambitious mechanistic interpretability requires understanding the structure that neural networks uncover from data. A quantitative theoretical model of natural data's organizing structure could help AI safety researchers build interpretability tools that decompose neural networks along their natural scales of abstraction. This project works towards this goal by developing a synthetic data model that reproduces qualitative features of natural data. The model is based on high-dimensional percolation theory and describes statistically self-similar, sparse, and power-law-distributed data distributional structure.

This repository provides code to generate synthetic datasets based on this data model. In particular, it employs a newly developed algorithm to construct a dataset in a way that explicitly and iteratively reveals its innate hierarchical structure. Increasing the number of data points corresponds to representing the same dataset at a more fine-grained level of abstraction.

## Percolation Theory

The branch of physics concerned with analyzing the properties of clusters of randomly occupied units on a lattice is called percolation theory ([Stauffer & Aharony, 1994](https://www.eng.uc.edu/~beaucag/Classes/Properties/Books/DietrichStaufferAmmonAharony-Introductiontopercolationtheory(1994).pdf)). In this framework, sites (or bonds) are occupied independently at random with probability $p$, and connected sites form clusters. While direct numerical simulation of percolation on a high-dimensional lattice is intractable due to the curse of dimensionality, the high-dimensional problem is exactly solvable analytically. Clusters are vanishingly unlikely to have loops, and the problem can be approximated by modeling the lattice as an infinite tree. This can be viewed as the mean-field approximation for percolation. In particular, percolation clusters on a high-dimensional lattice (at or above the upper critical dimension _d_ >= 6) that are at or near criticality can be modeled using the Bethe lattice, an infinite treelike graph in which each node has identical degree _z_. For site or bond percolation on the Bethe lattice, the percolation threshold is _p\_c_ = 1/(_z_- 1). Using the Bethe lattice as an approximate model of a hypercubic lattice of dimension _d_ gives _z_ = 2*_d_ and _p\_c_ = 1/(2*_d_ - 1). A brief self-contained review based on standard references can be found in [Brill (2025, App. A)](https://arxiv.org/abs/2504.20197).

## Algorithm

This repository implements an algorithm to simulate a data distribution modeled as a critical percolation cluster distribution on a large high-dimensional lattice, using an explicitly hierarchical approach. The algorithm consists of two stages. First, in the generation stage, a set of percolation clusters is generated iteratively. Each iteration represents a single "fine-graining" step in which a single data point (site) is decomposed into two related data points. The generation stage produces a set of undirected, treelike graphs representing the clusters, and a forest of binary latent features that denote each point's membership in a cluster or subcluster. Each point has an associated value that is a function of its latent subcluster membership features. Second, in the embedding stage, the graphs are embedded into a vector space following a branching random walk.

In the generation stage, each iteration follows one of two alternatives. With probability `create_prob`, a new cluster with one point is created. Otherwise, an existing point is selected at random and removed, becoming a latent feature. This parent is replaced by two new child points connected to each other by a new edge. Call these points a and b. The child points _a_ and _b_ are assigned values as a stochastic function of the parent's value. Each former neighbor of the parent is then connected to either _a_ with probability `split_prob`, or to _b_ with probability `1 - split_prob`. The parameter values that yield the correct cluster structure can be shown to be `create_prob = 1/3` and `split_prob = 0.2096414`. The derivations of these values and full details on the algorithm will be presented in a forthcoming publication.

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

This project is led by Ari Brill. Contact information is on my [website](https://www.aribrill.com/).

This research program is part of the [renormalization research group](https://www.lesswrong.com/posts/74wSgnCKPHAuqExe7/renormalization-roadmap) at [Principles of Intelligence](https://princint.ai/).

For more information on the percolation cluster model of data structure, see the papers:

[Brill (2024), Neural Scaling Laws Rooted in the Data Distribution](https://arxiv.org/abs/2412.07942)

[Brill (2025), Representation Learning on a Random Lattice](https://arxiv.org/abs/2504.20197)
