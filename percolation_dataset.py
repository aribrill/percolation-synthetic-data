from collections import defaultdict
from itertools import chain, islice
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

class Node:
    """Represents a node in the percolation graph."""
    def __init__(self, point_idx: int, cluster_idx: int,
                 parents: Optional[List['Node']] = None,
                 children: Optional[List['Node']] = None,
                 neighbors: Optional[List['Node']] = None,
                 value: float = 0.0, level: int = 0,
                 depth: int = 0, node_type: str = 'point'):
        self.point_idx = point_idx
        self.cluster_idx = cluster_idx
        self.node_type = node_type
        if self.node_type not in ('latent', 'point'):
            raise ValueError(f"Invalid node_type: {self.node_type}")

        self.parents: Set['Node'] = set(parents) if parents is not None else set()
        self.children: Set['Node'] = set(children) if children is not None else set()
        self.neighbors: Set['Node'] = set(neighbors) if neighbors is not None else set()
        self.value = value
        self.level = level
        self.depth = depth
        self.error = 0.0

    def __repr__(self) -> str:
        return f"Node(point_idx={self.point_idx}, type={self.node_type}, value={self.value:.2f})"

class PercolationDataset:
    """Generates a percolation dataset."""

    def __init__(self, create_prob: float = 0.3333333, split_prob: float = 0.2096414,
                 rng: Optional[np.random.Generator] = None,
                 value_generator: Optional[Callable[..., float]] = None,
                 value_generator_kwargs: Optional[Dict[str, Any]] = None):
        """
        Args:
            create_prob: Probability of creating a new cluster. Default is 1/3.
            split_prob: Probability of a neighbor connecting to the first child. Default is 0.2096414.
            rng: Random number generator.
            value_generator: Function to generate values for new nodes.
                             Signature: (*args, **kwargs) -> float.
            value_generator_kwargs: Additional keyword arguments for value_generator. Default is {'ratio': 0.5}.
        """
        if not (0 <= create_prob <= 1):
            raise ValueError(f"create_prob must be between 0 and 1, got {create_prob}")
        if not (0 <= split_prob <= 1):
            raise ValueError(f"split_prob must be between 0 and 1, got {split_prob}")

        self.create_prob = create_prob
        self.split_prob = split_prob
        self.rng = rng if rng is not None else np.random.default_rng()
        self.value_generator = value_generator if value_generator is not None else self._default_generate_value
        self.value_generator_kwargs = value_generator_kwargs if value_generator_kwargs is not None else {'ratio': 0.5}

    def _default_generate_value(self, parent_value: float, parent_depth: int, rng: np.random.Generator, **kwargs: Any) -> float:
        ratio = kwargs['ratio']
        variance = (1 - ratio) * ratio**(parent_depth + 1)
        std = np.sqrt(variance)
        return parent_value + rng.normal(0, std)

    def _split_node(self, node: 'Node', idx_1: int, idx_2: int, level: int) -> Tuple['Node', 'Node']:
        # Use the configured value generator, passing the node and the dataset's rng
        val1 = self.value_generator(parent_value=node.value, parent_depth=node.depth, rng=self.rng,
                                    **self.value_generator_kwargs)
        val2 = self.value_generator(parent_value=node.value, parent_depth=node.depth, rng=self.rng,
                                    **self.value_generator_kwargs)

        child1 = Node(idx_1, node.cluster_idx, parents=[node], value=val1,
                      level=level, depth=node.depth + 1)
        child2 = Node(idx_2, node.cluster_idx, parents=[node], value=val2,
                      level=level, depth=node.depth + 1)

        # Clear neighbors of the current node as they are being redistributed
        neighbors = list(node.neighbors)
        node.neighbors.clear()

        groups = self.rng.choice([1, 2], size=len(neighbors), p=[self.split_prob, 1 - self.split_prob])
        for n, group in zip(neighbors, groups):
            if group == 1:
                n.neighbors.add(child1)
                child1.neighbors.add(n)
            else:
                n.neighbors.add(child2)
                child2.neighbors.add(n)
            n.neighbors.remove(node)

        child1.neighbors.add(child2)
        child2.neighbors.add(child1)
        node.children.add(child1)
        node.children.add(child2)

        # Update the current node to be a latent node
        node.node_type = 'latent'
        node.value = 0.0
        node.level = level

        return child1, child2
    
    def _select_cluster(self, cluster_sizes: NDArray[np.int64], n_clusters: int) -> int:
        """
        Selects a cluster using size-proportional weighting.
        
        Args:
            cluster_sizes: Array of cluster sizes with trailing zeros.
            n_clusters: Number of clusters.

        Returns:
            idx: Index of the selected cluster. 
        """
        # TODO: implement efficient selection using Fenwick tree and numba
        weights = cluster_sizes[:n_clusters] / cluster_sizes[:n_clusters].sum()
        idx = self.rng.choice(n_clusters, p=weights)
        return idx

    def construct(self, size: int) -> Tuple[List[Dict[int, 'Node']], Dict[int, 'Node']]:
        """
        Constructs the dataset by iteratively splitting nodes.

        Args:
            size: The number of leaf nodes (points) to generate.

        Returns:
            points: Dictionary of leaf nodes.
            latents: Dictionary of latent (internal) nodes.
        """
        if size < 1:
            raise ValueError(f"size must be at least 1, got {size}")

        point_idx = 0
        cluster_idx = 0
        root = Node(point_idx, cluster_idx, value=self.value_generator(0, -1, self.rng, **self.value_generator_kwargs))
        points: List[Dict[int, Node]] = [{root.point_idx: root}]
        cluster_sizes = np.zeros(size, dtype=int)
        cluster_sizes[0] = 1
        keys: List[List[int]] = [[point_idx]]
        latents: Dict[int, Node] = {}

        rvs = self.rng.random(size=size)
        for i in range(1, size):
            if rvs[i] < self.create_prob:
                node = Node(point_idx + 1, cluster_idx + 1, level=i,
                            value=self.value_generator(0, -1, self.rng, **self.value_generator_kwargs))
                points.append({point_idx + 1: node})
                cluster_sizes[cluster_idx + 1] = 1
                keys.append([point_idx + 1])
                point_idx += 1
                cluster_idx += 1
            else:
                cluster_split_idx = self._select_cluster(cluster_sizes, cluster_idx + 1)
                keys_idx = self.rng.choice(len(keys[cluster_split_idx]))
                point_split_idx = keys[cluster_split_idx][keys_idx]
                split_node = points[cluster_split_idx][point_split_idx]

                child1, child2 = self._split_node(split_node, point_idx + 1, point_idx + 2, i)

                del points[cluster_split_idx][point_split_idx]
                points[cluster_split_idx][child1.point_idx] = child1
                points[cluster_split_idx][child2.point_idx] = child2
                latents[point_split_idx] = split_node

                # Swap-remove to update keys in O(1)
                keys[cluster_split_idx][keys_idx] = keys[cluster_split_idx][-1]
                keys[cluster_split_idx].pop()
                keys[cluster_split_idx].append(point_idx + 1)
                keys[cluster_split_idx].append(point_idx + 2)

                cluster_sizes[cluster_split_idx] += 1

                point_idx += 2

        points.sort(key=len, reverse=True)
        return points, latents

    @staticmethod
    def build_cluster_graph(cluster_points: Dict[int, 'Node']) -> Dict[int, List[int]]:
        """
        Builds the graph structure from points.
        Returns an adjacency dictionary mapping node idx to list of neighbor indices.
        """
        adj = {idx: [] for idx in cluster_points}
        for point in cluster_points.values():
            for neighbor in point.neighbors:
                adj[point.point_idx].append(neighbor.point_idx)
        return adj

    def embed(self, points: List[Dict[int, 'Node']], d: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embeds the points into a vector space using a tree random walk.

        Args:
            points: Dictionary of leaf nodes.
            d: Dimension of the embedding space.

        Returns:
            X: Embeddings matrix.
            y: Labels vector.
        """
        if d < 1:
             raise ValueError(f"Embedding dimension d must be at least 1, got {d}")

        if not points or not points[0]:
            return np.empty((0, d)), np.array([])
        
        all_nodes = []
        all_embeddings = {}
        all_labels = []

        size = sum(len(cluster_points) for cluster_points in points)
        scale = size**-0.25
        ratio = self.value_generator_kwargs['ratio']
        for cluster_points in points:
            adj = self.build_cluster_graph(cluster_points)

            for point in cluster_points.values():
                irreducible_variance = ratio**(point.depth  + 1)
                irreducible_std = np.sqrt(irreducible_variance)
                irreducible_error = self.rng.normal(0, irreducible_std)
                point.error = irreducible_error
                all_labels.append(point.value + point.error)

            nodes = [point.point_idx for point in cluster_points.values()]

            root = self.rng.choice(nodes)
            embeddings = {root: self.rng.uniform(low=-0.5, high=0.5, size=d)}

            # DFS using stack
            stack = [root]
            visited = {root}

            while stack:
                parent = stack.pop()
                for child in adj[parent]:
                    if child not in visited:
                        visited.add(child)
                        assert child not in embeddings
                        embeddings[child] = embeddings[parent] + self.rng.normal(0, scale, size=d)
                        stack.append(child)
            
            all_nodes.extend(nodes)
            all_embeddings.update(embeddings)

        X = np.stack([all_embeddings[node] for node in all_nodes])
        y = np.array(all_labels)

        return X, y

    def construct_embed(self, size: int, d: int) -> Tuple[List[Dict[int, 'Node']], Dict[int, 'Node'], np.ndarray, np.ndarray]:
        """Convenience method to construct the dataset and generate embeddings."""
        points, latents = self.construct(size)
        embeddings, labels = self.embed(points, d)
        return points, latents, embeddings, labels

class GroundTruthFeatures:
    """Generates ground truth features from generated percolation data."""

    def __init__(self, points: List[Dict[int, Node]], latents: Dict[int, Node]):
        self.points = points
        self.latents = latents
        
        self.pidx2lidx = defaultdict(list)
        self.lidx2pidx = defaultdict(list)
        sorted_latents = dict(sorted(latents.items(), key=lambda item: item[1].level))
        self.latent2lidx = {latent.point_idx: i for i, latent in enumerate(sorted_latents.values())}
        self.lidx2latent = {i: latent.point_idx for i, latent in enumerate(sorted_latents.values())}

        for i, point in enumerate(chain.from_iterable(cp.values() for cp in points)):
            parents = point.parents
            while parents:
                latent = list(parents)[0]
                lidx = self.latent2lidx[latent.point_idx]
                self.pidx2lidx[i].append(lidx)
                self.lidx2pidx[lidx].append(i)
                parents = latent.parents
            self.pidx2lidx[i].reverse()
        self.lidx2pidx = dict(sorted(self.lidx2pidx.items(), key=lambda item: item[0]))
        self.n_samples = sum(len(cp) for cp in self.points)
        self.n_latents = len(self.lidx2pidx)

    def get_features(self, n_features: Optional[int] = None) -> sparse.csr_matrix:
        """
        Returns sparse matrix of latent features for each data point.

        Args:
            n_features: Number of features to return, in generation order. If None, returns all features.
        Returns:            
            X: Sparse matrix of shape (n_points, n_features).
        """
        if n_features is None:
            n_features = self.n_latents
        data = []
        row_ind = []
        col_ind = []

        for lidx, pidx in islice(self.lidx2pidx.items(), n_features):
            data.extend(np.ones_like(pidx))
            row_ind.extend(pidx)
            col_ind.extend(np.full_like(pidx, lidx))

        X = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(self.n_samples, n_features))
        return X