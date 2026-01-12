from collections import Counter, defaultdict
from itertools import islice
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
        self.value: float = value
        self.level: int = level
        self.depth: int = depth
        self.error: float = 0.0

    def __repr__(self) -> str:
        return f"Node(point_idx={self.point_idx}, type={self.node_type}, value={self.value:.2f})"

class PercolationDataset:
    """Generates a percolation dataset."""

    def __init__(self, create_prob: float = 0.3333333, split_prob: float = 0.2096414,
                 graph_seed: Optional[int] = None,
                 embed_seed: Optional[int] = None,
                 value_seed: Optional[int] = None,
                 value_generator: Optional[Callable[..., float]] = None,
                 value_generator_kwargs: Optional[Dict[str, Any]] = None):
        """
        Args:
            create_prob: Probability of creating a new cluster. Default is 1/3.
            split_prob: Probability of a neighbor connecting to the first child. Default is 0.2096414.
            graph_seed: Seed for the random number generator used in graph construction.
            embed_seed: Seed for the random number generator used in embedding.
            value_seed: Seed for the random number generator used in value generation.
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
        self.graph_seed = graph_seed
        self.embed_seed = embed_seed
        self.value_seed = value_seed
        self.value_generator = value_generator if value_generator is not None else self._default_generate_value
        self.value_generator_kwargs = value_generator_kwargs if value_generator_kwargs is not None else {'ratio': 0.5}

    def _default_generate_value(self, base_value: float, depth: int, rng: np.random.Generator, **kwargs: Any) -> float:
        ratio = kwargs['ratio']
        variance = (1 - ratio) * ratio**depth
        std = np.sqrt(variance)
        return base_value + rng.normal(0, std)

    def _split_node(self, node: 'Node', rng: np.random.Generator, idx_1: int, idx_2: int) -> Tuple['Node', 'Node']:
        # Use the configured value generator, passing the node and the dataset's rng

        child1 = Node(idx_1, node.cluster_idx, parents=[node], depth=node.depth + 1)
        child2 = Node(idx_2, node.cluster_idx, parents=[node], depth=node.depth + 1)

        # Clear neighbors of the current node as they are being redistributed
        neighbors = sorted(list(node.neighbors), key=lambda n: n.point_idx)
        node.neighbors.clear()

        groups = rng.choice([1, 2], size=len(neighbors), p=[self.split_prob, 1 - self.split_prob])
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

        return child1, child2
    

    def construct(self, size: int) -> Tuple[List['Node'], Dict[int, 'Node']]:
        """
        Constructs the dataset by iteratively splitting nodes.

        Args:
            size: The number of leaf nodes (points) to generate.

        Returns:
            points: List of data points.
            latents: Dictionary of latent (internal) nodes.
        """
        if size < 1:
            raise ValueError(f"size must be at least 1, got {size}")
        
        rng = np.random.default_rng(seed=self.graph_seed) if self.graph_seed is not None else np.random.default_rng()

        point_idx = 0
        cluster_idx = 0
        root = Node(point_idx, cluster_idx, level=0)
        points: List[Node] = [root]
        latents: Dict[int, Node] = {}

        rvs = rng.random(size=(size, 2))
        for i in range(1, size):
            if rvs[i, 0] < self.create_prob:
                node = Node(point_idx + 1, cluster_idx + 1, level=i)
                points.append(node)
                point_idx += 1
                cluster_idx += 1
            else:
                split_idx = int(rvs[i, 1] * i)
                split_node = points[split_idx]
                child1, child2 = self._split_node(split_node, rng, point_idx + 1, point_idx + 2)
                child1.level, child2.level = i, i
                latents[split_node.point_idx] = split_node

                # Swap-remove to update points in O(1)
                points[split_idx], points[-1] = points[-1], points[split_idx]
                points.pop()
                points.append(child1)
                points.append(child2)                
                point_idx += 2

        # Sort points by cluster in descending order of cluster size
        counts = Counter(p.cluster_idx for p in points)
        points.sort(key=lambda p: (-counts[p.cluster_idx], p.cluster_idx, p.point_idx))

        return points, latents
    

    def embed_features(self, points: List['Node'], latents: Dict[int, 'Node'], d: int) -> np.ndarray:
        """
        Embeds the points into a vector space using a tree random walk.

        Args:
            points: List of data points.
            d: Dimension of the embedding space.

        Returns:
            X: Embeddings matrix.
        """
        if d < 1:
             raise ValueError(f"Embedding dimension d must be at least 1, got {d}")

        if not points:
            return np.empty((0, d))
        
        # Set up RNG
        rng = np.random.default_rng(seed=self.embed_seed) if self.embed_seed is not None else np.random.default_rng()
        
        # Compute base embedding direction for each cluster
        cluster_vectors = {}
        for cluster_idx in np.unique([point.cluster_idx for point in points]):
            vec = rng.normal(size=d)
            cluster_vectors[cluster_idx] = vec / np.linalg.norm(vec)

        # Assign directions using latents to embed points self-consistently with scale
        point_vectors = {}
        for point_idx, latent in latents.items():
            if point_idx not in point_vectors:
                point_vectors[point_idx] = cluster_vectors[latent.cluster_idx]
            u = rng.normal(size=d)
            w = u - (u @ point_vectors[point_idx])*point_vectors[point_idx] # project to orthogonal complement
            w /= np.linalg.norm(w)
            child_vectors = [np.sqrt(0.5)*(point_vectors[point_idx] + w), np.sqrt(0.5)*(point_vectors[point_idx] - w)]
            for child, vector in zip(sorted(latent.children, key=lambda c: c.point_idx), child_vectors):
                point_vectors[child.point_idx] = vector

        size = len(points)
        scale = size**-0.25
        start_idx, end_idx = 0, 0
        point_idx2idx = {point.point_idx: i for i, point in enumerate(points)}
        embeddings = {}

        while start_idx < size:
            # Find boundaries of current cluster
            while end_idx < size and points[end_idx].cluster_idx == points[start_idx].cluster_idx:
                end_idx += 1

            # Retrieve vector now for a single-point cluster with no latents
            if end_idx - start_idx == 1:
                point_vectors[points[start_idx].point_idx] = cluster_vectors[points[start_idx].cluster_idx]

            # Randomly choose root in [start_idx, end_idx)
            root_idx = rng.choice(end_idx - start_idx) + start_idx
            root = points[root_idx].point_idx

            # DFS using stack
            stack = [root]
            visited = {root}
            embeddings[root] = rng.uniform(low=-0.5, high=0.5, size=d) + scale*point_vectors[root]

            while stack:
                parent = stack.pop()
                for neighbor in sorted(points[point_idx2idx[parent]].neighbors, key=lambda n: n.point_idx):
                    if neighbor.point_idx not in visited:
                        visited.add(neighbor.point_idx)
                        embeddings[neighbor.point_idx] = embeddings[parent] + scale*point_vectors[neighbor.point_idx]
                        stack.append(neighbor.point_idx)
            
            start_idx = end_idx

        return np.stack([embeddings[point.point_idx] for point in points])


    def embed_labels(self, points: List['Node'], latents: Dict[int, 'Node']) -> np.ndarray:
        """Generates labels for the points."""
        rng = np.random.default_rng(seed=self.value_seed) if self.value_seed is not None else np.random.default_rng()

        # Compute base value for each cluster
        cluster_values = {}
        for cluster_idx in np.unique([point.cluster_idx for point in points]):
            base_value = self.value_generator(base_value=0.0, depth=0, rng=rng,
                                              **self.value_generator_kwargs)
            cluster_values[cluster_idx] = base_value

        # Compute values hierarchically using the latents
        for latent in latents.values():
            if not latent.parents:
                latent.value = cluster_values[latent.cluster_idx]
            for child in sorted(latent.children, key=lambda c: c.point_idx):
                child.value = self.value_generator(base_value=latent.value, depth=latent.depth + 1, rng=rng,
                                                  **self.value_generator_kwargs)
        for point in points:
            if not point.parents:
                point.value = cluster_values[point.cluster_idx]

        # Compute irreducible error
        ratio = self.value_generator_kwargs['ratio']
        for point in points:
            irreducible_variance = ratio**(point.depth + 1)
            irreducible_std = np.sqrt(irreducible_variance)
            irreducible_error = rng.normal(0, irreducible_std)
            point.error = irreducible_error

        return np.array([point.value + point.error for point in points])


    def construct_embed(self, size: int, d: int) -> Tuple[List['Node'], Dict[int, 'Node'], np.ndarray, np.ndarray]:
        """Convenience method to construct the dataset and generate embeddings."""
        points, latents = self.construct(size)
        X = self.embed_features(points, latents, d)
        y = self.embed_labels(points, latents)
        return points, latents, X, y


class GroundTruthFeatures:
    """Generates ground truth features from generated percolation data."""

    def __init__(self, points: List['Node'], latents: Dict[int, 'Node']):
        self.points = points
        self.latents = latents
        
        self.pidx2lidx = defaultdict(list)
        self.lidx2pidx = defaultdict(list)
        sorted_latents = dict(sorted(latents.items(), key=lambda item: item[1].level))
        self.latent2lidx = {latent.point_idx: i for i, latent in enumerate(sorted_latents.values())}
        self.lidx2latent = {i: latent.point_idx for i, latent in enumerate(sorted_latents.values())}

        for i, point in enumerate(points):
            parents = point.parents
            while parents:
                latent = list(parents)[0]
                lidx = self.latent2lidx[latent.point_idx]
                self.pidx2lidx[i].append(lidx)
                self.lidx2pidx[lidx].append(i)
                parents = latent.parents
            self.pidx2lidx[i].reverse()
        self.lidx2pidx = dict(sorted(self.lidx2pidx.items(), key=lambda item: item[0]))
        self.n_samples = len(self.points)
        self.n_latents = len(self.lidx2pidx)

    def get_features(self, n_features: Optional[int] = None) -> sparse.csr_matrix:
        """
        Returns sparse matrix of latent features for each data point.

        Args:
            n_features: Number of features to return, in generation order. If None, returns all features.
        Returns:            
            features: Sparse matrix of shape (n_points, n_features).
        """
        if n_features is None:
            n_features = self.n_latents
        if n_features < 1 or n_features > self.n_latents:
            raise ValueError(f"n_features must be between 1 and {self.n_latents}, got {n_features}")
        data = []
        row_ind = []
        col_ind = []

        for lidx, pidx in islice(self.lidx2pidx.items(), n_features):
            data.extend(np.ones_like(pidx))
            row_ind.extend(pidx)
            col_ind.extend(np.full_like(pidx, lidx))

        features = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(self.n_samples, n_features))
        return features