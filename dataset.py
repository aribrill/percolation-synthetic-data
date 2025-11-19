from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

class Node:
    """Represents a node in the percolation graph."""
    def __init__(self, idx: int, parents: Optional[List['Node']] = None,
                 children: Optional[List['Node']] = None,
                 neighbors: Optional[List['Node']] = None,
                 value: float = 0.0, level: int = 0, node_type: str = 'point'):
        self.idx = idx
        self.node_type = node_type
        if self.node_type not in ('latent', 'point'):
            raise ValueError(f"Invalid node_type: {self.node_type}")

        self.parents: Set['Node'] = set(parents) if parents is not None else set()
        self.children: Set['Node'] = set(children) if children is not None else set()
        self.neighbors: Set['Node'] = set(neighbors) if neighbors is not None else set()
        self.value = value
        self.level = level

    def __repr__(self) -> str:
        return f"Node(idx={self.idx}, type={self.node_type}, value={self.value:.2f})"

class PercolationDataset:
    """Generates a percolation dataset."""

    def __init__(self, split_prob: float = 0.2096414, rng: Optional[np.random.Generator] = None,
                 value_generator: Optional[Callable[['Node', np.random.Generator], float]] = None):
        """
        Args:
            split_prob: Probability of a neighbor connecting to the first child. Default is 0.2096414.
            rng: Random number generator.
            value_generator: Function to generate values for new nodes.
                             Signature: (parent, rng) -> float.
        """
        if not (0 <= split_prob <= 1):
            raise ValueError(f"split_prob must be between 0 and 1, got {split_prob}")

        self.split_prob = split_prob
        self.rng = rng if rng is not None else np.random.default_rng()
        self.value_generator = value_generator if value_generator is not None else self._default_generate_value

    def _default_generate_value(self, parent: 'Node', rng: np.random.Generator) -> float:
        variance = 0.8**(parent.level + 1)
        std = np.sqrt(variance)
        return parent.value + rng.normal(0, std)

    def _split_node(self, node: 'Node', idx_1: int, idx_2: int) -> Tuple['Node', 'Node']:
        node.node_type = 'latent'
        # Use the configured value generator, passing the node and the dataset's rng
        val1 = self.value_generator(node, self.rng)
        val2 = self.value_generator(node, self.rng)

        child1 = Node(idx_1, parents=[node], value=val1,
                      level=node.level + 1)
        child2 = Node(idx_2, parents=[node], value=val2,
                      level=node.level + 1)

        neighbors = list(node.neighbors)
        # Clear neighbors of the current node as they are being redistributed
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
        return child1, child2

    def construct(self, size: int) -> Tuple[Dict[int, 'Node'], Dict[int, 'Node']]:
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
        root = Node(point_idx)
        points: Dict[int, Node] = {root.idx: root}
        keys_list = [point_idx]
        latents: Dict[int, Node] = {}

        for i in range(1, size):
            keys_idx = self.rng.choice(len(keys_list))
            split_idx = keys_list[keys_idx]
            split_node = points[split_idx]

            child1, child2 = self._split_node(split_node, point_idx + 1, point_idx + 2)

            del points[split_idx]
            points[child1.idx] = child1
            points[child2.idx] = child2
            latents[split_idx] = split_node

            # Swap-remove to update keys_list in O(1)
            keys_list[keys_idx] = keys_list[-1]
            keys_list.pop()
            keys_list.append(point_idx + 1)
            keys_list.append(point_idx + 2)

            point_idx += 2

        return points, latents

    @staticmethod
    def build_cluster_graph(points: Dict[int, 'Node']) -> Dict[int, List[int]]:
        """
        Builds the graph structure from points.
        Returns an adjacency dictionary mapping node idx to list of neighbor indices.
        """
        adj = {idx: [] for idx in points}
        for point in points.values():
            for neighbor in point.neighbors:
                adj[point.idx].append(neighbor.idx)
        return adj

    def embed(self, points: Dict[int, 'Node'], d: int) -> Tuple[np.ndarray, np.ndarray]:
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

        adj = self.build_cluster_graph(points)
        labels = {point.idx: point.value for point in points.values()}

        nodes = list(adj.keys())
        if not nodes:
            return np.empty((0, d)), np.array([])

        root = self.rng.choice(nodes)
        embeddings = {root: np.zeros(d)}

        # DFS using stack
        stack = [root]
        visited = {root}

        while stack:
            parent = stack.pop()
            for child in adj[parent]:
                if child not in visited:
                    visited.add(child)
                    embeddings[child] = embeddings[parent] + self.rng.normal(0, 1, size=d)
                    stack.append(child)

        X = np.stack([embeddings[node] for node in nodes])
        if X.shape[0] > 1:
            X -= np.mean(X, axis=0)
            X /= np.std(X) + 1e-8

        y = np.array([labels[node] for node in nodes])
        if y.shape[0] > 1:
            y -= np.mean(y)
            y /= np.std(y) + 1e-8

        sorted_inds = np.argsort(nodes)
        X = X[sorted_inds]
        y = y[sorted_inds]

        return X, y

    def construct_embed(self, size: int, d: int) -> Tuple[Dict[int, 'Node'], Dict[int, 'Node'], np.ndarray, np.ndarray]:
        """Convenience method to construct the dataset and generate embeddings."""
        points, latents = self.construct(size)
        embeddings, labels = self.embed(points, d)
        return points, latents, embeddings, labels

import unittest
import networkx as nx
import numpy as np
from scipy import stats

class TestPercolationDataset(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.dataset = PercolationDataset(rng=self.rng)

    def test_initialization_validation(self):
        """Test that invalid split_prob values raise ValueError."""
        with self.assertRaises(ValueError):
            PercolationDataset(split_prob=-0.1)
        with self.assertRaises(ValueError):
            PercolationDataset(split_prob=1.1)

    def test_construct_structure(self):
        """Test that construct generates the correct number of points and latents."""
        size = 10
        points, latents = self.dataset.construct(size=size)

        # Verify counts
        self.assertEqual(len(points), size, "Incorrect number of points (leaves)")
        self.assertEqual(len(latents), size - 1, "Incorrect number of latents (internal nodes)")

        # Verify node types
        for p in points.values():
            self.assertEqual(p.node_type, 'point')
        for l in latents.values():
            self.assertEqual(l.node_type, 'latent')

    def test_embed_shapes(self):
        """Test that embed produces outputs with correct shapes."""
        size = 20
        d = 5
        points, latents = self.dataset.construct(size=size)
        X, y = self.dataset.embed(points, d=d)

        self.assertEqual(X.shape, (size, d))
        self.assertEqual(y.shape, (size,))

    def test_input_validation_methods(self):
        """Test validation in construct and embed methods."""
        with self.assertRaises(ValueError):
            self.dataset.construct(size=0)

        points, _ = self.dataset.construct(size=5)
        with self.assertRaises(ValueError):
            self.dataset.embed(points, d=0)

    def test_custom_value_generator(self):
        """Test that a custom value generator is correctly utilized."""
        def constant_gen(parent, rng):
            return 100.0

        ds = PercolationDataset(value_generator=constant_gen, rng=self.rng)
        points, latents = ds.construct(size=5)

        # Nodes generated via split (level > 0) should have value 100.0
        for p in points.values():
            if p.level > 0:
                self.assertEqual(p.value, 100.0)

    def test_neighbor_graph_is_tree(self):
        """Test that the graph formed by neighboring points is a tree."""
        # Generate a reasonably sized graph
        points, _ = self.dataset.construct(size=50)
        adj = self.dataset.build_cluster_graph(points)
        G = nx.Graph(adj)
        self.assertTrue(nx.is_tree(G), "Neighbor graph is not a tree (it should be connected and acyclic)")

    def test_hierarchy_is_directed_tree(self):
        """Test that parent-child relationships form a directed tree (arborescence)."""
        points, latents = self.dataset.construct(size=50)
        G = nx.DiGraph()

        # Collect all nodes involved
        all_nodes = list(points.values()) + list(latents.values())
        for node in all_nodes:
            for child in node.children:
                G.add_edge(node.idx, child.idx)

        # Check if the structure is an arborescence (a tree directed away from a root)
        self.assertTrue(nx.is_arborescence(G), "Hierarchy is not a directed tree/arborescence")

    def test_degree_distribution(self):
        """Test that degree distribution is well fit by shifted Poisson distribution using KL divergence."""
        size = 50000
        points, _ = self.dataset.construct(size=size)

        # Get degrees
        degrees = [len(node.neighbors) for node in points.values()]

        # Calculate empirical PMF
        val, counts = np.unique(degrees, return_counts=True)
        pk_dict = dict(zip(val, counts / size))

        max_degree = max(degrees)
        domain = np.arange(1, max_degree + 1)

        pk = np.array([pk_dict.get(k, 0.0) for k in domain])

        # Calculate expected PMF (Shifted Poisson: mu=1, loc=1)
        # P(k) = exp(-1) * 1^(k-1) / (k-1)!
        qk = stats.poisson.pmf(domain, mu=1, loc=1)

        # Normalize qk to ensure it sums to 1 over the domain (handling tail truncation)
        # This ensures we compare the shape effectively within the observed support
        qk = qk / qk.sum()

        # Compute KL Divergence
        kl_div = stats.entropy(pk, qk)

        threshold = 0.0001
        self.assertLess(kl_div, threshold, f"KL divergence {kl_div:.6f} exceeds threshold {threshold}")

# Run the tests
unittest.main(argv=[''], verbosity=2, exit=False)
