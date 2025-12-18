import unittest
import networkx as nx
import numpy as np
from scipy import stats

from percolation_dataset import PercolationDataset

class TestPercolationDataset(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.dataset = PercolationDataset(rng=self.rng)

    def test_initialization_validation(self):
        """Test that invalid create_prob and split_prob values raise ValueError."""
        with self.assertRaises(ValueError):
            PercolationDataset(create_prob=-0.1)
        with self.assertRaises(ValueError):
            PercolationDataset(create_prob=1.1)
        with self.assertRaises(ValueError):
            PercolationDataset(split_prob=-0.1)
        with self.assertRaises(ValueError):
            PercolationDataset(split_prob=1.1)

    def test_construct_structure(self):
        """Test that construct generates the correct number of points and latents."""
        size = 10
        points, latents = self.dataset.construct(size=size)

        # Verify counts
        point_count = sum(len(cluster_points) for cluster_points in points)
        cluster_count = len(points)
        self.assertEqual(point_count, size, "Incorrect number of points (leaves)")
        self.assertEqual(len(latents), size - cluster_count, "Incorrect number of latents (internal nodes)")

        # Verify node types
        for cluster_points in points:
            for p in cluster_points.values():
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
        for cluster_points in points:
            for p in cluster_points.values():
                if p.level > 0:
                    self.assertEqual(p.value, 100.0)

    def test_neighbor_graph_is_tree(self):
        """Test that the graph formed by neighboring points is a tree."""
        # Generate a reasonably sized graph
        points, _ = self.dataset.construct(size=100)
        adj = self.dataset.build_cluster_graph(points[0])
        G = nx.Graph(adj) # type: ignore[arg-type]
        self.assertTrue(nx.is_tree(G), "Neighbor graph is not a tree (it should be connected and acyclic)")

    def test_cluster_hierarchy_is_directed_tree(self):
        """Test that parent-child relationships form a directed tree (arborescence)."""
        points, latents = PercolationDataset(rng=self.rng, create_prob=0).construct(size=50)
        G = nx.DiGraph()

        # Collect all nodes involved
        for cluster_points in points:
            for node in cluster_points.values():
                for child in node.children:
                    G.add_edge(node.point_idx, child.point_idx)
        for node in latents.values():
            for child in node.children:
                G.add_edge(node.point_idx, child.point_idx)

        # Check if the structure is an arborescence (a tree directed away from a root)
        self.assertTrue(nx.is_arborescence(G), "Hierarchy is not an arborescence (directed tree)")

    def test_hierarchy_is_directed_forest(self):
        """Test that parent-child relationships form a directed forest (branching)."""
        points, latents = self.dataset.construct(size=50)
        G = nx.DiGraph()

        # Collect all nodes involved
        for cluster_points in points:
            for node in cluster_points.values():
                for child in node.children:
                    G.add_edge(node.point_idx, child.point_idx)
        for node in latents.values():
            for child in node.children:
                G.add_edge(node.point_idx, child.point_idx)

        # Check if the structure is a branching (a forest of trees directed away from a root)
        self.assertTrue(nx.is_branching(G), "Hierarchy is not a branching (directed forest)")

    def test_degree_distribution(self):
        """Test that degree distribution is well fit by shifted Poisson distribution using KL divergence."""
        size = 100000
        points, _ = self.dataset.construct(size=size)

        # Get degrees, skipping small clusters
        size_threshold = 100
        degrees = []
        for cluster_points in points:
            if len(cluster_points) >= size_threshold:
                degrees.extend([len(node.neighbors) for node in cluster_points.values()])

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

        threshold = 0.001
        self.assertLess(kl_div, threshold, f"KL divergence {kl_div:.6f} exceeds threshold {threshold}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
