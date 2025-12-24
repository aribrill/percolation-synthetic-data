import unittest
import networkx as nx
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from percolation_dataset import PercolationDataset

class TestPercolationDatasetBasic(unittest.TestCase):
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
        def constant_gen(parent_value, parent_depth, rng, **kwargs):
            return 100.0

        ds = PercolationDataset(value_generator=constant_gen, rng=self.rng)
        points, latents = ds.construct(size=5)

        # Nodes generated via split (level > 0) should have value 100.0
        for cluster_points in points:
            for p in cluster_points.values():
                if p.level > 0:
                    self.assertEqual(p.value, 100.0)

    def test_rng_reproducibility(self):
        """Test that using the same RNG seed produces identical datasets."""
        rng1 = np.random.default_rng(123)
        ds1 = PercolationDataset(rng=rng1)
        points1, latents1 = ds1.construct(size=10)

        rng2 = np.random.default_rng(123)
        ds2 = PercolationDataset(rng=rng2)
        points2, latents2 = ds2.construct(size=10)

        # Compare points
        for cluster1, cluster2 in zip(points1, points2):
            self.assertEqual(len(cluster1), len(cluster2))
            for idx in cluster1:
                node1 = cluster1[idx]
                node2 = cluster2[idx]
                self.assertEqual(node1.value, node2.value)
                self.assertEqual(node1.level, node2.level)

        # Compare latents
        self.assertEqual(len(latents1), len(latents2))
        for idx in latents1:
            node1 = latents1[idx]
            node2 = latents2[idx]
            self.assertEqual(node1.value, node2.value)
            self.assertEqual(node1.level, node2.level)

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

class TestPercolationDatasetProperties(unittest.TestCase):
    """Validate the properties of the generated dataset."""

    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(42)
        cls.dataset = PercolationDataset(rng=cls.rng)
        cls.size = 100000
        cls.d = 32
        points, latents, X, y = cls.dataset.construct_embed(size=cls.size, d=cls.d)
        cls.points = points
        cls.latents = latents
        cls.X = X
        cls.y = y

    def test_degree_distribution(self):
        """Test that degree distribution is well fit by shifted Poisson distribution using KL divergence."""
        # Get degrees, skipping small clusters
        size_threshold = 100
        degrees = []
        for cluster_points in self.points:
            if len(cluster_points) >= size_threshold:
                degrees.extend([len(node.neighbors) for node in cluster_points.values()])

        # Calculate empirical PMF
        val, counts = np.unique(degrees, return_counts=True)
        pk_dict = dict(zip(val, counts / self.size))

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

    def test_size_distribution(self):
        """Test that size distribution is well fit by a power law using maximum likelihood."""
        # Get sizes of clusters
        size_threshold = 10
        sizes = np.array([len(cluster_points) for cluster_points in self.points])
        sizes = sizes[sizes >= size_threshold]

        # Estimate power-law exponent using MLE
        alpha = 1 + len(sizes)*np.sum(np.log(sizes/size_threshold))**-1
        sigma = (alpha - 1)/np.sqrt(len(sizes))

        # Verify estimated alpha is close to the expected exponent
        expected_alpha = 2.5
        sigma_threshold = 1.5
        self.assertAlmostEqual(alpha, expected_alpha, delta=sigma_threshold*sigma,
                               msg=f"Estimated alpha {alpha:.4f} deviates more than {sigma_threshold} sigma "
                                   f"{sigma:.4f} from expected {expected_alpha}")
        
    def test_data_integrity(self):
        """Test that the embeddings and labels are valid data."""

        # Check shapes
        self.assertEqual(self.X.shape, (self.size, self.d))
        self.assertEqual(self.y.shape, (self.size,))

        # Check for NaNs and Infs in embeddings
        self.assertFalse(np.isnan(self.X).any(), "Embeddings contain NaN values")
        self.assertFalse(np.isinf(self.X).any(), "Embeddings contain Inf values")

        # Check for NaNs and Infs in labels
        self.assertFalse(np.isnan(self.y).any(), "Labels contain NaN values")
        self.assertFalse(np.isinf(self.y).any(), "Labels contain Inf values")

    def test_feature_variability(self):
        """Test that the features have reasonable variance."""
        variances = self.X.var(axis=0)
        min_var = 0.01
        max_var = 1.0
        self.assertTrue(np.all(variances > min_var), f"Feature variances are too small: {variances}")
        self.assertTrue(np.all(variances < max_var), f"Feature variances are too large: {variances}")

    def test_embedding_distribution(self):
        """Test that the embeddings have approximately zero mean and consistent standard deviations."""
        self.assertTrue(np.allclose(self.X.mean(axis=0), 0.0, atol=0.1), "Embeddings do not have mean close to 0")
        # ptp = "peak to peak" = max - min
        self.assertLess(np.ptp(self.X.std(axis=0)), 0.1, "Embeddings do not have consistent standard deviations")

    def test_label_distribution(self):
        """Test that the regression labels have approximately zero mean and unit standard deviation."""
        self.assertAlmostEqual(self.y.mean(), 0.0, delta=0.1, msg="Labels do not have mean close to 0")
        self.assertAlmostEqual(self.y.std(), 1.0, delta=0.01, msg="Labels do not have standard deviation close to 1")

    def test_feature_label_correlation(self):
        """Test that the features and labels are weakly correlated."""
        r = np.array([np.corrcoef(self.X[:, j], self.y)[0, 1] for j in range(self.X.shape[1])])
        self.assertTrue(np.max(np.abs(r)) > 0.01, "Features and labels are not correlated")
        self.assertTrue(np.max(np.abs(r)) < 0.5, "Features and labels are too strongly correlated")

    def test_baseline_performance(self):
        """Test that simple baseline models perform poorly."""
        model = DummyRegressor(strategy='mean')

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='r2')
        self.assertTrue(np.all(scores < 0.001), f"Mean baseline performance is too high, scores: {scores}")

        model = Ridge(alpha=1.0)
        scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='r2')
        self.assertTrue(np.all(scores < 0.05), f"Ridge baseline performance is too high, scores: {scores}")

    def test_knn_performance(self):
        """Test that a KNN model performs well on the dataset."""
        model = KNeighborsRegressor(n_neighbors=5)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='r2')
        self.assertTrue(np.all(scores > 0.2), f"KNN model performance is too low, scores: {scores}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
