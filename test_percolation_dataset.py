import unittest
import networkx as nx
import numpy as np
from scipy import stats

from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from percolation_dataset import PercolationDataset, GroundTruthFeatures

def ground_truth_1nn_baseline(points, rng):
    """Returns ground-truth 1-nearest-neighbor mean squared error"""
    error = []
    for cluster_points in points:
        for point in cluster_points.values():
            if len(point.neighbors) == 0:
                error.append(np.sqrt(2))
            else:
                neighbors = sorted(list(point.neighbors), key=lambda p: p.point_idx)
                idx = rng.choice(len(neighbors))
                neighbor = neighbors[idx]
                error.append((neighbor.value + neighbor.error) - (point.value + point.error))
    error = np.array(error)
    mse = (error**2).mean()
    return mse

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
        points1, latents1, X1, y1 = ds1.construct_embed(size=100, d=10)

        rng2 = np.random.default_rng(123)
        ds2 = PercolationDataset(rng=rng2)
        points2, latents2, X2, y2 = ds2.construct_embed(size=100, d=10)

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

        # Compare embeddings
        self.assertTrue(np.array_equal(X1, X2), "Embeddings are not reproducible with the same RNG seed")

        # Compare labels
        self.assertTrue(np.array_equal(y1, y2), "Labels are not reproducible with the same RNG seed")


    def test_neighbor_graph_is_tree(self):
        """Test that the graph formed by neighboring points is a tree."""
        points, _ = self.dataset.construct(size=1000)
        for cluster_points in points:
            adj = {idx: [] for idx in cluster_points}
            for point in cluster_points.values():
                for neighbor in point.neighbors:
                    adj[point.point_idx].append(neighbor.point_idx)
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

    def test_ground_truth_features_n_features(self):
        """Test that setting n_features has correct effect."""
        size = 1000
        points, latents = self.dataset.construct(size=size)
        gt_features = GroundTruthFeatures(points, latents)
        # Default n_features equal n_latents
        X_gt = gt_features.get_features()
        assert X_gt.shape is not None
        self.assertEqual(X_gt.shape[1], gt_features.n_latents,
                         "Ground truth feature matrix has incorrect number of columns")
        # Setting n_features yields correct shape
        n_features = gt_features.n_latents // 2
        X_gt = gt_features.get_features(n_features=n_features)
        assert X_gt.shape is not None
        self.assertEqual(X_gt.shape[1], n_features,
                         "Ground truth feature matrix has incorrect number of columns when n_features is set")
        # Setting n_features less than 1 raises error
        with self.assertRaises(ValueError):
            gt_features.get_features(n_features=0)
        # Setting n_features greater than n_latents raises error
        with self.assertRaises(ValueError):
            gt_features.get_features(n_features=gt_features.n_latents + 1)

    def test_ground_truth_feature_matrix(self):
        """Test that ground truth feature matrix has correct shape and content."""
        size = 1000
        points, latents = self.dataset.construct(size=size)
        gt_features = GroundTruthFeatures(points, latents)
        X_gt = gt_features.get_features()

        assert X_gt.shape is not None
        self.assertEqual(X_gt.shape[0], size, "Ground truth feature matrix has incorrect number of rows")
        self.assertEqual(X_gt.shape[1], gt_features.n_latents,
                         "Ground truth feature matrix has incorrect number of columns")

        # Verify that each row has exactly one non-zero entry per latent node it is associated with
        for pidx in range(size):
            row = X_gt.getrow(pidx).toarray().flatten()
            non_zero_indices = np.nonzero(row)[0]
            expected_latent_indices = gt_features.pidx2lidx[pidx]
            self.assertEqual(set(non_zero_indices), set(expected_latent_indices),
                             f"Row {pidx} of ground truth feature matrix has incorrect non-zero entries")
            
    def test_nearest_neighbor_ground_truth(self):
        """Test that 1-NN using the ground truth points matches the embedded dataset."""
        points, latents, X, y = self.dataset.construct_embed(size=10000, d=128)
        mse_gt_1nn = ground_truth_1nn_baseline(points, self.rng)
        nn = NearestNeighbors(n_neighbors=2).fit(X)
        distances, indices = nn.kneighbors(X)

        # First neighbor is the point itself (distance 0), so exclude it
        neighbor_indices = indices[:, 1:]  # shape (n_samples, k)
        neighbor_distances = distances[:, 1:]

        # Simple average of neighbor targets
        pred = y[neighbor_indices].mean(axis=1)
        mse_data_1nn = np.mean((y - pred)**2)
        self.assertAlmostEqual(mse_gt_1nn, mse_data_1nn, delta=0.005,
                               msg="1-NN MSE on ground truth points does not match that on embedded data")
        
    def test_ratio_effect_on_loss(self):
        """Test that increasing ratio increases baseline MSE."""
        size = 1000
        points_01, _latents01, _X01, _y01 = PercolationDataset(rng=self.rng, value_generator_kwargs={'ratio': 0.1}).construct_embed(size=size, d=16)
        points_05, _latents05, _X05, _y05 = PercolationDataset(rng=self.rng, value_generator_kwargs={'ratio': 0.5}).construct_embed(size=size, d=16)
        points_09, _latents09, _X09, _y09 = PercolationDataset(rng=self.rng, value_generator_kwargs={'ratio': 0.9}).construct_embed(size=size, d=16)

        mse_01 = ground_truth_1nn_baseline(points_01, self.rng)
        mse_05 = ground_truth_1nn_baseline(points_05, self.rng)
        mse_09 = ground_truth_1nn_baseline(points_09, self.rng)

        self.assertLess(mse_01, mse_05, "MSE with ratio=0.1 should be less than MSE with ratio=0.5")
        self.assertLess(mse_05, mse_09, "MSE with ratio=0.5 should be less than MSE with ratio=0.9")


class TestPercolationDatasetProperties(unittest.TestCase):
    """Validate the properties of the generated dataset."""

    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(42)
        cls.dataset = PercolationDataset(rng=cls.rng)
        cls.size = 100000
        cls.d = 128
        points, latents, X, y = cls.dataset.construct_embed(size=cls.size, d=cls.d)
        cls.points = points
        cls.latents = latents
        cls.X = X
        cls.y = y
        cls.ground_truth_features = GroundTruthFeatures(points, latents)

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
        relative_std = self.X.std(axis=0).std() / self.X.std(axis=0).mean()
        self.assertLess(relative_std, 0.1, "Embeddings do not have consistent standard deviations")

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
        cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

        model = DummyRegressor(strategy='mean')
        score = -cross_val_score(model, self.X, self.y, cv=cv, scoring='neg_mean_squared_error')
        self.assertGreater(score, 0.9, f"Mean baseline performance is too good, score: {score}")

        model = Ridge(alpha=1.0)
        score = -cross_val_score(model, self.X, self.y, cv=cv, scoring='neg_mean_squared_error')
        self.assertGreater(score, 0.9, f"Ridge baseline performance is too good, score: {score}")

    def test_ground_truth_features(self):
        """Test that ground truth features are correctly computed."""
        self.assertTrue(all(a <= b for a, b in zip(self.ground_truth_features.pidx2lidx.keys(),
                                                   list(self.ground_truth_features.pidx2lidx.keys())[1:])),
                        "pidx2lidx keys are not in sorted order")
        for lst in self.ground_truth_features.pidx2lidx.values():
            self.assertTrue(all(a <= b for a, b in zip(lst, lst[1:])), "pidx2lidx values are not in sorted order")
            self.assertEqual(len(np.unique(lst)), len(lst), "pidx2lidx values are not unique")

        self.assertTrue(all(a <= b for a, b in zip(self.ground_truth_features.lidx2pidx.keys(),
                                                   list(self.ground_truth_features.lidx2pidx.keys())[1:])),
                        "lidx2pidx keys are not in sorted order")
        for lst in self.ground_truth_features.lidx2pidx.values():
            self.assertTrue(all(a <= b for a, b in zip(lst, lst[1:])), "lidx2pidx values are not in sorted order")
            self.assertEqual(len(np.unique(lst)), len(lst), "lidx2pidx values are not unique")

        self.assertEqual(len(self.ground_truth_features.pidx2lidx), self.ground_truth_features.n_samples,
                         "pidx2lidx length does not match number of samples")
        self.assertEqual(len(self.ground_truth_features.lidx2pidx), self.ground_truth_features.n_latents,
                         "lidx2pidx length does not match number of latents")
        self.assertEqual(len(self.ground_truth_features.latent2lidx), self.ground_truth_features.n_latents,
                         "latent2lidx length does not match number of latents")
        self.assertEqual(len(self.ground_truth_features.lidx2latent), self.ground_truth_features.n_latents,
                         "lidx2latent length does not match number of latents")
        for k, v in self.ground_truth_features.latent2lidx.items():
            self.assertEqual(k, self.ground_truth_features.lidx2latent[self.ground_truth_features.latent2lidx[k]],
                             "latent2lidx and lidx2latent do not match")
        for k, v in self.ground_truth_features.lidx2latent.items():
            self.assertEqual(k, self.ground_truth_features.latent2lidx[self.ground_truth_features.lidx2latent[k]],
                             "lidx2latent and latent2lidx do not match")
                        
    def test_neighbor_graph_matches_embeddings(self):
        """Test that nearest neighbors in embeddings correspond to neighbor relationships in the graph."""
        n_sample_points = 1000
        sample_inds = self.rng.choice(self.size, size=n_sample_points, replace=False)
        X_sq = np.sum(self.X**2, axis=1)
        flat_points = [point for cluster_points in self.points for point in cluster_points.values()]
        point_idx2flat_idx = {p.point_idx: i for i, p in enumerate(flat_points)}
        for idx in sample_inds:
            point = flat_points[idx]
            neighbors = [p for p in point.neighbors]
            neighbor_inds = np.array([point_idx2flat_idx[p.point_idx] for p in neighbors], dtype=int)
            is_neighbor = np.zeros(self.size, dtype=bool)
            is_neighbor[neighbor_inds] = True
            is_neighbor[idx] = True # exclude self
            dists_sq = X_sq[idx] + X_sq - 2*self.X[idx] @ self.X.T
            max_neighbor = dists_sq[is_neighbor].max()
            min_non_neighbor = dists_sq[~is_neighbor].min()
            self.assertGreater(min_non_neighbor, max_neighbor,
                               f"Nearest embedded neighbors of point {point.point_idx} are not nearest graph neighbors")

    def test_point_values_match_labels(self):
        """Test that point values match labels."""
        point_labels = []
        for cluster_points in self.points:
            for p in cluster_points.values():
                point_labels.append(p.value + p.error)
        point_labels = np.array(point_labels)
        self.assertTrue(np.allclose(point_labels, self.y), "Point values do not match labels")

if __name__ == '__main__':
    unittest.main(verbosity=2)
