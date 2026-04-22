from collections import Counter
from typing import Dict, List, Optional

import unittest
import networkx as nx
import numpy as np
from scipy import stats

from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from percolation_dataset import Node, Seeds, PercolationDataset, GroundTruthFeatures

def ground_truth_1nn_baseline(points: List['Node'], seed: Optional[int] = None) -> float:
    """Returns ground-truth 1-nearest-neighbor mean squared error"""
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    error = []
    for point in points:
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
        self.dataset = PercolationDataset("distribution", seeds=Seeds(graph=0, embed=1, value=2))

    def test_initialization_validation(self):
        """Test that invalid parameter values raise ValueError."""
        with self.assertRaises(ValueError):
            PercolationDataset("invalid_structure") # type: ignore
        with self.assertRaises(ValueError):
            PercolationDataset("custom", create_prob=-0.1)
        with self.assertRaises(ValueError):
            PercolationDataset("custom", create_prob=1.1)
        with self.assertRaises(ValueError):
            PercolationDataset("custom")
        with self.assertRaises(ValueError):
            PercolationDataset("one_cluster", create_prob=0.5)
        with self.assertRaises(ValueError):
            PercolationDataset("distribution", create_prob=0.5)


    def test_construct_structure(self):
        """Test that construct generates the correct number of points and latents."""
        size = 10
        points, latents = self.dataset.construct(size=size)

        # Verify counts
        point_count = len(points)
        cluster_count = len(np.unique([p.cluster_idx for p in points]))
        self.assertEqual(point_count, size, "Incorrect number of points (leaves)")
        self.assertEqual(len(latents), size - cluster_count, "Incorrect number of latents (internal nodes)")

        # Verify node types
        for p in points:
            self.assertEqual(p.node_type, 'point')
        for l in latents.values():
            self.assertEqual(l.node_type, 'latent')

    def test_embed_shapes(self):
        """Test that embed produces outputs with correct shapes."""
        size = 20
        d = 5
        points, latents = self.dataset.construct(size=size)

        X = self.dataset.embed_features(points, latents, d=d)
        self.assertEqual(X.shape, (size, d))

        y = self.dataset.embed_labels(points, latents)
        self.assertEqual(y.shape, (size,))

    def test_input_validation_methods(self):
        """Test validation in construct and embed methods."""
        with self.assertRaises(ValueError):
            self.dataset.construct(size=0)

        points, latents = self.dataset.construct(size=5)
        with self.assertRaises(ValueError):
            self.dataset.embed_features(points, latents, d=0)


    def test_rng_reproducibility_for_same_dataset(self):
        """Test that multiple calls to construct_embed produce identical datasets."""
        ds = PercolationDataset("distribution", seeds=Seeds(graph=10, embed=20, value=30))
        points1, latents1, X1, y1 = ds.construct_embed(size=100, d=10)
        points2, latents2, X2, y2 = ds.construct_embed(size=100, d=10)

        # Compare points
        for p1, p2 in zip(points1, points2):
            self.assertEqual(p1.point_idx, p2.point_idx)
            self.assertEqual(p1.cluster_idx, p2.cluster_idx)
            self.assertEqual(p1.value, p2.value)
            self.assertEqual(p1.level, p2.level)

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

    def test_rng_reproducibility_for_different_datasets(self):
        """Test that using the same RNG seed produces identical datasets."""
        seeds = Seeds(graph=10, embed=20, value=30)

        ds1 = PercolationDataset("distribution", seeds=seeds)
        points1, latents1, X1, y1 = ds1.construct_embed(size=100, d=10)

        ds2 = PercolationDataset("distribution", seeds=seeds)
        points2, latents2, X2, y2 = ds2.construct_embed(size=100, d=10)

        # Compare points
        for p1, p2 in zip(points1, points2):
            self.assertEqual(p1.point_idx, p2.point_idx)
            self.assertEqual(p1.cluster_idx, p2.cluster_idx)
            self.assertEqual(p1.value, p2.value)
            self.assertEqual(p1.level, p2.level)

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

    def test_different_seeds_produce_different_datasets(self):
        """Test that different RNG seeds produce different datasets."""
        size = 100
        d = 10

        ds1 = PercolationDataset("distribution", seeds=Seeds(graph=10, embed=20, value=30))
        _points1, _latents1, X1, y1 = ds1.construct_embed(size=size, d=d)

        ds2 = PercolationDataset("distribution", seeds=Seeds(graph=11, embed=20, value=30))
        _points2, _latents2, X2, y2 = ds2.construct_embed(size=size, d=d)

        ds3 = PercolationDataset("distribution", seeds=Seeds(graph=10, embed=21, value=30))
        _points3, _latents3, X3, y3 = ds3.construct_embed(size=size, d=d)

        ds4 = PercolationDataset("distribution", seeds=Seeds(graph=10, embed=20, value=31))
        _points4, _latents4, X4, y4 = ds4.construct_embed(size=size, d=d)

        # Compare embeddings
        self.assertFalse(np.array_equal(X1, X2), "Embeddings are identical with only different graph seeds")
        self.assertFalse(np.array_equal(X1, X3), "Embeddings are identical with only different embed seeds")
        self.assertTrue(np.array_equal(X1, X4), "Embeddings are not identical with only different value seeds")

        # Compare labels
        self.assertFalse(np.array_equal(y1, y2), "Labels are identical with only different graph seeds")
        self.assertTrue(np.array_equal(y1, y3), "Labels are not identical with only different embed seeds")
        self.assertFalse(np.array_equal(y1, y4), "Labels are identical with only different value seeds")

    @unittest.skip(reason="cyclic coalescent algorithm not consistent across sizes, skip for now")
    def test_overlapping_points_consistent_across_sizes(self):
        """Test that datasets generated with different sizes are consistent for overlapping points."""
        ds = PercolationDataset("distribution", seeds=Seeds(graph=5, embed=6, value=7))
        points_small, latents_small, _X_small, y_small = ds.construct_embed(size=50, d=10)
        points_large, latents_large, _X_large, y_large = ds.construct_embed(size=100, d=10)

        # Create mapping from point_idx to index in large dataset
        point_idx_to_large_idx = {p.point_idx: i for i, p in enumerate(points_large)}

        # Every point in small dataset should be either in large dataset's points or latents
        for i, p_small in enumerate(points_small):
            point_idx = p_small.point_idx
            self.assertTrue((point_idx in point_idx_to_large_idx) ^ (point_idx in latents_large),
                            f"Point {point_idx} not in exactly one of large dataset's points or latents")
            
            if point_idx in point_idx_to_large_idx:
                idx_large = point_idx_to_large_idx[p_small.point_idx]
                p_large = points_large[idx_large]

                # Compare point properties
                self.assertEqual(p_small.cluster_idx, p_large.cluster_idx)
                self.assertEqual(p_small.level, p_large.level)
                self.assertEqual(p_small.value, p_large.value)
                self.assertEqual(p_small.error, p_large.error)

                # Compare labels
                self.assertEqual(y_small[i], y_large[idx_large],
                                f"Labels for point {p_small.point_idx} differ between sizes")
            
            if point_idx in latents_large:
                l_large = latents_large[point_idx]

                # Compare latent properties
                self.assertEqual(p_small.cluster_idx, l_large.cluster_idx)
                self.assertEqual(p_small.value, l_large.value)
                self.assertEqual(p_small.level, l_large.level)

        # Every latent in small dataset should be in large dataset's latents
        for point_idx, latent in latents_small.items():
            self.assertIn(point_idx, latents_large)
            l_large = latents_large[point_idx]

            # Compare latent properties
            self.assertEqual(latent.cluster_idx, l_large.cluster_idx)
            self.assertEqual(latent.value, l_large.value)
            self.assertEqual(latent.level, l_large.level)


    def test_contiguous_cluster_indices(self):
        """Test that cluster indices are contiguous and start from 0."""
        points, _latents = self.dataset.construct(size=1000)
        cluster_indices = sorted(set(p.cluster_idx for p in points))
        expected_indices = list(range(len(cluster_indices)))
        self.assertEqual(cluster_indices, expected_indices,
                         "Cluster indices are not contiguous and starting from 0")
        

    def test_point_depths(self):
        """Test that point depths are correctly assigned based on the hierarchy."""
        points, _latents = self.dataset.construct(size=1000)
        for point in points:
            expected_depth = 0
            current = point
            while current.parents:
                expected_depth += 1
                current = next(iter(current.parents)) # Get the single parent latent
            self.assertEqual(point.depth, expected_depth,
                             f"Point {point.point_idx} has incorrect depth {point.depth}, expected {expected_depth}")

    def test_neighbor_graph_is_tree(self):
        """Test that the graph formed by neighboring points is a tree."""
        points, _ = self.dataset.construct(size=1000)
        start_idx, end_idx = 0, 0
        while start_idx < len(points):
            while end_idx < len(points) and points[end_idx].cluster_idx == points[start_idx].cluster_idx:
                end_idx += 1
            adj = {points[i].point_idx: [] for i in range(start_idx, end_idx)}
            for point in points[start_idx:end_idx]:
                for neighbor in point.neighbors:
                    adj[point.point_idx].append(neighbor.point_idx)
            G = nx.Graph(adj) # type: ignore[arg-type]
            self.assertTrue(nx.is_tree(G), "Neighbor graph is not a tree (it should be connected and acyclic)")
            start_idx = end_idx

    def test_isomorphism_class_frequencies(self):
        """Test that the frequencies of isomorphism classes of small clusters match the expected distribution."""

        # Shapes on n=6 and their labeled-tree counts (= 6! / |Aut|).
        # Sum = 6^4 = 1296 (Cayley). Shape key is sorted degree sequence, with a
        # leaf-neighbor count of the unique deg-3 vertex to split the two spiders.
        _TOTAL = 1296
        _EXPECTED_COUNTS = {
            (1, 1, 2, 2, 2, 2):      360,   # P_6                  |Aut| = 2
            (1, 1, 1, 1, 1, 5):        6,   # star K_{1,5}         |Aut| = 120
            (1, 1, 1, 1, 2, 4):      120,   # broom (K_{1,4}+edge) |Aut| = 6
            (1, 1, 1, 1, 3, 3):       90,   # double star          |Aut| = 8
            ((1, 1, 1, 2, 2, 3), 1): 360,   # spider S(2,2,1)      |Aut| = 2
            ((1, 1, 1, 2, 2, 3), 2): 360,   # spider S(3,1,1)      |Aut| = 2
        }
        assert sum(_EXPECTED_COUNTS.values()) == _TOTAL

        size = 6
        n_datasets = 3000
        alpha = 1e-3

        obs = {}
        for i in range(n_datasets):
            points, _ = PercolationDataset("one_cluster", seeds=Seeds(graph=i)).construct(size=size)
            edges = [(p.point_idx, nbr.point_idx) for p in points for nbr in p.neighbors] # duplicates edges
            degrees = {p.point_idx: len(p.neighbors) for p in points}
            ds = tuple(sorted(degrees.values()))
            if ds != (1, 1, 1, 2, 2, 3):
                obs[ds] = obs.get(ds, 0) + 1
            else:
                c = next(k for k, v in degrees.items() if v == 3)
                leaf_nbrs = sum(1 for u, v in edges if u == c and degrees[v] == 1)
                obs[(ds, leaf_nbrs)] = obs.get((ds, leaf_nbrs), 0) + 1
        
        keys = list(_EXPECTED_COUNTS)
        f_obs = np.fromiter((obs.get(k, 0) for k in keys), float, len(keys))
        f_exp = np.fromiter((_EXPECTED_COUNTS[k] * n_datasets / _TOTAL for k in keys), float, len(keys))
        chi2, p = stats.chisquare(f_obs, f_exp)
        self.assertGreater(p, alpha, f"Observed isomorphism class frequencies {f_obs} deviate significantly from expected distribution {f_exp}, (chi2={chi2:.2f}, p={p:.4e})")


    def test_cluster_hierarchy_is_directed_tree(self):
        """Test that parent-child relationships form a directed tree (arborescence)."""
        points, latents = PercolationDataset("one_cluster", seeds=Seeds(graph=0)).construct(size=50)
        G = nx.DiGraph()

        # Collect all nodes involved
        for node in points:
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
        for node in points:
            for child in node.children:
                G.add_edge(node.point_idx, child.point_idx)
        for node in latents.values():
            for child in node.children:
                G.add_edge(node.point_idx, child.point_idx)

        # Check if the structure is a branching (a forest of trees directed away from a root)
        self.assertTrue(nx.is_branching(G), "Hierarchy is not a branching (directed forest)")

    def test_embedding_single_cluster(self):
        """Test that embedding a single cluster works correctly."""
        points, latents, X, y = PercolationDataset("one_cluster", seeds=Seeds(graph=0, embed=1)).construct_embed(size=50, d=10)
        # Check for NaNs and Infs in embeddings
        self.assertFalse(np.isnan(X).any(), "Embeddings contain NaN values")
        self.assertFalse(np.isinf(X).any(), "Embeddings contain Inf values")

        # Check for NaNs and Infs in labels
        self.assertFalse(np.isnan(y).any(), "Labels contain NaN values")
        self.assertFalse(np.isinf(y).any(), "Labels contain Inf values")


    def test_ground_truth_features_n_features(self):
        """Test that setting n_features has correct effect."""
        size = 1000
        points, latents = self.dataset.construct(size=size)
        gt_features = GroundTruthFeatures(points, latents)
        # Default n_features equal n_latents
        X_gt = gt_features.get_latent_features()
        assert X_gt.shape is not None
        self.assertEqual(X_gt.shape[1], gt_features.n_latents,
                         "Ground truth feature matrix has incorrect number of columns")
        # Setting n_features yields correct shape
        n_features = gt_features.n_latents // 2
        X_gt = gt_features.get_latent_features(n_features=n_features)
        assert X_gt.shape is not None
        self.assertEqual(X_gt.shape[1], n_features,
                         "Ground truth feature matrix has incorrect number of columns when n_features is set")
        # Setting n_features less than 1 raises error
        with self.assertRaises(ValueError):
            gt_features.get_latent_features(n_features=0)
        # Setting n_features greater than n_latents raises error
        with self.assertRaises(ValueError):
            gt_features.get_latent_features(n_features=gt_features.n_latents + 1)

    def test_ground_truth_feature_matrix(self):
        """Test that ground truth feature matrix has correct shape and content."""
        size = 1000
        points, latents = self.dataset.construct(size=size)
        gt_features = GroundTruthFeatures(points, latents)
        X_gt = gt_features.get_latent_features()

        assert X_gt.shape is not None
        self.assertEqual(X_gt.shape[0], size, "Ground truth feature matrix has incorrect number of rows")
        self.assertEqual(X_gt.shape[1], gt_features.n_latents,"Ground truth feature matrix has incorrect number of columns")

        # Verify that each row has exactly one non-zero entry per latent node it is associated with
        for pidx in range(size):
            row = X_gt.getrow(pidx).toarray().flatten()
            non_zero_indices = np.nonzero(row)[0]
            expected_latent_indices = gt_features.pidx2lidx[pidx]
            self.assertEqual(set(non_zero_indices), set(expected_latent_indices),
                             f"Row {pidx} of ground truth feature matrix has incorrect non-zero entries")
            
    def test_get_summary_features(self):
        """Test that get_summary_features returns correct per-point metadata aligned with feature matrix rows."""
        size = 1000
        points, latents = self.dataset.construct(size=size)
        gt_features = GroundTruthFeatures(points, latents)
        metadata = gt_features.get_summary_features()

        self.assertIsInstance(metadata, dict)
        self.assertListEqual(list(metadata.keys()), ['cluster_id', 'cluster_size', 'latent_tree_height'])
        for value in metadata.values():
            self.assertEqual(len(value), size)

        counts = Counter(p.cluster_idx for p in points)
        for i, point in enumerate(points):
            self.assertEqual(metadata['cluster_id'][i], point.cluster_idx)
            self.assertEqual(metadata['cluster_size'][i], counts[point.cluster_idx])
            self.assertEqual(metadata['latent_tree_height'][i], len(gt_features.pidx2lidx[i]))

    def test_get_latent_features_use_values(self):
        """Test that get_latent_features(use_values=True) stores correct z_u values."""
        size = 100
        points, latents, X, y = self.dataset.construct_embed(size=size, d=16)
        gt_features = GroundTruthFeatures(points, latents)

        X_binary = gt_features.get_latent_features(use_values=False)
        X_values = gt_features.get_latent_features(use_values=True)

        # Sparsity pattern must be identical
        self.assertEqual(X_binary.nnz, X_values.nnz)

        # Binary mode still returns 1s
        self.assertTrue(np.all(X_binary.data == 1))

        # Check z_u values are correct for each latent
        for lidx in range(gt_features.n_latents):
            latent_node = gt_features.latents[gt_features.lidx2latent[lidx]]
            if latent_node.parents:
                parent = list(latent_node.parents)[0]
                expected_z_u = latent_node.value - parent.value
            else:
                expected_z_u = latent_node.value
            col = X_values.getcol(lidx)
            nonzero_vals = col.data
            if len(nonzero_vals) > 0:
                # Note: np.testing.assert_allclose would give richer failure messages
                self.assertTrue(np.allclose(nonzero_vals, expected_z_u),
                    msg=f"Incorrect z_u value for latent {lidx}")

        # Sum of z_u telescopes to the deepest ancestor latent's value
        # (point.value also includes the leaf's own Gaussian draw, not stored in features)
        for pidx, point in enumerate(points):
            row = X_values.getrow(pidx)
            ancestors = gt_features.pidx2lidx[pidx]
            if ancestors:
                deepest_latent = gt_features.latents[gt_features.lidx2latent[ancestors[-1]]]
                self.assertAlmostEqual(row.sum(), deepest_latent.value, places=10,
                    msg=f"Sum of z_u for point {pidx} does not equal deepest ancestor latent value")

    def test_nearest_neighbor_ground_truth(self):
        """Test that 1-NN using the ground truth points matches the embedded dataset."""
        points, _latents, X, y = self.dataset.construct_embed(size=10000, d=128)
        mse_gt_1nn = ground_truth_1nn_baseline(points, seed=0)
        nn = NearestNeighbors(n_neighbors=2).fit(X)
        _distances, indices = nn.kneighbors(X)

        # First neighbor is the point itself (distance 0), so exclude it
        neighbor_index = indices[:, 1:]  # shape (n_samples, k)
        pred = y[neighbor_index].squeeze()
        mse_data_1nn = np.mean((y - pred)**2)
        self.assertAlmostEqual(mse_gt_1nn, mse_data_1nn, delta=0.05,
                               msg="1-NN MSE on ground truth points does not match that on embedded data")
        
    def test_ratio_effect_on_loss(self):
        """Test that increasing ratio increases baseline MSE."""
        size = 1000
        points_01, _latents01, _X01, _y01 = PercolationDataset("distribution", ratio=0.1, seeds=Seeds(graph=0)).construct_embed(size=size, d=16)
        points_05, _latents05, _X05, _y05 = PercolationDataset("distribution", ratio=0.5, seeds=Seeds(graph=0)).construct_embed(size=size, d=16)
        points_09, _latents09, _X09, _y09 = PercolationDataset("distribution", ratio=0.9, seeds=Seeds(graph=0)).construct_embed(size=size, d=16)

        mse_01 = ground_truth_1nn_baseline(points_01, seed=42)
        mse_05 = ground_truth_1nn_baseline(points_05, seed=42)
        mse_09 = ground_truth_1nn_baseline(points_09, seed=42)

        self.assertLess(mse_01, mse_05, "MSE with ratio=0.1 should be less than MSE with ratio=0.5")
        self.assertLess(mse_05, mse_09, "MSE with ratio=0.5 should be less than MSE with ratio=0.9")

    @unittest.skip(reason="cyclic coalescent algorithm not consistent across sizes, skip for now")
    def test_cluster_consistency_across_size(self):
        """Test that clusters are consistent when generating datasets of different sizes."""
        d = 100
        size = 1000

        dataset = PercolationDataset("one_cluster", ratio=0.5, seeds=Seeds(graph=20, embed=21, value=22))
        _points_large, _latents_large, X_large, y_large = dataset.construct_embed(size, d)
        nn = NearestNeighbors(n_neighbors=1).fit(X_large)

        size_deltas = (1, 10, 100)
        bounds = (1e-5, 1e-4, 1e-3) # Loose bounds, MSE can vary significantly with different seeds

        for size_delta, bound in zip(size_deltas, bounds):
            _points_small, _latents_small, X_small, y_small = dataset.construct_embed(size - size_delta, d)
            _distances, indices = nn.kneighbors(X_small)
            pred = y_large[indices].squeeze()
            mse = np.mean((y_small - pred)**2)
            self.assertLessEqual(mse, bound, f"Mean squared error {mse} not less than bound {bound}")

    @unittest.skip(reason="cyclic coalescent algorithm not consistent across sizes, skip for now")
    def test_distribution_consistency_across_size(self):
        """Test that distributions are consistent when generating datasets of different sizes."""
        d = 100
        size = 3000
        size_delta = size // 10
        bound = 0.03 # Fairly tight bound, variance is small across seeds

        dataset = PercolationDataset("distribution", ratio=0.5, seeds=Seeds(graph=20, embed=21, value=22))
        _points_large, _latents_large, X_large, y_large = dataset.construct_embed(size, d)
        _points_small, _latents_small, X_small, y_small = dataset.construct_embed(size - size_delta, d)

        nn = NearestNeighbors(n_neighbors=1).fit(X_large)
        _distances, indices = nn.kneighbors(X_small)
        pred = y_large[indices].squeeze()
        mse = np.mean((y_small - pred)**2)
        self.assertLessEqual(mse, bound, f"Mean squared error {mse} not less than bound {bound}")

class BaseTestWrapper: # Wrapper prevents unittest from trying to run DatasetPropertiesTests directly
    class DatasetPropertiesTests(unittest.TestCase):
        """Tests that validate the properties of the generated dataset, such as degree distribution, embedding properties, and data integrity."""

        # Set up by subclasses in setUpClass to avoid expensive dataset generation for each test method
        dataset: PercolationDataset
        size: int
        d: int
        points: List['Node']
        latents: Dict[int, 'Node']
        cluster_sizes: Counter
        X: np.ndarray
        y: np.ndarray
        ground_truth_features: GroundTruthFeatures

        def test_degree_distribution(self):
            """Test that degree distribution is well fit by shifted Poisson distribution using KL divergence."""
            # Get degrees, skipping small clusters
            size_threshold = 100
            degrees = [len(node.neighbors) for node in self.points if self.cluster_sizes[node.cluster_idx] >= size_threshold]

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

        def test_embedding_distribution_mean(self):
            """Test that the embeddings have approximately zero mean."""
            self.assertLessEqual(np.mean(np.abs(self.X.mean(axis=0))), 0.1, f"Typical embedding mean is not close to 0, mean of means: {np.mean(np.abs(self.X.mean(axis=0)))}")
            self.assertTrue(np.allclose(self.X.mean(axis=0), 0.0, atol=0.25), f"Embeddings do not have mean close to 0, means: {self.X.mean(axis=0)}")

        def test_embedding_distribution_std(self):
            """Test that the embeddings have consistent standard deviations."""
            relative_std = self.X.std(axis=0).std() / self.X.std(axis=0).mean()
            self.assertLess(relative_std, 0.1, f"Embeddings do not have consistent standard deviations, relative std: {relative_std}")

        def test_label_distribution_mean(self):
            """Test that the regression labels have approximately zero mean."""
            self.assertAlmostEqual(self.y.mean(), 0.0, delta=0.1, msg=f"Labels do not have mean close to 0, mean: {self.y.mean()}")
        
        def test_label_distribution_std(self):
            """Test that the regression labels have approximately unit standard deviation."""
            self.assertAlmostEqual(self.y.std(), 1.0, delta=0.1, msg=f"Labels do not have standard deviation close to 1, std: {self.y.std()}")

        def test_feature_label_correlation(self):
            """Test that the features and labels are weakly correlated."""
            r = np.array([np.corrcoef(self.X[:, j], self.y)[0, 1] for j in range(self.X.shape[1])])
            self.assertTrue(np.max(np.abs(r)) > 0.01, f"Features and labels are not correlated, correlation coefficients: {r}")
            self.assertTrue(np.max(np.abs(r)) < 0.5, f"Features and labels are too strongly correlated, correlation coefficients: {r}")

        def test_mean_performance(self):
            """Test that simple mean baseline model performs poorly."""
            cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
            model = DummyRegressor(strategy='mean')
            score = -cross_val_score(model, self.X, self.y, cv=cv, scoring='neg_mean_squared_error')
            self.assertGreater(score, 0.8, f"Mean baseline performance is too good, score: {score}")

        def test_ridge_performance(self):
            """Test that simple ridge regression model performs poorly."""
            cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
            model = Ridge(alpha=1.0)
            score = -cross_val_score(model, self.X, self.y, cv=cv, scoring='neg_mean_squared_error')
            self.assertGreater(score, 0.3, f"Ridge baseline performance is too good, score: {score}")

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
            n_sample_points = 250
            rng = np.random.default_rng(42)
            sample_inds = rng.choice(self.size, size=n_sample_points, replace=False)
            X_sq = np.sum(self.X**2, axis=1)
            point_idx2idx = {p.point_idx: i for i, p in enumerate(self.points)}
            for idx in sample_inds:
                point = self.points[idx]
                neighbors = [p for p in point.neighbors]
                neighbor_inds = np.array([point_idx2idx[p.point_idx] for p in neighbors], dtype=int)
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
            point_labels = np.array([p.value + p.error for p in self.points])
            self.assertTrue(np.allclose(point_labels, self.y), "Point values do not match labels")


class TestOneCluster(BaseTestWrapper.DatasetPropertiesTests, unittest.TestCase):
    """Validate the properties of the generated dataset."""

    @classmethod
    def setUpClass(cls):
        print("Setting up dataset for TestOneCluster...")
        cls.dataset = PercolationDataset("one_cluster", seeds=Seeds(graph=10, embed=11, value=12))
        cls.size = 10000
        cls.d = 128
        points, latents, X, y = cls.dataset.construct_embed(size=cls.size, d=cls.d)
        cls.points = points
        cls.latents = latents
        cls.cluster_sizes = Counter([p.cluster_idx for p in points])
        cls.X = X
        cls.y = y
        cls.ground_truth_features = GroundTruthFeatures(points, latents)


class TestLargeDistribution(BaseTestWrapper.DatasetPropertiesTests, unittest.TestCase):
    """Validate the properties of the generated dataset."""

    @classmethod
    def setUpClass(cls):
        print("Setting up dataset for TestLargeDistribution...")
        cls.dataset = PercolationDataset("distribution", seeds=Seeds(graph=42, embed=43, value=44))
        cls.size = 100000
        cls.d = 128
        points, latents, X, y = cls.dataset.construct_embed(size=cls.size, d=cls.d)
        cls.points = points
        cls.latents = latents
        cls.cluster_sizes = Counter([p.cluster_idx for p in points])
        cls.X = X
        cls.y = y
        cls.ground_truth_features = GroundTruthFeatures(points, latents)

    def test_size_distribution(self):
        """Test that size distribution is well fit by a power law using maximum likelihood."""
        # Get sizes of clusters
        size_threshold = 10
        sizes = np.array([size for size in self.cluster_sizes.values() if size >= size_threshold])

        # Estimate power-law exponent using MLE
        alpha = 1 + len(sizes)*np.sum(np.log(sizes/size_threshold))**-1
        sigma = (alpha - 1)/np.sqrt(len(sizes))

        # Verify estimated alpha is close to the expected exponent
        expected_alpha = 2.5
        sigma_threshold = 2.0
        self.assertAlmostEqual(alpha, expected_alpha, delta=sigma_threshold*sigma,
                            msg=f"Estimated alpha {alpha:.4f} deviates more than {sigma_threshold} sigma "
                                f"{sigma:.4f} from expected {expected_alpha}")
        

if __name__ == '__main__':
    unittest.main(verbosity=2)
