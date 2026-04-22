from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import groupby, islice
from operator import attrgetter
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
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


@dataclass(frozen=True)
class Seeds:
    """Seeds for reproducibility of different random processes in dataset generation.

        graph_seed: Seed for the random number generator used in graph construction.
        embed_seed: Seed for the random number generator used in embedding.
        value_seed: Seed for the random number generator used in value generation.
    """
    graph: Optional[int] = None
    embed: Optional[int] = None
    value: Optional[int] = None


class PercolationDataset:
    """Generates a percolation dataset."""    

    def __init__(self, mode: Literal["one_cluster", "distribution", "custom"], create_prob: Optional[float] = None, ratio: float = 0.9, seeds: Optional[Seeds] = None):
        """Initializes the dataset generator.
        Args:
            mode: The mode of dataset generation, either "one_cluster", "distribution", or "custom". Determines create_prob.
            create_prob: Probability of creating a new cluster instead of splitting an existing one (only used in "custom" mode).
            ratio: The ratio for value generation (only used in "custom" mode).
            seeds: Seeds for reproducibility of different random processes.
        """
        
        self.mode = mode
        self.ratio = ratio
        self.seeds = seeds if seeds is not None else Seeds()

        if self.mode not in ("one_cluster", "distribution", "custom"):
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.mode == "custom" and create_prob is None:
            raise ValueError("create_prob must be specified in custom mode")
        if self.mode != "custom" and create_prob is not None:
            raise ValueError("create_prob should not be specified in non-custom mode")
        if self.mode == "one_cluster":
            self.create_prob = 0.0
        elif self.mode == "distribution":
            self.create_prob = 1/3
        elif self.mode == "custom":
            self.create_prob = create_prob

        if not (0 <= self.create_prob <= 1): # type: ignore
            raise ValueError(f"create_prob must be between 0 and 1, got {self.create_prob}")

    
    def _cyclic_coalescent(self, points: List['Node'], rng: np.random.Generator) -> Dict[int, 'Node']:
        """Constructs a percolation cluster using a cyclic coalescent algorithm."""

        size = len(points)
        if size == 1:
            return {}

        cluster_idx = points[0].cluster_idx
        latents: Dict[int, Node] = {}

        # Set up RNG
        perm_rng, u_rng, v_rng = rng.spawn(3)

        # Random initial arrangement in the cyclic array
        perm_rng.shuffle(points)

        # Cyclic linked list of active subtrees
        nxt = list(range(1, size)) + [0]
        st_size = [1] * size

        # Union-find: position in array to subtree representative
        uf_parent = list(range(size))
        uf_rank   = [0] * size
        ll_of_root = list(range(size))

        def find(x):
            while uf_parent[x] != x:
                uf_parent[x] = uf_parent[uf_parent[x]]
                x = uf_parent[x]
            return x

        def union(a, b, survivor):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if uf_rank[ra] < uf_rank[rb]:
                ra, rb = rb, ra
            uf_parent[rb] = ra
            if uf_rank[ra] == uf_rank[rb]:
                uf_rank[ra] += 1
            ll_of_root[ra] = survivor

        # Binary latent tree representative for each linked-list node
        btree_rep = [points[i] for i in range(size)]

        # Generate all random variables at once for efficiency
        all_u = u_rng.integers(size, size=size - 1)
        all_v_frac = v_rng.random(size - 1)

        for step in range(size - 1):
            u_pos = all_u[step]
            ll_a  = ll_of_root[find(u_pos)]
            ll_b  = nxt[ll_a]

            v_pos = (ll_b + int(all_v_frac[step] * st_size[ll_b])) % size

            points[u_pos].neighbors.add(points[v_pos])
            points[v_pos].neighbors.add(points[u_pos])

            # hack to ensure unique latent node indices that don't overlap with point indices
            latent_idx_base = 10_000_000_000 + cluster_idx * 1_000_000_000
            internal_point_idx = latent_idx_base - step # iterate backwards since we're working up the tree from the leaves
            child_a, child_b = btree_rep[ll_a], btree_rep[ll_b]
            internal = Node(internal_point_idx, cluster_idx, node_type='latent', children=[child_a, child_b])
            latents[internal_point_idx] = internal
            child_a.parents.add(internal)
            child_b.parents.add(internal)

            st_size[ll_a] += st_size[ll_b]
            btree_rep[ll_a] = internal

            nxt[ll_a] = nxt[ll_b]

            union(u_pos, v_pos, ll_a)

        # Breadth-first search to set latent node depths
        latents = dict(reversed(latents.items())) # reverse insertion order
        _point_idx, root = next(iter(latents.items()))
        queue = deque([(root, 0)])
        while queue:
            node, depth = queue.popleft()
            node.depth = depth
            for child in node.children:
                queue.append((child, depth + 1))

        return latents


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
        
        # Set up RNG
        ss = np.random.SeedSequence(self.seeds.graph) if self.seeds.graph is not None else np.random.SeedSequence()
        create_seed, grow_seed = ss.spawn(2)
        create_rng = np.random.default_rng(seed=create_seed)
        grow_rng = np.random.default_rng(seed=grow_seed)

        point_idx: int = 0
        cluster_idx: int = 0
        points: List[Node] = []
        latents: Dict[int, Node] = {}

        # Generate points assigned to clusters following preferential attachment 
        new_cluster = create_rng.random(size) < self.create_prob # type: ignore
        u = grow_rng.random(size)

        for i in range(size):
            if new_cluster[i] or point_idx == 0:
                node = Node(point_idx, cluster_idx, level=i)
                points.append(node)
                cluster_idx += 1
            else:
                grow_cluster_idx = points[int(u[i] * point_idx)].cluster_idx
                node = Node(point_idx, grow_cluster_idx, level=i)
                points.append(node)
            point_idx += 1

        # Apply cyclic coalescent to generate latent binary trees and connect neighboring points within each cluster
        points.sort(key=lambda p: (p.cluster_idx, p.point_idx))
        streams = ss.spawn(cluster_idx)
        for cluster_idx, group in groupby(points, key=attrgetter('cluster_idx')):
            cluster_points = list(group)
            cluster_latents = self._cyclic_coalescent(cluster_points, rng=np.random.default_rng(seed=streams[cluster_idx]))
            latents.update(cluster_latents)

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
        seed_seq = np.random.SeedSequence(self.seeds.embed) if self.seeds.embed is not None else np.random.SeedSequence()
        cluster_vec_seed, cluster_uniform_seed, cluster_root_seed, latent_dir_seed = seed_seq.spawn(4)

        # Compute base embedding direction for each cluster
        cluster_vec_rng = np.random.default_rng(seed=cluster_vec_seed)
        cluster_vectors = {}
        for cluster_idx in np.unique([point.cluster_idx for point in points]):
            vec = cluster_vec_rng.normal(size=d)
            cluster_vectors[cluster_idx] = vec / np.linalg.norm(vec)

        # Assign directions using latents to embed points self-consistently with scale
        point_vectors = {}
        latent_dir_rng = np.random.default_rng(seed=latent_dir_seed)
        for point_idx, latent in latents.items():
            if point_idx not in point_vectors:
                point_vectors[point_idx] = cluster_vectors[latent.cluster_idx]
            u = latent_dir_rng.normal(size=d)
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

        # Retrieve the top-level latent for each cluster
        cluster_latents = {}
        for point_idx, latent in latents.items():
            if not latent.parents:
                cluster_latents[latent.cluster_idx] = point_idx

        while start_idx < size:
            # Find boundaries of current cluster
            while end_idx < size and points[end_idx].cluster_idx == points[start_idx].cluster_idx:
                end_idx += 1

            # For a single-point cluster with no latents, assign its vector and root node directly
            if end_idx - start_idx == 1:
                point_vectors[points[start_idx].point_idx] = cluster_vectors[points[start_idx].cluster_idx]
                root = points[start_idx].point_idx
            else:
                # Randomly choose root following the latent hierarchy for consistency across dataset sizes
                ss = np.random.SeedSequence(entropy=cluster_root_seed.entropy,
                                            spawn_key=cluster_root_seed.spawn_key + (points[start_idx].cluster_idx,))
                cluster_root_rng = np.random.default_rng(ss)
                latent_idx = cluster_latents[points[start_idx].cluster_idx]
                while latent_idx in latents:
                    children = sorted(latents[latent_idx].children, key=lambda c: c.point_idx)
                    latent_idx = cluster_root_rng.choice(children).point_idx
                root = latent_idx

            # DFS using stack
            stack = [root]
            visited = {root}
            ss = np.random.SeedSequence(entropy=cluster_uniform_seed.entropy,
                                        spawn_key=cluster_uniform_seed.spawn_key + (points[start_idx].cluster_idx,))
            cluster_uniform_rng = np.random.default_rng(ss)
            embeddings[root] = scale*point_vectors[root]
            if self.mode == "distribution":
                embeddings[root] += cluster_uniform_rng.uniform(low=-0.5, high=0.5, size=d)

            while stack:
                parent = stack.pop()
                for neighbor in sorted(points[point_idx2idx[parent]].neighbors, key=lambda n: n.point_idx):
                    if neighbor.point_idx not in visited:
                        visited.add(neighbor.point_idx)
                        embeddings[neighbor.point_idx] = embeddings[parent] + scale*point_vectors[neighbor.point_idx]
                        stack.append(neighbor.point_idx)
            
            start_idx = end_idx

        return np.stack([embeddings[point.point_idx] for point in points])
    
    def _generate_value(self, base_value: float, depth: int, rng: np.random.Generator) -> float:
        variance = (1 - self.ratio) * self.ratio**depth
        std = np.sqrt(variance)
        return base_value + rng.normal(0, std)


    def embed_labels(self, points: List['Node'], latents: Dict[int, 'Node']) -> np.ndarray:
        """Generates labels for the points."""
        cluster_inds = [point.cluster_idx for point in points]
        cluster_counts = Counter(cluster_inds)
        n_clusters = len(cluster_counts)

        seed_seq = np.random.SeedSequence(self.seeds.value) if self.seeds.value is not None else np.random.SeedSequence()
        cluster_seed, error_seed = seed_seq.spawn(2)

        rngs = [np.random.default_rng(s) for s in cluster_seed.spawn(n_clusters)]

        # Compute base value for each cluster
        cluster_values = np.zeros(n_clusters)
        for cluster_idx in range(n_clusters):
            base_value = self._generate_value(base_value=0.0, depth=0, rng=rngs[cluster_idx])
            cluster_values[cluster_idx] = base_value

        # Assign values for single points with no latents
        for point in points:
            if not point.parents:
                point.value = cluster_values[point.cluster_idx]

        # Compute values hierarchically using the latents
        for latent in latents.values():
            if not latent.parents:
                latent.value = cluster_values[latent.cluster_idx]
            for child in sorted(latent.children, key=lambda c: c.point_idx):
                child.value = self._generate_value(base_value=latent.value, depth=latent.depth + 1,
                                                   rng=rngs[latent.cluster_idx])

        # Compute irreducible error for each point
        for point in points:
            irreducible_variance = self.ratio**(point.depth + 1)
            irreducible_std = np.sqrt(irreducible_variance)
            ss = np.random.SeedSequence(entropy=error_seed.entropy,
                            spawn_key=error_seed.spawn_key + (point.point_idx,))
            rng = np.random.default_rng(ss)
            point.error = rng.normal(0, irreducible_std)

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


    def get_summary_features(self) -> dict:
        """
        Returns a dict of per-point metadata aligned with the feature matrix rows.

        Returns:
            dict with keys:
                - cluster_id: cluster index of each point
                - cluster_size: number of points in each point's cluster
                - latent_tree_height: number of latent ancestors of each point
        """

        counts = Counter(p.cluster_idx for p in self.points)
        return {
            'cluster_id':        [p.cluster_idx for p in self.points],
            'cluster_size':      [counts[p.cluster_idx] for p in self.points],
            'latent_tree_height': [len(self.pidx2lidx[i]) for i, _ in enumerate(self.points)],
        }

    def get_latent_features(self, n_features: Optional[int] = None, use_values: bool = False) -> sparse.csr_matrix:
        """
        Returns sparse matrix of latent features for each data point.

        Args:
            n_features: Number of features to return, in generation order. If None, returns all features.
            use_values: If True, store the actual z_u value for each latent instead of 1.
                        Requires embed_labels() to have been called (i.e. use construct_embed(), not construct()).
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
            if use_values:
                latent_node = self.latents[self.lidx2latent[lidx]]
                if latent_node.parents:
                    parent = list(latent_node.parents)[0]
                    z_u = latent_node.value - parent.value
                else:
                    z_u = latent_node.value
                data.extend([z_u] * len(pidx))
            else:
                data.extend(np.ones_like(pidx))
            row_ind.extend(pidx)
            col_ind.extend(np.full_like(pidx, lidx))

        features = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(self.n_samples, n_features))
        return features