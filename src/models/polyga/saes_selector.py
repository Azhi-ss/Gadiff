"""
Structure-Aware Environmental Selection (SAES) for PolyGA.

Replaces simple elite() selection with diversity-preserving selection using
composite structural distance (Morgan fingerprint Jaccard + USR shape distance).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

_HAS_RDKIT = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


# ---------------------------------------------------------------------------
# USR helpers (lazy import to keep rdkit dependency optional at module level)
# ---------------------------------------------------------------------------

def _compute_usr(smiles: str) -> Optional[np.ndarray]:
    """Compute Ultra-fast Shape Recognition (USR) descriptor for a SMILES.

    Returns 60-dimensional vector (12 moments x 5 summary stats), or None
    if 3D conformer generation fails.
    """
    if not _HAS_RDKIT:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        # Embed 3D conformer -- may fail for invalid or strained molecules
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params)
        if status < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        positions = conf.GetPositions()  # (n_atoms, 3)
        centroid = positions.mean(axis=0)
        # Distance to centroid
        d_cent = np.linalg.norm(positions - centroid, axis=1)
        # C+ = atom farthest from centroid
        farthest_idx = np.argmax(d_cent)
        c_plus = positions[farthest_idx]
        d_cp = np.linalg.norm(positions - c_plus, axis=1)
        # C- = atom closest to centroid
        closest_idx = np.argmin(d_cent)
        c_minus = positions[closest_idx]
        d_cm = np.linalg.norm(positions - c_minus, axis=1)
        # 12 descriptors: {mean, var, skew, kurt, max} for each set
        # Order: centroid distances, C+ distances, C- distances
        sets = [d_cent, d_cp, d_cm]
        usr = []
        for s in sets:
            if len(s) == 0:
                return None
            mean = float(np.mean(s))
            var = float(np.var(s))
            skew = float(np.mean(((s - mean) / (np.sqrt(var) + 1e-10)) ** 3))
            kurt = float(np.mean(((s - mean) / (np.sqrt(var) + 1e-10)) ** 4))
            mx = float(np.max(s))
            usr.extend([mean, var, skew, kurt, mx])
        return np.array(usr, dtype=np.float64)
    except Exception:
        return None


_USR_CACHE: dict[str, Optional[np.ndarray]] = {}


def _get_usr_safe(smiles: str) -> np.ndarray:
    """Cached USR computation returning zeros(60) on failure."""
    if smiles in _USR_CACHE:
        result = _USR_CACHE[smiles]
        return result if result is not None else np.zeros(60, dtype=np.float64)
    usr = _compute_usr(smiles)
    _USR_CACHE[smiles] = usr
    return usr if usr is not None else np.zeros(60, dtype=np.float64)


# ---------------------------------------------------------------------------
# SAESSelector
# ---------------------------------------------------------------------------

class SAESSelector:
    """Structure-Aware Environmental Selection.

    Maintains both high fitness and structural diversity to prevent premature
    convergence. Uses a composite distance combining Morgan fingerprint
    (2D topology) and USR descriptors (3D shape).
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        tau: float = 0.05,
        fingerprint_radius: int = 2,
        fp_size: int = 2048,
        use_3d: bool = True,
        random_seed: int = 42,
    ) -> None:
        """
        Args:
            k_neighbors: Number of neighbors for k-NN adjusted fitness.
            tau: Minimum structural distance threshold for diversity.
            fingerprint_radius: Morgan fingerprint radius.
            fp_size: Morgan fingerprint size in bits.
            use_3d: Whether to include USR 3D shape distance.
            random_seed: Random seed for fallback selection.
        """
        self.k_neighbors = k_neighbors
        self.tau = tau
        self.fingerprint_radius = fingerprint_radius
        self.fp_size = fp_size
        self.use_3d = use_3d and _HAS_RDKIT
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        population_df: pd.DataFrame,
        n_select: int,
        fitness_col: str = "fitness",
    ) -> pd.DataFrame:
        """Select individuals from population using SAES.

        Args:
            population_df: Population with 'smiles_string' and fitness_col.
            n_select: Number of individuals to select.
            fitness_col: Column name for fitness values.

        Returns:
            DataFrame with selected individuals.
        """
        if len(population_df) == 0:
            return population_df
        if n_select >= len(population_df):
            # Just return a copy of all
            result = population_df.copy()
            result = result.reset_index(drop=True)
            return result

        # Graceful degradation: if RDKit unavailable, random selection
        if not _HAS_RDKIT:
            n = min(n_select, len(population_df))
            idx = self._rng.choice(len(population_df), size=n, replace=False)
            return population_df.iloc[idx].reset_index(drop=True)

        df = population_df.copy().reset_index(drop=True)
        # Sort by fitness descending
        df = df.sort_values(by=fitness_col, ascending=False).reset_index(drop=True)

        # Precompute distance matrix
        dist_matrix = self._compute_distance_matrix(df)
        selected_indices: list[int] = []
        tau_current = self.tau

        # Greedy selection with adaptive threshold
        max_attempts = 3
        for attempt in range(max_attempts):
            selected_indices = []
            for i in range(len(df)):
                if len(selected_indices) >= n_select:
                    break
                if not selected_indices:
                    # Always accept the best individual
                    selected_indices.append(i)
                    continue
                # Check minimum distance to already-selected
                dists_to_selected = dist_matrix[i, selected_indices]
                min_dist = float(dists_to_selected.min())
                if min_dist < tau_current:
                    continue  # Too similar, skip
                # Adjusted fitness
                k = min(self.k_neighbors, len(df))
                dists_to_all = dist_matrix[i]
                # k-th nearest neighbor distance (excluding self)
                sorted_dists = np.sort(dists_to_all)
                d_k = sorted_dists[min(k, len(sorted_dists) - 1)]
                fitness_rank = float(i)  # 0 = best
                adjusted_f = fitness_rank + 1.0 / (d_k + 2.0)
                # Accept if adjusted fitness beats threshold (rank + diversity)
                if adjusted_f < float(n_select) * (1.0 + tau_current):
                    selected_indices.append(i)
            if len(selected_indices) >= n_select:
                break
            # Relax threshold if too few selected
            tau_current *= 0.5

        if len(selected_indices) > n_select:
            selected_indices = selected_indices[:n_select]
        result = df.iloc[selected_indices].reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

    def _compute_distance_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Compute composite structural distance matrix.

        D = 0.5 * Jaccard(MorganFP) + 0.5 * normalized_Euclidean(USR)

        Returns N x N symmetric matrix with zero diagonal.
        """
        n = len(df)
        morgan = self._get_morgan_fingerprints(df["smiles_string"].tolist())

        # Morgan Jaccard distance
        fp_dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = self._jaccard_distance(morgan[i], morgan[j])
                fp_dist[i, j] = d
                fp_dist[j, i] = d

        if self.use_3d:
            usr = self._get_usr_descriptors(df["smiles_string"].tolist())
            usr_dist = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(i + 1, n):
                    d = self._normalized_euclidean(usr[i], usr[j])
                    usr_dist[i, j] = d
                    usr_dist[j, i] = d
            dist = 0.5 * fp_dist + 0.5 * usr_dist
        else:
            # Fall back to fingerprint distance only when USR unavailable
            dist = fp_dist
        return dist

    def _get_morgan_fingerprints(self, smiles_list: list[str]) -> np.ndarray:
        """Compute Morgan fingerprints for a list of SMILES.

        Returns (n, fp_size) array; zero vector for invalid SMILES.
        """
        n = len(smiles_list)
        fps = np.zeros((n, self.fp_size), dtype=np.float64)
        if not _HAS_RDKIT:
            return fps
        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fingerprint_radius, nBits=self.fp_size
                )
                fps[i] = np.array(fp, dtype=np.float64)
            except Exception:
                continue
        return fps

    def _get_usr_descriptors(self, smiles_list: list[str]) -> np.ndarray:
        """Compute USR shape descriptors.

        Returns (n, 60) array; zero rows for failed SMILES.
        """
        n = len(smiles_list)
        usrs = np.zeros((n, 60), dtype=np.float64)
        if not _HAS_RDKIT or not self.use_3d:
            return usrs
        for i, smi in enumerate(smiles_list):
            usrs[i] = _get_usr_safe(smi)
        return usrs

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Jaccard distance = 1 - |intersection| / |union| for bit vectors."""
        intersection = float(np.dot(a, b))
        union = float(np.sum(a) + np.sum(b) - intersection)
        if union == 0.0:
            return 0.0
        return 1.0 - intersection / union

    @staticmethod
    def _normalized_euclidean(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance normalized by sum of norms."""
        diff = a - b
        d = float(np.sqrt(np.dot(diff, diff)))
        norm_sum = float(np.linalg.norm(a) + np.linalg.norm(b))
        if norm_sum < 1e-10:
            return 0.0
        return d / norm_sum


# ---------------------------------------------------------------------------
# Convenience function matching selection_schemes signature
# ---------------------------------------------------------------------------

def saes_select(
    population: pd.DataFrame,
    num_parents_per_nationality: dict[str, int],
    selector: Optional[SAESSelector] = None,
) -> pd.DataFrame:
    """SAES-based selection wrapped for PolyGA compatibility.

    Selects per nationality using the SAES algorithm then concatenates.

    Args:
        population: Population dataframe.
        num_parents_per_nationality: Mapping of birth_nation -> count.
        selector: SAESSelector instance. Created with defaults if None.

    Returns:
        Selected parents dataframe.
    """
    if selector is None:
        selector = SAESSelector()
    result = pd.DataFrame()
    for nation, n_parents in num_parents_per_nationality.items():
        sub = population[population["birth_nation"] == nation].copy()
        if len(sub) == 0:
            continue
        selected = selector.select(sub, n_select=n_parents)
        result = pd.concat([result, selected], ignore_index=True)
    return result
