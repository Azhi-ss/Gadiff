"""
Tests for SAESSelector -- structure-aware environmental selection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from polyga.saes_selector import SAESSelector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def selector() -> SAESSelector:
    return SAESSelector(
        k_neighbors=3,
        tau=0.05,
        fingerprint_radius=2,
        fp_size=2048,
        use_3d=False,  # avoid USR dependency in unit tests
        random_seed=42,
    )


@pytest.fixture
def simple_population() -> pd.DataFrame:
    """Small population with distinct scaffolds."""
    return pd.DataFrame({
        "smiles_string": [
            "c1ccccc1C(=O)O",       # benzoic acid
            "c1ccccc1CC(=O)O",      # phenylacetic acid
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin-like
            "c1ccncc1",             # pyridine
            "c1ccccc1",             # benzene
            "CCO",                  # ethanol (aliphatic)
        ],
        "fitness": [10.0, 8.0, 6.0, 4.0, 2.0, 0.5],
        "birth_nation": ["A", "A", "B", "B", "C", "C"],
        "chromosome_ids": [[1], [2], [3], [4], [5], [6]],
    })


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestSAESSelectorInit:
    def test_stores_parameters(self) -> None:
        sel = SAESSelector(k_neighbors=7, tau=0.1, fingerprint_radius=3, fp_size=1024)
        assert sel.k_neighbors == 7
        assert sel.tau == 0.1
        assert sel.fingerprint_radius == 3
        assert sel.fp_size == 1024


# ---------------------------------------------------------------------------
# Morgan fingerprints
# ---------------------------------------------------------------------------

class TestMorganFingerprints:
    def test_returns_correct_shape(self, selector: SAESSelector) -> None:
        smiles = ["c1ccccc1", "CCO", "CC(=O)O"]
        fps = selector._get_morgan_fingerprints(smiles)
        assert fps.shape == (3, 2048)
        assert fps.dtype == np.float64

    def test_returns_zeros_for_invalid_smiles(self, selector: SAESSelector) -> None:
        smiles = ["not_a_smiles", "", "CC"]
        fps = selector._get_morgan_fingerprints(smiles)
        assert fps.shape == (3, 2048)
        # First two are invalid -> all zeros
        assert fps[0].sum() == 0.0
        assert fps[1].sum() == 0.0
        # Third ("CC") is valid -> non-zero fingerprint
        assert fps[2].sum() > 0.0

    def test_handles_empty_list(self, selector: SAESSelector) -> None:
        fps = selector._get_morgan_fingerprints([])
        assert fps.shape == (0, 2048)

    @patch("polyga.saes_selector._HAS_RDKIT", False)
    def test_graceful_degradation_no_rdkit(self) -> None:
        sel = SAESSelector()
        smiles = ["c1ccccc1", "CCO"]
        fps = sel._get_morgan_fingerprints(smiles)
        assert fps.shape == (2, 2048)
        assert fps.sum() == 0.0


# ---------------------------------------------------------------------------
# USR descriptors
# ---------------------------------------------------------------------------

class TestUSRDescriptors:
    def test_returns_correct_shape_valid(self, selector: SAESSelector) -> None:
        """USR with valid SMILES returns (n, 60)."""
        smiles = ["c1ccccc1", "CCO"]
        usrs = selector._get_usr_descriptors(smiles)
        assert usrs.shape == (2, 60)
        assert usrs.dtype == np.float64

    def test_returns_zeros_for_invalid_smiles(self, selector: SAESSelector) -> None:
        """Graceful degradation: invalid SMILES -> zeros(60)."""
        smiles = ["not_valid", ""]
        usrs = selector._get_usr_descriptors(smiles)
        assert usrs.shape == (2, 60)
        assert usrs.sum() == 0.0

    def test_handles_empty_list(self, selector: SAESSelector) -> None:
        usrs = selector._get_usr_descriptors([])
        assert usrs.shape == (0, 60)


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------

class TestDistanceMatrix:
    def test_symmetric_with_zero_diagonal(self, selector: SAESSelector) -> None:
        df = pd.DataFrame({
            "smiles_string": ["c1ccccc1", "CCO", "CC(=O)O"],
        })
        dist = selector._compute_distance_matrix(df)
        assert dist.shape == (3, 3)
        # Diagonal should be zero
        assert np.allclose(np.diag(dist), 0.0)
        # Symmetric
        assert np.allclose(dist, dist.T)

    def test_distance_between_identical_is_zero(self, selector: SAESSelector) -> None:
        df = pd.DataFrame({
            "smiles_string": ["c1ccccc1", "c1ccccc1"],
        })
        dist = selector._compute_distance_matrix(df)
        assert dist[0, 1] < 1e-6

    def test_aliphatic_vs_aromatic_nonzero(self, selector: SAESSelector) -> None:
        df = pd.DataFrame({
            "smiles_string": ["c1ccccc1", "CCO"],
        })
        dist = selector._compute_distance_matrix(df)
        # Benzene and ethanol should be structurally distinct
        assert dist[0, 1] > 0.1


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

class TestSelect:
    def test_returns_correct_number(self, selector: SAESSelector,
                                    simple_population: pd.DataFrame) -> None:
        result = selector.select(simple_population, n_select=3)
        assert len(result) == 3

    def test_select_none_when_population_empty(self, selector: SAESSelector) -> None:
        df = pd.DataFrame(columns=["smiles_string", "fitness"])
        result = selector.select(df, n_select=5)
        assert len(result) == 0

    def test_select_all_when_n_gt_population(
        self, selector: SAESSelector, simple_population: pd.DataFrame
    ) -> None:
        result = selector.select(simple_population, n_select=100)
        assert len(result) == len(simple_population)

    def test_preserves_diversity(self, selector: SAESSelector) -> None:
        """Different scaffolds should not be collapsed into same selection."""
        df = pd.DataFrame({
            "smiles_string": [
                "c1ccccc1",       # aromatic
                "CCO",             # aliphatic
                "CCCC",            # alkane
                "c1ccncc1",        # heteroaromatic
                "CC(=O)O",         # carboxylic acid
                "c1ccccc1O",       # phenol
            ],
            "fitness": [100.0, 90.0, 80.0, 70.0, 60.0, 50.0],
        })
        result = selector.select(df, n_select=4)
        # All 4 selected should have different SMILES
        assert result["smiles_string"].nunique() == 4

    def test_select_best_is_always_selected(
        self, selector: SAESSelector, simple_population: pd.DataFrame
    ) -> None:
        result = selector.select(simple_population, n_select=2)
        # The highest fitness individual (benzoic acid, fitness=10) should be in result
        assert 10.0 in result["fitness"].values

    @patch("polyga.saes_selector._HAS_RDKIT", False)
    def test_fallback_random_selection_when_no_rdkit(self) -> None:
        """Without RDKit, selection should still return correct number."""
        sel = SAESSelector(random_seed=0)
        df = pd.DataFrame({
            "smiles_string": ["a", "b", "c", "d"],
            "fitness": [1.0, 2.0, 3.0, 4.0],
        })
        result = sel.select(df, n_select=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Jaccard / Euclidean helpers
# ---------------------------------------------------------------------------

class TestDistanceHelpers:
    def test_jaccard_same_vectors(self) -> None:
        a = np.array([1, 0, 1, 0], dtype=np.float64)
        d = SAESSelector._jaccard_distance(a, a)
        assert d == 0.0

    def test_jaccard_orthogonal(self) -> None:
        a = np.array([1, 0], dtype=np.float64)
        b = np.array([0, 1], dtype=np.float64)
        d = SAESSelector._jaccard_distance(a, b)
        assert d == 1.0

    def test_normalized_euclidean_same(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        d = SAESSelector._normalized_euclidean(a, a)
        assert d == 0.0

    def test_normalized_euclidean_zero_vectors(self) -> None:
        a = np.zeros(3)
        b = np.zeros(3)
        d = SAESSelector._normalized_euclidean(a, b)
        assert d == 0.0
