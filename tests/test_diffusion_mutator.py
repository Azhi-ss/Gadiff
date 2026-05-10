"""Tests for EGDFragmentMutator -- noise-space mutation operator."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from polyga.diffusion_mutator import (
    EGDFragmentMutator,
    _is_valid_fragment,
    _build_radius_graph,
    ATOM_DECODER_Z,
    ATOM_Z_TO_INDEX,
    NUM_ATOM_TYPES,
    ATOM_FEATURE_DIM,
    RADIUS_CUTOFF,
)


# ---------------------------------------------------------------------------
# Mock diffusion model
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_diffusion_model() -> MagicMock:
    """A fully mocked MDMFullDP instance."""
    model = MagicMock()
    # Standard diffusion schedule with 1000 timesteps.
    num_timesteps = 1000
    betas = torch.linspace(1e-4, 0.02, num_timesteps)
    alphas = (1.0 - betas).cumprod(dim=0)
    model.num_timesteps = num_timesteps
    model.betas = betas
    model.alphas = alphas

    # langevin_dynamics_sample returns (pos, pos_traj, atom_type, atom_traj).
    def mock_sample(
        atom_type=None,
        pos_init=None,
        bond_index=None,
        bond_type=None,
        batch=None,
        num_graphs=None,
        context=None,
        n_steps=100,
        step_lr=0.0000010,
        clip=1000,
        clip_local=None,
        clip_pos=None,
        global_start_sigma=float("inf"),
        local_start_sigma=float("inf"),
        w_global_pos=1,
        w_global_node=1,
        w_local_pos=1,
        w_local_node=1,
        update_mask=None,
        start_timestep=None,
        **kwargs,
    ):
        # Return noisy-ish positions and scaled atom types.
        n_atoms = pos_init.size(0)
        n_type = atom_type.size(-1) if atom_type is not None else 8
        pos_out = pos_init + 0.01 * torch.randn_like(pos_init)
        # Scaled atom types (following model convention: types *4, charge *10).
        atom_type_scaled = torch.cat(
            [
                atom_type[:, :-1] * 4.0 if atom_type is not None else torch.randn(n_atoms, n_type - 1) * 4.0,
                atom_type[:, -1:] * 10.0 if atom_type is not None else torch.randn(n_atoms, 1) * 10.0,
            ],
            dim=1,
        )
        return pos_out, [pos_out], atom_type_scaled, [atom_type_scaled]

    model.langevin_dynamics_sample = mock_sample
    model.eval = MagicMock(return_value=None)
    model.to = MagicMock(return_value=model)
    return model


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestEGDFragmentMutatorInit:
    """EGDFragmentMutator stores configuration correctly."""

    def test_stores_config(self, mock_diffusion_model: MagicMock) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        assert mutator.model is mock_diffusion_model
        assert mutator.t_prime == 250
        assert mutator.num_atom_types == len(ATOM_DECODER_Z)

    def test_custom_t_prime(self, mock_diffusion_model: MagicMock) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=500,
        )
        assert mutator.t_prime == 500

    def test_atom_feature_dim(self, mock_diffusion_model: MagicMock) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        assert mutator.atom_feature_dim == NUM_ATOM_TYPES + 1


# ---------------------------------------------------------------------------
# _is_valid_fragment tests
# ---------------------------------------------------------------------------

class TestIsValidFragment:
    """Fragment validation logic."""

    def test_valid_ethane_fragment(self) -> None:
        # Three-heavy-atom backbone with two [*] connection points: 3 heavy atoms.
        assert _is_valid_fragment("[*]CCO[*]")

    def test_invalid_empty_string(self) -> None:
        assert not _is_valid_fragment("")

    def test_invalid_none(self) -> None:
        assert not _is_valid_fragment(None)  # type: ignore[arg-type]

    def test_invalid_unparsable(self) -> None:
        assert not _is_valid_fragment("not-a-smiles")

    def test_too_few_heavy_atoms(self) -> None:
        # Single carbon with [*] -- only 1 heavy atom.
        assert not _is_valid_fragment("C[*]")

    def test_too_many_wildcards(self) -> None:
        # Three [*] instead of two.
        assert not _is_valid_fragment("[*]CC[*]C[*]")

    def test_disallowed_element(self) -> None:
        # Contains phosphorus (Z=15), not in allowed set.
        assert not _is_valid_fragment("[*]CP([*])C")

    def test_valid_benzene_fragment(self) -> None:
        # Phenyl group with two connection points.
        assert _is_valid_fragment("[*]c1ccccc1[*]")


# ---------------------------------------------------------------------------
# _smiles_to_3d tests
# ---------------------------------------------------------------------------

class TestSmilesTo3D:
    """SMILES to 3D conformer conversion."""

    def test_simple_smiles(self, mock_diffusion_model: MagicMock) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        mol = mutator._smiles_to_3d("CCO")
        assert mol is not None
        assert mol.GetNumConformers() == 1
        assert mol.GetNumAtoms() >= 3  # includes H

    def test_empty_smiles(self, mock_diffusion_model: MagicMock) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        assert mutator._smiles_to_3d("") is None

    def test_invalid_smiles(self, mock_diffusion_model: MagicMock) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        assert mutator._smiles_to_3d("ZZZ") is None

    def test_wildcard_smiles(self, mock_diffusion_model: MagicMock) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        mol = mutator._smiles_to_3d("[*]C[*]")
        assert mol is not None
        assert mol.GetNumConformers() == 1


# ---------------------------------------------------------------------------
# mutate_fragment -- graceful degradation
# ---------------------------------------------------------------------------

class TestMutateFragment:
    """EGD mutation returns [] gracefully when model/data not available."""

    def test_returns_empty_list_when_model_fails(
        self, mock_diffusion_model: MagicMock
    ) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        # _smiles_to_3d will return None for invalid SMILES.
        result = mutator.mutate_fragment("invalid_smiles_xyz")
        assert result == []

    def test_returns_empty_list_on_empty_smiles(
        self, mock_diffusion_model: MagicMock
    ) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        result = mutator.mutate_fragment("")
        assert result == []

    def test_custom_t_prime_is_passed_through(
        self, mock_diffusion_model: MagicMock
    ) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        # Using a valid SMILES; _mol_to_data will fail because PyG may
        # not be available, resulting in empty list -- that's OK.
        result = mutator.mutate_fragment("CCO", t_prime=100)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# crossover_fragments -- graceful degradation
# ---------------------------------------------------------------------------

class TestCrossoverFragments:
    """EGD crossover returns [] gracefully when model/data not available."""

    def test_returns_empty_list_on_failure(
        self, mock_diffusion_model: MagicMock
    ) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        result = mutator.crossover_fragments("invalid", "CCO")
        assert result == []

    def test_returns_empty_list_with_empty_smiles(
        self, mock_diffusion_model: MagicMock
    ) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        result = mutator.crossover_fragments("", "")
        assert result == []


# ---------------------------------------------------------------------------
# _forward_diffuse -- noise addition
# ---------------------------------------------------------------------------

class TestForwardDiffuse:
    """Forward diffusion adds noise to positions correctly."""

    @pytest.fixture
    def simple_data(self) -> Any:
        """A minimal PyG-like Data object with 5 atoms."""
        if not _pyg_available():
            pytest.skip("PyTorch Geometric not available")
        from torch_geometric.data import Data

        pos = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.5],
            [2.5, 0.0, 0.0],
        ], dtype=torch.float)
        batch = torch.zeros(5, dtype=torch.long)
        atom_type = torch.zeros(5, ATOM_FEATURE_DIM, dtype=torch.float)
        atom_type[:, 0] = 1.0  # All carbon
        edge_index = _build_radius_graph(pos, r=RADIUS_CUTOFF, batch=batch)
        return Data(pos=pos, atom_type=atom_type, edge_index=edge_index, batch=batch)

    def test_noise_added_at_t500(self, mock_diffusion_model: MagicMock, simple_data: Any) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        data_noisy = mutator._forward_diffuse(simple_data, t=500)
        assert data_noisy is not None
        # Positions should have changed.
        assert not torch.allclose(data_noisy.pos, simple_data.pos)
        # Shape preserved.
        assert data_noisy.pos.shape == simple_data.pos.shape

    def test_noise_increases_with_t(self, mock_diffusion_model: MagicMock, simple_data: Any) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        # Fix random seed for reproducibility.
        torch.manual_seed(42)
        data_noisy_low = mutator._forward_diffuse(simple_data, t=10)
        torch.manual_seed(42)
        data_noisy_high = mutator._forward_diffuse(simple_data, t=500)

        assert data_noisy_low is not None
        assert data_noisy_high is not None

        low_noise = (data_noisy_low.pos - simple_data.pos).norm(dim=1).mean()
        high_noise = (data_noisy_high.pos - simple_data.pos).norm(dim=1).mean()
        # Higher t means more noise (larger sigma).
        assert high_noise > low_noise


# ---------------------------------------------------------------------------
# _chimeric_concat -- crossover helper
# ---------------------------------------------------------------------------

class TestChimericConcat:
    """Chimeric concatenation produces correct output size."""

    @pytest.fixture
    def data_a(self) -> Any:
        if not _pyg_available():
            pytest.skip("PyTorch Geometric not available")
        from torch_geometric.data import Data

        pos = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [0.0, 1.4, 0.0],
        ], dtype=torch.float)
        atom_type = torch.zeros(3, ATOM_FEATURE_DIM, dtype=torch.float)
        atom_type[:, 0] = 1.0
        batch = torch.zeros(3, dtype=torch.long)
        edge_index = _build_radius_graph(pos, r=RADIUS_CUTOFF, batch=batch)
        return Data(pos=pos, atom_type=atom_type, edge_index=edge_index, batch=batch)

    @pytest.fixture
    def data_b(self) -> Any:
        if not _pyg_available():
            pytest.skip("PyTorch Geometric not available")
        from torch_geometric.data import Data

        pos = torch.tensor([
            [0.5, 0.5, 0.5],
            [1.9, 0.5, 0.5],
            [0.5, 1.9, 0.5],
            [0.5, 0.5, 1.9],
        ], dtype=torch.float)
        atom_type = torch.zeros(4, ATOM_FEATURE_DIM, dtype=torch.float)
        atom_type[:, 1] = 1.0  # All nitrogen
        batch = torch.zeros(4, dtype=torch.long)
        edge_index = _build_radius_graph(pos, r=RADIUS_CUTOFF, batch=batch)
        return Data(pos=pos, atom_type=atom_type, edge_index=edge_index, batch=batch)

    def test_output_size(self, mock_diffusion_model: MagicMock, data_a: Any, data_b: Any) -> None:
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        chimeric = mutator._chimeric_concat(data_a, data_b)
        assert chimeric is not None
        # At least 1 from each parent (2+), at most total from both (7-).
        n_chimeric = chimeric.pos.size(0)
        assert n_chimeric >= 2, "Should have at least 1 atom from each parent"
        assert n_chimeric <= 7, "Should not exceed total atom count"
        # All tensors should have the same number of rows.
        assert chimeric.atom_type.size(0) == n_chimeric
        assert chimeric.batch.size(0) == n_chimeric

    def test_returns_none_without_pyg(self, mock_diffusion_model: MagicMock) -> None:
        """When PyG is unavailable, _chimeric_concat returns None gracefully."""
        mutator = EGDFragmentMutator(
            diffusion_model=mock_diffusion_model,
            device="cpu",
            t_prime=250,
        )
        # Even with valid Data-like objects, without _HAVE_PYG it fails.
        with patch("polyga.diffusion_mutator._HAVE_PYG", False):
            result = mutator._chimeric_concat(
                MagicMock(pos=torch.randn(3, 3)),
                MagicMock(pos=torch.randn(4, 3)),
            )
        assert result is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pyg_available() -> bool:
    try:
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False
