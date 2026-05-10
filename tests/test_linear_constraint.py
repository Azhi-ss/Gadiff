"""Tests for linear polymer validation in PolyDiff-3D's PolyGA.

MMPolymer can only predict properties for linear polymers (exactly 2 [*]
wildcard endpoints). These tests verify that validation and enforcement
work correctly.
"""

import pytest
from src.models.polyga.utils import validate_linear_polymer


class TestValidateLinearPolymer:
    """Tests for validate_linear_polymer() -- a pure function that checks
    whether a SMILES string has exactly two [*] wildcard atoms."""

    def test_validate_linear_polymer_with_linear(self):
        """Simple linear polymer: exactly 2 endpoints."""
        assert validate_linear_polymer("[*]CC[*]") is True

    def test_validate_linear_polymer_with_linear_complex(self):
        """Linear polymer with functional groups: exactly 2 endpoints."""
        assert validate_linear_polymer("[*]CC(=O)NC[*]") is True

    def test_validate_linear_polymer_with_branched(self):
        """Branched polymer: 3 endpoints (should be rejected)."""
        assert validate_linear_polymer("[*]C([*])[*]") is False

    def test_validate_linear_polymer_with_cyclic(self):
        """Cyclic polymer with 2 endpoints (still linear from MMPolymer's
        perspective -- only the number of endpoints matters)."""
        assert validate_linear_polymer("[*]C1CC1[*]") is True

    def test_validate_linear_polymer_with_single_star(self):
        """Single endpoint: incomplete chain (should be rejected)."""
        assert validate_linear_polymer("[*]CCC") is False

    def test_validate_linear_polymer_with_no_star(self):
        """No endpoints: not a polymer chain (should be rejected)."""
        assert validate_linear_polymer("CCCC") is False

    def test_validate_linear_polymer_none(self):
        """None input: should return False without crashing."""
        assert validate_linear_polymer(None) is False

    def test_validate_linear_polymer_with_empty_string(self):
        """Empty string: should return False."""
        assert validate_linear_polymer("") is False

    def test_validate_linear_polymer_with_more_than_three_stars(self):
        """Heavily branched: 4 endpoints (should be rejected)."""
        assert validate_linear_polymer("[*]C([*])C([*])[*]") is False

    def test_validate_linear_polymer_with_bracket_star_no_bracket(self):
        """A bare * is not the same as [*] -- should not be counted."""
        # A bare * is a different SMILES notation; our function counts [*]
        assert validate_linear_polymer("*CC*") is False

    def test_validate_linear_polymer_star_in_side_chain(self):
        """[*] appearing in a branch context, total > 2."""
        assert validate_linear_polymer("[*]CC(C[*])[*]") is False

    def test_validate_linear_polymer_four_stars(self):
        """Four endpoints: should be rejected."""
        assert validate_linear_polymer("[*]C([*])C([*])C[*]") is False


class TestChromosomeIdsToSmilesIntegration:
    """Integration tests: verify that chromosome_ids_to_smiles() rejects
    fragment combinations that would produce branched polymers."""

    def test_chromosome_ids_to_smiles_rejects_branched(self, monkeypatch):
        """Verify the assembly function returns None for branched output.

        We mock the internal helpers that finalize the SMILES to simulate
        a branched polymer, then check that the validation catches it.
        """
        import src.models.polyga.utils as utils

        # Mock _finalize_two_star_psmiles to return a branched SMILES
        monkeypatch.setattr(
            utils,
            "_finalize_two_star_psmiles",
            lambda m: "[*]C([*])[*]",
        )
        # Prevent the recovery path (longest_smiles / _linearize_two_stars)
        # from running -- we want to test the final validation gate.
        monkeypatch.setattr(
            utils,
            "longest_smiles",
            lambda s: "",
        )
        monkeypatch.setattr(
            utils,
            "_linearize_two_stars",
            lambda s: "",
        )

        from numpy.random import default_rng

        rng = default_rng(seed=42)
        chromosomes = {
            "frag1": "C[*]",
            "frag2": "[*]C[*]",
            "frag3": "[*]C",
        }
        result = utils.chromosome_ids_to_smiles(
            ["frag1", "frag2", "frag3"],
            chromosomes,
            rng,
        )
        assert result is None, (
            "Branched polymer should be rejected by validate_linear_polymer"
        )
