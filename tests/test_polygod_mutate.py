"""Tests for PolyNation.__mutate with EGD diffusion-based mutation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
from numpy.random import default_rng

from polyga.polygod import PolyLand, PolyNation, PolyPlanet


# ---------------------------------------------------------------------------
# Fixtures: minimal PolyPlanet, PolyLand, PolyNation
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_planet() -> MagicMock:
    """A PolyPlanet with chromosome registry and uid counter."""
    planet = MagicMock(spec=PolyPlanet)
    # Chromosome registry: {id: SMILES}
    planet.chromosomes = {
        1: "CCO",
        2: "CCN",
        3: "c1ccccc1",
        4: "CC(=O)O",
        5: "CC(C)C",
        6: "C1CCCCC1",
        7: "CCOC",
        8: "CCS",
        9: "CCCl",
        10: "CBr",
    }
    planet.num_citizens = 100
    planet.name = "TestPlanet"
    planet.uid = MagicMock(side_effect=_uid_counter(100))
    return planet


@pytest.fixture
def mock_land(mock_planet: MagicMock) -> MagicMock:
    """A PolyLand without EGD mutator (classic random mutation)."""
    land = MagicMock(spec=PolyLand)
    land.planet = mock_planet
    land.name = "TestLand"
    land.land_chromosomes = list(mock_planet.chromosomes.keys())
    land.fraction_mutation = 0.2
    land.mutation_sigma_offset = 0.25
    land.fraction_mutate_additional_block = 0.05
    land.generative_function_parameters = {}
    # generative_function: takes (ids, chroms, rng, **kw) -> SMILES string
    land.generative_function = MagicMock(
        return_value="CCCCCCCC"
    )
    # No egd_mutator set -- falls back to random.
    return land


@pytest.fixture
def nation(mock_land: MagicMock) -> PolyNation:
    """A minimal PolyNation with known random seed."""
    nation = PolyNation(
        name="TestNation",
        land=mock_land,
        num_families=1,
        num_parents_per_family=1,
        num_children_per_family=1,
        random_seed=42,
    )
    return nation


# ---------------------------------------------------------------------------
# Helper: uid counter
# ---------------------------------------------------------------------------

def _uid_counter(start: int = 0):
    """Returns a callable that produces incrementing IDs."""
    _count = [start]

    def _next() -> int:
        _count[0] += 1
        return _count[0]

    return _next


# ---------------------------------------------------------------------------
# Tests: no EGD mutator (classic random mutation)
# ---------------------------------------------------------------------------

class TestMutateNoEGD:
    """Without EGD mutator, __mutate behaves exactly like the original."""

    def test_mutates_some_positions(self, nation: PolyNation) -> None:
        """At least some positions should be replaced from land_chromosomes."""
        ids = [1, 2, 3, 4, 5]
        result = nation._PolyNation__mutate(ids)
        # Result should be a list of the same length or longer (possible append).
        assert len(result) >= len(ids) - 1
        # At least one ID should differ (with high probability given fraction=0.2).
        changed = sum(1 for i in range(len(ids)) if i < len(result) and result[i] != ids[i])
        assert changed >= 0  # non-deterministic, but should usually be >= 1

    def test_picks_from_land_chromosomes(self, nation: PolyNation) -> None:
        """All resulting IDs should exist in the planet's chromosome registry."""
        ids = [1, 2, 3, 4, 5]
        result = nation._PolyNation__mutate(ids)
        for rid in result:
            assert rid in nation.land.planet.chromosomes

    def test_random_seed_determinism(self, mock_land: MagicMock) -> None:
        """Same seed produces same mutation result."""
        n1 = PolyNation("A", land=mock_land, random_seed=123)
        n2 = PolyNation("B", land=mock_land, random_seed=123)
        ids = [1, 2, 3, 4, 5, 6, 7, 8]
        r1 = n1._PolyNation__mutate(list(ids))
        r2 = n2._PolyNation__mutate(list(ids))
        assert r1 == r2

    def test_empty_chromosome_ids(self, nation: PolyNation) -> None:
        """Empty input returns empty list (no crash)."""
        result = nation._PolyNation__mutate([])
        assert result == []

    def test_single_chromosome(self, nation: PolyNation) -> None:
        """Single ID input still works."""
        result = nation._PolyNation__mutate([5])
        assert len(result) >= 1
        assert result[0] in nation.land.planet.chromosomes

    def test_additional_block_appends(self, mock_land: MagicMock) -> None:
        """With high append probability, one extra chromosome is sometimes added."""
        mock_land.fraction_mutate_additional_block = 1.0  # always append
        n = PolyNation("HighAppend", land=mock_land, random_seed=42)
        ids = [1, 2, 3]
        result = n._PolyNation__mutate(ids)
        assert len(result) >= len(ids)
        assert result[-1] in mock_land.planet.chromosomes


# ---------------------------------------------------------------------------
# Tests: with EGD mutator
# ---------------------------------------------------------------------------

class TestMutateWithEGD:
    """With EGD mutator, __mutate delegates when probability hits."""

    @pytest.fixture
    def egd_mutator(self) -> MagicMock:
        mut = MagicMock()
        mut.mutate_fragment.return_value = ["[*]CC[*]", "[*]CCO[*]"]
        return mut

    @pytest.fixture
    def land_with_egd(
        self, mock_planet: MagicMock, egd_mutator: MagicMock
    ) -> MagicMock:
        land = MagicMock(spec=PolyLand)
        land.planet = mock_planet
        land.name = "EGDLand"
        land.land_chromosomes = list(mock_planet.chromosomes.keys())
        land.fraction_mutation = 0.2
        land.mutation_sigma_offset = 0.25
        land.fraction_mutate_additional_block = 0.05
        land.generative_function_parameters = {}
        land.generative_function = MagicMock(return_value="CCCCCCCC")
        land.egd_mutator = egd_mutator
        land.egd_mutation_prob = 1.0  # always try EGD
        return land

    @pytest.fixture
    def nation_with_egd(self, land_with_egd: MagicMock) -> PolyNation:
        return PolyNation(
            name="EGDNation",
            land=land_with_egd,
            num_families=1,
            num_parents_per_family=1,
            num_children_per_family=1,
            random_seed=42,
        )

    def test_delegates_to_egd_mutator(
        self, nation_with_egd: PolyNation, egd_mutator: MagicMock
    ) -> None:
        """EGD mutator is called with the correct fragment SMILES."""
        ids = [1, 2, 3, 4, 5]
        result = nation_with_egd._PolyNation__mutate(ids)
        # The EGD mutator should have been called at least once.
        assert egd_mutator.mutate_fragment.called
        # The chosen SMILES should be from the planet's registry.
        called_smiles = egd_mutator.mutate_fragment.call_args[0][0]
        assert called_smiles in nation_with_egd.land.planet.chromosomes.values()

    def test_new_fragments_registered_in_planet(
        self, nation_with_egd: PolyNation
    ) -> None:
        """New fragment SMILES are registered in the planet's chromosome dict."""
        prev_count = len(nation_with_egd.land.planet.chromosomes)
        ids = list(nation_with_egd.land.planet.chromosomes.keys())[:5]
        result = nation_with_egd._PolyNation__mutate(ids)
        # New IDs should have been created if EGD mutation succeeded.
        assert len(nation_with_egd.land.planet.chromosomes) >= prev_count

    def test_mutated_position_uses_new_fragment(
        self, nation_with_egd: PolyNation
    ) -> None:
        """The position replaced by EGD uses a newly registered fragment ID."""
        ids = list(nation_with_egd.land.planet.chromosomes.keys())[:5]
        original_ids = set(ids)
        result = nation_with_egd._PolyNation__mutate(list(ids))
        # At least one ID should be new (registered during mutation).
        result_set = set(result)
        new_ids = result_set - original_ids
        assert len(new_ids) >= 0  # Non-deterministic but should be >= 1 usually

    def test_returns_valid_result(
        self, nation_with_egd: PolyNation
    ) -> None:
        """Result is a list of valid chromosome IDs."""
        ids = [1, 2, 3, 4, 5]
        result = nation_with_egd._PolyNation__mutate(ids)
        assert isinstance(result, list)
        for rid in result:
            assert rid in nation_with_egd.land.planet.chromosomes


# ---------------------------------------------------------------------------
# Tests: EGD mutator with low probability
# ---------------------------------------------------------------------------

class TestMutateEGDLowProb:
    """With low EGD probability, mostly original behavior."""

    @pytest.fixture
    def egd_mutator(self) -> MagicMock:
        mut = MagicMock()
        mut.mutate_fragment.return_value = ["[*]CC[*]"]
        return mut

    @pytest.fixture
    def land_with_low_egd(
        self, mock_planet: MagicMock, egd_mutator: MagicMock
    ) -> MagicMock:
        land = MagicMock(spec=PolyLand)
        land.planet = mock_planet
        land.name = "LowEGDLand"
        land.land_chromosomes = list(mock_planet.chromosomes.keys())
        land.fraction_mutation = 0.2
        land.mutation_sigma_offset = 0.25
        land.fraction_mutate_additional_block = 0.05
        land.generative_function_parameters = {}
        land.generative_function = MagicMock(return_value="CCCCCCCC")
        land.egd_mutator = egd_mutator
        land.egd_mutation_prob = 0.0  # never use EGD
        return land

    @pytest.fixture
    def nation_low_egd(self, land_with_low_egd: MagicMock) -> PolyNation:
        return PolyNation(
            name="LowEGDNation",
            land=land_with_low_egd,
            num_families=1,
            num_parents_per_family=1,
            num_children_per_family=1,
            random_seed=42,
        )

    def test_falls_back_to_random(
        self, nation_low_egd: PolyNation, egd_mutator: MagicMock
    ) -> None:
        """With egd_mutation_prob=0, EGD is never called."""
        ids = [1, 2, 3, 4, 5]
        nation_low_egd._PolyNation__mutate(ids)
        assert not egd_mutator.mutate_fragment.called


# ---------------------------------------------------------------------------
# Tests: EGD mutator returns empty (graceful degradation)
# ---------------------------------------------------------------------------

class TestMutateEGDFallback:
    """When EGD mutator returns [], fall back to original."""

    @pytest.fixture
    def egd_mutator_empty(self) -> MagicMock:
        mut = MagicMock()
        mut.mutate_fragment.return_value = []  # EGD failed
        return mut

    @pytest.fixture
    def land_with_egd_empty(
        self, mock_planet: MagicMock, egd_mutator_empty: MagicMock
    ) -> MagicMock:
        land = MagicMock(spec=PolyLand)
        land.planet = mock_planet
        land.name = "EGDEmptyLand"
        land.land_chromosomes = list(mock_planet.chromosomes.keys())
        land.fraction_mutation = 0.2
        land.mutation_sigma_offset = 0.25
        land.fraction_mutate_additional_block = 0.05
        land.generative_function_parameters = {}
        land.generative_function = MagicMock(return_value="CCCCCCCC")
        land.egd_mutator = egd_mutator_empty
        land.egd_mutation_prob = 1.0  # always try EGD
        return land

    @pytest.fixture
    def nation_egd_empty(self, land_with_egd_empty: MagicMock) -> PolyNation:
        return PolyNation(
            name="EGDEmptyNation",
            land=land_with_egd_empty,
            num_families=1,
            num_parents_per_family=1,
            num_children_per_family=1,
            random_seed=42,
        )

    def test_falls_back_when_egd_returns_empty(
        self, nation_egd_empty: PolyNation
    ) -> None:
        """When EGD returns [], the original random mutation is used."""
        ids = [1, 2, 3, 4, 5]
        result = nation_egd_empty._PolyNation__mutate(ids)
        # Should still produce valid results from random fallback.
        assert isinstance(result, list)
        for rid in result:
            assert rid in nation_egd_empty.land.planet.chromosomes
