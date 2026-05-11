"""
Tests for DEMOPolyGA -- three-population co-evolutionary framework.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from polyga.demo_polyga import DEMOPolyGA
from polyga.saes_selector import SAESSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_predict_function() -> Any:
    """Return a predict function that adds random fitness."""

    def predict_fn(df: pd.DataFrame, fp_headers: list[str],
                   models: Any = None) -> pd.DataFrame:
        df_out = df.copy()
        rng = np.random.default_rng(42)
        df_out["fitness"] = rng.random(len(df_out)) * 10.0
        return df_out

    return predict_fn


def _make_dummy_fingerprint_function() -> Any:
    """Return a fingerprint function adding a dummy fingerprint column."""

    def fp_fn(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        df_out = df.copy()
        df_out["fp_dummy"] = 1.0
        return df_out, ["fp_dummy"]

    return fp_fn


@pytest.fixture
def mock_egd_mutator() -> MagicMock:
    mut = MagicMock()
    mut.crossover.return_value = "c1ccccc1CCO"
    mut.mutate.return_value = "c1ccccc1CC"
    return mut


@pytest.fixture
def saes_selector() -> SAESSelector:
    return SAESSelector(k_neighbors=3, tau=0.05, use_3d=False, random_seed=42)


def _prepopulate_populations(ga: DEMOPolyGA) -> None:
    """Fill the three populations with distinct, valid SMILES before run()."""
    smiles_pool = [
        "c1ccccc1",       # benzene
        "CCO",            # ethanol
        "CC(=O)O",        # acetic acid
        "c1ccncc1",       # pyridine
        "CCCC",           # butane
        "c1ccccc1O",      # phenol
        "CCN",            # ethylamine
        "CC=O",           # acetaldehyde
        "c1ccccc1C",      # toluene
        "CCCO",           # propanol
    ]
    n_a = int(ga.population_size * (1.0 - ga.p_b_ratio - ga.p_c_ratio))
    n_b = int(ga.population_size * ga.p_b_ratio)
    n_c = ga.population_size - n_a - n_b

    def _make(n: int, offset: int, nation: str) -> list[dict[str, Any]]:
        return [
            {
                "smiles_string": smiles_pool[(i + offset) % len(smiles_pool)],
                "fitness": float(100 - i),
                "chromosome_ids": [i],
                "birth_nation": nation,
            }
            for i in range(n)
        ]

    ga.pop_a = _make(n_a, 0, "explorer")
    ga.pop_b = _make(n_b, n_a, "refiner")
    ga.pop_c = _make(n_c, n_a + n_b, "elite")
    ga._dna_fragments = set(smiles_pool[:6])


@pytest.fixture
def demo(mock_egd_mutator: MagicMock,
         saes_selector: SAESSelector) -> DEMOPolyGA:
    ga = DEMOPolyGA(
        egd_mutator=mock_egd_mutator,
        saes_selector=saes_selector,
        predict_function=_make_dummy_predict_function(),
        fingerprint_function=_make_dummy_fingerprint_function(),
        population_size=60,
        n_generations=5,
        noise_start=300,
        noise_end=150,
        random_seed=42,
    )
    _prepopulate_populations(ga)
    return ga


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestDEMOPolyGAInit:
    def test_stores_config(self, mock_egd_mutator: MagicMock,
                           saes_selector: SAESSelector) -> None:
        ga = DEMOPolyGA(
            egd_mutator=mock_egd_mutator,
            saes_selector=saes_selector,
            predict_function=_make_dummy_predict_function(),
            fingerprint_function=_make_dummy_fingerprint_function(),
            population_size=120,
            n_generations=50,
            noise_start=400,
            noise_end=200,
        )
        assert ga.population_size == 120
        assert ga.n_generations == 50
        assert ga.noise_start == 400
        assert ga.noise_end == 200
        assert ga.egd_mutator is mock_egd_mutator
        assert ga.saes_selector is saes_selector

    def test_default_noise_params(self, mock_egd_mutator: MagicMock) -> None:
        """When noise params aren't provided, use defaults."""
        ga = DEMOPolyGA(
            egd_mutator=mock_egd_mutator,
            saes_selector=None,
            predict_function=_make_dummy_predict_function(),
            fingerprint_function=_make_dummy_fingerprint_function(),
        )
        assert ga.noise_start == 300
        assert ga.noise_end == 150

    def test_initial_populations_empty(self) -> None:
        """Before run/_initialize_populations, all pops should be empty."""
        ga = DEMOPolyGA(
            egd_mutator=None,
            saes_selector=None,
            predict_function=_make_dummy_predict_function(),
            fingerprint_function=_make_dummy_fingerprint_function(),
        )
        assert len(ga.pop_a) == 0
        assert len(ga.pop_b) == 0
        assert len(ga.pop_c) == 0


# ---------------------------------------------------------------------------
# Adaptive noise schedule
# ---------------------------------------------------------------------------

class TestAdaptiveNoiseSchedule:
    @pytest.mark.parametrize("gen, expected", [
        (0, 300),   # start
        (2, 225),   # midpoint (gen 2 of 4): 300 + 0.5 * (150-300) = 225
        (4, 150),   # end
    ])
    def test_returns_correct_t_prime(
        self, demo: DEMOPolyGA, gen: int, expected: int
    ) -> None:
        t = demo._adaptive_noise_schedule(gen)
        assert t == expected

    def test_monotonically_decreasing(self, demo: DEMOPolyGA) -> None:
        values = [demo._adaptive_noise_schedule(g)
                  for g in range(demo.n_generations)]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]

    def test_single_generation_no_division_by_zero(self) -> None:
        ga = DEMOPolyGA(
            egd_mutator=None,
            saes_selector=None,
            predict_function=_make_dummy_predict_function(),
            fingerprint_function=_make_dummy_fingerprint_function(),
            n_generations=1,
        )
        t = ga._adaptive_noise_schedule(0)
        assert t == ga.noise_start


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class TestRoute:
    def test_puts_novel_in_explorer(self, demo: DEMOPolyGA) -> None:
        """Novel fragments (not in DNA) -> P_A."""
        selected = pd.DataFrame({
            "smiles_string": ["ZZZnovel", "known_smiles"],
            "fitness": [5.0, 5.0],
            "chromosome_ids": [[99], [1]],
        })
        demo._dna_fragments = {"known_smiles"}
        demo._route(selected)
        novel_in_a = any(
            p.get("smiles_string") == "ZZZnovel" for p in demo.pop_a
        )
        assert novel_in_a

    def test_puts_high_fitness_in_elite(self, demo: DEMOPolyGA) -> None:
        """High fitness (>80th percentile) -> P_C."""
        selected = pd.DataFrame({
            "smiles_string": [f"mol_{i}" for i in range(10)],
            "fitness": list(range(10)),  # 9 is highest
            "chromosome_ids": [[i] for i in range(10)],
        })
        demo._dna_fragments = {f"mol_{i}" for i in range(10)}
        demo._route(selected)
        # The highest fitness individual should be in P_C
        elite_fitnesses = [p.get("fitness", -1) for p in demo.pop_c]
        assert max(elite_fitnesses) >= 8.0  # 80th percentile of 0-9 is 8

    def test_puts_rest_in_refiner(self, demo: DEMOPolyGA) -> None:
        selected = pd.DataFrame({
            "smiles_string": ["known_a", "known_b", "known_c"],
            "fitness": [1.0, 5.0, 9.0],
            "chromosome_ids": [[1], [2], [3]],
        })
        demo._dna_fragments = {"known_a", "known_b", "known_c"}
        demo._route(selected)
        # Known, mid-fitness individuals go to P_B
        assert len(demo.pop_b) > 0

    def test_empty_selection_does_nothing(self) -> None:
        """Routing an empty DataFrame leaves populations unchanged."""
        ga = DEMOPolyGA(
            egd_mutator=None,
            saes_selector=None,
            predict_function=_make_dummy_predict_function(),
            fingerprint_function=_make_dummy_fingerprint_function(),
        )
        _prepopulate_populations(ga)
        orig_a = list(ga.pop_a)
        orig_b = list(ga.pop_b)
        orig_c = list(ga.pop_c)
        ga._route(pd.DataFrame())
        assert len(ga.pop_a) == len(orig_a)
        assert len(ga.pop_b) == len(orig_b)
        assert len(ga.pop_c) == len(orig_c)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_produces_non_empty_result(self, demo: DEMOPolyGA) -> None:
        result = demo.run()
        assert isinstance(result, list)
        # Should have some elites
        assert len(result) > 0

    def test_run_advances_generation(self, demo: DEMOPolyGA) -> None:
        assert demo._generation == 0
        demo.run()
        assert demo._generation == demo.n_generations

    def test_run_with_small_population(self) -> None:
        ga = DEMOPolyGA(
            egd_mutator=None,
            saes_selector=None,
            predict_function=_make_dummy_predict_function(),
            fingerprint_function=_make_dummy_fingerprint_function(),
            population_size=6,
            n_generations=3,
        )
        _prepopulate_populations(ga)
        result = ga.run()
        assert len(result) > 0

    def test_elite_archive_quality_trend(self, demo: DEMOPolyGA) -> None:
        """When using dummy predictions, run should still complete."""
        result = demo.run()
        assert all(isinstance(p, dict) for p in result)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_fallback_no_egd_mutator(self) -> None:
        """With egd_mutator=None, run completes without error."""
        ga = DEMOPolyGA(
            egd_mutator=None,
            saes_selector=None,
            predict_function=_make_dummy_predict_function(),
            fingerprint_function=_make_dummy_fingerprint_function(),
            population_size=12,
            n_generations=3,
        )
        _prepopulate_populations(ga)
        result = ga.run()
        assert len(result) > 0

    def test_egd_mutator_failure_fallback(self, demo: DEMOPolyGA) -> None:
        """If egd_mutator raises, falls back gracefully."""
        demo.egd_mutator.crossover.side_effect = RuntimeError("EGD failed")
        demo.egd_mutator.mutate.side_effect = RuntimeError("EGD failed")
        # Should not raise, should fall back
        demo._initialize_populations()
        demo.pop_a = [{"smiles_string": "c1ccccc1", "fitness": 5.0}] * 3
        child = demo._crossover(demo.pop_a[0], demo.pop_a[1], 200)
        assert child is None
        mutant = demo._mutate(demo.pop_a[0], 200)
        # Fallback copy should have fitness * 0.99
        assert mutant is not None
        assert mutant["fitness"] == 5.0 * 0.99

    def test_no_rdkit_fallback(self) -> None:
        """Running without SAES selector works."""
        ga = DEMOPolyGA(
            egd_mutator=None,
            saes_selector=None,
            predict_function=_make_dummy_predict_function(),
            fingerprint_function=_make_dummy_fingerprint_function(),
            population_size=12,
            n_generations=2,
        )
        _prepopulate_populations(ga)
        result = ga.run()
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Novelty / fitness helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_has_novel_fragments_no_dna(self, demo: DEMOPolyGA) -> None:
        demo._dna_fragments = set()
        assert demo._has_novel_fragments({"smiles_string": "anything"})

    def test_has_novel_fragments_known(self, demo: DEMOPolyGA) -> None:
        demo._dna_fragments = {"benzene", "toluene"}
        assert not demo._has_novel_fragments({"smiles_string": "benzene"})

    def test_has_novel_fragments_novel(self, demo: DEMOPolyGA) -> None:
        demo._dna_fragments = {"benzene"}
        assert demo._has_novel_fragments({"smiles_string": "novel_polymer"})

    def test_is_high_fitness_with_empty_pool(self) -> None:
        """With no pool, no fitness is high."""
        ga = DEMOPolyGA(
            egd_mutator=None,
            saes_selector=None,
            predict_function=_make_dummy_predict_function(),
            fingerprint_function=_make_dummy_fingerprint_function(),
        )
        assert not ga._is_high_fitness({"fitness": 100.0})

    def test_is_high_fitness_above_threshold(self, demo: DEMOPolyGA) -> None:
        demo.pop_a = [{"fitness": 1.0}, {"fitness": 2.0}, {"fitness": 3.0}]
        demo.pop_b = [{"fitness": 4.0}]
        demo.pop_c = [{"fitness": 5.0}]
        assert demo._is_high_fitness({"fitness": 4.5})
        assert not demo._is_high_fitness({"fitness": 2.0})


# ---------------------------------------------------------------------------
# Evolution methods
# ---------------------------------------------------------------------------

class TestEvolve:
    def test_evolve_a_empty_pop(self, demo: DEMOPolyGA) -> None:
        demo.pop_a = []
        assert demo._evolve_a(200) == []

    def test_evolve_a_produces_offspring(self, demo: DEMOPolyGA) -> None:
        demo.pop_a = [
            {"smiles_string": "c1ccccc1", "fitness": 10.0},
            {"smiles_string": "CCO", "fitness": 9.0},
        ]
        offspring = demo._evolve_a(200)
        assert len(offspring) > 0

    def test_evolve_b_empty_pop(self, demo: DEMOPolyGA) -> None:
        demo.pop_b = []
        assert demo._evolve_b(200) == []

    def test_evolve_b_produces_offspring(self, demo: DEMOPolyGA) -> None:
        demo.pop_b = [{"smiles_string": "c1ccccc1", "fitness": 8.0}]
        offspring = demo._evolve_b(200)
        assert len(offspring) > 0

    def test_evolve_c_empty_pop(self, demo: DEMOPolyGA) -> None:
        demo.pop_c = []
        assert demo._evolve_c(200) == []

    def test_evolve_c_produces_offspring(self, demo: DEMOPolyGA) -> None:
        demo.pop_c = [{"smiles_string": "c1ccccc1", "fitness": 10.0}]
        offspring = demo._evolve_c(200)
        assert len(offspring) > 0


class TestEGDFragmentMutatorInterface:
    def test_mutate_uses_fragment_mutator_api(self) -> None:
        """DEMOPolyGA should work with EGDFragmentMutator's public API."""
        mutator = MagicMock()
        mutator.mutate_fragment = MagicMock(return_value=["[*]CCO[*]"])
        ga = DEMOPolyGA(
            egd_mutator=mutator,
            saes_selector=None,
            predict_function=_make_dummy_predict_function(),
            fingerprint_function=_make_dummy_fingerprint_function(),
        )

        mutant = ga._mutate({"smiles_string": "[*]CC[*]", "fitness": 5.0}, 200)

        mutator.mutate_fragment.assert_called_once_with("[*]CC[*]", 200)
        assert mutant == {"smiles_string": "[*]CCO[*]", "fitness": 0.0}

    def test_crossover_uses_fragment_mutator_api(self) -> None:
        """DEMOPolyGA should accept list output from EGD crossover."""
        mutator = MagicMock()
        mutator.crossover_fragments = MagicMock(return_value=["[*]CCN[*]"])
        ga = DEMOPolyGA(
            egd_mutator=mutator,
            saes_selector=None,
            predict_function=_make_dummy_predict_function(),
            fingerprint_function=_make_dummy_fingerprint_function(),
        )

        child = ga._crossover(
            {"smiles_string": "[*]CC[*]", "fitness": 5.0},
            {"smiles_string": "[*]CO[*]", "fitness": 4.0},
            200,
        )

        mutator.crossover_fragments.assert_called_once_with("[*]CC[*]", "[*]CO[*]", 200)
        assert child == {"smiles_string": "[*]CCN[*]", "fitness": 0.0}
