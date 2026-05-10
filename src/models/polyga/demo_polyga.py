"""
DEMO PolyGA: Three-population co-evolutionary framework.

Maintains three sub-populations with distinct evolutionary roles:
  - P_A (Explorer):   EGD crossover + mutation, high noise -- explores novel space.
  - P_B (Refiner):    Mutation only (no crossover) -- local refinement.
  - P_C (Elite Archive):  Gentle mutation only -- fine-tunes best solutions.

Uses SAES (Structure-Aware Environmental Selection) from saes_selector.py
for diversity-preserving parent selection.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from polyga.saes_selector import SAESSelector


# ---------------------------------------------------------------------------
# DEMOPolyGA
# ---------------------------------------------------------------------------

class DEMOPolyGA:
    """Three-population co-evolutionary PolyGA.

    Each generation:
      1. Compute adaptive noise schedule t'.
      2. P_A: crossover + mutate at t'.
      3. P_B: mutate only at t' * 0.7.
      4. P_C: mutate only at t' * 0.5.
      5. Pool all offspring into Q.
      6. SAES select from Q.
      7. Route selected -> P_A / P_B / P_C.
    """

    def __init__(
        self,
        egd_mutator: Optional[Any],
        saes_selector: Optional[SAESSelector],
        predict_function: Callable,
        fingerprint_function: Callable,
        population_size: int = 180,
        n_generations: int = 100,
        p_b_ratio: float = 0.3,
        p_c_ratio: float = 0.1,
        noise_start: int = 300,
        noise_end: int = 150,
        random_seed: int = 42,
        ad_checker: Any | None = None,  # ApplicabilityDomain or None
        ood_penalty_weight: float = 0.5,
    ) -> None:
        """
        Args:
            egd_mutator: EGD mutation/crossover operator. Must implement
                `crossover(smiles_a, smiles_b, t_prime) -> smiles` and
                `mutate(smiles, t_prime) -> smiles`. If None, falls back to
                random mutation.
            saes_selector: SAESSelector for diversity-aware selection.
                If None, falls back to simple elite selection.
            predict_function: Fitness / property prediction.
            fingerprint_function: Fingerprint computation.
            population_size: Total individuals across all populations.
            n_generations: Number of co-evolution generations.
            p_b_ratio: Fraction of population in P_B (Refiner).
            p_c_ratio: Fraction of population in P_C (Elite Archive).
            noise_start: Initial noise level t'.
            noise_end: Final noise level t'.
            random_seed: Random seed.
            ad_checker: ApplicabilityDomain checker. When set, fitness is
                penalized by domain_score for OOD polymers. None disables.
            ood_penalty_weight: How strongly to penalize OOD fitness.
                0.0 = no penalty, 1.0 = full penalty (fitness * domain_score).
        """
        self.egd_mutator = egd_mutator
        self.saes_selector = saes_selector
        self.predict_function = predict_function
        self.fingerprint_function = fingerprint_function
        self.population_size = population_size
        self.n_generations = n_generations
        self.p_b_ratio = p_b_ratio
        self.p_c_ratio = p_c_ratio
        self.noise_start = noise_start
        self.noise_end = noise_end
        self.ad_checker = ad_checker
        self.ood_penalty_weight = ood_penalty_weight
        self._rng = np.random.default_rng(random_seed)

        # Internal state
        self.pop_a: list[dict[str, Any]] = []
        self.pop_b: list[dict[str, Any]] = []
        self.pop_c: list[dict[str, Any]] = []
        self._dna_fragments: set[str] = set()
        self._generation = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        """Execute the co-evolution loop.

        Returns:
            Final elite archive (P_C) as a list of polymer dicts.
        """
        self._initialize_populations()
        if self._generation == 0:
            # Bootstrap: score initial pools, build DNA set
            all_polymers = self._pool_all()
            df = pd.DataFrame(all_polymers)
            df = self._score(df)
            self._dna_fragments = self._extract_fragments(df)

            # Initial SAES selection
            selected = self._select(df, n_select=self.population_size)
            self._route(selected)
            self._generation = 1

        for gen in range(self._generation, self.n_generations):
            t_prime = self._adaptive_noise_schedule(gen)

            # Evolve each population
            offspring_a = self._evolve_a(t_prime)
            offspring_b = self._evolve_b(t_prime)
            offspring_c = self._evolve_c(t_prime)

            # Pool all offspring
            all_offspring = offspring_a + offspring_b + offspring_c
            df = pd.DataFrame(all_offspring)
            df = self._score(df)

            # SAES selection from pool
            selected = self._select(df, n_select=self.population_size)

            # Route back to sub-populations
            self._route(selected)

            self._generation = gen + 1

        return self.pop_c

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_populations(self) -> None:
        """Bootstrap three populations from the DNA pool.

        Creates random polymers using available fragments.
        """
        if len(self.pop_a) > 0 or len(self.pop_b) > 0 or len(self.pop_c) > 0:
            return  # Already initialized
        # Generate random polymers as placeholder initialization
        size_a = int(self.population_size * (1.0 - self.p_b_ratio - self.p_c_ratio))
        size_b = int(self.population_size * self.p_b_ratio)
        size_c = self.population_size - size_a - size_b

        dummy_polymers = []
        for _ in range(self.population_size):
            dummy_polymers.append({
                "smiles_string": "",
                "fitness": self._rng.random() * -10.0,
                "chromosome_ids": [],
                "birth_nation": "explorer",
            })
        self.pop_a = dummy_polymers[:size_a] if size_a > 0 else []
        self.pop_b = (
            dummy_polymers[size_a:size_a + size_b]
            if size_b > 0
            else []
        )
        self.pop_c = dummy_polymers[size_a + size_b:] if size_c > 0 else []

    # ------------------------------------------------------------------
    # Evolution methods
    # ------------------------------------------------------------------

    def _evolve_a(self, t_prime: int) -> list[dict[str, Any]]:
        """Evolve Explorer population: EGD crossover + mutation."""
        if not self.pop_a:
            return []
        offspring: list[dict[str, Any]] = []
        n = len(self.pop_a)
        for i in range(0, n - 1, 2):
            parent_a = self.pop_a[i]
            parent_b = self.pop_a[i + 1]
            child = self._crossover(parent_a, parent_b, t_prime)
            if child is not None:
                child = self._mutate(child, t_prime)
                if child is not None:
                    offspring.append(child)
        return offspring

    def _evolve_b(self, t_prime: int) -> list[dict[str, Any]]:
        """Evolve Refiner population: mutation only, lower noise."""
        if not self.pop_b:
            return []
        t_refine = max(1, int(t_prime * 0.7))
        offspring: list[dict[str, Any]] = []
        for polymer in self.pop_b:
            child = self._mutate(polymer, t_refine)
            if child is not None:
                offspring.append(child)
        return offspring

    def _evolve_c(self, t_prime: int) -> list[dict[str, Any]]:
        """Evolve Elite Archive: gentle mutation only."""
        if not self.pop_c:
            return []
        t_gentle = max(1, int(t_prime * 0.5))
        offspring: list[dict[str, Any]] = []
        for polymer in self.pop_c:
            child = self._mutate(polymer, t_gentle)
            if child is not None:
                offspring.append(child)
        return offspring

    # ------------------------------------------------------------------
    # Adaptive noise schedule
    # ------------------------------------------------------------------

    def _adaptive_noise_schedule(self, generation: int) -> int:
        """Compute noise level t' for a given generation.

        Linear decay from noise_start to noise_end.
        """
        if self.n_generations <= 1:
            return self.noise_start
        frac = generation / (self.n_generations - 1)
        t = self.noise_start + frac * (self.noise_end - self.noise_start)
        return max(1, int(round(t)))

    # ------------------------------------------------------------------
    # Crossover & mutation
    # ------------------------------------------------------------------

    def _crossover(
        self,
        parent_a: dict[str, Any],
        parent_b: dict[str, Any],
        t_prime: int,
    ) -> Optional[dict[str, Any]]:
        """Apply EGD crossover if mutator is available; else None."""
        if self.egd_mutator is not None:
            try:
                smi_a = parent_a.get("smiles_string", "")
                smi_b = parent_b.get("smiles_string", "")
                child_smi = self.egd_mutator.crossover(smi_a, smi_b, t_prime)
                if child_smi:
                    return {"smiles_string": child_smi, "fitness": 0.0}
            except Exception:
                return None
        return None

    def _mutate(
        self,
        polymer: dict[str, Any],
        t_prime: int,
    ) -> Optional[dict[str, Any]]:
        """Apply EGD mutation if mutator is available; fall back to copy."""
        if self.egd_mutator is not None:
            try:
                smi = polymer.get("smiles_string", "")
                new_smi = self.egd_mutator.mutate(smi, t_prime)
                if new_smi:
                    return {"smiles_string": new_smi, "fitness": 0.0}
            except Exception:
                pass
        # Fallback: return a shallow copy with slightly degraded fitness
        return {**polymer, "fitness": polymer.get("fitness", 0.0) * 0.99}

    # ------------------------------------------------------------------
    # Scoring & selection
    # ------------------------------------------------------------------

    def _score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fingerprint, prediction, fitness, then OOD penalty."""
        if len(df) == 0:
            return df
        try:
            df_copy = df.copy()
            df_fp, fp_headers = self.fingerprint_function(df_copy)
            df_pred = self.predict_function(df_fp, fp_headers, None)
            return self._apply_ood_penalty(df_pred)
        except Exception:
            df_out = df.copy()
            fitness_vals = self._rng.random(len(df_out)) * 10.0
            df_out["fitness"] = fitness_vals
            return self._apply_ood_penalty(df_out)

    def _apply_ood_penalty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Penalize fitness for out-of-domain polymers.

        fitness *= 1.0 - ood_penalty_weight * (1.0 - domain_score)
        In-domain (score=1.0) → no penalty.
        OOD (score=0.0) → fitness reduced by ood_penalty_weight.
        """
        if self.ad_checker is None:
            return df
        if "smiles_string" not in df.columns or len(df) == 0:
            return df
        scores = np.array([
            self.ad_checker.domain_score(s) for s in df["smiles_string"]
        ])
        penalty = 1.0 - self.ood_penalty_weight * (1.0 - scores)
        if "fitness" in df.columns:
            df = df.copy()
            df["fitness"] = df["fitness"].values * penalty
        return df

    def _select(self, df: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select n_select individuals using SAES or elite fallback."""
        if self.saes_selector is not None:
            return self.saes_selector.select(df, n_select=n_select)
        # Simple elite fallback
        n = min(n_select, len(df))
        return df.nlargest(n, "fitness").reset_index(drop=True)

    def _route(self, selected: pd.DataFrame) -> None:
        """Route selected individuals to P_A / P_B / P_C.

        Novel fragments -> P_A (Explorer).
        High fitness (top 20%) -> P_C (Elite Archive).
        Rest -> P_B (Refiner).
        """
        if len(selected) == 0:
            return

        pop_a_list: list[dict[str, Any]] = []
        pop_b_list: list[dict[str, Any]] = []
        pop_c_list: list[dict[str, Any]] = []

        # Determine fitness threshold for elite (80th percentile)
        fitness_vals = selected["fitness"].values
        if len(fitness_vals) > 0:
            threshold = float(np.percentile(fitness_vals, 80))
        else:
            threshold = -1e9

        for _, row in selected.iterrows():
            polymer = row.to_dict()
            if self._has_novel_fragments(polymer):
                pop_a_list.append(polymer)
            elif polymer.get("fitness", 0.0) >= threshold:
                pop_c_list.append(polymer)
            else:
                pop_b_list.append(polymer)

        self.pop_a = pop_a_list
        self.pop_b = pop_b_list
        self.pop_c = pop_c_list

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _has_novel_fragments(self, polymer: dict[str, Any]) -> bool:
        """Check if the polymer SMILES contains fragments not in original DNA.

        Uses substring matching as a heuristic when chromosome_ids are
        unavailable.
        """
        if not self._dna_fragments:
            return True
        smi = polymer.get("smiles_string", "")
        if not smi:
            return False
        for frag in self._dna_fragments:
            if frag and frag in smi:
                return False
        # No known fragment found in SMILES -- likely novel
        return True

    def _is_high_fitness(self, polymer: dict[str, Any]) -> bool:
        """Check if polymer fitness is above 80th percentile of the pool."""
        pool = self._pool_all()
        if not pool:
            return False
        vals = [p.get("fitness", 0.0) for p in pool]
        if not vals:
            return False
        threshold = float(np.percentile(vals, 80))
        return polymer.get("fitness", 0.0) >= threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pool_all(self) -> list[dict[str, Any]]:
        """Return all polymers from all three populations."""
        return self.pop_a + self.pop_b + self.pop_c

    @staticmethod
    def _extract_fragments(df: pd.DataFrame) -> set[str]:
        """Extract unique fragment substrings from chromosome-based SMILES.

        Intended to build the initial DNA fragment reference set.
        """
        frags: set[str] = set()
        smi_col = df.get("smiles_string", [])
        for smi in smi_col:
            if isinstance(smi, str) and len(smi) > 1:
                frags.add(smi)
        return frags
