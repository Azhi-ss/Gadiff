from unittest.mock import MagicMock

import pandas as pd

from polyga.polygod import PolyLand, PolyNation, PolyPlanet


def _uid_counter(start: int = 100):
    current = [start]

    def next_uid() -> int:
        current[0] += 1
        return current[0]

    return next_uid


def _make_land() -> MagicMock:
    planet = MagicMock(spec=PolyPlanet)
    planet.name = "Planet"
    planet.species = "polymers"
    planet.num_nations = 1
    planet.chromosomes = {1: "[Bi]CC[Bi]", 2: "[Bi]NN[Bi]", 3: "[Bi]OO[Bi]"}
    planet.uid = MagicMock(side_effect=_uid_counter())

    land = MagicMock(spec=PolyLand)
    land.name = "Land"
    land.planet = planet
    land.land_chromosomes = list(planet.chromosomes)
    land.fitness_function = MagicMock(side_effect=lambda df, fp_headers: df)
    land.generative_function = MagicMock(return_value="[*]CC[*]")
    land.generative_function_parameters = {}
    land.fraction_mutation = 0.0
    land.mutation_sigma_offset = 0.0
    land.fraction_mutate_additional_block = 0.0
    land.crossover_position = "center"
    land.crossover_sigma_offset = 0.0
    land.elite_retention_count = 1
    land.target_population_size = 4
    land.refill_num_chromosomes_initial = 1
    return land


def test_propagate_species_retains_elite_and_refills_population(monkeypatch) -> None:
    land = _make_land()
    nation = PolyNation(
        name="Nation",
        land=land,
        num_population_initial=0,
        num_chromosomes_initial=1,
        num_families=1,
        num_parents_per_family=2,
        num_children_per_family=1,
        random_seed=42,
    )
    nation.fp_headers = []
    nation.population = pd.DataFrame(
        [
            {
                "chromosome_ids": [1],
                "num_chromosomes": 1,
                "planetary_id": 1,
                "parent_1_id": 0,
                "parent_2_id": 0,
                "smiles_string": "[*]elite[*]",
                "birth_land": "Land",
                "birth_nation": "Nation",
                "birth_planet": "Planet",
                "fitness": 10.0,
            },
            {
                "chromosome_ids": [2],
                "num_chromosomes": 1,
                "planetary_id": 2,
                "parent_1_id": 0,
                "parent_2_id": 0,
                "smiles_string": "[*]weak[*]",
                "birth_land": "Land",
                "birth_nation": "Nation",
                "birth_planet": "Planet",
                "fitness": 1.0,
            },
        ]
    )
    monkeypatch.setattr(nation, "_PolyNation__selection", lambda: [[1, 2]])
    monkeypatch.setattr(nation, "_PolyNation__crossover", lambda families: ([[1]], [[1, 2]]))
    monkeypatch.setattr(nation, "_PolyNation__mutate", lambda child: child)

    nation.propagate_species(take_census=False, narrate=False)

    assert len(nation.population) == 4
    assert "[*]elite[*]" in set(nation.population["smiles_string"])
    elite_clone = nation.population[nation.population["smiles_string"] == "[*]elite[*]"].iloc[0]
    assert elite_clone["parent_1_id"] == 1
    assert elite_clone["planetary_id"] != 1
