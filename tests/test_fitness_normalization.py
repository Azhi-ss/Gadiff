import pandas as pd

from src.optimization import run_optimization


def test_balanced_fitness_prefers_high_tg_low_dc_low_sa() -> None:
    fitness_fn = run_optimization.create_fitness_function(
        weight_tg=1.0,
        weight_dc=1.0,
        weight_sa=1.0,
    )
    df = pd.DataFrame(
        {
            "Tg": [700.0, 300.0],
            "DC": [2.0, 6.0],
            "SA_score": [2.0, 8.0],
        }
    )

    result = fitness_fn(df, [])

    assert result.loc[0, "tg_norm"] == 1.0
    assert result.loc[0, "dc_norm"] == 1.0
    assert result.loc[0, "sa_norm"] == 1.0
    assert result.loc[0, "fitness"] > result.loc[1, "fitness"]


def test_fitness_penalizes_missing_sa_score() -> None:
    fitness_fn = run_optimization.create_fitness_function()
    df = pd.DataFrame({"Tg": [500.0], "DC": [3.0], "SA_score": [float("nan")]})

    result = fitness_fn(df, [])

    assert result.loc[0, "fitness"] < -1e8


def test_fitness_applies_soft_gates() -> None:
    fitness_fn = run_optimization.create_fitness_function(
        weight_tg=1.0,
        weight_dc=1.0,
        weight_sa=1.0,
    )
    df = pd.DataFrame(
        {
            "Tg": [450.0, 650.0],
            "DC": [2.5, 4.5],
            "SA_score": [2.5, 6.0],
        }
    )

    result = fitness_fn(df, [])

    assert result.loc[0, "fitness_gate"] == 1.0
    assert result.loc[1, "fitness_gate"] < 1.0
    assert result.loc[1, "fitness"] < result.loc[1, "base_fitness"]


def test_normalization_cache_is_locked_after_first_call(monkeypatch) -> None:
    monkeypatch.setitem(run_optimization.CONFIG, "fitness_normalization", "minmax")
    fitness_fn = run_optimization.create_fitness_function()
    first = pd.DataFrame(
        {"Tg": [300.0, 700.0], "DC": [6.0, 2.0], "SA_score": [8.0, 2.0]}
    )
    second = pd.DataFrame(
        {"Tg": [1000.0], "DC": [1.0], "SA_score": [1.0]}
    )

    fitness_fn(first, [])
    result = fitness_fn(second, [])

    assert run_optimization._NORMALIZATION_CACHE["tg_L"] == 300.0
    assert result.loc[0, "tg_norm"] == 1.0
