import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.optimization import run_optimization


def test_summary_includes_sa_and_normalized_audit_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "planetary_database.sqlite"
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE polymer (generation INTEGER, smiles_string TEXT, properties TEXT)"
    )
    rows = [
        (0, "[*]CC[*]", {"Tg": 300.0, "DC": 6.0, "SA_score": 8.0, "fitness_gate": 0.1}),
        (0, "[*]c1ccccc1[*]", {"Tg": 450.0, "DC": 2.0, "SA_score": 2.0, "fitness_gate": 1.0}),
    ]
    con.executemany(
        "INSERT INTO polymer VALUES (?, ?, ?)",
        [(gen, smi, json.dumps(props)) for gen, smi, props in rows],
    )
    con.commit()
    con.close()

    out_csv = tmp_path / "summary.csv"
    run_optimization._NORMALIZATION_CACHE.clear()
    run_optimization._summarize_generation_to_csv(db_path, 0, out_csv, run_optimization.CONFIG)

    summary = pd.read_csv(out_csv)
    for column in ["sa_mean", "sa_min", "sa_max", "best_sa", "best_tg_norm", "best_dc_norm", "best_sa_norm", "best_fitness_gate"]:
        assert column in summary.columns
    assert summary.loc[0, "best_sa"] == 2.0
    assert summary.loc[0, "best_fitness_gate"] == 1.0


def test_summarize_generation_propagates_errors(monkeypatch, tmp_path: Path) -> None:
    def fail_summary(*args, **kwargs):
        raise RuntimeError("summary failed")

    monkeypatch.setattr(run_optimization, "_summarize_generation_to_csv", fail_summary)

    with pytest.raises(RuntimeError, match="summary failed"):
        run_optimization.summarize_generation(tmp_path / "missing.sqlite", 0, tmp_path / "summary.csv", {})
