from pathlib import Path

import pytest

from src.optimization import run_optimization


def test_default_config_uses_enriched_dna_and_real_weights() -> None:
    assert run_optimization.CONFIG["dna_file"] == "data/enriched_dna.csv"
    assert run_optimization.CONFIG["weight_dir"] == "/internfs/Zy/polygen_assets/mm_polymer/finetune_data"
    assert run_optimization.CONFIG["allow_heuristic_predictor"] is False


def test_resolve_dna_path_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="DNA file not found"):
        run_optimization.resolve_dna_path(tmp_path, "missing.csv")


def test_resolve_dna_path_accepts_repo_relative_path(tmp_path: Path) -> None:
    dna = tmp_path / "data" / "enriched_dna.csv"
    dna.parent.mkdir()
    dna.write_text("chromosome_id,chromosome,num_connections\n1,[Bi]CC[Bi],2\n")

    assert run_optimization.resolve_dna_path(tmp_path, "data/enriched_dna.csv") == dna


def test_mmpolymer_predictor_requires_real_weights(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(run_optimization.CONFIG, "allow_heuristic_predictor", False)
    cfg = {
        **run_optimization.CONFIG,
        "weight_dir": str(tmp_path / "ckpt"),
        "properties": ["Tg", "DC"],
    }

    with pytest.raises(FileNotFoundError, match="MMPolymer weight files not found"):
        run_optimization.maybe_create_mmpolymer_predictor(cfg)
