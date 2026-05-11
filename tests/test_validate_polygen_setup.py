from pathlib import Path

import pytest

from scripts.validate_polygen_setup import validate_polygen_setup


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_validate_polygen_setup_accepts_required_files(tmp_path: Path) -> None:
    _write(tmp_path / "data" / "enriched_dna.csv", "chromosome_id,chromosome,num_connections\n1,[Bi]CC[Bi],2\n")
    _write(tmp_path / "src" / "models" / "MMPolymer" / "dict.txt")
    _write(tmp_path / "data" / "tokenizer" / "vocab.json", "{}")
    _write(tmp_path / "data" / "tokenizer" / "merges.txt")
    _write(tmp_path / "ckpt" / "Tg" / "checkpoint_best.pt")
    _write(tmp_path / "ckpt" / "DC" / "checkpoint_best.pt")

    validate_polygen_setup(tmp_path, tmp_path / "ckpt")


def test_validate_polygen_setup_rejects_non_bi_dna(tmp_path: Path) -> None:
    _write(tmp_path / "data" / "enriched_dna.csv", "chromosome_id,chromosome,num_connections\n1,[*]CC[*],2\n")
    _write(tmp_path / "src" / "models" / "MMPolymer" / "dict.txt")
    _write(tmp_path / "data" / "tokenizer" / "vocab.json", "{}")
    _write(tmp_path / "data" / "tokenizer" / "merges.txt")
    _write(tmp_path / "ckpt" / "Tg" / "checkpoint_best.pt")
    _write(tmp_path / "ckpt" / "DC" / "checkpoint_best.pt")

    with pytest.raises(ValueError, match=r"\[Bi\]"):
        validate_polygen_setup(tmp_path, tmp_path / "ckpt")


def test_validate_polygen_setup_accepts_nested_weight_layout(tmp_path: Path) -> None:
    _write(tmp_path / "data" / "enriched_dna.csv", "chromosome_id,chromosome,num_connections\n1,[Bi]CC[Bi],2\n")
    _write(tmp_path / "src" / "models" / "MMPolymer" / "dict.txt")
    _write(tmp_path / "data" / "tokenizer" / "vocab.json", "{}")
    _write(tmp_path / "data" / "tokenizer" / "merges.txt")
    _write(tmp_path / "ckpt" / "Tg" / "ckpt" / "Tg" / "checkpoint_best.pt")
    _write(tmp_path / "ckpt" / "DC" / "ckpt" / "DC" / "checkpoint_best.pt")

    validate_polygen_setup(tmp_path, tmp_path / "ckpt")
