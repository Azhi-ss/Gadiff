from pathlib import Path

from MMPolymer.models.PolymerSmilesTokenization import PolymerSmilesTokenizer
from polyga.mmpolymer_predict import _resolve_weight_path


def test_tokenizer_loads_repo_local_files() -> None:
    tokenizer = PolymerSmilesTokenizer.from_pretrained()

    assert tokenizer.vocab_size > 0
    assert tokenizer.encode("[*]CC[*]")


def test_resolve_weight_path_accepts_flat_checkpoint_layout(tmp_path: Path) -> None:
    weight_file = tmp_path / "Tg" / "checkpoint_best.pt"
    weight_file.parent.mkdir(parents=True)
    weight_file.write_bytes(b"checkpoint")

    assert _resolve_weight_path(tmp_path, "Tg") == weight_file


def test_resolve_weight_path_accepts_nested_checkpoint_layout(tmp_path: Path) -> None:
    weight_file = tmp_path / "Tg" / "ckpt" / "Tg" / "checkpoint_best.pt"
    weight_file.parent.mkdir(parents=True)
    weight_file.write_bytes(b"checkpoint")

    assert _resolve_weight_path(tmp_path, "Tg") == weight_file
from pathlib import Path

from MMPolymer.models.PolymerSmilesTokenization import PolymerSmilesTokenizer
from polyga.mmpolymer_predict import _resolve_weight_path


def test_tokenizer_loads_repo_local_files() -> None:
    tokenizer = PolymerSmilesTokenizer.from_pretrained()

    assert tokenizer.vocab_size > 0
    assert tokenizer.encode("[*]CC[*]")


def test_resolve_weight_path_accepts_flat_checkpoint_layout(tmp_path: Path) -> None:
    weight_file = tmp_path / "Tg" / "checkpoint_best.pt"
    weight_file.parent.mkdir(parents=True)
    weight_file.write_bytes(b"checkpoint")

    assert _resolve_weight_path(tmp_path, "Tg") == weight_file


def test_resolve_weight_path_accepts_nested_checkpoint_layout(tmp_path: Path) -> None:
    weight_file = tmp_path / "Tg" / "ckpt" / "Tg" / "checkpoint_best.pt"
    weight_file.parent.mkdir(parents=True)
    weight_file.write_bytes(b"checkpoint")

    assert _resolve_weight_path(tmp_path, "Tg") == weight_file
