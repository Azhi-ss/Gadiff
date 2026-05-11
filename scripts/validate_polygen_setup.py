#!/usr/bin/env python3
"""Validate files required for formal PolyGen runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def resolve_weight_path(weight_dir: Path, property_name: str) -> Path:
    candidates = [
        weight_dir / property_name / "checkpoint_best.pt",
        weight_dir / property_name / "ckpt" / property_name / "checkpoint_best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_polygen_setup(
    repo_root: Path,
    weight_dir: Path,
    dna_file: Path | None = None,
) -> None:
    dna_path = dna_file or repo_root / "data" / "enriched_dna.csv"
    required_files = [
        dna_path,
        repo_root / "src" / "models" / "MMPolymer" / "dict.txt",
        repo_root / "data" / "tokenizer" / "vocab.json",
        repo_root / "data" / "tokenizer" / "merges.txt",
        resolve_weight_path(weight_dir, "Tg"),
        resolve_weight_path(weight_dir, "DC"),
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required PolyGen setup files:\n"
            + "\n".join(str(path) for path in missing)
        )

    dna = pd.read_csv(dna_path)
    if "chromosome" not in dna.columns or not dna["chromosome"].astype(str).str.contains(r"\[Bi\]").all():
        raise ValueError("DNA chromosomes must use [Bi] connection placeholders.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate formal PolyGen setup files.")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument("--weight-dir", default="/internfs/Zy/polygen_assets/mm_polymer/finetune_data")
    parser.add_argument("--dna-file", default=None)
    args = parser.parse_args()

    validate_polygen_setup(
        Path(args.repo_root).resolve(),
        Path(args.weight_dir),
        Path(args.dna_file) if args.dna_file else None,
    )
    print("PolyGen setup validation passed")


if __name__ == "__main__":
    main()
