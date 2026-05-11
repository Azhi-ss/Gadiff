#!/usr/bin/env python3
"""Generate new BRICS fragments via EvoDiffMol and append them to the PolyGA DNA pool."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
EVODIFFMOL_ROOT = REPO_ROOT / "EvoDiffMol"
EVODIFFMOL_MODEL_CONFIG = EVODIFFMOL_ROOT / "configs" / "general_without_h.yml"
EVODIFFMOL_GA_CONFIG = EVODIFFMOL_ROOT / "ga_config" / "moses_production.yml"

try:
    from evodiffmol import MoleculeGenerator as _MoleculeGenerator

    EVODIFFMOL_AVAILABLE = True
except ImportError:
    EVODIFFMOL_AVAILABLE = False
    _MoleculeGenerator = None  # type: ignore[assignment]

from rdkit import Chem
from rdkit.Chem import BRICS

try:
    from torch import load as torch_load
except ImportError:
    torch_load = None  # type: ignore[assignment]

ALLOWED_ELEMENTS: Set[int] = {6, 7, 8, 9, 16, 17, 35}  # C, N, O, F, S, Cl, Br
MIN_HEAVY_ATOMS: int = 3
MAX_HEAVY_ATOMS: int = 15
TARGET_CONNECTIONS: int = 2
DNA_COLUMNS: List[str] = ["chromosome_id", "chromosome", "num_connections"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pregenerate BRICS fragments via EvoDiffMol and merge into PolyGA DNA CSV."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to EvoDiffMol model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=100,
        help="Number of molecules to generate via EvoDiffMol (default: 100)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the merged DNA CSV",
    )
    parser.add_argument(
        "--existing-dna",
        default=None,
        help="Optional path to existing PolyGA DNA CSV to merge with",
    )
    return parser.parse_args(argv)


def generate_molecules(
    checkpoint_path: str,
    n_molecules: int,
    strict: bool = False,
) -> List[str]:
    """Generate small molecule SMILES via EvoDiffMol's MoleculeGenerator."""
    if not EVODIFFMOL_AVAILABLE:
        print(
            "WARNING: evodiffmol package is not available; skipping molecule generation",
            file=sys.stderr,
        )
        return []

    if _MoleculeGenerator is None:
        return []

    try:
        if strict:
            validate_evodiffmol_checkpoint(checkpoint_path)
        gen = _MoleculeGenerator(
            checkpoint_path=checkpoint_path,
            model_config=str(EVODIFFMOL_MODEL_CONFIG),
            ga_config=str(EVODIFFMOL_GA_CONFIG),
            dataset=[],
            device="cuda" if _cuda_available() else "cpu",
            verbose=False,
        )
        if hasattr(gen, "generate"):
            result = gen.generate(n=n_molecules)
        else:
            result = gen.optimize(
                target_properties={"qed": 0.9},
                population_size=n_molecules,
                generations=0,
                verbose=False,
            )
        return list(result)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"EvoDiffMol generation failed: {exc}") from exc
        print(
            f"WARNING: EvoDiffMol generation failed: {exc}",
            file=sys.stderr,
        )
        return []


def validate_evodiffmol_checkpoint(checkpoint_path: str) -> None:
    """Fail early when a MMPolymer checkpoint is passed as a diffusion checkpoint."""
    if torch_load is None:
        return

    checkpoint = torch_load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(model_state, dict):
        raise ValueError(
            f"{checkpoint_path} is not an EvoDiffMol diffusion checkpoint: "
            "expected a dict with a 'model' state_dict."
        )

    keys = set(model_state.keys())
    evodiffmol_markers = {
        "betas",
        "alphas",
        "edge_encoder_global.bond_emb.weight",
        "model_global.0.bond_emb.weight",
    }
    if keys & evodiffmol_markers:
        return

    mmpolymer_markers = (
        "PretrainedModel.",
        "classification_head.",
        "embed_tokens.",
    )
    looks_like_mmpolymer = any(
        key.startswith(mmpolymer_markers) for key in keys
    )
    detail = " It looks like an MMPolymer predictor checkpoint." if looks_like_mmpolymer else ""
    raise ValueError(
        f"{checkpoint_path} is not an EvoDiffMol diffusion checkpoint.{detail} "
        "Use the EvoDiffMol checkpoint such as moses_without_h_80.pt."
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def brics_decompose(smiles: str) -> List[str]:
    """Decompose a molecule into BRICS fragments, returning each as a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    fragments = BRICS.BRICSDecompose(mol)
    return list(fragments)


def count_connection_points(fragment_smiles: str) -> int:
    """Return the number of [*] dummy-atom connection points in a fragment."""
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)


def heavy_atom_count(fragment_smiles: str) -> int:
    """Return the number of heavy (non-H, non-dummy) atoms in a fragment."""
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()


def has_allowed_elements_only(fragment_smiles: str) -> bool:
    """Check that every heavy atom in the fragment belongs to the allowed set."""
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z == 0:
            continue
        if z > 1 and z not in ALLOWED_ELEMENTS:
            return False
    return True


def is_valid_fragment(fragment_smiles: str) -> bool:
    """Return True if the fragment has exactly 2 connection points, 3-15 heavy
    atoms, and only allowed elements."""
    if count_connection_points(fragment_smiles) != TARGET_CONNECTIONS:
        return False
    hvy = heavy_atom_count(fragment_smiles)
    if hvy < MIN_HEAVY_ATOMS or hvy > MAX_HEAVY_ATOMS:
        return False
    if not has_allowed_elements_only(fragment_smiles):
        return False
    return True


def normalize_fragment_for_polyga(fragment_smiles: str) -> str:
    """Convert BRICS dummy atoms into PolyGA's Bi connection placeholders."""
    normalized = re.sub(r"\[\d+\*\]", "[Bi]", fragment_smiles)
    normalized = normalized.replace("[*]", "[Bi]")
    return normalized


def process_molecules(smiles_list: List[str]) -> List[Tuple[str, int]]:
    """BRICS-decompose each molecule, filter fragments, and deduplicate.

    Returns a list of (SMILES, num_connections) tuples.
    """
    seen: Set[str] = set()
    valid: List[Tuple[str, int]] = []

    for input_smiles in smiles_list:
        fragments = brics_decompose(input_smiles)
        for frag in fragments:
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(frag))
            if is_valid_fragment(canonical):
                normalized = normalize_fragment_for_polyga(canonical)
                if normalized in seen:
                    continue
                seen.add(normalized)
                valid.append((canonical, TARGET_CONNECTIONS))

    return valid


def read_existing_dna(path: str) -> List[Dict[str, str]]:
    """Read an existing PolyGA DNA CSV and return its rows."""
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_dna_csv(path: str, fragments: List[Tuple[str, int]]) -> None:
    """Write fragments to a CSV file in PolyGA DNA format."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(DNA_COLUMNS)
        for idx, (smiles, n_conn) in enumerate(fragments, start=1):
            writer.writerow([idx, normalize_fragment_for_polyga(smiles), n_conn])


def merge_with_existing(
    existing_path: str,
    new_fragments: List[Tuple[str, int]],
) -> List[Tuple[str, int]]:
    """Deduplicate new fragments against an existing DNA CSV and merge.

    Returns a combined list of (SMILES, num_connections) tuples with sequential IDs.
    """
    existing_rows = read_existing_dna(existing_path)
    existing_smiles: Set[str] = set()
    combined: List[Tuple[str, int]] = []

    for row in existing_rows:
        smiles = row["chromosome"]
        n_conn = int(row["num_connections"])
        combined.append((smiles, n_conn))
        existing_smiles.add(normalize_fragment_for_polyga(smiles))

    for smiles, n_conn in new_fragments:
        normalized = normalize_fragment_for_polyga(smiles)
        if normalized not in existing_smiles:
            combined.append((smiles, n_conn))
            existing_smiles.add(normalized)

    return combined


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    output_path = Path(args.output)

    if not EVODIFFMOL_AVAILABLE:
        print(
            "WARNING: evodiffmol package is not available; "
            "no new molecules will be generated",
            file=sys.stderr,
        )

    try:
        generated = generate_molecules(
            checkpoint_path=args.checkpoint,
            n_molecules=args.n_molecules,
            strict=True,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    if generated:
        print(f"Generated {len(generated)} molecules via EvoDiffMol")
    else:
        print("No molecules generated; output will be empty unless --existing-dna is provided")

    new_fragments = process_molecules(generated)
    print(f"Obtained {len(new_fragments)} valid BRICS fragments after decomposition and filtering")

    if args.existing_dna is not None:
        merged = merge_with_existing(args.existing_dna, new_fragments)
        print(f"Merged with existing DNA ({len(merged)} total entries)")
        write_dna_csv(str(output_path), merged)
    else:
        write_dna_csv(str(output_path), new_fragments)

    print(f"Written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
