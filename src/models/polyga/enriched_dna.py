"""Helpers for EvoDiffMol-enriched PolyGA DNA fragments."""

from __future__ import annotations

from typing import Any

from polyga.utils import validate_linear_polymer


def enriched_fragment_to_psmiles(fragment_smiles: str | None) -> str | None:
    """Convert a two-ended enriched DNA fragment into MMPolymer PSMILES."""
    if not fragment_smiles or fragment_smiles.count("[Bi]") != 2:
        return None
    psmiles = fragment_smiles.replace("[Bi]", "[*]")
    if not validate_linear_polymer(psmiles):
        return None
    return psmiles


def enriched_chromosome_to_psmiles(
    chromosome_ids: list[Any],
    chromosomes: dict[Any, str],
    rng: Any,
    **kwargs: Any,
) -> str | None:
    """Generate a linear PSMILES from the first enriched DNA chromosome."""
    if not chromosome_ids:
        return None
    return enriched_fragment_to_psmiles(chromosomes.get(chromosome_ids[0]))
