"""Synthetic accessibility scoring for polymer-like SMILES."""

from __future__ import annotations

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def _load_sascorer():
    try:
        from rdkit.Contrib.SA_Score import sascorer

        return sascorer
    except Exception:
        import sascorer  # type: ignore

        return sascorer


def psmiles_for_sa(smiles: str | None) -> str | None:
    """Cap polymer wildcard endpoints so RDKit SA scoring sees a molecule."""
    if not smiles:
        return None
    return smiles.replace("[*]", "C").replace("*", "C")


def calculate_sa_score(smiles: str | None, penalty: float = 10.0) -> float:
    """Return raw SA score where lower is easier to synthesize."""
    capped = psmiles_for_sa(smiles)
    if not capped:
        return penalty
    mol = Chem.MolFromSmiles(capped)
    if mol is None:
        return penalty
    try:
        score = _load_sascorer().calculateScore(mol)
        if score is None:
            return penalty
        return float(score)
    except Exception:
        return penalty


def add_sa_scores(
    df: pd.DataFrame,
    smiles_col: str = "smiles_string",
    score_col: str = "SA_score",
) -> pd.DataFrame:
    """Return a copy of df with raw synthetic accessibility scores."""
    df_out = df.copy()
    df_out[score_col] = [
        calculate_sa_score(smiles) for smiles in df_out[smiles_col].astype(str)
    ]
    return df_out

