import pandas as pd

from src.models.polyga.sa_score import add_sa_scores, calculate_sa_score, psmiles_for_sa


def test_psmiles_for_sa_caps_star_endpoints_with_carbon() -> None:
    assert psmiles_for_sa("[*]c1cccc([*])c1") == "Cc1cccc(C)c1"


def test_calculate_sa_score_returns_finite_score_for_linear_psmiles() -> None:
    score = calculate_sa_score("[*]CCO[*]")

    assert 1.0 <= score <= 10.0


def test_calculate_sa_score_returns_penalty_for_invalid_smiles() -> None:
    assert calculate_sa_score("not-a-smiles") == 10.0


def test_add_sa_scores_adds_column() -> None:
    df = pd.DataFrame({"smiles_string": ["[*]CCO[*]", "bad"]})

    result = add_sa_scores(df)

    assert "SA_score" in result.columns
    assert result.loc[0, "SA_score"] < 10.0
    assert result.loc[1, "SA_score"] == 10.0
