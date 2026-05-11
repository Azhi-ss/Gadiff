from numpy.random import default_rng

from src.models.polyga.enriched_dna import enriched_fragment_to_psmiles, enriched_chromosome_to_psmiles


def test_enriched_fragment_to_psmiles_converts_bi_to_star_endpoints() -> None:
    assert enriched_fragment_to_psmiles("[Bi]c1cccc([Bi])c1") == "[*]c1cccc([*])c1"


def test_enriched_fragment_to_psmiles_rejects_missing_endpoints() -> None:
    assert enriched_fragment_to_psmiles("[Bi]CC") is None
    assert enriched_fragment_to_psmiles("CCC") is None


def test_enriched_chromosome_to_psmiles_uses_first_chromosome_id() -> None:
    chromosomes = {
        1: "[Bi]CC[Bi]",
        2: "[Bi]NN[Bi]",
    }

    assert enriched_chromosome_to_psmiles([2, 1], chromosomes, default_rng(1)) == "[*]NN[*]"
