"""Tests for applicability_domain.py — OOD detection for MMPolymer predictions."""

import numpy as np
import pandas as pd
import pytest

from polyga.applicability_domain import ApplicabilityDomain


# Reference fragments mimicking a DNA chromosome pool.
REF_SMILES = [
    "[*]CC[*]",
    "[*]CCO[*]",
    "[*]CC(=O)N[*]",
    "[*]c1ccccc1[*]",
    "[*]CCC[*]",
    "[*]COC[*]",
    "[*]CCS[*]",
]


class TestApplicabilityDomain:
    def test_init_empty(self):
        ad = ApplicabilityDomain()
        assert ad.reference_count == 0
        assert ad.threshold == 0.4

    def test_fit_stores_references(self):
        ad = ApplicabilityDomain(REF_SMILES)
        assert ad.reference_count == len(REF_SMILES)

    def test_max_similarity_identical(self):
        ad = ApplicabilityDomain(REF_SMILES)
        sim = ad.max_similarity("[*]CC[*]")
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_max_similarity_similar(self):
        ad = ApplicabilityDomain(REF_SMILES)
        sim = ad.max_similarity("[*]CCCO[*]")
        assert sim > 0.3  # similar to [*]CCC[*] / [*]CCO[*]

    def test_max_similarity_dissimilar(self):
        ad = ApplicabilityDomain(REF_SMILES)
        sim = ad.max_similarity("[*]c1ncnc1[*]")
        assert sim < 0.3  # heterocycle, nothing like the refs

    def test_max_similarity_empty_reference(self):
        ad = ApplicabilityDomain()
        assert ad.max_similarity("[*]CC[*]") == 0.0

    def test_max_similarity_invalid_smiles(self):
        ad = ApplicabilityDomain(REF_SMILES)
        assert ad.max_similarity("not_a_smiles") == 0.0

    def test_max_similarity_none(self):
        ad = ApplicabilityDomain(REF_SMILES)
        assert ad.max_similarity(None) == 0.0

    def test_is_in_domain_above_threshold(self):
        ad = ApplicabilityDomain(REF_SMILES, threshold=0.4)
        assert ad.is_in_domain("[*]CCCO[*]") is True

    def test_is_in_domain_below_threshold(self):
        ad = ApplicabilityDomain(REF_SMILES, threshold=0.8)
        assert ad.is_in_domain("[*]CCCO[*]") is False  # similar but not >0.8

    def test_is_in_domain_dissimilar(self):
        ad = ApplicabilityDomain(REF_SMILES, threshold=0.4)
        assert ad.is_in_domain("[*]c1ncnc1[*]") is False

    def test_domain_score_in_domain(self):
        ad = ApplicabilityDomain(REF_SMILES, threshold=0.4)
        score = ad.domain_score("[*]CC[*]")
        assert score == 1.0

    def test_domain_score_partial(self):
        ad = ApplicabilityDomain(REF_SMILES, threshold=0.4)
        # A heterocycle should have low sim, producing a low domain score.
        score = ad.domain_score("[*]c1ncnc1[*]")
        assert 0.0 <= score < 0.5

    def test_domain_score_zero_threshold(self):
        ad = ApplicabilityDomain(REF_SMILES, threshold=0.0)
        assert ad.domain_score("[*]c1ncnc1[*]") == 1.0

    def test_domain_score_no_references(self):
        ad = ApplicabilityDomain(threshold=0.4)
        assert ad.domain_score("[*]CC[*]") == 0.0

    def test_filter_in_domain_all_pass(self):
        ad = ApplicabilityDomain(REF_SMILES, threshold=0.3)
        candidates = ["[*]CC[*]", "[*]CCO[*]", "[*]CCC[*]"]
        assert ad.filter_in_domain(candidates) == candidates

    def test_filter_in_domain_some_rejected(self):
        ad = ApplicabilityDomain(REF_SMILES, threshold=0.7)
        candidates = ["[*]CC[*]", "[*]c1ncnc1[*]"]
        result = ad.filter_in_domain(candidates)
        assert "[*]CC[*]" in result
        assert "[*]c1ncnc1[*]" not in result

    def test_filter_in_domain_empty_input(self):
        ad = ApplicabilityDomain(REF_SMILES)
        assert ad.filter_in_domain([]) == []

    def test_custom_fp_params(self):
        ad = ApplicabilityDomain(REF_SMILES, fp_radius=3, fp_bits=4096)
        assert ad.reference_count == len(REF_SMILES)
        sim = ad.max_similarity("[*]CC[*]")
        assert sim > 0.9  # identical should still score high with different params


class TestIntegrationEGDMutator:
    """Test that EGDFragmentMutator applies AD filtering."""

    def test_mutator_accepts_ad_checker(self):
        from polyga.diffusion_mutator import EGDFragmentMutator

        ad = ApplicabilityDomain(REF_SMILES, threshold=0.4)
        mutator = EGDFragmentMutator(
            diffusion_model=None, ad_checker=ad
        )
        assert mutator.ad_checker is ad

    def test_mutator_without_ad_checker(self):
        from polyga.diffusion_mutator import EGDFragmentMutator

        mutator = EGDFragmentMutator(diffusion_model=None)
        assert mutator.ad_checker is None

    def test_apply_ad_filter_passes_through_when_no_checker(self):
        from polyga.diffusion_mutator import EGDFragmentMutator

        mutator = EGDFragmentMutator(diffusion_model=None)
        frags = ["[*]CC[*]", "[*]CCO[*]"]
        assert mutator._apply_ad_filter(frags) == frags

    def test_apply_ad_filter_with_checker(self):
        from polyga.diffusion_mutator import EGDFragmentMutator

        ad = ApplicabilityDomain(REF_SMILES, threshold=0.9)
        mutator = EGDFragmentMutator(diffusion_model=None, ad_checker=ad)
        # "[*]CC[*]" passes (identical), "[*]c1ncnc1[*]" fails
        frags = ["[*]CC[*]", "[*]c1ncnc1[*]"]
        result = mutator._apply_ad_filter(frags)
        assert result == ["[*]CC[*]"]

    def test_apply_ad_filter_empty(self):
        from polyga.diffusion_mutator import EGDFragmentMutator

        ad = ApplicabilityDomain(REF_SMILES)
        mutator = EGDFragmentMutator(diffusion_model=None, ad_checker=ad)
        assert mutator._apply_ad_filter([]) == []


class TestIntegrationDEMO:
    """Test that DEMOPolyGA applies OOD fitness penalty."""

    def test_demo_accepts_ad_checker(self):
        from polyga.demo_polyga import DEMOPolyGA

        ad = ApplicabilityDomain(REF_SMILES, threshold=0.4)

        def dummy_predict(df, headers, _):
            df["fitness"] = 1.0
            return df

        def dummy_fp(df):
            return df, []

        demo = DEMOPolyGA(
            egd_mutator=None,
            saes_selector=None,
            predict_function=dummy_predict,
            fingerprint_function=dummy_fp,
            ad_checker=ad,
            ood_penalty_weight=0.5,
        )
        assert demo.ad_checker is ad
        assert demo.ood_penalty_weight == 0.5

    def test_ood_penalty_in_domain_no_effect(self):
        from polyga.demo_polyga import DEMOPolyGA

        ad = ApplicabilityDomain(REF_SMILES, threshold=0.3)

        demo = DEMOPolyGA(
            egd_mutator=None,
            saes_selector=None,
            predict_function=lambda d, h, _: d,
            fingerprint_function=lambda d: (d, []),
            ad_checker=ad,
            ood_penalty_weight=0.5,
        )
        df = pd.DataFrame({
            "smiles_string": ["[*]CC[*]", "[*]CCO[*]"],
            "fitness": [1.0, 2.0],
        })
        result = demo._apply_ood_penalty(df)
        assert result["fitness"].iloc[0] == pytest.approx(1.0, abs=0.05)
        assert result["fitness"].iloc[1] == pytest.approx(2.0, abs=0.05)

    def test_ood_penalty_ood_reduced(self):
        from polyga.demo_polyga import DEMOPolyGA

        ad = ApplicabilityDomain(REF_SMILES, threshold=1.0)

        demo = DEMOPolyGA(
            egd_mutator=None,
            saes_selector=None,
            predict_function=lambda d, h, _: d,
            fingerprint_function=lambda d: (d, []),
            ad_checker=ad,
            ood_penalty_weight=0.5,
        )
        df = pd.DataFrame({
            "smiles_string": ["[*]c1ncnc1[*]"],
            "fitness": [2.0],
        })
        result = demo._apply_ood_penalty(df)
        # Far OOD with threshold=1.0 -> domain_score near 0 -> fitness ~ 1.0
        assert result["fitness"].iloc[0] < 1.5

    def test_ood_penalty_disabled_without_checker(self):
        from polyga.demo_polyga import DEMOPolyGA

        demo = DEMOPolyGA(
            egd_mutator=None,
            saes_selector=None,
            predict_function=lambda d, h, _: d,
            fingerprint_function=lambda d: (d, []),
            ad_checker=None,
        )
        df = pd.DataFrame({
            "smiles_string": ["[*]c1ncnc1[*]"],
            "fitness": [2.0],
        })
        result = demo._apply_ood_penalty(df)
        assert result["fitness"].iloc[0] == 2.0

    def test_ood_penalty_zero_weight_no_effect(self):
        from polyga.demo_polyga import DEMOPolyGA

        ad = ApplicabilityDomain(REF_SMILES, threshold=1.0)

        demo = DEMOPolyGA(
            egd_mutator=None,
            saes_selector=None,
            predict_function=lambda d, h, _: d,
            fingerprint_function=lambda d: (d, []),
            ad_checker=ad,
            ood_penalty_weight=0.0,
        )
        df = pd.DataFrame({
            "smiles_string": ["[*]c1ncnc1[*]"],
            "fitness": [2.0],
        })
        result = demo._apply_ood_penalty(df)
        assert result["fitness"].iloc[0] == 2.0
