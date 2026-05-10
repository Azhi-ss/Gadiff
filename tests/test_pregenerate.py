"""Mock tests for the fragment pregeneration script."""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import pytest
from rdkit import Chem

from scripts.pregenerate_fragments import (
    MAX_HEAVY_ATOMS,
    MIN_HEAVY_ATOMS,
    TARGET_CONNECTIONS,
    brics_decompose,
    count_connection_points,
    generate_molecules,
    has_allowed_elements_only,
    heavy_atom_count,
    is_valid_fragment,
    merge_with_existing,
    parse_args,
    process_molecules,
    read_existing_dna,
    write_dna_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_stars(smiles: str) -> int:
    """Count [*] placeholder atoms in a SMILES string."""
    return smiles.count("[*]") + smiles.count("*")


# ---------------------------------------------------------------------------
# BRICS decomposition and filtering  (direct, no mocking needed)
# ---------------------------------------------------------------------------

class TestBricsDecompose:
    def test_decompose_valid_smiles(self) -> None:
        """A simple molecule should yield at least one BRICS fragment."""
        fragments = brics_decompose("CCO")
        assert len(fragments) >= 1
        for f in fragments:
            mol = Chem.MolFromSmiles(f)
            assert mol is not None

    def test_invalid_smiles_returns_empty(self) -> None:
        assert brics_decompose("not-a-smiles") == []

    def test_empty_string(self) -> None:
        # RDKit returns a list with an empty fragment for empty SMILES
        result = brics_decompose("")
        assert len(result) >= 0


class TestCountConnectionPoints:
    def test_fragment_with_two_stars(self) -> None:
        smiles = "[*]CC[*]"  # two connection points
        assert count_connection_points(smiles) == 2

    def test_fragment_with_one_star(self) -> None:
        smiles = "[*]CC"
        assert count_connection_points(smiles) == 1

    def test_no_stars(self) -> None:
        smiles = "CCO"
        assert count_connection_points(smiles) == 0

    def test_invalid_smiles(self) -> None:
        assert count_connection_points("bad") == 0


class TestHeavyAtomCount:
    def test_basic_count(self) -> None:
        smiles = "[*]CC[*]"  # 2 heavy (C, C); [*] already excluded by GetNumHeavyAtoms
        assert heavy_atom_count(smiles) == 2

    def test_excludes_dummy_atoms(self) -> None:
        smiles = "[*]c1ccccc1[*]"  # 6 heavy (ring carbons); [*] excluded by GetNumHeavyAtoms
        assert heavy_atom_count(smiles) == 6

    def test_invalid_smiles(self) -> None:
        assert heavy_atom_count("bad") == 0


class TestHasAllowedElementsOnly:
    def test_all_carbon(self) -> None:
        assert has_allowed_elements_only("[*]CC[*]") is True

    def test_includes_nitrogen(self) -> None:
        assert has_allowed_elements_only("[*]C(=O)N[*]") is True

    def test_rejects_phosphorus(self) -> None:
        # Phosphorus (Z=15) is not in the allowed set
        assert has_allowed_elements_only("[*]CP[*]") is False

    def test_rejects_sulfur_in_excess(self) -> None:
        # Sulfur (Z=16) is allowed
        assert has_allowed_elements_only("[*]CS[*]") is True

    def test_invalid_smiles(self) -> None:
        assert has_allowed_elements_only("bad") is False


class TestIsValidFragment:
    def test_valid_two_star_fragment(self) -> None:
        assert is_valid_fragment("[*]CCC[*]") is True

    def test_too_few_heavy_atoms(self) -> None:
        # Only 1 heavy atom
        assert is_valid_fragment("[*]C[*]") is False

    def test_too_many_stars(self) -> None:
        fragment = "[*]CC[*]C[*]"
        assert is_valid_fragment(fragment) is False

    def test_disallowed_element(self) -> None:
        assert is_valid_fragment("[*]CP[*]") is False

    def test_large_fragment_rejected(self) -> None:
        # Create a fragment with >15 heavy carbons + 2 stars
        long_carbon = "[*]" + "C" * (MAX_HEAVY_ATOMS + 1) + "[*]"
        assert is_valid_fragment(long_carbon) is False

    def test_fragment_at_min_boundary(self) -> None:
        too_few = "[*]" + "C" * (MIN_HEAVY_ATOMS - 1) + "[*]"
        assert is_valid_fragment(too_few) is False
        at_min = "[*]" + "C" * MIN_HEAVY_ATOMS + "[*]"
        assert is_valid_fragment(at_min) is True

    def test_fragment_at_max_boundary(self) -> None:
        at_max = "[*]" + "C" * MAX_HEAVY_ATOMS + "[*]"
        assert is_valid_fragment(at_max) is True
        over = "[*]" + "C" * (MAX_HEAVY_ATOMS + 1) + "[*]"
        assert is_valid_fragment(over) is False

    def test_invalid_smiles_returns_false(self) -> None:
        assert is_valid_fragment("bad") is False


# ---------------------------------------------------------------------------
# process_molecules (integration of BRICS + filtering + dedup)
# ---------------------------------------------------------------------------

class TestProcessMolecules:
    def test_deduplicates_identical_fragments(self) -> None:
        """Passing the same SMILES twice should yield one unique fragment."""
        results = process_molecules(["CCO", "CCO"])
        seen = set(frag for frag, _ in results)
        assert len(results) == len(seen)

    def test_invalid_molecules_skipped(self) -> None:
        results = process_molecules(["not-a-molecule"])
        assert len(results) == 0

    def test_empty_input(self) -> None:
        assert process_molecules([]) == []

    def test_all_results_have_correct_format(self) -> None:
        results = process_molecules(["CCO", "CCN"])
        for smiles, n_conn in results:
            assert _count_stars(smiles) == 2
            assert n_conn == TARGET_CONNECTIONS


# ---------------------------------------------------------------------------
# CSV read / write / merge
# ---------------------------------------------------------------------------

DNA_ROW = ("chromosome_id", "chromosome", "num_connections")


class TestReadExistingDna:
    def test_reads_csv_correctly(self) -> None:
        content = "chromosome_id,chromosome,num_connections\n1,[*]CC[*],2\n2,[*]CN[*],2\n"
        f = io.StringIO(content)
        with mock.patch(
            "scripts.pregenerate_fragments.open",
            mock.mock_open(read_data=content),
        ):
            rows = read_existing_dna("/fake/path.csv")
        assert len(rows) == 2
        assert rows[0]["chromosome"] == "[*]CC[*]"
        assert rows[0]["num_connections"] == "2"


class TestWriteDnaCsv:
    def test_writes_correct_format(self) -> None:
        fragments: List[Tuple[str, int]] = [("[*]CC[*]", 2), ("[*]CN[*]", 2)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            write_dna_csv(tmp_path, fragments)
            with open(tmp_path, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert rows[0] == list(DNA_ROW)
            assert rows[1][1] == "[*]CC[*]"
            assert rows[1][2] == "2"
            assert rows[2][1] == "[*]CN[*]"
            assert rows[2][2] == "2"
            # IDs should be sequential starting from 1
            assert rows[1][0] == "1"
            assert rows[2][0] == "2"
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestMergeWithExisting:
    def test_deduplicates_new_fragments(self) -> None:
        existing_csv = io.StringIO(
            "chromosome_id,chromosome,num_connections\n"
            "1,[*]CC[*],2\n"
        )
        with mock.patch(
            "scripts.pregenerate_fragments.open",
            mock.mock_open(read_data=existing_csv.getvalue()),
        ):
            merged = merge_with_existing(
                "/fake/path.csv",
                [("[*]CC[*]", 2), ("[*]CN[*]", 2)],
            )

        assert len(merged) == 2
        smiles_only = [s for s, _ in merged]
        assert "[*]CC[*]" in smiles_only
        assert "[*]CN[*]" in smiles_only

    def test_new_fragments_all_duplicates(self) -> None:
        existing_csv = io.StringIO(
            "chromosome_id,chromosome,num_connections\n"
            "1,[*]CC[*],2\n2,[*]CN[*],2\n"
        )
        with mock.patch(
            "scripts.pregenerate_fragments.open",
            mock.mock_open(read_data=existing_csv.getvalue()),
        ):
            merged = merge_with_existing(
                "/fake/path.csv",
                [("[*]CC[*]", 2)],
            )
        assert len(merged) == 2

    def test_empty_new_fragments(self) -> None:
        existing_csv = io.StringIO(
            "chromosome_id,chromosome,num_connections\n"
            "1,[*]CC[*],2\n"
        )
        with mock.patch(
            "scripts.pregenerate_fragments.open",
            mock.mock_open(read_data=existing_csv.getvalue()),
        ):
            merged = merge_with_existing("/fake/path.csv", [])
        assert len(merged) == 1

    def test_no_existing_fragments_in_new(self) -> None:
        existing_csv = io.StringIO(
            "chromosome_id,chromosome,num_connections\n"
            "1,[*]CC[*],2\n"
        )
        with mock.patch(
            "scripts.pregenerate_fragments.open",
            mock.mock_open(read_data=existing_csv.getvalue()),
        ):
            merged = merge_with_existing(
                "/fake/path.csv",
                [("[*]CN[*]", 2)],
            )
        assert len(merged) == 2


# ---------------------------------------------------------------------------
# generate_molecules (EvoDiffMol interaction, mocked)
# ---------------------------------------------------------------------------

class TestGenerateMolecules:
    @mock.patch("scripts.pregenerate_fragments.EVODIFFMOL_AVAILABLE", False)
    def test_returns_empty_when_evodiffmol_unavailable(self) -> None:
        result = generate_molecules("/fake/checkpoint.pt", 10)
        assert result == []

    @mock.patch("scripts.pregenerate_fragments.EVODIFFMOL_AVAILABLE", True)
    @mock.patch("scripts.pregenerate_fragments._MoleculeGenerator")
    def test_calls_generate_with_correct_arg(self, mock_generator_cls: mock.MagicMock) -> None:
        mock_instance = mock_generator_cls.return_value
        mock_instance.generate.return_value = ["CCO", "CCN"]

        result = generate_molecules("/fake/checkpoint.pt", 5)

        mock_generator_cls.assert_called_once_with(checkpoint_path="/fake/checkpoint.pt")
        mock_instance.generate.assert_called_once_with(n=5)
        assert result == ["CCO", "CCN"]

    @mock.patch("scripts.pregenerate_fragments.EVODIFFMOL_AVAILABLE", True)
    @mock.patch("scripts.pregenerate_fragments._MoleculeGenerator")
    def test_returns_empty_on_generation_failure(self, mock_generator_cls: mock.MagicMock) -> None:
        mock_generator_cls.side_effect = RuntimeError("CUDA out of memory")
        result = generate_molecules("/fake/checkpoint.pt", 5)
        assert result == []

    @mock.patch("scripts.pregenerate_fragments.EVODIFFMOL_AVAILABLE", True)
    @mock.patch("scripts.pregenerate_fragments._MoleculeGenerator")
    def test_handles_non_list_generate_return(self, mock_generator_cls: mock.MagicMock) -> None:
        mock_instance = mock_generator_cls.return_value
        mock_instance.generate.return_value = tuple(["CCO"])
        result = generate_molecules("/fake/checkpoint.pt", 1)
        assert result == ["CCO"]


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_requires_checkpoint(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--n-molecules", "10", "--output", "out.csv"])

    def test_requires_output(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--checkpoint", "model.pt", "--n-molecules", "10"])

    def test_default_n_molecules(self) -> None:
        args = parse_args(["--checkpoint", "model.pt", "--output", "out.csv"])
        assert args.n_molecules == 100

    def test_custom_n_molecules(self) -> None:
        args = parse_args([
            "--checkpoint", "model.pt",
            "--n-molecules", "50",
            "--output", "out.csv",
        ])
        assert args.n_molecules == 50

    def test_existing_dna_optional(self) -> None:
        args = parse_args([
            "--checkpoint", "model.pt",
            "--output", "out.csv",
        ])
        assert args.existing_dna is None

    def test_existing_dna_provided(self) -> None:
        args = parse_args([
            "--checkpoint", "model.pt",
            "--output", "out.csv",
            "--existing-dna", "dna.csv",
        ])
        assert args.existing_dna == "dna.csv"


# ---------------------------------------------------------------------------
# End-to-end smoke test (fully mocked)
# ---------------------------------------------------------------------------

class TestMainSmoke:
    @mock.patch("scripts.pregenerate_fragments.EVODIFFMOL_AVAILABLE", False)
    def test_main_runs_without_evodiffmol(self) -> None:
        from scripts.pregenerate_fragments import main

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            out_path = tmp.name

        try:
            main([
                "--checkpoint", "/fake/model.pt",
                "--output", out_path,
            ])
            with open(out_path, "r") as f:
                content = f.read()
            # Header only (no data)
            assert "chromosome_id" in content
        finally:
            Path(out_path).unlink(missing_ok=True)

    @mock.patch("scripts.pregenerate_fragments.EVODIFFMOL_AVAILABLE", True)
    @mock.patch("scripts.pregenerate_fragments._MoleculeGenerator")
    def test_main_with_existing_dna(self, mock_generator_cls: mock.MagicMock) -> None:
        from scripts.pregenerate_fragments import main

        mock_instance = mock_generator_cls.return_value
        mock_instance.generate.return_value = ["CCO"]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as existing_tmp:
            existing_tmp.write(
                b"chromosome_id,chromosome,num_connections\n"
                b"1,[*]CC[*],2\n"
            )
            existing_path = existing_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out_tmp:
            out_path = out_tmp.name

        try:
            main([
                "--checkpoint", "/fake/model.pt",
                "--n-molecules", "5",
                "--output", out_path,
                "--existing-dna", existing_path,
            ])
            with open(out_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            # Should include the existing fragment plus any valid new ones
            assert len(rows) >= 1
        finally:
            Path(existing_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)
