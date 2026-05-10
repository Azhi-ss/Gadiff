"""Applicability Domain checker for MMPolymer predictions.

MMPolymer is trained on a specific polymer dataset. Polymers assembled from
diffusion-generated fragments may fall outside this training distribution,
leading to unreliable property predictions with unknown error.

This module provides Morgan fingerprint-based similarity checking against a
reference fragment pool (the DNA chromosomes), which serves as a proxy for
MMPolymer's training domain.
"""

from typing import Dict, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity


class ApplicabilityDomain:
    """Checks whether a polymer SMILES is within MMPolymer's applicability domain.

    Uses Morgan fingerprint Tanimoto similarity to the reference fragment pool.
    Polymers with low maximum similarity to any known fragment are likely OOD.

    Reference: OECD QSAR Validation Principles, Principle 3 (defined AD).
    """

    def __init__(
        self,
        reference_smiles: Optional[list] = None,
        threshold: float = 0.4,
        fp_radius: int = 2,
        fp_bits: int = 2048,
    ):
        """
        Args:
            reference_smiles: List of SMILES representing the training domain.
                              Typically the DNA chromosome pool.
            threshold: Minimum Tanimoto similarity to be considered in-domain.
                       OECD defaults: 0.3-0.5. Higher = stricter.
            fp_radius: Morgan fingerprint radius.
            fp_bits: Morgan fingerprint bit length.
        """
        self.threshold = threshold
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self._reference_fps: list = []
        self._reference_smiles: list = []

        if reference_smiles:
            self.fit(reference_smiles)

    def fit(self, reference_smiles: list) -> None:
        """Build fingerprint index from reference SMILES."""
        self._reference_smiles = list(reference_smiles)
        self._reference_fps = []
        for smi in reference_smiles:
            fp = self._smiles_to_fp(smi)
            if fp is not None:
                self._reference_fps.append(fp)

    def max_similarity(self, smiles: str) -> float:
        """Return max Tanimoto similarity to any reference fragment."""
        fp = self._smiles_to_fp(smiles)
        if fp is None or not self._reference_fps:
            return 0.0
        scores = BulkTanimotoSimilarity(fp, self._reference_fps)
        return float(max(scores)) if scores else 0.0

    def is_in_domain(self, smiles: str) -> bool:
        """Check if polymer is within the applicability domain."""
        return self.max_similarity(smiles) >= self.threshold

    def domain_score(self, smiles: str) -> float:
        """Return domain adherence score in [0, 1].

        1.0 = fully in-domain (similarity >= threshold).
        <1.0 = penalty proportional to distance from threshold.
        """
        sim = self.max_similarity(smiles)
        if sim >= self.threshold:
            return 1.0
        if self.threshold == 0:
            return 1.0
        # Linear ramp from 0 at sim=0 to 1 at sim=threshold
        return sim / self.threshold

    def filter_in_domain(self, smiles_list: list) -> list:
        """Return only SMILES that pass the domain check."""
        return [s for s in smiles_list if self.is_in_domain(s)]

    def _smiles_to_fp(self, smiles: str):
        """Convert SMILES to Morgan fingerprint. Returns None on failure."""
        if smiles is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, self.fp_radius, nBits=self.fp_bits
        )

    @property
    def reference_count(self) -> int:
        return len(self._reference_fps)
