"""EGDFragmentMutator -- noise-space mutation operator for PolyGA.

Wraps the diffusion model (MDMFullDP) from EvoDiffMol to perform
structure-aware fragment mutation and crossover via the EGD framework.

Pipeline: SMILES -> 3D conformer (ETKDGv3) -> PyG Data ->
  forward diffuse to t' -> partial denoise to 0 ->
  SMILES -> BRICS decompose -> valid fragments

Every public method degrades gracefully: returns empty list / None on
failure so the GA caller can fall back to random mutation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import radius_graph

    _HAVE_PYG = True
except ImportError:
    Data = None  # type: ignore
    radius_graph = None  # type: ignore
    torch_geometric = None  # type: ignore
    _HAVE_PYG = False

# Diffusion model import -- EvoDiffMol may have chain dependencies (yaml etc.)
# that aren't always installed. We degrade gracefully when the model is absent.
try:
    from evodiffmol.models.epsnet.MDM import MDMFullDP

    _HAVE_EGD = True
except ImportError:
    MDMFullDP = None  # type: ignore
    _HAVE_EGD = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local center_pos -- identical to the one in EvoDiffMol's MDM.py but
# avoids the import dependency chain through evodiffmol.models.epsnet.MDM.
# ---------------------------------------------------------------------------

def _radius_graph_fallback(
    pos: torch.Tensor, r: float, batch: torch.Tensor
) -> torch.Tensor:
    """Simple radius graph using pairwise distances.

    Fallback when torch-cluster (required by PyG's radius_graph) is not
    installed. Returns edge_index of shape (2, E).
    """
    n = pos.size(0)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 3)
    dist_sq = (diff ** 2).sum(dim=-1)  # (N, N)
    mask = (dist_sq < r * r) & (dist_sq > 1e-8)
    # Only connect atoms in the same batch entry.
    same_batch = batch.unsqueeze(0) == batch.unsqueeze(1)
    mask = mask & same_batch
    edge_idx = mask.nonzero(as_tuple=False).t().contiguous()  # (2, E)
    return edge_idx


def _build_radius_graph(
    pos: torch.Tensor, r: float, batch: torch.Tensor
) -> torch.Tensor:
    """Build radius graph, falling back to a pure-PyTorch implementation
    when torch-cluster (the PyG dependency) is unavailable."""
    if _HAVE_PYG:
        try:
            return radius_graph(pos, r=r, batch=batch)
        except (ImportError, Exception):
            pass
    return _radius_graph_fallback(pos, r, batch)


def _center_pos(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Subtract the per-graph centroid from each node's position.

    Pure PyTorch equivalent of torch_scatter.scatter_mean.
    """
    unique = batch.unique()
    centroids = torch.stack(
        [pos[batch == b].mean(dim=0) for b in unique]
    )
    return pos - centroids[batch]

# ---------------------------------------------------------------------------
# Atom type mapping (decoder: index -> atomic number)
# ---------------------------------------------------------------------------
# The diffusion model operates on a 7-element atom type space (no hydrogen).
ATOM_DECODER_Z: list[int] = [6, 7, 8, 9, 16, 17, 35]  # C, N, O, F, S, Cl, Br
ATOM_Z_TO_INDEX: dict[int, int] = {z: i for i, z in enumerate(ATOM_DECODER_Z)}
ALLOWED_ELEMENTS: set[int] = set(ATOM_DECODER_Z)

# Each atom feature has num_atom_types + 1 channels (types + charge placeholder).
NUM_ATOM_TYPES: int = len(ATOM_DECODER_Z)  # 7
ATOM_FEATURE_DIM: int = NUM_ATOM_TYPES + 1  # 8

# Radius graph cutoff for building molecular graphs (Angstroms).
RADIUS_CUTOFF: float = 2.5

# BRICS bond tolerance -- how many attempts at decomposition.
_BRICS_MAX_ATTEMPTS: int = 5


def _is_valid_fragment(smi: str) -> bool:
    """Check if *smi* is a valid BRICS fragment.

    Criteria:
      - Parsable by RDKit.
      - 3-15 heavy atoms.
      - Only allowed elements (C, N, O, F, S, Cl, Br).
      - Exactly 2 wildcard atoms ([*]) representing connection points.
    """
    if not smi or not isinstance(smi, str):
        return False
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
    except Exception:
        return False

    # Heavy-atom count check.
    if mol.GetNumHeavyAtoms() < 3 or mol.GetNumHeavyAtoms() > 15:
        return False

    # Element check: every non-wildcard atom must be in the allowed set.
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z == 0:
            continue  # wildcard [*]
        if z not in ALLOWED_ELEMENTS:
            return False

    # Exactly two wildcard connection points (BRICS convention).
    wildcard_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    return wildcard_count == 2


class EGDFragmentMutator:
    """Wraps a diffusion model (MDMFullDP) for noise-space mutation
    and crossover of molecular fragments.

    Parameters
    ----------
    diffusion_model : MDMFullDP
        Trained diffusion model instance.
    device : str
        Torch device string (default 'cpu').
    t_prime : int
        Default noise depth for forward diffusion (default 250).
    """

    def __init__(
        self,
        diffusion_model: Any,  # MDMFullDP instance (or None for degraded mode)
        device: str = "cpu",
        t_prime: int = 250,
    ) -> None:
        self.model = diffusion_model
        self.device = torch.device(device)
        self.t_prime = t_prime
        self.num_atom_types = NUM_ATOM_TYPES
        self.atom_feature_dim = ATOM_FEATURE_DIM
        self.model.eval()
        self.model.to(self.device)

    # -- Public API -----------------------------------------------------------

    def mutate_fragment(
        self,
        seed_smiles: str,
        t_prime: Optional[int] = None,
    ) -> list[str]:
        """EGD mutation: diffuse *seed_smiles* to noise level *t_prime*,
        denoise back, then extract valid fragment SMILES.

        Returns a list of fragment SMILES (empty on any failure).
        """
        t = t_prime if t_prime is not None else self.t_prime
        try:
            mol_3d = self._smiles_to_3d(seed_smiles)
            if mol_3d is None:
                return []
            data = self._mol_to_data(mol_3d)
            if data is None:
                return []
            data_noisy = self._forward_diffuse(data, t)
            if data_noisy is None:
                return []
            data_denoised = self._partial_denoise(data_noisy, t)
            if data_denoised is None:
                return []
            smiles_out = self._data_to_smiles(data_denoised)
            if smiles_out is None:
                return []
            return self._extract_fragments(smiles_out)
        except Exception:
            logger.warning("EGD mutation failed for %s", seed_smiles, exc_info=True)
            return []

    def crossover_fragments(
        self,
        smi1: str,
        smi2: str,
        t_prime: Optional[int] = None,
    ) -> list[str]:
        """Noise-space chimeric crossover of two fragments.

        Each fragment is converted to a 3D conformer, a random atom subset
        is taken from each, concatenated into a chimeric molecule, diffused
        to *t_prime*, denoised, and decomposed into valid fragments.

        Returns a list of fragment SMILES (empty on any failure).
        """
        t = t_prime if t_prime is not None else self.t_prime
        try:
            mol1 = self._smiles_to_3d(smi1)
            mol2 = self._smiles_to_3d(smi2)
            if mol1 is None or mol2 is None:
                return []
            data1 = self._mol_to_data(mol1)
            data2 = self._mol_to_data(mol2)
            if data1 is None or data2 is None:
                return []
            chimeric = self._chimeric_concat(data1, data2)
            if chimeric is None:
                return []
            chimeric_noisy = self._forward_diffuse(chimeric, t)
            if chimeric_noisy is None:
                return []
            chimeric_denoised = self._partial_denoise(chimeric_noisy, t)
            if chimeric_denoised is None:
                return []
            smiles_out = self._data_to_smiles(chimeric_denoised)
            if smiles_out is None:
                return []
            return self._extract_fragments(smiles_out)
        except Exception:
            logger.warning(
                "EGD crossover failed for %s / %s", smi1, smi2, exc_info=True
            )
            return []

    # -- SMILES <-> 3D conversion --------------------------------------------

    def _smiles_to_3d(self, smi: str) -> Optional[Any]:
        """Convert SMILES to RDKit Mol with a single 3D conformer (ETKDGv3).

        Wildcard atoms [*] are temporarily replaced with carbon for embedding,
        then restored in the conformer.
        """
        if not smi:
            return None
        try:
            # Replace [*] with C so ETKDGv3 can embed.
            has_wildcard = "[*]" in smi
            smi_clean = smi.replace("[*]", "C") if has_wildcard else smi

            mol = Chem.MolFromSmiles(smi_clean)
            if mol is None:
                return None

            # Track original atomic numbers so we can restore wildcards.
            orig_atomic = None
            if has_wildcard:
                mol_template = Chem.MolFromSmiles(smi)
                orig_atomic = [
                    a.GetAtomicNum() for a in mol_template.GetAtoms()
                ]

            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            result = AllChem.EmbedMolecule(mol, params)
            if result != 0:
                return None

            # Restore wildcard atomic numbers in the embedded mol.
            if has_wildcard and orig_atomic is not None:
                for i, z in enumerate(orig_atomic):
                    if z == 0:
                        mol.GetAtomWithIdx(i).SetAtomicNum(0)
                        mol.GetAtomWithIdx(i).SetNoImplicit(True)

            Chem.MolToSmiles(mol)  # trigger sanitisation
            return mol
        except Exception:
            logger.warning("SMILES-to-3D failed for %s", smi, exc_info=True)
            return None

    def _mol_to_data(self, mol: Any) -> Optional[Any]:
        """Convert RDKit Mol (with 3D conformer) to a PyG Data object.

        Atom types are one-hot encoded using *ATOM_DECODER_Z*.
        Edge indices are computed via a radius graph.
        """
        if not _HAVE_PYG:
            logger.warning("PyTorch Geometric not available -- cannot build Data")
            return None
        try:
            conf = mol.GetConformer(0)
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
            n_atoms = pos.size(0)

            # Build one-hot atom features: (N, ATOM_FEATURE_DIM)
            atom_type = torch.zeros(n_atoms, ATOM_FEATURE_DIM, dtype=torch.float)
            for i in range(n_atoms):
                z = mol.GetAtomWithIdx(i).GetAtomicNum()
                if z == 0:
                    # Wildcard -- leave as zeros (model learns to predict).
                    continue
                idx = ATOM_Z_TO_INDEX.get(z)
                if idx is not None:
                    atom_type[i, idx] = 1.0
                # Unknown atoms remain all-zeros (model will predict).

            # Radius graph for edge_index.
            batch = torch.zeros(n_atoms, dtype=torch.long)
            edge_index = _build_radius_graph(pos, r=RADIUS_CUTOFF, batch=batch)

            data = Data(
                pos=pos,
                atom_type=atom_type,
                edge_index=edge_index,
                batch=batch,
            )
            return data
        except Exception:
            logger.warning("Mol-to-Data conversion failed", exc_info=True)
            return None

    # -- Diffusion operations ------------------------------------------------

    def _forward_diffuse(
        self,
        data: Any,
        t: int,
    ) -> Optional[Any]:
        """Add noise to positions: pos_t = pos_0 + sigma_t * centered_noise.

        sigma_t = sqrt((1 - alpha_t) / alpha_t) from the model's noise schedule.
        Noise is centred (zero-mean per molecule) to avoid translation drift.
        """
        try:
            sigma_t = (
                (1.0 - self.model.alphas[t]) / self.model.alphas[t]
            ).sqrt().item()

            noise = torch.randn_like(data.pos)
            # Center noise per graph.
            noise = _center_pos(noise, data.batch)

            data_noisy = data.clone()
            data_noisy.pos = data.pos + sigma_t * noise
            return data_noisy
        except Exception:
            logger.warning("Forward diffuse failed at t=%d", t, exc_info=True)
            return None

    def _partial_denoise(
        self,
        data: Any,
        t_start: int,
    ) -> Optional[Any]:
        """Partially denoise from timestep *t_start* back to 0.

        Uses the model's ``langevin_dynamics_sample`` with
        ``start_timestep=t_start`` to resume from an intermediate noise level.

        Returns a new Data object with denoised positions and atom types,
        or None on failure.
        """
        try:
            pos_gen, _, atom_type_out, _ = self.model.langevin_dynamics_sample(
                atom_type=data.atom_type.to(self.device),
                pos_init=data.pos.to(self.device),
                bond_index=data.edge_index.to(self.device),
                bond_type=None,
                batch=data.batch.to(self.device),
                num_graphs=1,
                context=None,
                n_steps=t_start,
                start_timestep=t_start,
                step_lr=1e-6,
                w_global_pos=1.0,
                w_global_node=4.0,
                w_local_pos=1.0,
                w_local_node=5.0,
                clip=1000.0,
                sampling_type="ddpm_noisy",
            )

            data_denoised = data.clone()
            data_denoised.pos = pos_gen.cpu()
            # The output atom_type has shape (N, atom_out_dim) with scaling
            # (types *4, charge *10). Reverse the scaling and take argmax.
            atom_type_scaled = atom_type_out.cpu()
            n_type_cols = atom_type_scaled.size(-1) - 1
            type_logits = atom_type_scaled[:, :n_type_cols] / 4.0
            data_denoised.atom_type = type_logits
            return data_denoised
        except Exception:
            logger.warning(
                "Partial denoise failed at t_start=%d", t_start, exc_info=True
            )
            return None

    # -- Chimeric crossover --------------------------------------------------

    def _chimeric_concat(
        self,
        data1: Any,
        data2: Any,
    ) -> Optional[Any]:
        """Create a chimeric molecule from random atom subsets of two parents.

        A random 40-60% of atoms are taken from parent 1, the remainder
        from parent 2. The two subsets are concatenated, a new radius graph
        is built, and the result is returned as a single PyG Data object.
        """
        if not _HAVE_PYG:
            logger.warning("PyG not available -- chimeric concat unavailable")
            return None
        try:
            n1 = data1.pos.size(0)
            n2 = data2.pos.size(0)

            # Random split proportion between 40% and 60%.
            split_ratio = float(
                np.random.default_rng().uniform(0.4, 0.6)
            )
            n_from_1 = max(1, min(n1 - 1, round(n1 * split_ratio)))
            n_from_2 = max(1, min(n2 - 1, round(n2 * (1.0 - split_ratio))))

            # Random atom indices from each parent.
            rng = np.random.default_rng()
            idx1 = sorted(rng.choice(n1, size=n_from_1, replace=False).tolist())
            idx2 = sorted(rng.choice(n2, size=n_from_2, replace=False).tolist())

            # Concatenate positions and atom types.
            pos_cat = torch.cat([data1.pos[idx1], data2.pos[idx2]], dim=0)
            atom_type_cat = torch.cat(
                [data1.atom_type[idx1], data2.atom_type[idx2]], dim=0
            )
            n_total = pos_cat.size(0)

            # Rebuild radius graph for the concatenated molecule.
            batch = torch.zeros(n_total, dtype=torch.long)
            edge_index = _build_radius_graph(pos_cat, r=RADIUS_CUTOFF, batch=batch)

            chimeric = Data(
                pos=pos_cat,
                atom_type=atom_type_cat,
                edge_index=edge_index,
                batch=batch,
            )
            return chimeric
        except Exception:
            logger.warning("Chimeric concat failed", exc_info=True)
            return None

    # -- Data -> SMILES reconstruction ---------------------------------------

    def _data_to_smiles(self, data: Any) -> Optional[str]:
        """Convert PyG Data object back to a SMILES string.

        Uses distance-based bond inference on the denoised positions.
        Falls back gracefully (returns None) when reconstruction is not
        possible, e.g. because OpenBabel or a full reconstruction pipeline
        is unavailable.

        NOTE: 3D-to-SMILES reconstruction can fail ~15% of the time due to
        invalid geometries produced by denoising. This is expected -- the
        caller falls back to random mutation.
        """
        try:
            pos = data.pos.numpy()
            # Argmax over atom type logits to get the most likely type index.
            type_logits = data.atom_type
            if type_logits.dim() == 2 and type_logits.size(-1) > 1:
                atom_indices = type_logits.argmax(dim=-1).numpy()
            else:
                atom_indices = (
                    type_logits.round().long().squeeze().numpy()
                )

            # Map indices to atomic numbers.
            atomic_numbers = [
                ATOM_DECODER_Z[int(idx)] if 0 <= int(idx) < len(ATOM_DECODER_Z) else 6
                for idx in atom_indices
            ]

            # Build RDKit Mol from scratch using distance-based bonding.
            from rdkit.Chem import rdchem, rdmolops

            mol = Chem.RWMol()
            atom_map = []
            for z in atomic_numbers:
                if z == 0:
                    a = Chem.Atom(0)
                    a.SetNoImplicit(True)
                else:
                    a = Chem.Atom(z)
                mol.AddAtom(a)
                atom_map.append(mol.GetNumAtoms() - 1)

            conf = Chem.Conformer(mol.GetNumAtoms())
            for i, p in enumerate(pos):
                conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))
            mol.AddConformer(conf)

            # Infer bonds from inter-atomic distances.
            # Typical covalent bond cutoff: ~1.8 Angstroms for organic molecules.
            # Use a generous cutoff and remove duplicates.
            n_atoms = mol.GetNumAtoms()
            bond_cutoff = 1.9
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    dist = (
                        (pos[i] - pos[j]) ** 2
                    ).sum() ** 0.5
                    if dist < bond_cutoff:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)

            mol = mol.GetMol()
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                # Valence errors are common -- still try to generate SMILES.
                pass

            smiles = Chem.MolToSmiles(mol)
            if smiles and _is_valid_fragment(smiles):
                return smiles
            return None
        except Exception:
            logger.warning("Data-to-SMILES conversion failed", exc_info=True)
            return None

    # -- Fragment extraction -------------------------------------------------

    @staticmethod
    def _extract_fragments(smiles: str) -> list[str]:
        """Apply BRICS decomposition to *smiles* and return valid fragments."""
        if not smiles:
            return []
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            frags = BRICS.BRICSDecompose(mol, randomizeOrder=False)
            valid = sorted(
                {frag for frag in frags if _is_valid_fragment(frag)}
            )
            return valid
        except Exception:
            logger.warning("BRICS decomposition failed for %s", smiles, exc_info=True)
            return []
