import os
import sys
import itertools
import re

import numpy as np
import pandas as pd
from numpy.random import default_rng
from rdkit import Chem


def validate_linear_polymer(smiles: str) -> bool:
    """Check if a polymer SMILES represents a linear chain (exactly 2 endpoints).

    MMPolymer can only predict properties for linear polymers.
    Branched polymers (>2 [*] wildcards) are rejected.

    Returns True if linear (valid), False otherwise.
    """
    if smiles is None:
        return False
    # Count [*] wildcard atoms -- these mark connection endpoints
    # A linear polymer has exactly 2 endpoints
    star_count = smiles.count('[*]')
    return star_count == 2


def longest_smiles(smiles):
    """Returns longest chain of polymer with more than two stars.
        
    Done so only a linear chain is returned.
    """
    smiles = smiles.replace('*', 'Bi')
    mol_temp = Chem.MolFromSmiles(smiles)
    # Find endatom index
    at_idx = []  # Indices of Bi atoms in this fragment
    try:
        num_atom_m = mol_temp.GetNumAtoms()  # Number of atoms in this fragment
    except:
        return('') # This smiles is not wrong in syntax
    atom_i = 0
    for atom in mol_temp.GetAtoms():
        atom_i = atom_i + 1
        if atom.GetSymbol() == 'Bi':
            at_idx.append(atom.GetIdx())
        if atom_i == num_atom_m:
            break

    # Make all combination of 2
    star_pairs = list(itertools.combinations(at_idx, 2))
    # Calculate length
    idx1 = 0
    idx2 = 0
    len_path_max = -999
    for star_pair in star_pairs:
        path_temp = Chem.GetShortestPath(mol_temp, star_pair[0], star_pair[1])
        path_temp_ = list(path_temp)

        len_path = len(path_temp_)
        if len_path > len_path_max:
            len_path_max = len_path
            idx1 = star_pair[0]
            idx2 = star_pair[1]

    for idx in at_idx:
        if idx == idx1 or idx == idx2:
            mol_temp.GetAtomWithIdx(idx).SetAtomicNum(82)

    smiles = Chem.MolToSmiles(mol_temp)

    smiles = smiles.replace('([Bi])', '')
    smiles = smiles.replace('[Bi]', '')
    smiles = smiles.replace('[Pb]', '[*]')

    return smiles

def chromosome_ids_to_smiles(chromosome_ids: list, chromosomes: dict,
        rng: default_rng, **kwargs) -> str:
    """Combined chromosome ids to create new polymer smiles
       
    Combines chromosomes sequentially according to index, but combines
    joints randomly.

    Args:
        chromosome_ids (list): 
            list of chromosome ids
        chromosomes (dict):
            dict with key of chromosome id and value of the chromosome
        rng (default_rng):
            random number generator
            

    Returns (str):
        smiles string of combined chromosomes
    """
    chromosomes = [chromosomes[chromosome_id] for 
                   chromosome_id in chromosome_ids]
    mols_of_chromosomes = list()
    for chromosome in chromosomes:
        m = Chem.MolFromSmiles(chromosome)
        if m is not None:
            mols_of_chromosomes.append(m)
        # Error occurs
        else:
            return None

    # Get Ids of 'Bi' atoms in each fragment
    # edtg stands for symbols used in ladder polymers
    mols_edtg = list()
    mols_at_idx = list()

    # Iterate for each fragment (m) in the list ms
    for m, i in zip(mols_of_chromosomes, range(len(mols_of_chromosomes))):
        at_idx = []  # Indices of Bi atoms in this fragment
        num_atom_m = m.GetNumAtoms()  # Number of atoms in this fragment
        atom_i = 0
        for atom in m.GetAtoms():
            atom_i = atom_i + 1
            if atom.GetSymbol() == 'Bi':
                at_idx.append(atom.GetIdx())
            if atom_i == num_atom_m:
                break
        edtg = [''] * len(at_idx)
        temp_list_idx = list(range(len(at_idx)))

        if len(at_idx) == 0:
            # No Bi atoms, skip this fragment
            mols_at_idx.append([])
            mols_edtg.append([])
            continue
        elif len(at_idx) == 1:
            # Only one Bi atom, mark it as 'e' (end) for connection
            edtg[0] = 'e'
        else:
            # Two or more Bi atoms, randomly select one as 'e' and one as 't'
            temp_rnd_e = rng.choice(temp_list_idx)
            edtg[temp_rnd_e] = 'e'
            temp_list_idx.remove(temp_rnd_e)

            # Select random index and set the element of the index as 't'
            temp_rnd_t = rng.choice(temp_list_idx)
            edtg[temp_rnd_t] = 't'
            temp_list_idx.remove(temp_rnd_t)

        mols_at_idx.append(at_idx)
        mols_edtg.append(edtg)

    # Convert edtg to endatom symbols
    endatom = {
               'e': ['Sb', 51], 't': ['Po', 84], 
              }

    for m, i in zip(mols_of_chromosomes, range(len(mols_of_chromosomes))):
        for key in endatom.keys():
            try:
                m.GetAtomWithIdx(
                    mols_at_idx[i][
                        mols_edtg[i].index(key)
                                ]
                                ).SetAtomicNum(endatom[key][1])
            except ValueError as e:
                pass

    # Time to connect fragments.

    L_mol = mols_of_chromosomes[0]
    L_n_atoms = mols_of_chromosomes[0].GetNumAtoms()
    L_ms_edtg = mols_edtg[0]
    L_ms_at_idx = mols_at_idx[0]

    for i in range(1, len(mols_of_chromosomes)):
        R_mol = mols_of_chromosomes[i]
        R_ms_edtg = mols_edtg[i]
        R_ms_at_idx = mols_at_idx[i]

        # Find connection points: L needs 't' (tail), R needs 'e' (end)
        # If L has only one Bi atom, use it as 't'
        # If R has only one Bi atom, use it as 'e'
        if 't' in L_ms_edtg:
            L_index_of_t = L_ms_at_idx[L_ms_edtg.index('t')]
        elif len(L_ms_at_idx) == 1:
            # Only one Bi atom in L, use it as connection point and set as 't' (Po)
            L_index_of_t = L_ms_at_idx[0]
            try:
                L_mol.GetAtomWithIdx(L_index_of_t).SetAtomicNum(84)
            except:
                pass
        else:
            # No valid connection point in L, skip this fragment
            continue

        if 'e' in R_ms_edtg:
            R_index_of_e = R_ms_at_idx[R_ms_edtg.index('e')]
        elif len(R_ms_at_idx) == 1:
            # Only one Bi atom in R, use it as connection point and set as 'e' (Sb)
            R_index_of_e = R_ms_at_idx[0]
            try:
                R_mol.GetAtomWithIdx(R_index_of_e).SetAtomicNum(51)
            except:
                pass
        else:
            # No valid connection point in R, skip this fragment
            continue

        combo = Chem.CombineMols(L_mol, R_mol)
        edcombo = Chem.EditableMol(combo)

        edcombo.AddBond(L_index_of_t, R_index_of_e + L_n_atoms, order=Chem.rdchem.BondType.SINGLE)

        L_mol = edcombo.GetMol()
        # Sanitize the molecule to ensure proper structure
        try:
            Chem.SanitizeMol(L_mol)
        except:
            pass
        L_ms_edtg = [''] * len(L_ms_edtg)
        L_ms_edtg.extend(mols_edtg[i])
        L_ms_at_idx.extend([x + L_n_atoms for x in mols_at_idx[i]])
        L_n_atoms = L_mol.GetNumAtoms()

    # Sanitize final molecule before generating SMILES
    try:
        Chem.SanitizeMol(L_mol)
    except:
        pass
    L_mol = _contract_connection_atoms(L_mol)
    try:
        Chem.SanitizeMol(L_mol)
    except:
        pass
    SMILES_connected = _finalize_two_star_psmiles(L_mol)
    if SMILES_connected is None:
        return None
    
    # Remove explicit hydrogen atoms from SMILES
    # RDKit sometimes generates SMILES with explicit H (e.g., [CH2], [NH2])
    # We remove these to get standard SMILES format where H is implicit
    # Replace [XHn] where X is an element and n is a number (e.g., [CH2] -> [C])
    # But preserve [*] for wildcards and other bracketed atoms
    SMILES_connected = re.sub(r'\[([A-Z][a-z]?)H(\d+)\]', r'[\1]', SMILES_connected)
    # Handle cases like [CH], [NH], [OH] (single H, no number)
    SMILES_connected = re.sub(r'\[([A-Z][a-z]?)H\]', r'[\1]', SMILES_connected)

    if SMILES_connected is not None and SMILES_connected.count('*') > 2:
        temp_res = longest_smiles(SMILES_connected)
        if temp_res is not None and temp_res != '' and temp_res.count('*') == 2:
            SMILES_connected = temp_res
        else:
            SMILES_connected = _linearize_two_stars(SMILES_connected)

    # Ensure exactly two stars for downstream MMPolymer compatibility
    if SMILES_connected is None or not validate_linear_polymer(SMILES_connected):
        return None
    return SMILES_connected

def _linearize_two_stars(smiles):
    s = smiles.replace('*', '[Bi]')
    m = Chem.MolFromSmiles(s)
    if m is None:
        return ''
    at_idx = []
    num_atom_m = m.GetNumAtoms()
    atom_i = 0
    for atom in m.GetAtoms():
        atom_i = atom_i + 1
        if atom.GetSymbol() == 'Bi':
            at_idx.append(atom.GetIdx())
        if atom_i == num_atom_m:
            break
    if len(at_idx) < 2:
        return ''
    star_pairs = list(itertools.combinations(at_idx, 2))
    idx1 = 0
    idx2 = 0
    len_path_max = -999
    for star_pair in star_pairs:
        path_temp = Chem.GetShortestPath(m, star_pair[0], star_pair[1])
        path_temp_ = list(path_temp)
        len_path = len(path_temp_)
        if len_path > len_path_max:
            len_path_max = len_path
            idx1 = star_pair[0]
            idx2 = star_pair[1]
    for idx in at_idx:
        if idx == idx1 or idx == idx2:
            m.GetAtomWithIdx(idx).SetAtomicNum(82)
    s2 = Chem.MolToSmiles(m)
    s2 = s2.replace('([Bi])', '')
    s2 = s2.replace('[Bi]', '')
    s2 = s2.replace('[Pb]', '[*]')
    return s2

def _contract_connection_atoms(m):
    pairs = []
    for b in m.GetBonds():
        a1 = b.GetBeginAtom()
        a2 = b.GetEndAtom()
        nums = (a1.GetAtomicNum(), a2.GetAtomicNum())
        s = set(nums)
        if s == {51, 84} or s == {51} or s == {84}:
            pairs.append((a1.GetIdx(), a2.GetIdx()))
    if not pairs:
        return m
    add_bonds = []
    to_delete = set()
    for po_idx, sb_idx in pairs:
        a_po = m.GetAtomWithIdx(po_idx)
        a_sb = m.GetAtomWithIdx(sb_idx)
        neigh_po = [n.GetIdx() for n in a_po.GetNeighbors() if n.GetIdx() != sb_idx]
        neigh_sb = [n.GetIdx() for n in a_sb.GetNeighbors() if n.GetIdx() != po_idx]
        if len(neigh_po) == 1 and len(neigh_sb) == 1:
            add_bonds.append((neigh_po[0], neigh_sb[0]))
            to_delete.add(po_idx)
            to_delete.add(sb_idx)
    ed = Chem.EditableMol(m)
    existing = set()
    for b in m.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        existing.add(tuple(sorted((i, j))))
    for i, j in add_bonds:
        if tuple(sorted((i, j))) not in existing:
            ed.AddBond(i, j, order=Chem.rdchem.BondType.SINGLE)
            existing.add(tuple(sorted((i, j))))
    for idx in sorted(to_delete, reverse=True):
        ed.RemoveAtom(idx)
    return ed.GetMol()

def _finalize_two_star_psmiles(m):
    try:
        Chem.SanitizeMol(m)
    except:
        pass
    # Remove existing star atoms and contract placeholder atoms
    ed = Chem.EditableMol(m)
    star_idxs = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(star_idxs, reverse=True):
        ed.RemoveAtom(idx)
    m2 = ed.GetMol()
    # Contract any remaining Bi/Sb/Po singletons or linears
    m2 = _contract_placeholders_simple(m2)
    try:
        Chem.SanitizeMol(m2)
    except:
        pass
    # Choose two atoms that can accept a star without breaking valence
    heavy = [a.GetIdx() for a in m2.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) < 2:
        return None
    def can_accept_star(idx: int) -> bool:
        a = m2.GetAtomWithIdx(idx)
        try:
            imp = a.GetImplicitValence()
        except Exception:
            imp = a.GetNumImplicitHs()
        # require at least one implicit valence/hydrogen to replace
        if imp is None:
            return False
        return imp >= 1
    terminals = [i for i in heavy if m2.GetAtomWithIdx(i).GetDegree() == 1 and can_accept_star(i)]
    cand_pool = terminals if len(terminals) >= 2 else [i for i in heavy if can_accept_star(i)]
    if len(cand_pool) < 2:
        return None
    best = (cand_pool[0], cand_pool[1])
    best_len = -1
    for i in range(len(cand_pool)):
        for j in range(i+1, len(cand_pool)):
            try:
                p = Chem.GetShortestPath(m2, cand_pool[i], cand_pool[j])
                Lp = len(p)
                if Lp > best_len:
                    best_len = Lp
                    best = (cand_pool[i], cand_pool[j])
            except:
                continue
    ed2 = Chem.EditableMol(m2)
    s1 = Chem.Atom(0)
    s2 = Chem.Atom(0)
    idx_s1 = ed2.AddAtom(s1)
    idx_s2 = ed2.AddAtom(s2)
    ed2.AddBond(idx_s1, best[0], order=Chem.rdchem.BondType.SINGLE)
    ed2.AddBond(idx_s2, best[1], order=Chem.rdchem.BondType.SINGLE)
    m3 = ed2.GetMol()
    try:
        Chem.SanitizeMol(m3)
    except:
        pass
    return Chem.MolToSmiles(m3)

def _contract_placeholders_simple(m):
    ed = Chem.EditableMol(m)
    to_remove = []
    to_add_bonds = []
    for a in m.GetAtoms():
        if a.GetAtomicNum() in (83, 51, 84):
            idx = a.GetIdx()
            neigh = [n.GetIdx() for n in a.GetNeighbors()]
            if len(neigh) == 1:
                to_remove.append(idx)
            elif len(neigh) == 2:
                to_add_bonds.append(tuple(neigh))
                to_remove.append(idx)
    existing = set()
    for b in m.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        existing.add(tuple(sorted((i, j))))
    for i, j in to_add_bonds:
        if tuple(sorted((i, j))) not in existing:
            ed.AddBond(i, j, order=Chem.rdchem.BondType.SINGLE)
            existing.add(tuple(sorted((i, j))))
    for idx in sorted(set(to_remove), reverse=True):
        ed.RemoveAtom(idx)
    return ed.GetMol()
