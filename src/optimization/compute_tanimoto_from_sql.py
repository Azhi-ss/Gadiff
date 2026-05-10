#!/usr/bin/env python3
"""
从 SQLite 数据库读取指定代数的 SMILES，计算 Morgan 指纹与塔尼莫相似度矩阵，
并将结果保存到 CSV 文件，用于后续可视化化学多样性收敛。
"""
from __future__ import annotations

import sqlite3
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def compute_morgan_fingerprints(
    smiles_list: List[str],
    radius: int = 2,
    nbits: int = 1024,
    use_chirality: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """返回 (N, nbits) 矩阵与有效 SMILES 的索引列表"""
    vectors: List[np.ndarray] = []
    valid_indices: List[int] = []
    
    def _process_polymer_smiles_like_mmpolymer(s: str) -> str:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return s
        star_ids = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        if len(star_ids) != 2:
            return s
        try:
            n1 = mol.GetAtomWithIdx(star_ids[0]).GetNeighbors()
            n2 = mol.GetAtomWithIdx(star_ids[1]).GetNeighbors()
            if len(n1) != 1 or len(n2) != 1:
                return s
            star_pair_list = [star_ids[0], n1[0].GetIdx(), star_ids[1], n2[0].GetIdx()]
            mol.GetAtomWithIdx(star_pair_list[0]).SetAtomicNum(
                mol.GetAtomWithIdx(star_pair_list[3]).GetAtomicNum()
            )
            mol.GetAtomWithIdx(star_pair_list[2]).SetAtomicNum(
                mol.GetAtomWithIdx(star_pair_list[1]).GetAtomicNum()
            )
            return Chem.MolToSmiles(mol)
        except Exception:
            return s
    for idx, smi in enumerate(smiles_list):
        smi_proc = _process_polymer_smiles_like_mmpolymer(smi)
        mol = Chem.MolFromSmiles(smi_proc)
        if mol is None:
            continue
        bit_vect = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=nbits,
            useChirality=use_chirality,
        )
        arr = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bit_vect, arr)
        if np.sum(arr) == 0.0:
            continue
        vectors.append(arr)
        valid_indices.append(idx)
    if not vectors:
        return np.empty((0, nbits), dtype=np.float32), []
    return np.vstack(vectors), valid_indices


def pairwise_tanimoto(fp_matrix: np.ndarray) -> np.ndarray:
    """计算塔尼莫相似度矩阵（对角线为 1）"""
    fp = np.asarray(fp_matrix, dtype=np.float32)
    dot = fp @ fp.T
    bit_sums = fp.sum(axis=1, keepdims=True)
    denom = bit_sums + bit_sums.T - dot
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = np.where(denom > 0.0, dot / denom, 0.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def load_generation_smiles(db_path: Path, generation: int) -> List[str]:
    """从 SQLite 读取指定代数的 SMILES 列表"""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    rows = cur.execute(
        'SELECT smiles_string FROM polymer WHERE generation = ? ORDER BY planetary_id',
        (generation,),
    ).fetchall()
    con.close()
    return [r['smiles_string'] for r in rows if r['smiles_string']]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='从 SQLite 读取 SMILES，计算塔尼莫相似度矩阵并保存到 CSV'
    )
    parser.add_argument(
        '--db',
        type=Path,
        default=Path('results_2096dna_standard/BRICSPolyPlanet/planetary_database.sqlite'),
        help='SQLite 数据库路径',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('results_2096dna_standard/diversity'),
        help='输出目录路径',
    )
    parser.add_argument(
        '--generations',
        type=int,
        nargs='+',
        default=list(range(10)),
        help='要处理的代数列表（默认 0-9）',
    )
    parser.add_argument(
        '--radius',
        type=int,
        default=2,
        help='Morgan 指纹半径',
    )
    parser.add_argument(
        '--nbits',
        type=int,
        default=1024,
        help='Morgan 指纹位数',
    )
    parser.add_argument(
        '--use-chirality',
        action='store_true',
        default=True,
        help='是否使用手性信息',
    )
    args = parser.parse_args()

    db_path = args.db.resolve()
    if not db_path.exists():
        raise FileNotFoundError(f'数据库不存在: {db_path}')

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []

    for gen in args.generations:
        print(f'--- Generation {gen} ---')
        smiles_list = load_generation_smiles(db_path, gen)
        if not smiles_list:
            print('  无有效 SMILES，跳过')
            continue

        fp_matrix, valid_idx = compute_morgan_fingerprints(
            smiles_list,
            radius=args.radius,
            nbits=args.nbits,
            use_chirality=args.use_chirality,
        )
        if fp_matrix.size == 0:
            print('  无有效指纹，跳过')
            continue

        sim_matrix = pairwise_tanimoto(fp_matrix)
        # 取上三角（不含对角线）统计
        if sim_matrix.shape[0] > 1:
            upper = sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]
            stats = {
                'tanimoto_mean': float(np.mean(upper)),
                'tanimoto_std': float(np.std(upper)),
                'tanimoto_min': float(np.min(upper)),
                'tanimoto_max': float(np.max(upper)),
            }
        else:
            stats = {
                'tanimoto_mean': 1.0,
                'tanimoto_std': 0.0,
                'tanimoto_min': 1.0,
                'tanimoto_max': 1.0,
            }

        # 保存矩阵 CSV
        labels = [
            f"{valid_idx[i]}_{smiles_list[valid_idx[i]]}"
            for i in range(len(valid_idx))
        ]
        matrix_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
        matrix_path = out_dir / f'tanimoto_gen_{gen:03d}.csv'
        matrix_df.to_csv(matrix_path)
        print(f'  矩阵已保存: {matrix_path}')
        print(f"  统计: mean={stats['tanimoto_mean']:.3f}, std={stats['tanimoto_std']:.3f}")

        summary_rows.append({'generation': gen, **stats})

    # 保存汇总 CSV
    summary_df = pd.DataFrame(summary_rows).sort_values('generation')
    summary_path = out_dir / 'tanimoto_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'汇总已保存: {summary_path}')


if __name__ == '__main__':
    main()
