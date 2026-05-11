#!/usr/bin/env python3
"""
正式项目：使用 PolyGA + MMPolymer，对扩散增强片段 DNA 进行聚合物进化优化。

- DNA 来源：data/enriched_dna.csv
- 生成函数：polyga.enriched_dna.enriched_chromosome_to_psmiles
- 预测器：必须使用真实 MMPolymer 权重
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import sqlite3
import json

import numpy as np
import pandas as pd
from datetime import datetime
import shutil

# 将仓库内模型源码加入路径
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / 'src' / 'models'
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from polyga.polygod import PolyPlanet, PolyLand, PolyNation  # type: ignore
from polyga.selection_schemes import elite  # type: ignore
from polyga import utils  # type: ignore
from polyga.enriched_dna import enriched_chromosome_to_psmiles  # type: ignore
from polyga.sa_score import add_sa_scores  # type: ignore

# 可选 MMPolymer 预测器
# 延迟导入 MMPolymer 预测器（在 maybe_create_mmpolymer_predictor 内部）


CONFIG: Dict[str, object] = {
    # 路径
    'dna_file': 'data/enriched_dna.csv',
    'results_dir': 'results_polygen_formal',
    'planet_name': 'PolyGenFormalPlanet',

    # MMPolymer 设置
    'weight_dir': '/internfs/Zy/polygen_assets/mm_polymer/finetune_data',
    'properties': ['Tg', 'DC'],
    'use_gpu': True,
    'batch_size': 128,
    'num_cpus': 1,
    'allow_heuristic_predictor': False,

    # 目标权重（示例：最大化 Tg，最小化 DC）
    'weight_tg': 1.5,
    'weight_dc': 1.0,
    'weight_sa': 1.0,

    # GA 参数（正式运行）
    'random_seed': 43,
    'num_generations': 100,
    'population_size': 120,
    'num_chromosomes_initial': 1,
    'num_families': 20,
    'num_parents_per_family': 3,
    'num_children_per_family': 6,
    'elite_retention_count': 5,

    # 遗传操作参数
    'fraction_mutation': 0.01,
    'fraction_mutate_additional_block': 0.10,
    'crossover_position': 'relative_center',
    'crossover_sigma_offset': 0.25,
    'mutation_sigma_offset': 0.20,

    # 指纹与多样性分析配置
    'fingerprint_radius': 2,    
    'fingerprint_nbits': 1024,
    'fingerprint_use_chirality': True,
    'diversity_similarity_threshold_start': 0.50,
    'diversity_similarity_threshold_end': 0.20,
    'diversity_similarity_decay_start_gen': 10,
    'diversity_similarity_decay_end_gen': 90,

    # 适应度归一化与惩罚配置
    # normalization 可选: 'quantile' 或 'minmax'
    'fitness_normalization': 'quantile',
    # 分位数归一化边界（Tg 越大越好，DC 越小越好）
    'tg_q_low': 0.10,
    'tg_q_high': 0.90,
    'dc_q_low': 0.10,
    'dc_q_high': 0.90,
    'sa_q_low': 0.10,
    'sa_q_high': 0.90,
    'tg_gate_soft': 500.0,
    'tg_gate_hard': 600.0,
    'dc_gate_soft': 3.0,
    'dc_gate_hard': 4.0,
    'sa_gate_soft': 3.0,
    'sa_gate_hard': 5.0,
    'gate_floor': 0.10,
    # DC 阈值惩罚：支持 'quantile' 或 'value'
    'dc_penalty_mode': 'quantile',
    'dc_max_quantile': 0.80,
    'dc_max_value': 9999.0,
    # 触发惩罚后的缩放系数（乘法因子）
    'dc_penalty_factor': 0.10,

    # 动态遗传操作调节
    'fraction_mutation_start': 0.30,
    'fraction_mutation_end': 0.1,
    'mutation_decay_start_gen': 10,
    'mutation_decay_end_gen': 80,
}


_NORMALIZATION_CACHE: Dict[str, float] = {}


def _configure_thread_env(max_threads: int) -> None:
    """限制线程并行度，避免 CPU 负载过高。"""
    max_threads = max(1, int(max_threads))
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = str(max_threads)


def resolve_dna_path(project_dir: Path, dna_file: str | Path) -> Path:
    """Resolve and validate the configured DNA CSV path."""
    dna_path = Path(dna_file)
    if not dna_path.is_absolute():
        dna_path = project_dir / dna_path
    if not dna_path.exists():
        raise FileNotFoundError(f"DNA file not found: {dna_path}")
    return dna_path


def _linear_schedule(
    current_gen: int,
    start_gen: int,
    end_gen: int,
    start_value: float,
    end_value: float,
) -> float:
    """线性插值调度函数，用于衰减阈值或变异率。"""
    if end_gen <= start_gen:
        return end_value
    if current_gen <= start_gen:
        return start_value
    if current_gen >= end_gen:
        return end_value
    ratio = (current_gen - start_gen) / float(end_gen - start_gen)
    return start_value + ratio * (end_value - start_value)


def create_fitness_function(
    weight_tg: float | None = None,
    weight_dc: float | None = None,
    weight_sa: float | None = None,
) -> Callable[[pd.DataFrame, List[str]], pd.DataFrame]:
    """构建高 Tg、低 DC、低 SA 的平衡归一化适应度函数。"""
    # 清空缓存，确保每次运行使用新的基准
    _NORMALIZATION_CACHE.clear()

    norm_scheme: str = str(CONFIG.get('fitness_normalization', 'quantile'))
    weights = {
        'tg': float(CONFIG.get('weight_tg', 1.0) if weight_tg is None else weight_tg),
        'dc': float(CONFIG.get('weight_dc', 1.0) if weight_dc is None else weight_dc),
        'sa': float(CONFIG.get('weight_sa', 1.0) if weight_sa is None else weight_sa),
    }
    weight_total = sum(max(v, 0.0) for v in weights.values()) or 1.0
    q_bounds = {
        'tg': (float(CONFIG.get('tg_q_low', 0.10)), float(CONFIG.get('tg_q_high', 0.90))),
        'dc': (float(CONFIG.get('dc_q_low', 0.10)), float(CONFIG.get('dc_q_high', 0.90))),
        'sa': (float(CONFIG.get('sa_q_low', 0.10)), float(CONFIG.get('sa_q_high', 0.90))),
    }
    gate_floor = float(CONFIG.get('gate_floor', 0.10))
    gate_thresholds = {
        'tg': (float(CONFIG.get('tg_gate_soft', 500.0)), float(CONFIG.get('tg_gate_hard', 600.0))),
        'dc': (float(CONFIG.get('dc_gate_soft', 3.0)), float(CONFIG.get('dc_gate_hard', 4.0))),
        'sa': (float(CONFIG.get('sa_gate_soft', 3.0)), float(CONFIG.get('sa_gate_hard', 5.0))),
    }

    def _clip01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def _safe_div(num: float, den: float) -> float:
        if den == 0 or np.isclose(den, 0.0):
            return 0.0
        return num / den

    def _bounds(series: pd.Series, key: str) -> tuple[float, float]:
        q_low, q_high = q_bounds[key]
        if norm_scheme == 'quantile':
            low = float(series.dropna().quantile(q_low)) if series.notna().any() else 0.0
            high = float(series.dropna().quantile(q_high)) if series.notna().any() else 1.0
        else:
            low = float(series.min()) if series.notna().any() else 0.0
            high = float(series.max()) if series.notna().any() else 1.0
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return 0.0, 1.0
        return low, high

    def _normalize(df_in: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        tg_series = pd.to_numeric(df_in.get('Tg'), errors='coerce')
        dc_series = pd.to_numeric(df_in.get('DC'), errors='coerce')
        sa_series = pd.to_numeric(df_in.get('SA_score'), errors='coerce')

        if not all(key in _NORMALIZATION_CACHE for key in ('tg_L', 'tg_U', 'dc_L', 'dc_U', 'sa_L', 'sa_U')):
            tg_L, tg_U = _bounds(tg_series, 'tg')
            dc_L, dc_U = _bounds(dc_series, 'dc')
            sa_L, sa_U = _bounds(sa_series, 'sa')
            _NORMALIZATION_CACHE.update({
                'tg_L': tg_L,
                'tg_U': tg_U,
                'dc_L': dc_L,
                'dc_U': dc_U,
                'sa_L': sa_L,
                'sa_U': sa_U,
            })
        tg_L = _NORMALIZATION_CACHE['tg_L']
        tg_U = _NORMALIZATION_CACHE['tg_U']
        dc_L = _NORMALIZATION_CACHE['dc_L']
        dc_U = _NORMALIZATION_CACHE['dc_U']
        sa_L = _NORMALIZATION_CACHE['sa_L']
        sa_U = _NORMALIZATION_CACHE['sa_U']

        # 归一化：Tg 越大越好，DC/SA 越小越好
        tg_norm = ((tg_series - tg_L) / (tg_U - tg_L)).apply(lambda v: _clip01(float(v)) if pd.notna(v) else np.nan)
        dc_norm = ((dc_U - dc_series) / (dc_U - dc_L)).apply(lambda v: _clip01(float(v)) if pd.notna(v) else np.nan)
        sa_norm = ((sa_U - sa_series) / (sa_U - sa_L)).apply(lambda v: _clip01(float(v)) if pd.notna(v) else np.nan)

        return tg_norm, dc_norm, sa_norm

    def _soft_upper_gate(series: pd.Series, key: str) -> pd.Series:
        soft, hard = gate_thresholds[key]
        values = pd.to_numeric(series, errors='coerce')
        if hard <= soft:
            return pd.Series(1.0, index=values.index)
        gates = []
        for value in values:
            if pd.isna(value):
                gates.append(0.0)
            elif float(value) <= soft:
                gates.append(1.0)
            elif float(value) >= hard:
                gates.append(gate_floor)
            else:
                ratio = (float(value) - soft) / (hard - soft)
                gates.append(1.0 - ratio * (1.0 - gate_floor))
        return pd.Series(gates, index=values.index, dtype=float)

    def fitness_function(df: pd.DataFrame, fp_headers: List[str]) -> pd.DataFrame:
        if 'SA_score' not in df.columns and 'smiles_string' in df.columns:
            df = add_sa_scores(df)
        else:
            df = df.copy()
        tg_norm, dc_norm, sa_norm = _normalize(df)
        df['tg_norm'] = tg_norm
        df['dc_norm'] = dc_norm
        df['sa_norm'] = sa_norm
        df['tg_gate'] = _soft_upper_gate(df['Tg'], 'tg')
        df['dc_gate'] = _soft_upper_gate(df['DC'], 'dc')
        df['sa_gate'] = _soft_upper_gate(df['SA_score'], 'sa')
        df['fitness_gate'] = df['tg_gate'] * df['dc_gate'] * df['sa_gate']

        scores: List[float] = []
        base_scores: List[float] = []
        for idx, row in df.iterrows():
            tg = row.get('Tg', np.nan)
            dc = row.get('DC', np.nan)
            sa = row.get('SA_score', np.nan)
            if pd.isna(tg) or pd.isna(dc) or pd.isna(sa):
                base_scores.append(np.nan)
                scores.append(-9.99e8)
                continue
            tgn = float(tg_norm.loc[idx]) if np.isfinite(tg_norm.loc[idx]) else 0.0
            dcn = float(dc_norm.loc[idx]) if np.isfinite(dc_norm.loc[idx]) else 0.0
            san = float(sa_norm.loc[idx]) if np.isfinite(sa_norm.loc[idx]) else 0.0
            base_fit = (weights['tg'] * tgn + weights['dc'] * dcn + weights['sa'] * san) / weight_total
            fit = base_fit * float(df.loc[idx, 'fitness_gate'])
            base_scores.append(base_fit)
            scores.append(fit)

        df['base_fitness'] = base_scores
        df['fitness'] = scores
        return df

    return fitness_function


def _compute_morgan_fingerprints(
    smiles_series: pd.Series,
    radius: int,
    nbits: int,
    use_chirality: bool,
) -> Tuple[np.ndarray, List[int]]:
    from rdkit import Chem, DataStructs  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

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

    vectors: List[np.ndarray] = []
    valid_indices: List[int] = []
    for idx, smi in smiles_series.astype(str).items():
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


def _pairwise_tanimoto(fp_matrix: np.ndarray) -> np.ndarray:
    fp = np.asarray(fp_matrix, dtype=np.float32)
    dot = fp @ fp.T
    bit_sums = fp.sum(axis=1, keepdims=True)
    denom = bit_sums + bit_sums.T - dot
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = np.where(denom > 0.0, dot / denom, 0.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def _summarize_generation_to_csv(db_path: Path, gen: int, out_csv: Path, cfg: Dict[str, object]) -> None:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    rows = cur.execute('SELECT generation, smiles_string, properties FROM polymer WHERE generation = ?', (gen,)).fetchall()
    if not rows:
        con.close()
        return
    recs: List[Dict[str, object]] = []
    for r in rows:
        try:
            props = json.loads(r['properties']) if r['properties'] else {}
        except Exception:
            props = {}
        recs.append({
            'generation': int(r['generation']) if r['generation'] is not None else gen,
            'smiles': r['smiles_string'],
            'Tg': props.get('Tg', None),
            'DC': props.get('DC', None),
            'SA_score': props.get('SA_score', None),
            'tg_norm': props.get('tg_norm', None),
            'dc_norm': props.get('dc_norm', None),
            'sa_norm': props.get('sa_norm', None),
            'base_fitness': props.get('base_fitness', None),
            'fitness_gate': props.get('fitness_gate', None),
            'tg_gate': props.get('tg_gate', None),
            'dc_gate': props.get('dc_gate', None),
            'sa_gate': props.get('sa_gate', None),
        })
    df = pd.DataFrame(recs)
    if 'SA_score' not in df.columns or df['SA_score'].isna().any():
        df = add_sa_scores(df.rename(columns={'smiles': 'smiles_string'})).rename(columns={'smiles_string': 'smiles'})
    df = df[df['Tg'].notna() & df['DC'].notna() & df['SA_score'].notna()].copy()
    if len(df) == 0:
        con.close()
        return
    weight_tg = float(cfg.get('weight_tg', 1.0))
    weight_dc = float(cfg.get('weight_dc', 0.5))
    weight_sa = float(cfg.get('weight_sa', 0.5))
    weight_total = max(weight_tg, 0.0) + max(weight_dc, 0.0) + max(weight_sa, 0.0) or 1.0
    norm_scheme = str(cfg.get('fitness_normalization', 'quantile'))
    tg_q_low = float(cfg.get('tg_q_low', 0.50))
    tg_q_high = float(cfg.get('tg_q_high', 0.90))
    dc_q_low = float(cfg.get('dc_q_low', 0.10))
    dc_q_high = float(cfg.get('dc_q_high', 0.50))
    tg = pd.to_numeric(df['Tg'], errors='coerce')
    dc = pd.to_numeric(df['DC'], errors='coerce')
    radius = int(cfg.get('fingerprint_radius', 2))
    nbits = int(cfg.get('fingerprint_nbits', 1024))
    use_chirality = bool(cfg.get('fingerprint_use_chirality', True))
    sa = pd.to_numeric(df['SA_score'], errors='coerce')
    if gen == 0 or not all(
        key in _NORMALIZATION_CACHE for key in ('tg_L', 'tg_U', 'dc_L', 'dc_U', 'sa_L', 'sa_U')
    ):
        # 仅在 Gen0 计算基准，后续代数强制使用缓存
        if gen != 0:
            raise ValueError(f"Normalization cache missing for generation {gen}")
        if norm_scheme == 'quantile':
            tg_L = float(tg.dropna().quantile(tg_q_low)) if tg.notna().any() else 0.0
            tg_U = float(tg.dropna().quantile(tg_q_high)) if tg.notna().any() else 1.0
            dc_L = float(dc.dropna().quantile(dc_q_low)) if dc.notna().any() else 0.0
            dc_U = float(dc.dropna().quantile(dc_q_high)) if dc.notna().any() else 1.0
            sa_L = float(sa.dropna().quantile(float(cfg.get('sa_q_low', 0.10)))) if sa.notna().any() else 0.0
            sa_U = float(sa.dropna().quantile(float(cfg.get('sa_q_high', 0.90)))) if sa.notna().any() else 1.0
        else:
            tg_L = float(tg.min()) if tg.notna().any() else 0.0
            tg_U = float(tg.max()) if tg.notna().any() else 1.0
            dc_L = float(dc.min()) if dc.notna().any() else 0.0
            dc_U = float(dc.max()) if dc.notna().any() else 1.0
            sa_L = float(sa.min()) if sa.notna().any() else 0.0
            sa_U = float(sa.max()) if sa.notna().any() else 1.0
        if not np.isfinite(tg_L) or not np.isfinite(tg_U) or tg_U <= tg_L:
            tg_L, tg_U = 0.0, 1.0
        if not np.isfinite(dc_L) or not np.isfinite(dc_U) or dc_U <= dc_L:
            dc_L, dc_U = 0.0, 1.0
        if not np.isfinite(sa_L) or not np.isfinite(sa_U) or sa_U <= sa_L:
            sa_L, sa_U = 0.0, 1.0
        _NORMALIZATION_CACHE['tg_L'] = tg_L
        _NORMALIZATION_CACHE['tg_U'] = tg_U
        _NORMALIZATION_CACHE['dc_L'] = dc_L
        _NORMALIZATION_CACHE['dc_U'] = dc_U
        _NORMALIZATION_CACHE['sa_L'] = sa_L
        _NORMALIZATION_CACHE['sa_U'] = sa_U
    else:
        tg_L = _NORMALIZATION_CACHE['tg_L']
        tg_U = _NORMALIZATION_CACHE['tg_U']
        dc_L = _NORMALIZATION_CACHE['dc_L']
        dc_U = _NORMALIZATION_CACHE['dc_U']
        sa_L = _NORMALIZATION_CACHE['sa_L']
        sa_U = _NORMALIZATION_CACHE['sa_U']
    tg_den = tg_U - tg_L
    dc_den = dc_U - dc_L
    sa_den = sa_U - sa_L
    tg_norm = ((tg - tg_L) / tg_den if tg_den != 0 else pd.Series(0.0, index=tg.index)).clip(0.0, 1.0)
    dc_norm = ((dc_U - dc) / dc_den if dc_den != 0 else pd.Series(0.0, index=dc.index)).clip(0.0, 1.0)
    sa_norm = ((sa_U - sa) / sa_den if sa_den != 0 else pd.Series(0.0, index=sa.index)).clip(0.0, 1.0)
    fitness = (weight_tg * tg_norm + weight_dc * dc_norm + weight_sa * sa_norm) / weight_total
    df['tg_norm'] = tg_norm
    df['dc_norm'] = dc_norm
    df['sa_norm'] = sa_norm

    # 计算并持久化塔尼莫相似度矩阵
    fp_matrix, valid_idx = _compute_morgan_fingerprints(df['smiles'], radius, nbits, use_chirality)
    diversity_stats = {
        'tanimoto_mean': float('nan'),
        'tanimoto_std': float('nan'),
        'tanimoto_min': float('nan'),
        'tanimoto_max': float('nan'),
    }
    if fp_matrix.size > 0:
        sim_matrix = _pairwise_tanimoto(fp_matrix)
        if sim_matrix.shape[0] > 1:
            upper = sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]
            diversity_stats = {
                'tanimoto_mean': float(np.mean(upper)),
                'tanimoto_std': float(np.std(upper)),
                'tanimoto_min': float(np.min(upper)),
                'tanimoto_max': float(np.max(upper)),
            }
        else:
            diversity_stats = {
                'tanimoto_mean': 1.0,
                'tanimoto_std': 0.0,
                'tanimoto_min': 1.0,
                'tanimoto_max': 1.0,
            }
        diversity_dir = out_csv.parent / 'diversity'
        diversity_dir.mkdir(parents=True, exist_ok=True)
        df_valid = df.iloc[valid_idx].copy()
        labels = [f"{idx}_{df_valid.loc[idx, 'smiles']}" for idx in df_valid.index]
        matrix_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
        matrix_path = diversity_dir / f'tanimoto_gen_{gen:03d}.csv'
        matrix_df.to_csv(matrix_path)
    def stat(s: pd.Series) -> Dict[str, float]:
        s = s.dropna().astype(float)
        if s.empty:
            return {'mean': float('nan'), 'median': float('nan'), 'p75': float('nan'), 'min': float('nan'), 'max': float('nan')}
        return {
            'mean': float(s.mean()),
            'median': float(s.quantile(0.5)),
            'p75': float(s.quantile(0.75)),
            'min': float(s.min()),
            'max': float(s.max()),
        }
    fs = stat(fitness)
    ts = stat(tg)
    ds = stat(dc)
    ss = stat(sa)
    best_idx = int(fitness.idxmax())
    best_row = {
        'best_fitness': float(fitness.loc[best_idx]) if pd.notna(fitness.loc[best_idx]) else np.nan,
        'best_tg': float(tg.loc[best_idx]) if pd.notna(tg.loc[best_idx]) else np.nan,
        'best_dc': float(dc.loc[best_idx]) if pd.notna(dc.loc[best_idx]) else np.nan,
        'best_sa': float(sa.loc[best_idx]) if pd.notna(sa.loc[best_idx]) else np.nan,
        'best_tg_norm': float(tg_norm.loc[best_idx]) if pd.notna(tg_norm.loc[best_idx]) else np.nan,
        'best_dc_norm': float(dc_norm.loc[best_idx]) if pd.notna(dc_norm.loc[best_idx]) else np.nan,
        'best_sa_norm': float(sa_norm.loc[best_idx]) if pd.notna(sa_norm.loc[best_idx]) else np.nan,
        'best_fitness_gate': float(df.loc[best_idx, 'fitness_gate']) if 'fitness_gate' in df.columns and pd.notna(df.loc[best_idx, 'fitness_gate']) else np.nan,
        'best_smiles': str(df.loc[best_idx, 'smiles']),
    }
    out_row = {
        'generation': int(gen),
        'n': int(len(df)),
        'fitness_mean': fs['mean'],
        'fitness_median': fs['median'],
        'fitness_p75': fs['p75'],
        'fitness_max': fs['max'],
        'tg_mean': ts['mean'],
        'tg_median': ts['median'],
        'tg_p75': ts['p75'],
        'tg_max': ts['max'],
        'dc_mean': ds['mean'],
        'dc_median': ds['median'],
        'dc_p75': ds['p75'],
        'dc_min': ds['min'],
        'dc_max': ds['max'],
        'sa_mean': ss['mean'],
        'sa_median': ss['median'],
        'sa_p75': ss['p75'],
        'sa_min': ss['min'],
        'sa_max': ss['max'],
        **diversity_stats,
        **best_row,
    }
    out_df = pd.DataFrame([out_row])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在相同代的数据，避免重复写入
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        if not existing_df.empty and gen in existing_df['generation'].values:
            print(f"    警告：代 {gen} 数据已存在于 {out_csv.name}，跳过重复写入")
            con.close()
            return
    
    header = not out_csv.exists()
    out_df.to_csv(out_csv, mode='a', header=header, index=False)
    con.close()


def summarize_generation(db_path: Path, gen: int, out_csv: Path, cfg: Dict[str, object]) -> None:
    """Write generation summary and propagate failures to the caller."""
    _summarize_generation_to_csv(db_path, gen, out_csv, cfg)


def morgan_fingerprint_function(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """生成 Morgan 指纹，供 PolyGA 进行多样性选择。"""
    radius = int(CONFIG.get('fingerprint_radius', 2))
    nbits = int(CONFIG.get('fingerprint_nbits', 1024))
    use_chirality = bool(CONFIG.get('fingerprint_use_chirality', True))

    fp_matrix, valid_idx = _compute_morgan_fingerprints(
        df['smiles_string'],
        radius,
        nbits,
        use_chirality,
    )
    if fp_matrix.size == 0:
        return df.iloc[0:0].copy(), []
    df_valid = df.loc[valid_idx].copy()
    fp_headers = [f'fp_{i}' for i in range(fp_matrix.shape[1])]
    fp_df = pd.DataFrame(fp_matrix, index=df_valid.index, columns=fp_headers)
    df_valid = pd.concat([df_valid, fp_df], axis=1)
    return df_valid, fp_headers


def maybe_create_mmpolymer_predictor(cfg: Dict[str, object]) -> Callable[[pd.DataFrame, List[str], Dict | None], pd.DataFrame]:
    """构建真实 MMPolymer 预测器；默认不允许回退到启发式预测器。"""
    weight_dir = Path(str(cfg['weight_dir']))
    properties: List[str] = list(cfg['properties'])  # type: ignore
    allow_fallback = bool(cfg.get('allow_heuristic_predictor', False))
    from polyga.mmpolymer_predict import _resolve_weight_path  # type: ignore

    # 核心权重路径检查
    missing: List[str] = []
    for prop in properties:
        expect = _resolve_weight_path(weight_dir, prop)
        if not expect.exists():
            missing.append(str(expect))

    if missing:
        msg = "MMPolymer weight files not found:\n" + "\n".join(missing)
        if not allow_fallback:
            raise FileNotFoundError(msg)
        print(msg)
        print('启用回退预测器。')
        return heuristic_predictor(properties)

    try:
        # 尝试延迟导入
        from polyga.mmpolymer_predict import create_mmpolymer_predictor  # type: ignore
        # 使用稳定缓存目录，避免多进程下临时目录被其他进程清理
        cache_dir = Path(str(cfg.get('results_dir', 'results_basic'))) / 'mmpolymer_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return create_mmpolymer_predictor(
            weight_dir=str(weight_dir),
            properties=properties,
            use_gpu=bool(cfg['use_gpu']),
            batch_size=int(cfg['batch_size']),
            cache_dir=str(cache_dir),
            cleanup_cache=True,
        )
    except Exception as e:  # 兜底保证可运行
        if not allow_fallback:
            raise RuntimeError(f'创建 MMPolymer 预测器失败: {e}') from e
        print(f'创建 MMPolymer 预测器失败，使用回退预测器。原因: {e}')
        return heuristic_predictor(properties)


def heuristic_predictor(properties: List[str]) -> Callable[[pd.DataFrame, List[str], Dict | None], pd.DataFrame]:
    """轻量级回退预测器：
    - 用 RDKit 解析聚合物 SMILES（可能含两个星号 *）
    - 以原子数、芳香环计数等简易特征经线性缩放生成占位预测
    """
    from rdkit import Chem  # 延迟导入

    def predict(df: pd.DataFrame, fp_headers: List[str], models: Dict | None = None) -> pd.DataFrame:
        tg_vals: List[float] = []
        dc_vals: List[float] = []
        for smi in df['smiles_string'].astype(str).tolist():
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    tg_vals.append(np.nan)
                    dc_vals.append(np.nan)
                    continue
                n_atoms = mol.GetNumAtoms()
                n_rings = Chem.GetSSSR(mol)
                # 简单启发：更多原子 -> 更高 Tg；更多环 -> 更高 DC（仅占位）
                tg = 50.0 + 0.6 * n_atoms + 5.0 * float(n_rings)
                dc = 2.5 + 0.01 * n_atoms + 0.03 * float(n_rings)
                tg_vals.append(float(tg))
                dc_vals.append(float(dc))
            except Exception:
                tg_vals.append(np.nan)
                dc_vals.append(np.nan)
        if 'Tg' in properties:
            df['Tg'] = tg_vals
        if 'DC' in properties:
            df['DC'] = dc_vals
        return df

    return predict


def main() -> None:
    print('=' * 70)
    print('PolyGA Optimization (BRICS DNA)')
    print('=' * 70)

    project_dir = REPO_ROOT
    os.chdir(project_dir)
    from scripts.validate_polygen_setup import validate_polygen_setup
    validate_polygen_setup(project_dir, Path(str(CONFIG['weight_dir'])), project_dir / str(CONFIG['dna_file']))
    os.makedirs(str(CONFIG['results_dir']), exist_ok=True)

    dna_file = resolve_dna_path(project_dir, str(CONFIG['dna_file']))

    # 预测器与适应度
    predict_fn = maybe_create_mmpolymer_predictor(CONFIG)
    fitness_fn = create_fitness_function(
        weight_tg=float(CONFIG['weight_tg']),
        weight_dc=float(CONFIG['weight_dc']),
        weight_sa=float(CONFIG['weight_sa']),
    )

    _configure_thread_env(CONFIG.get('num_cpus', os.cpu_count()))

    # 初始化 PolyGA 组件
    # 若已存在旧数据库，先备份以避免表已存在冲突
    existing_db = Path(CONFIG['results_dir']) / str(CONFIG['planet_name']) / 'planetary_database.sqlite'
    if existing_db.exists():
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_db = existing_db.with_name(f"planetary_database_{ts}.sqlite.bak")
        backup_db.parent.mkdir(parents=True, exist_ok=True)
        existing_db.rename(backup_db)
        print(f"已备份旧数据库: {backup_db}")
    planet = PolyPlanet(
        name=str(CONFIG['planet_name']),
        predict_function=predict_fn,
        fingerprint_function=morgan_fingerprint_function,
        path_to_dna=str(dna_file),
        models=None,
        num_cpus=int(CONFIG.get('num_cpus', os.cpu_count())),
        random_seed=int(CONFIG['random_seed']),
        save_folder=str(CONFIG['results_dir']),
    )

    land = PolyLand(
        name='OptimizationLand',
        planet=planet,
        generative_function=enriched_chromosome_to_psmiles,
        fitness_function=fitness_fn,
        crossover_position=str(CONFIG['crossover_position']),
        fraction_mutation=float(CONFIG.get('fraction_mutation_start', CONFIG.get('fraction_mutation', 0.25))),
        fraction_mutate_additional_block=float(CONFIG['fraction_mutate_additional_block']),
        crossover_sigma_offset=float(CONFIG['crossover_sigma_offset']),
        mutation_sigma_offset=float(CONFIG['mutation_sigma_offset']),
    )
    land.elite_retention_count = int(CONFIG.get('elite_retention_count', 0))
    land.target_population_size = int(CONFIG['population_size'])
    land.refill_num_chromosomes_initial = int(CONFIG['num_chromosomes_initial'])

    nation = PolyNation(
        name='OptimizerNation',
        land=land,
        num_population_initial=int(CONFIG['population_size']),
        num_chromosomes_initial=int(CONFIG['num_chromosomes_initial']),
        num_families=int(CONFIG['num_families']),
        num_parents_per_family=int(CONFIG['num_parents_per_family']),
        num_children_per_family=int(CONFIG['num_children_per_family']),
        selection_scheme=elite,
        partner_selection='diversity',
        diversity_similarity_threshold=float(CONFIG.get('diversity_similarity_threshold_start', CONFIG.get('diversity_similarity_threshold', 0.5))),
        random_seed=int(CONFIG['random_seed']),
    )

    # ---------- CFM diffusion mutation setup ----------
    checkpoint_path = str(CONFIG.get(
        'evodiffmol_checkpoint',
        'EvoDiffMol/assets/checkpoints/moses_without_h_80.pt',
    ))
    if os.path.exists(checkpoint_path):
        try:
            from setup_diffusion import setup_diffusion_modules  # type: ignore
            land = setup_diffusion_modules(
                land,
                checkpoint_path=checkpoint_path,
                device='cuda' if CONFIG.get('use_gpu', True) else 'cpu',
                ad_threshold=float(CONFIG.get('ad_threshold', 0.4)),
                egd_mutation_prob=float(CONFIG.get('egd_mutation_prob', 0.3)),
                t_prime=int(CONFIG.get('t_prime', 250)),
                enable_cbsg=bool(CONFIG.get('enable_cbsg', True)),
            )
            print(f"CFM diffusion mutation: ENABLED (checkpoint={checkpoint_path})")
        except Exception as e:
            print(f"CFM diffusion mutation: FAILED to load ({e}) — falling back to random mutation")
    else:
        print(f"CFM diffusion mutation: SKIPPED (checkpoint not found: {checkpoint_path})")
    # ----------------------------------------------------

    print(f"Planet: {CONFIG['planet_name']} | DNA: {dna_file} | Chromosomes: {len(planet.chromosomes)}")
    print(
        "Fitness: high Tg, low DC, low SA normalized blend "
        f"(Tg={CONFIG['weight_tg']}, DC={CONFIG['weight_dc']}, SA={CONFIG['weight_sa']})"
    )

    # 运行进化
    div_start_val = float(CONFIG.get('diversity_similarity_threshold_start', 0.5))
    div_end_val = float(CONFIG.get('diversity_similarity_threshold_end', div_start_val))
    div_decay_start = int(CONFIG.get('diversity_similarity_decay_start_gen', 0))
    div_decay_end = int(CONFIG.get('diversity_similarity_decay_end_gen', int(CONFIG['num_generations']) - 1))

    mut_start_val = float(CONFIG.get('fraction_mutation_start', CONFIG.get('fraction_mutation', land.fraction_mutation)))
    mut_end_val = float(CONFIG.get('fraction_mutation_end', mut_start_val))
    mut_decay_start = int(CONFIG.get('mutation_decay_start_gen', 0))
    mut_decay_end = int(CONFIG.get('mutation_decay_end_gen', int(CONFIG['num_generations']) - 1))

    for gen in range(int(CONFIG['num_generations'])):
        print(f"\n--- Generation {gen + 1}/{CONFIG['num_generations']} ---")
        # 动态调整多样性阈值与变异率
        nation.diversity_similarity_threshold = max(
            0.0,
            min(
                1.0,
                _linear_schedule(gen, div_decay_start, div_decay_end, div_start_val, div_end_val),
            ),
        )
        land.fraction_mutation = max(
            0.0,
            _linear_schedule(gen, mut_decay_start, mut_decay_end, mut_start_val, mut_end_val),
        )
        print(
            f"    diversity_threshold={nation.diversity_similarity_threshold:.3f} | fraction_mutation={land.fraction_mutation:.3f}"
        )
        planet.advance_time()
        db_path = Path(CONFIG['results_dir']) / str(CONFIG['planet_name']) / 'planetary_database.sqlite'
        summarize_generation(db_path, nation.generation - 1, Path(CONFIG['results_dir']) / 'summary.csv', CONFIG)

    # 简要结果输出
    db_path = Path(CONFIG['results_dir']) / str(CONFIG['planet_name']) / 'planetary_database.sqlite'
    print('\n' + '=' * 70)
    if db_path.exists():
        print('进化完成。数据库: ', db_path)
    else:
        print('进化结束，但未找到数据库（可能没有有效个体存活）。')


if __name__ == '__main__':
    main()
