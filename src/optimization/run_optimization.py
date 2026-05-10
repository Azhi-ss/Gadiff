#!/usr/bin/env python3
"""
正式项目：使用 PolyGA + (可选)MMPolymer，对 BRICS 片段 DNA 进行聚合物进化优化。

- DNA 来源：/root/code/BRICS_DNA/fragments_output1_clean.csv -> 转换为本目录 dna_polymers.csv
- 生成函数：polyga.utils.chromosome_ids_to_smiles
- 预测器：优先使用 MMPolymer；若权重缺失则启用回退预测器（启发式），保证可运行
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

# 将 polyga 源码加入路径
sys.path.insert(0, '/root/code/polyga')

from polyga.polygod import PolyPlanet, PolyLand, PolyNation  # type: ignore
from polyga.selection_schemes import elite  # type: ignore
from polyga import utils  # type: ignore

# 可选 MMPolymer 预测器
# 延迟导入 MMPolymer 预测器（在 maybe_create_mmpolymer_predictor 内部）


CONFIG: Dict[str, object] = {
    # 路径
    'dna_file': 'dna_polymers.csv',
    'results_dir': 'results_100gen_run',
    'planet_name': 'BRICSPolyPlanet',

    # MMPolymer 设置
    'weight_dir': '/internfs/Zy/dataset/finetune_data',
    'properties': ['Tg', 'DC'],
    'use_gpu': True,
    'batch_size': 128,
    'num_cpus': 1,

    # 目标权重（示例：最大化 Tg，最小化 DC）
    'weight_tg': 0.5,
    'weight_dc': 0.5,

    # GA 参数（正式运行）
    'random_seed': 43,
    'num_generations': 100,
    'population_size': 120,
    'num_chromosomes_initial': 4,
    'num_families': 20,
    'num_parents_per_family': 3,
    'num_children_per_family': 6,

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


def create_fitness_function(weight_tg: float = 1.0, weight_dc: float = 0.5) -> Callable[[pd.DataFrame, List[str]], pd.DataFrame]:
    """构建线性加权+归一化的适应度函数，并对 DC 超阈值施加惩罚。"""
    # 清空缓存，确保每次运行使用新的基准
    _NORMALIZATION_CACHE.clear()

    norm_scheme: str = str(CONFIG.get('fitness_normalization', 'quantile'))
    tg_q_low: float = float(CONFIG.get('tg_q_low', 0.50))
    tg_q_high: float = float(CONFIG.get('tg_q_high', 0.90))
    dc_q_low: float = float(CONFIG.get('dc_q_low', 0.10))
    dc_q_high: float = float(CONFIG.get('dc_q_high', 0.50))
    dc_penalty_mode: str = str(CONFIG.get('dc_penalty_mode', 'quantile'))
    dc_max_quantile: float = float(CONFIG.get('dc_max_quantile', 0.90))
    dc_max_value: float = float(CONFIG.get('dc_max_value', 9999.0))
    dc_penalty_factor: float = float(CONFIG.get('dc_penalty_factor', 0.10))

    def _clip01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def _safe_div(num: float, den: float) -> float:
        if den == 0 or np.isclose(den, 0.0):
            return 0.0
        return num / den

    def _normalize(df_in: pd.DataFrame, generation: int | None = None) -> Tuple[pd.Series, pd.Series, float]:
        tg_series = pd.to_numeric(df_in.get('Tg'), errors='coerce')
        dc_series = pd.to_numeric(df_in.get('DC'), errors='coerce')

        # 只要缓存已填充，就使用缓存（锁定基准），不再依赖 generation 字段
        if all(
            key in _NORMALIZATION_CACHE
            for key in ('tg_L', 'tg_U', 'dc_L', 'dc_U', 'dc_max_th')
        ):
            tg_L = _NORMALIZATION_CACHE['tg_L']
            tg_U = _NORMALIZATION_CACHE['tg_U']
            dc_L = _NORMALIZATION_CACHE['dc_L']
            dc_U = _NORMALIZATION_CACHE['dc_U']
            dc_max_th = _NORMALIZATION_CACHE['dc_max_th']
        else:
            if norm_scheme == 'quantile':
                tg_L = tg_series.quantile(tg_q_low)
                tg_U = tg_series.quantile(tg_q_high)
                dc_L = dc_series.quantile(dc_q_low)
                dc_U = dc_series.quantile(dc_q_high)
            else:  # 'minmax'
                tg_L, tg_U = float(tg_series.min()), float(tg_series.max())
                dc_L, dc_U = float(dc_series.min()), float(dc_series.max())

            if not np.isfinite(tg_L) or not np.isfinite(tg_U) or tg_U <= tg_L:
                tg_L = float(pd.Series(tg_series.dropna()).quantile(0.25)) if tg_series.notna().any() else 0.0
                tg_U = float(pd.Series(tg_series.dropna()).quantile(0.75)) if tg_series.notna().any() else 1.0
                if tg_U <= tg_L:
                    tg_L, tg_U = 0.0, 1.0
            if not np.isfinite(dc_L) or not np.isfinite(dc_U) or dc_U <= dc_L:
                dc_L = float(pd.Series(dc_series.dropna()).quantile(0.25)) if dc_series.notna().any() else 0.0
                dc_U = float(pd.Series(dc_series.dropna()).quantile(0.75)) if dc_series.notna().any() else 1.0
                if dc_U <= dc_L:
                    dc_L, dc_U = 0.0, 1.0

            if dc_penalty_mode == 'quantile':
                dc_max_th = float(dc_series.quantile(dc_max_quantile)) if dc_series.notna().any() else float('inf')
            else:
                dc_max_th = dc_max_value
            if not np.isfinite(dc_max_th):
                dc_max_th = float('inf')

            _NORMALIZATION_CACHE['tg_L'] = float(tg_L)
            _NORMALIZATION_CACHE['tg_U'] = float(tg_U)
            _NORMALIZATION_CACHE['dc_L'] = float(dc_L)
            _NORMALIZATION_CACHE['dc_U'] = float(dc_U)
            _NORMALIZATION_CACHE['dc_max_th'] = float(dc_max_th)

        dc_max_th = _NORMALIZATION_CACHE['dc_max_th']

        # 归一化：Tg 越大越好，DC 越小越好
        tg_norm = (tg_series - tg_L).apply(lambda v: _clip01(_safe_div(float(v), 1.0)) if False else _clip01(_safe_div(float(v - tg_L), float(tg_U - tg_L))))
        dc_norm = (dc_U - dc_series).apply(lambda v: _clip01(_safe_div(float(v), float(dc_U - dc_L))))

        return tg_norm, dc_norm, dc_max_th

    def fitness_function(df: pd.DataFrame, fp_headers: List[str]) -> pd.DataFrame:
        # 计算当前群体的归一化与阈值
        generation = int(df['generation'].iloc[0]) if 'generation' in df.columns and not df['generation'].isna().all() else 0
        tg_norm, dc_norm, dc_max_th = _normalize(df, generation)

        scores: List[float] = []
        for idx, row in df.iterrows():
            tg = row.get('Tg', np.nan)
            dc = row.get('DC', np.nan)
            if pd.isna(tg) or pd.isna(dc):
                scores.append(-9.99e8)
                continue
            tgn = float(tg_norm.loc[idx]) if np.isfinite(tg_norm.loc[idx]) else 0.0
            dcn = float(dc_norm.loc[idx]) if np.isfinite(dc_norm.loc[idx]) else 0.0
            fit = float(weight_tg) * tgn + float(weight_dc) * dcn
            # DC 超阈值惩罚（乘法缩放）
            try:
                if float(dc) > dc_max_th:
                    fit *= dc_penalty_factor
            except Exception:
                pass
            scores.append(fit)

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
        })
    df = pd.DataFrame(recs)
    df = df[df['Tg'].notna() & df['DC'].notna()].copy()
    if len(df) == 0:
        con.close()
        return
    weight_tg = float(cfg.get('weight_tg', 1.0))
    weight_dc = float(cfg.get('weight_dc', 0.5))
    norm_scheme = str(cfg.get('fitness_normalization', 'quantile'))
    tg_q_low = float(cfg.get('tg_q_low', 0.50))
    tg_q_high = float(cfg.get('tg_q_high', 0.90))
    dc_q_low = float(cfg.get('dc_q_low', 0.10))
    dc_q_high = float(cfg.get('dc_q_high', 0.50))
    dc_penalty_mode = str(cfg.get('dc_penalty_mode', 'quantile'))
    dc_max_quantile = float(cfg.get('dc_max_quantile', 0.90))
    dc_max_value = float(cfg.get('dc_max_value', 9999.0))
    dc_penalty_factor = float(cfg.get('dc_penalty_factor', 0.10))
    tg = pd.to_numeric(df['Tg'], errors='coerce')
    dc = pd.to_numeric(df['DC'], errors='coerce')
    radius = int(cfg.get('fingerprint_radius', 2))
    nbits = int(cfg.get('fingerprint_nbits', 1024))
    use_chirality = bool(cfg.get('fingerprint_use_chirality', True))
    if gen == 0 or not all(
        key in _NORMALIZATION_CACHE for key in ('tg_L', 'tg_U', 'dc_L', 'dc_U', 'dc_max_th')
    ):
        # 仅在 Gen0 计算基准，后续代数强制使用缓存
        if gen != 0:
            raise ValueError(f"Normalization cache missing for generation {gen}")
        if norm_scheme == 'quantile':
            tg_L = float(tg.dropna().quantile(tg_q_low)) if tg.notna().any() else 0.0
            tg_U = float(tg.dropna().quantile(tg_q_high)) if tg.notna().any() else 1.0
            dc_L = float(dc.dropna().quantile(dc_q_low)) if dc.notna().any() else 0.0
            dc_U = float(dc.dropna().quantile(dc_q_high)) if dc.notna().any() else 1.0
        else:
            tg_L = float(tg.min()) if tg.notna().any() else 0.0
            tg_U = float(tg.max()) if tg.notna().any() else 1.0
            dc_L = float(dc.min()) if dc.notna().any() else 0.0
            dc_U = float(dc.max()) if dc.notna().any() else 1.0
        if not np.isfinite(tg_L) or not np.isfinite(tg_U) or tg_U <= tg_L:
            tg_L, tg_U = 0.0, 1.0
        if not np.isfinite(dc_L) or not np.isfinite(dc_U) or dc_U <= dc_L:
            dc_L, dc_U = 0.0, 1.0
        if dc_penalty_mode == 'quantile':
            dc_p90 = float(dc.dropna().quantile(dc_max_quantile)) if dc.notna().any() else float('inf')
        else:
            dc_p90 = dc_max_value
        if not np.isfinite(dc_p90):
            dc_p90 = float('inf')
        _NORMALIZATION_CACHE['tg_L'] = tg_L
        _NORMALIZATION_CACHE['tg_U'] = tg_U
        _NORMALIZATION_CACHE['dc_L'] = dc_L
        _NORMALIZATION_CACHE['dc_U'] = dc_U
        _NORMALIZATION_CACHE['dc_max_th'] = dc_p90
    else:
        tg_L = _NORMALIZATION_CACHE['tg_L']
        tg_U = _NORMALIZATION_CACHE['tg_U']
        dc_L = _NORMALIZATION_CACHE['dc_L']
        dc_U = _NORMALIZATION_CACHE['dc_U']
        dc_p90 = _NORMALIZATION_CACHE['dc_max_th']
    tg_den = tg_U - tg_L
    dc_den = dc_U - dc_L
    tg_norm = ((tg - tg_L) / tg_den if tg_den != 0 else pd.Series(0.0, index=tg.index)).clip(0.0, 1.0)
    dc_norm = ((dc_U - dc) / dc_den if dc_den != 0 else pd.Series(0.0, index=dc.index)).clip(0.0, 1.0)
    fitness_base = weight_tg * tg_norm + weight_dc * dc_norm
    penalized = dc > dc_p90
    fitness = fitness_base.copy()
    fitness.loc[penalized] = fitness.loc[penalized] * dc_penalty_factor

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
            return {'mean': float('nan'), 'median': float('nan'), 'p75': float('nan'), 'max': float('nan')}
        return {
            'mean': float(s.mean()),
            'median': float(s.quantile(0.5)),
            'p75': float(s.quantile(0.75)),
            'max': float(s.max()),
        }
    fs = stat(fitness)
    ts = stat(tg)
    ds = stat(dc)
    best_idx = int(fitness.idxmax())
    best_row = {
        'best_fitness': float(fitness.loc[best_idx]) if pd.notna(fitness.loc[best_idx]) else np.nan,
        'best_tg': float(tg.loc[best_idx]) if pd.notna(tg.loc[best_idx]) else np.nan,
        'best_dc': float(dc.loc[best_idx]) if pd.notna(dc.loc[best_idx]) else np.nan,
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
        'dc_max': ds['max'],
        'dc_p90_threshold': None if not np.isfinite(dc_p90) else float(dc_p90),
        'penalized_ratio': float(penalized.mean()) if len(penalized) > 0 else 0.0,
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
    """优先构建 MMPolymer 预测器；若权重缺失或不可用则回退到启发式预测器。"""
    weight_dir = Path(str(cfg['weight_dir']))
    properties: List[str] = list(cfg['properties'])  # type: ignore

    # 核心权重路径检查
    missing: List[str] = []
    for prop in properties:
        expect = weight_dir / prop / 'ckpt' / prop / 'checkpoint_best.pt'
        if not expect.exists():
            missing.append(str(expect))

    if missing:
        print('未找到 MMPolymer 权重，启用回退预测器：')
        for m in missing:
            print('  -', m)
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


def ensure_dna_exists(project_dir: Path, source_fragments_csv: Path) -> Path:
    """若 dna_polymers.csv 不存在，则调用 create_dna.py 生成。"""
    dna_path = project_dir / 'dna_polymers.csv'
    if dna_path.exists():
        return dna_path
    # 调用同目录下的 create_dna.py
    import subprocess
    cmd = [sys.executable, str(project_dir / 'create_dna.py')]
    print('生成 DNA:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    assert dna_path.exists(), 'dna_polymers.csv 未生成'
    return dna_path


def main() -> None:
    print('=' * 70)
    print('PolyGA Optimization (BRICS DNA)')
    print('=' * 70)

    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    os.makedirs(str(CONFIG['results_dir']), exist_ok=True)

    # 确保 DNA 存在（从 BRICS 片段转换）
    source_fragments = Path('/root/code/BRICS_DNA/brics_fragments.csv')
    dna_file = ensure_dna_exists(script_dir, source_fragments)

    # 预测器与适应度
    predict_fn = maybe_create_mmpolymer_predictor(CONFIG)
    fitness_fn = create_fitness_function(
        weight_tg=float(CONFIG['weight_tg']),
        weight_dc=float(CONFIG['weight_dc']),
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
        generative_function=utils.chromosome_ids_to_smiles,
        fitness_function=fitness_fn,
        crossover_position=str(CONFIG['crossover_position']),
        fraction_mutation=float(CONFIG.get('fraction_mutation_start', CONFIG.get('fraction_mutation', 0.25))),
        fraction_mutate_additional_block=float(CONFIG['fraction_mutate_additional_block']),
        crossover_sigma_offset=float(CONFIG['crossover_sigma_offset']),
        mutation_sigma_offset=float(CONFIG['mutation_sigma_offset']),
    )

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

    print(f"Planet: {CONFIG['planet_name']} | DNA: {dna_file} | Chromosomes: {len(planet.chromosomes)}")

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
        try:
            _summarize_generation_to_csv(db_path, nation.generation - 1, Path(CONFIG['results_dir']) / 'summary.csv', CONFIG)
        except Exception:
            pass

    # 简要结果输出
    db_path = Path(CONFIG['results_dir']) / str(CONFIG['planet_name']) / 'planetary_database.sqlite'
    print('\n' + '=' * 70)
    if db_path.exists():
        print('进化完成。数据库: ', db_path)
    else:
        print('进化结束，但未找到数据库（可能没有有效个体存活）。')


if __name__ == '__main__':
    main()
