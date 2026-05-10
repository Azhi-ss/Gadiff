#!/usr/bin/env python3
"""
可视化化学多样性收敛：
- 折线图：塔尼莫均值/方差随代变化
- 热力图：每代相似度矩阵
- MDS 2D 投影：化学空间分布（按代着色）
"""
from __future__ import annotations

import argparse
import sqlite3
import pathlib
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def load_tanimoto_matrices(div_dir: pathlib.Path, generations: List[int]) -> dict[int, pd.DataFrame]:
    """加载各代塔尼莫矩阵"""
    mats = {}
    for gen in generations:
        path = div_dir / f'tanimoto_gen_{gen:03d}.csv'
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            mats[gen] = df
    return mats


def load_smiles_from_db(db_path: pathlib.Path, gen: int) -> List[str]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    rows = cur.execute(
        'SELECT smiles_string FROM polymer WHERE generation = ? ORDER BY planetary_id',
        (gen,),
    ).fetchall()
    con.close()
    return [r['smiles_string'] for r in rows if r['smiles_string']]


def plot_tanimoto_trend(summary_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    """绘制塔尼莫统计趋势折线图，采用现代科学可视化风格"""
    df = pd.read_csv(summary_path)
    
    # 设置seaborn样式
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.3)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 使用更现代的配色 - 深紫色风格 (参考用户提供的图片)
    mean_color = '#440154'  # Dark Purple (Viridis style)
    fill_color = '#9fa8da'  # Desaturated light purple/blue
    
    # 绘制均值线
    ax.plot(df['generation'], df['tanimoto_mean'], 
            marker='o', markersize=7, linewidth=2.5, 
            color=mean_color, label='Mean', zorder=3)
    
    # 填充标准差区域
    ax.fill_between(
        df['generation'],
        df['tanimoto_mean'] - df['tanimoto_std'],
        df['tanimoto_mean'] + df['tanimoto_std'],
        alpha=0.3,
        color=fill_color,
        label='±1 Std Dev',
        zorder=2
    )
    
    # 设置轴标签和标题
    ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tanimoto Similarity', fontsize=12, fontweight='bold')
    # ax.set_title('Chemical Diversity Convergence', fontsize=14, fontweight='bold', pad=15)
    
    # 优化图例
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # 优化网格
    ax.grid(False)
    ax.set_axisbelow(True)
    
    # 设置背景色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 添加边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1.2)
    
    out_path = out_dir / 'tanimoto_trend.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'趋势图已保存: {out_path}')
    plt.close(fig)


def plot_heatmaps(mats: dict[int, pd.DataFrame], out_dir: pathlib.Path,
                  use_mol_index: bool = False) -> None:
    """绘制每代相似度矩阵热力图，采用现代科学可视化风格"""
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.2)

    for gen, df in mats.items():
        fig, ax = plt.subplots(figsize=(7, 6))

        n = len(df)
        if use_mol_index:
            # 用简洁编号替换长 SMILES 标签
            mol_labels = [f'Mol-{i+1}' for i in range(n)]
            plot_df = df.copy()
            plot_df.index = mol_labels
            plot_df.columns = mol_labels
        else:
            plot_df = df

        sns.heatmap(
            plot_df.astype(float),
            cmap='RdYlBu_r',
            vmin=0.0,
            vmax=0.6,
            ax=ax,
            cbar_kws={
                'label': 'Tanimoto Similarity',
                'shrink': 0.8,
                'aspect': 20
            },
            square=True,
            linewidths=0.0,
            rasterized=True
        )

        # 根据分子数决定 tick 显示密度
        if use_mol_index:
            step = max(1, n // 10)  # 最多显示 ~10 个刻度
            tick_indices = list(range(0, n, step))
            tick_labels_show = [mol_labels[i] for i in tick_indices]
            ax.set_xticks([i + 0.5 for i in tick_indices])
            ax.set_xticklabels(tick_labels_show, rotation=45, ha='right', fontsize=8)
            ax.set_yticks([i + 0.5 for i in tick_indices])
            ax.set_yticklabels(tick_labels_show, rotation=0, fontsize=8)

        ax.set_title(f'Similarity Matrix (Generation {gen})',
                     fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Molecule Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Molecule Index', fontsize=11, fontweight='bold')

        fig.patch.set_facecolor('white')

        out_path = out_dir / f'heatmap_gen_{gen:02d}.png'
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'热力图已保存: {out_path}')
        plt.close(fig)


def plot_si_heatmaps(div_dir: pathlib.Path, out_dir: pathlib.Path,
                     all_generations: List[int],
                     si_gens: List[int] | None = None) -> None:
    """生成 SI 用代表性热力图（精简 tick，统一色条，出版质量）"""
    # 默认选 6 个代表性时间点
    if si_gens is None:
        max_gen = max(all_generations)
        candidates = [0, max_gen // 5, 2 * max_gen // 5,
                      3 * max_gen // 5, 4 * max_gen // 5, max_gen]
        # 对齐到实际存在的代数
        si_gens = [min(all_generations, key=lambda g: abs(g - c)) for c in candidates]
        si_gens = sorted(set(si_gens))  # 去重排序

    print(f'SI 热力图选定代数: {si_gens}')
    mats = load_tanimoto_matrices(div_dir, si_gens)
    if not mats:
        print('警告: 未找到 SI 热力图数据')
        return

    sns.set_style('white')
    sns.set_context('paper', font_scale=1.3)

    si_out = out_dir / 'SI_heatmaps'
    si_out.mkdir(parents=True, exist_ok=True)

    for gen, df in mats.items():
        n = len(df)
        mol_labels = [f'Mol-{i+1}' for i in range(n)]
        plot_df = df.copy()
        plot_df.index = mol_labels
        plot_df.columns = mol_labels

        fig, ax = plt.subplots(figsize=(6, 5.5))

        hm = sns.heatmap(
            plot_df.astype(float),
            cmap='RdYlBu_r',
            vmin=0.0,
            vmax=0.6,
            ax=ax,
            cbar_kws={
                'label': 'Tanimoto Similarity',
                'shrink': 0.75,
                'aspect': 18,
                'ticks': [0.0, 0.2, 0.4, 0.6]
            },
            square=True,
            linewidths=0.0,
            rasterized=True
        )

        # 最多显示 8 个均匀分布的刻度
        n_ticks = min(8, n)
        tick_pos = np.linspace(0, n - 1, n_ticks, dtype=int)
        tick_labels_show = [mol_labels[i] for i in tick_pos]

        ax.set_xticks([i + 0.5 for i in tick_pos])
        ax.set_xticklabels(tick_labels_show, rotation=45, ha='right', fontsize=9)
        ax.set_yticks([i + 0.5 for i in tick_pos])
        ax.set_yticklabels(tick_labels_show, rotation=0, fontsize=9)

        ax.set_title(f'Generation {gen}', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Molecule Index', fontsize=11)
        ax.set_ylabel('Molecule Index', fontsize=11)

        fig.patch.set_facecolor('white')
        fig.tight_layout()

        out_path = si_out / f'SI_heatmap_gen_{gen:02d}.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'SI 热力图已保存: {out_path}')
        plt.close(fig)

    print(f'SI 热力图共 {len(mats)} 张，保存于: {si_out}')


def load_smiles_from_matrix(div_dir: pathlib.Path, gen: int) -> List[str]:
    """从矩阵文件名提取 SMILES（假设标签格式为 idx_{smiles}）"""
    path = div_dir / f'tanimoto_gen_{gen:03d}.csv'
    df = pd.read_csv(path, index_col=0)
    labels = df.index.astype(str)
    smiles_list = [lbl.split('_', 1)[1] if '_' in lbl else '' for lbl in labels]
    return smiles_list


def compute_fingerprints(smiles_list: List[str], radius: int = 3, nbits: int = 2048) -> np.ndarray:
    """批量计算 Morgan 指纹矩阵"""
    fps = []
    for smi in smiles_list:
        # 按 MMPolymer 方式处理星号（dummy 原子）以获得更稳定的结构表示
        def _process_polymer_smiles_like_mmpolymer(s: str) -> str:
            mol0 = Chem.MolFromSmiles(s)
            if mol0 is None:
                return s
            star_ids = [a.GetIdx() for a in mol0.GetAtoms() if a.GetAtomicNum() == 0]
            if len(star_ids) != 2:
                return s
            try:
                n1 = mol0.GetAtomWithIdx(star_ids[0]).GetNeighbors()
                n2 = mol0.GetAtomWithIdx(star_ids[1]).GetNeighbors()
                if len(n1) != 1 or len(n2) != 1:
                    return s
                star_pair_list = [star_ids[0], n1[0].GetIdx(), star_ids[1], n2[0].GetIdx()]
                mol0.GetAtomWithIdx(star_pair_list[0]).SetAtomicNum(
                    mol0.GetAtomWithIdx(star_pair_list[3]).GetAtomicNum()
                )
                mol0.GetAtomWithIdx(star_pair_list[2]).SetAtomicNum(
                    mol0.GetAtomWithIdx(star_pair_list[1]).GetAtomicNum()
                )
                return Chem.MolToSmiles(mol0)
            except Exception:
                return s

        smi_proc = _process_polymer_smiles_like_mmpolymer(smi)
        mol = Chem.MolFromSmiles(smi_proc)
        if mol is None:
            fps.append(np.zeros(nbits, dtype=np.float32))
            continue
        bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useChirality=True)
        arr = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bit_vect, arr)
        fps.append(arr)
    return np.vstack(fps)


def plot_pca_trajectory(div_dir: pathlib.Path, generations: List[int], out_dir: pathlib.Path, highlight_last: bool = True, db_path: pathlib.Path | None = None, method: str = 'pca', perplexity: int | None = None, tune_tsne: bool = False) -> None:
    """绘制降维可视化轨迹 (PCA 或 t-SNE)"""
    all_fp: List[np.ndarray] = []
    sizes: List[int] = []
    gens_has_data: List[int] = []
    for gen in generations:
        smiles_list = []
        if db_path is not None and db_path.exists():
            smiles_list = load_smiles_from_db(db_path, gen)
        if not smiles_list:
            p = div_dir / f'tanimoto_gen_{gen:03d}.csv'
            if p.exists():
                smiles_list = load_smiles_from_matrix(div_dir, gen)
        if not smiles_list:
            sizes.append(0)
            continue
        fp_matrix = compute_fingerprints(smiles_list)
        if fp_matrix.size == 0:
            sizes.append(0)
            continue
        all_fp.append(fp_matrix)
        sizes.append(fp_matrix.shape[0])
        gens_has_data.append(gen)

    if not all_fp:
        print(f'无有效数据用于 {method.upper()} 投影')
        return

    X = np.vstack(all_fp)
    n_samples = X.shape[0]

    # 准备要运行的参数列表
    run_configs = []
    if method == 'tsne' and tune_tsne:
        candidates = [5, 30, 50, 100]
        for p in candidates:
            if p < n_samples:
                run_configs.append({'perplexity': p})
        if not run_configs:
             run_configs.append({'perplexity': max(1, min(30, n_samples - 1))})
    elif method == 'tsne':
        if perplexity is not None:
            perp = perplexity
        else:
            perp = min(30, n_samples - 1) if n_samples > 1 else 1
        run_configs.append({'perplexity': perp})
    else:
        # PCA 不需要 perplexity
        run_configs.append({})

    # 循环运行配置
    for config in run_configs:
        current_perp = config.get('perplexity')
        
        if method == 'tsne':
            print(f"Running t-SNE with perplexity={current_perp}, n_iter=3000...")
            reducer = TSNE(n_components=2, random_state=0, perplexity=current_perp, 
                           init='pca', learning_rate='auto', n_iter=3000)
            coords_all = reducer.fit_transform(X)
            explained_var = None 
        else:
            reducer = PCA(n_components=2, random_state=0)
            coords_all = reducer.fit_transform(X)
            explained_var = reducer.explained_variance_ratio_

        # 按代切片坐标
        per_gen_coords: List[np.ndarray] = []
        idx = 0
        for s in sizes:
            if s <= 0:
                continue
            per_gen_coords.append(coords_all[idx: idx + s])
            idx += s
            
        # --- 绘图部分开始 ---
        sns.set_style('whitegrid')
        sns.set_context('paper', font_scale=1.8)
        
        # 配色方案
        if highlight_last and gens_has_data:
            n_early = len(gens_has_data) - 1
            if n_early > 0:
                base_palette = sns.color_palette('viridis', n_colors=n_early)
                early_colors = base_palette
            else:
                early_colors = []
            highlight_color = '#e74c3c' 
            color_list = list(early_colors) + [highlight_color]
            size_list = [40] * n_early + [100] 
            alpha_list = [0.4] * n_early + [0.9] 
            zorder_list = list(range(10, 10 + n_early)) + [100] 
        else:
            color_list = sns.color_palette('viridis', n_colors=len(gens_has_data))
            size_list = [60] * len(gens_has_data)
            alpha_list = [0.6] * len(gens_has_data)
            zorder_list = [10 + i for i in range(len(gens_has_data))]

        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, (coords, gen) in enumerate(zip(per_gen_coords, gens_has_data)):
            c = color_list[i]
            alpha = alpha_list[i]
            s = size_list[i]
            z = zorder_list[i]
            is_last = (highlight_last and i == len(gens_has_data) - 1)
            
            ax.scatter(
                coords[:, 0], coords[:, 1], 
                c=[c] * len(coords), alpha=alpha, s=s,
                edgecolor='white' if not is_last else 'black', 
                linewidth=0.5 if not is_last else 1.5,
                label=f'Gen {gen}' if (i == 0 or i == len(gens_has_data)-1 or i % 10 == 0) else None,
                zorder=z
            )
        
        # centroids = np.array([c.mean(axis=0) for c in per_gen_coords])
        # ax.plot(centroids[:, 0], centroids[:, 1], 
        #         color='#34495e', linestyle='--', linewidth=1.5, alpha=0.6, 
        #         label='Evolutionary Trajectory (Centroid)', zorder=200)
        
        # ax.text(centroids[0, 0], centroids[0, 1], 'Start', fontsize=14, fontweight='bold')
        # ax.text(centroids[-1, 0], centroids[-1, 1], 'End', fontsize=14, fontweight='bold', color='#e74c3c')
        
        if method == 'tsne':
            ax.set_xlabel('t-SNE Dimension 1', fontsize=18, fontweight='bold')
            ax.set_ylabel('t-SNE Dimension 2', fontsize=18, fontweight='bold')
            # ax.set_title(f'Chemical Space Trajectory (t-SNE, perp={current_perp})', fontsize=14, fontweight='bold', pad=15)
            filename = f'tsne_trajectory_perp{current_perp}.png'
        else:
            ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=18, fontweight='bold')
            ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=18, fontweight='bold')
            # ax.set_title('Chemical Space Trajectory (PCA)', fontsize=14, fontweight='bold', pad=15)
            filename = 'pca_trajectory.png'
        
        # 图例逻辑
        if highlight_last and gens_has_data and len(gens_has_data) > 5:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_list[0], markersize=9, 
                          markeredgecolor='white', markeredgewidth=0.5, label=f'Gen {gens_has_data[0]}'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_list[len(color_list)//2], markersize=9,
                          markeredgecolor='white', markeredgewidth=0.5, label=f'Gen {gens_has_data[len(gens_has_data)//2]}'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_list[-1], markersize=9,
                          markeredgecolor='white', markeredgewidth=0.5, label=f'Gen {gens_has_data[-1]} (Latest)'),
            ]
            ax.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True, shadow=True, fontsize=14)
        else:
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=13, ncol=2 if len(gens_has_data) > 6 else 1)
        
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(1.2)
        
        fig.tight_layout()
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'{method.upper()} 轨迹图已保存: {out_path}')
        
        if method == 'pca':
            print(f'  - PC1 解释方差: {explained_var[0]*100:.2f}%')
            print(f'  - PC2 解释方差: {explained_var[1]*100:.2f}%')
            print(f'  - 累计解释方差: {sum(explained_var)*100:.2f}%')
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='化学多样性收敛可视化')
    parser.add_argument(
        '--div-dir',
        type=pathlib.Path,
        default=pathlib.Path('results_formal_run/diversity'),
        help='diversity 目录路径',
    )
    parser.add_argument(
        '--summary',
        type=pathlib.Path,
        default=pathlib.Path('results_formal_run/summary.csv'),
        help='tanimoto 汇总 CSV 路径',
    )
    parser.add_argument(
        '--db',
        type=pathlib.Path,
        default=pathlib.Path('results_formal_run/BRICSPolyPlanet/planetary_database.sqlite'),
        help='SQLite 数据库路径（PCA-only 可直接从 DB 读取 SMILES）',
    )
    parser.add_argument(
        '--out-dir',
        type=pathlib.Path,
        default=pathlib.Path('results_formal_run/figures'),
        help='图片输出目录',
    )
    parser.add_argument(
        '--generations',
        type=int,
        nargs='+',
        default=None,
        help='要可视化的代数列表 (默认: 自动检测所有代)',
    )
    parser.add_argument(
        '--pca-only',
        action='store_true',
        default=False,
        help='仅生成降维轨迹图，跳过 tanimoto 趋势与热力图',
    )
    parser.add_argument(
        '--si-heatmaps',
        action='store_true',
        default=False,
        help='生成 SI 用代表性热力图（使用 Mol-N 编号替代 SMILES，输出至 figures/SI_heatmaps/）',
    )
    parser.add_argument(
        '--si-gens',
        type=int,
        nargs='+',
        default=None,
        help='SI 热力图指定代数列表 (默认: 自动选 6 个代表性时间点)',
    )
    parser.add_argument(
        '--method',
        type=str,
        default='pca',
        choices=['pca', 'tsne'],
        help='降维方法: pca 或 tsne (默认: pca)',
    )
    parser.add_argument(
        '--perplexity',
        type=int,
        default=None,
        help='t-SNE 的 perplexity 参数 (默认: 根据样本数自动设置)',
    )
    parser.add_argument(
        '--tune-tsne',
        action='store_true',
        default=False,
        help='自动尝试多个 perplexity 值进行 t-SNE 调优',
    )
    args = parser.parse_args()

    div_dir = args.div_dir.resolve()
    summary_path = args.summary.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 自动检测代数
    generations = args.generations
    if generations is None:
        if summary_path.exists():
            try:
                df_sum = pd.read_csv(summary_path)
                generations = sorted(df_sum['generation'].unique().tolist())
                print(f'自动检测到 {len(generations)} 代: {generations[0]} -> {generations[-1]}')
            except Exception as e:
                print(f'无法从 summary 自动检测代数: {e}')
                generations = list(range(10))
        else:
            print('Summary文件不存在，默认使用 0-9 代')
            generations = list(range(10))

    # 采样代数以避免绘图过于拥挤 (例如只取 10 个点，加上最后一代)
    if len(generations) > 15:
        step = len(generations) // 10
        sampled_gens = generations[::step]
        if generations[-1] not in sampled_gens:
            sampled_gens.append(generations[-1])
        print(f'代数过多 ({len(generations)})，采样显示: {sampled_gens}')
        generations = sampled_gens

    if not args.pca_only:
        if summary_path.exists():
            plot_tanimoto_trend(summary_path, out_dir)
        # 热力图也只画采样后的
        mats = load_tanimoto_matrices(div_dir, generations)
        if mats:
            plot_heatmaps(mats, out_dir)
        else:
            print('警告: 未找到任何 tanimoto 矩阵文件，跳过热力图')

    if args.si_heatmaps:
        # 重新获取完整代数列表用于 SI 选点
        full_gens = args.generations
        if full_gens is None and summary_path.exists():
            try:
                df_sum = pd.read_csv(summary_path)
                full_gens = sorted(df_sum['generation'].unique().tolist())
            except Exception:
                full_gens = generations
        plot_si_heatmaps(div_dir, out_dir, full_gens or generations, si_gens=args.si_gens)

    plot_pca_trajectory(div_dir, generations, out_dir, highlight_last=True, db_path=args.db, 
                        method=args.method, perplexity=args.perplexity, tune_tsne=args.tune_tsne)

    print('全部可视化图已生成，可在', out_dir, '查看')


if __name__ == '__main__':
    main()
