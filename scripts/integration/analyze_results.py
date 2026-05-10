#!/usr/bin/env python3
"""
分析遗传算法结果并生成可视化报告
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_results(results_csv='/root/code/MMga/results/MMPolymer_Planet_all_polymers.csv',
                    output_dir='/root/code/MMga/results/analysis'):
    """
    分析遗传算法结果
    
    Args:
        results_csv: 结果CSV文件路径
        output_dir: 输出目录
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print("📖 读取结果数据...")
    df = pd.read_csv(results_csv)
    
    print(f"\n📊 数据概览:")
    print(f"  总聚合物数: {len(df)}")
    print(f"  总代数: {df['generation'].max() + 1}")
    print(f"  列数: {len(df.columns)}")
    
    # 基本统计
    print(f"\n📈 性质统计:")
    if 'Tg' in df.columns:
        print(f"  Tg: {df['Tg'].min():.2f} - {df['Tg'].max():.2f} K")
        print(f"  Tg 平均: {df['Tg'].mean():.2f} K")
        print(f"  Tg 标准差: {df['Tg'].std():.2f} K")
    
    if 'DC' in df.columns:
        print(f"  DC: {df['DC'].min():.4f} - {df['DC'].max():.4f}")
        print(f"  DC 平均: {df['DC'].mean():.4f}")
        print(f"  DC 标准差: {df['DC'].std():.4f}")
    
    if 'fitness' in df.columns:
        print(f"  适应度: {df['fitness'].min():.2f} - {df['fitness'].max():.2f}")
        print(f"  适应度平均: {df['fitness'].mean():.2f}")
    
    # 每代统计
    print(f"\n📅 每代统计:")
    gen_stats = df.groupby('generation').agg({
        'fitness': ['mean', 'max', 'std', 'count']
    }).round(4)
    print(gen_stats)
    
    # ========================================================================
    # 可视化
    # ========================================================================
    print(f"\n🎨 生成可视化图表...")
    
    # 图1: 适应度进化曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PolyGA + MMPolymer Evolution Analysis', fontsize=16, fontweight='bold')
    
    # 1.1 适应度进化
    ax = axes[0, 0]
    gen_groups = df.groupby('generation')
    generations = sorted(df['generation'].unique())
    
    if 'fitness' in df.columns:
        mean_fitness = [gen_groups.get_group(g)['fitness'].mean() for g in generations]
        max_fitness = [gen_groups.get_group(g)['fitness'].max() for g in generations]
        min_fitness = [gen_groups.get_group(g)['fitness'].min() for g in generations]
        
        ax.plot(generations, max_fitness, 'ro-', label='Max Fitness', linewidth=2)
        ax.plot(generations, mean_fitness, 'bs-', label='Mean Fitness', linewidth=2)
        ax.fill_between(generations, min_fitness, max_fitness, alpha=0.2)
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title('Fitness Evolution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 1.2 Tg 进化
    ax = axes[0, 1]
    if 'Tg' in df.columns:
        mean_tg = [gen_groups.get_group(g)['Tg'].mean() for g in generations]
        max_tg = [gen_groups.get_group(g)['Tg'].max() for g in generations]
        
        ax.plot(generations, max_tg, 'r^-', label='Max Tg', linewidth=2)
        ax.plot(generations, mean_tg, 'bv-', label='Mean Tg', linewidth=2)
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Tg (K)', fontsize=12)
        ax.set_title('Glass Transition Temperature Evolution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 1.3 DC 进化
    ax = axes[1, 0]
    if 'DC' in df.columns:
        mean_dc = [gen_groups.get_group(g)['DC'].mean() for g in generations]
        max_dc = [gen_groups.get_group(g)['DC'].max() for g in generations]
        
        ax.plot(generations, max_dc, 'r*-', label='Max DC', linewidth=2)
        ax.plot(generations, mean_dc, 'b+-', label='Mean DC', linewidth=2)
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Dielectric Constant', fontsize=12)
        ax.set_title('Dielectric Constant Evolution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 1.4 种群大小变化
    ax = axes[1, 1]
    pop_size = [len(gen_groups.get_group(g)) for g in generations]
    
    ax.bar(generations, pop_size, color='skyblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Population Size', fontsize=12)
    ax.set_title('Population Size per Generation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'evolution_analysis.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {fig_path}")
    
    # 图2: 性质分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Property Distributions', fontsize=16, fontweight='bold')
    
    # 2.1 Tg 分布
    ax = axes[0]
    if 'Tg' in df.columns:
        ax.hist(df['Tg'], bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(df['Tg'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Tg"].mean():.2f} K')
        ax.set_xlabel('Tg (K)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Tg Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # 2.2 DC 分布
    ax = axes[1]
    if 'DC' in df.columns:
        ax.hist(df['DC'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.axvline(df['DC'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {df["DC"].mean():.4f}')
        ax.set_xlabel('Dielectric Constant', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('DC Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'property_distributions.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {fig_path}")
    
    # 图3: Tg vs DC 散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'Tg' in df.columns and 'DC' in df.columns and 'fitness' in df.columns:
        scatter = ax.scatter(df['Tg'], df['DC'], c=df['fitness'], 
                            s=100, cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Fitness', fontsize=12)
        
        ax.set_xlabel('Tg (K)', fontsize=12)
        ax.set_ylabel('Dielectric Constant', fontsize=12)
        ax.set_title('Tg vs DC (colored by Fitness)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标记最佳点
        best_idx = df['fitness'].idxmax()
        best = df.loc[best_idx]
        ax.scatter([best['Tg']], [best['DC']], c='red', s=300, marker='*', 
                  edgecolors='black', linewidth=2, label='Best', zorder=5)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'tg_vs_dc.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {fig_path}")
    
    # 图4: 染色体数量统计
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'num_chromosomes' in df.columns:
        chrom_counts = df['num_chromosomes'].value_counts().sort_index()
        ax.bar(chrom_counts.index, chrom_counts.values, color='plum', edgecolor='purple', alpha=0.7)
        ax.set_xlabel('Number of Chromosomes', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Chromosome Numbers', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'chromosome_distribution.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {fig_path}")
    
    # ========================================================================
    # 生成报告
    # ========================================================================
    print(f"\n📝 生成文本报告...")
    
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PolyGA + MMPolymer 遗传算法运行报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. 数据概览\n")
        f.write("-" * 40 + "\n")
        f.write(f"总聚合物数: {len(df)}\n")
        f.write(f"总代数: {df['generation'].max() + 1}\n")
        f.write(f"列数: {len(df.columns)}\n\n")
        
        f.write("2. 性质统计\n")
        f.write("-" * 40 + "\n")
        if 'Tg' in df.columns:
            f.write(f"Tg:\n")
            f.write(f"  范围: {df['Tg'].min():.2f} - {df['Tg'].max():.2f} K\n")
            f.write(f"  平均: {df['Tg'].mean():.2f} K\n")
            f.write(f"  标准差: {df['Tg'].std():.2f} K\n\n")
        
        if 'DC' in df.columns:
            f.write(f"DC:\n")
            f.write(f"  范围: {df['DC'].min():.4f} - {df['DC'].max():.4f}\n")
            f.write(f"  平均: {df['DC'].mean():.4f}\n")
            f.write(f"  标准差: {df['DC'].std():.4f}\n\n")
        
        if 'fitness' in df.columns:
            f.write(f"适应度:\n")
            f.write(f"  范围: {df['fitness'].min():.2f} - {df['fitness'].max():.2f}\n")
            f.write(f"  平均: {df['fitness'].mean():.2f}\n")
            f.write(f"  标准差: {df['fitness'].std():.2f}\n\n")
        
        f.write("3. 最佳聚合物 Top 10\n")
        f.write("-" * 40 + "\n")
        top10 = df.nlargest(10, 'fitness')
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            f.write(f"\n{i}. ID={row['planetary_id']}, 适应度={row['fitness']:.4f}\n")
            if 'Tg' in row:
                f.write(f"   Tg={row['Tg']:.2f} K\n")
            if 'DC' in row:
                f.write(f"   DC={row['DC']:.4f}\n")
            f.write(f"   染色体数={row['num_chromosomes']}\n")
            f.write(f"   SMILES={row['smiles_string'][:80]}...\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")
    
    print(f"  ✓ 保存: {report_path}")
    
    print(f"\n✅ 分析完成！")
    print(f"\n📁 结果保存在: {output_dir}/")
    print(f"  - evolution_analysis.png")
    print(f"  - property_distributions.png")
    print(f"  - tg_vs_dc.png")
    print(f"  - chromosome_distribution.png")
    print(f"  - analysis_report.txt")


if __name__ == '__main__':
    analyze_results()

