import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3
import json

def get_min_max_from_db(db_path):
    """从数据库获取每一代的真实最小值、最大值和 P90"""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    
    # 获取总代数
    max_gen = cur.execute('SELECT MAX(generation) FROM polymer').fetchone()[0]
    
    data = []
    for gen in range(max_gen + 1):
        # 解析该代所有个体的属性
        rows = cur.execute('SELECT properties FROM polymer WHERE generation = ?', (gen,)).fetchall()
        tgs = []
        dcs = []
        for r in rows:
            try:
                props = json.loads(r['properties']) if r['properties'] else {}
                if 'Tg' in props and props['Tg'] is not None:
                    tgs.append(float(props['Tg']))
                if 'DC' in props and props['DC'] is not None:
                    dcs.append(float(props['DC']))
            except:
                pass
        
        if tgs and dcs:
            data.append({
                'generation': gen,
                'tg_min': min(tgs),
                'tg_max_real': max(tgs),
                'dc_min': min(dcs),
                'dc_max_real': max(dcs),
                'dc_p90_real': np.percentile(dcs, 90)
            })
            
    con.close()
    return pd.DataFrame(data)

def visualize_trends(summary_path):
    # 读取汇总数据
    df = pd.read_csv(summary_path)
    
    # 尝试读取数据库以获取 Min/Max
    results_dir = Path(summary_path).parent
    # 查找可能的数据库路径 (过滤空文件)
    db_path = [p for p in results_dir.rglob('planetary_database.sqlite') if p.stat().st_size > 0]
    if db_path:
        print(f"Found database: {db_path[0]}")
        df_db = get_min_max_from_db(db_path[0])
        # 合并数据
        df = pd.merge(df, df_db, on='generation', how='left')
    else:
        print("Warning: Database not found, estimating ranges...")
        # 如果没有数据库，使用粗略估计或仅用已有数据
        df['tg_min'] = df['tg_mean'] - (df['tg_max'] - df['tg_mean']) # 粗略对称估计
        df['dc_min'] = df['best_dc'] # 近似
        df['dc_p90_real'] = df['dc_mean'] * 1.5 # 粗略估计
    
    # -------------------------------------------------------------------------
    # 统一的期刊格式绘图风格
    # -------------------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 统一字号
    plt.rcParams['font.size'] = 14               
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    # 统一网格线和边框样式
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#e0e0e0'       
    plt.rcParams['grid.linestyle'] = '--'        
    plt.rcParams['grid.linewidth'] = 0.8
    plt.rcParams['grid.alpha'] = 0.8
    # -------------------------------------------------------------------------
    
    # 创建画布：2行1列
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True, dpi=600)
    
    # --- 子图 1: Tg (Maximize) ---
    ax_tg = axes[0]
    
    # 绘制均值线
    sns.lineplot(data=df, x='generation', y='tg_mean', ax=ax_tg, 
                label='Mean $T_g$', color='#1f77b4', linewidth=1.5)
    
    # 绘制最大值线
    sns.lineplot(data=df, x='generation', y='tg_max', ax=ax_tg, 
                label='Max $T_g$', color='#d62728', linestyle='--', linewidth=1.0)
                
    # 仅填充区域：Min 到 Max (Population Range) 作为范围展示
    
    # 填充区域：Min 到 Max (Population Range)
    ax_tg.fill_between(df['generation'], df['tg_min'], df['tg_max'], 
                      color='#1f77b4', alpha=0.15, label='Population range (min-max)')
    
    ax_tg.set_ylabel(r'Glass transition temperature, $T_g$ (°C)', fontweight='bold')
    ax_tg.legend(loc='upper left', frameon=True)
    
    # --- 子图 2: DC (Minimize) ---
    ax_dc = axes[1]
    
    # 绘制均值线
    sns.lineplot(data=df, x='generation', y='dc_mean', ax=ax_dc, 
                label='Mean $\epsilon$', color='#9467bd', linewidth=1.5)
    
    # 绘制最优个体 DC 线
    sns.lineplot(data=df, x='generation', y='best_dc', ax=ax_dc, 
                label='Best candidate $\epsilon$', color='#ff7f0e', linestyle='--', linewidth=1.0)
                
    # 填充区域：Min 到 P90 (Population Range)
    ax_dc.fill_between(df['generation'], df['dc_min'], df['dc_p90_real'], 
                      color='#9467bd', alpha=0.15, label='Population range (min-P90)')
    
    ax_dc.set_ylabel(r'Dielectric constant, $\epsilon$ (–)', fontweight='bold')
    ax_dc.set_xlabel('Generation', fontweight='bold')
    
    # 确保 Y 轴范围合理
    # 使用 P90 的最大值稍微放宽一点作为上限
    if 'dc_p90_real' in df.columns:
        max_val = df['dc_p90_real'].max()
        max_view_dc = min(max_val * 1.2, 50.0) # 稍微留点余地，但不超过50
    else:
        max_view_dc = 30.0
        
    ax_dc.set_ylim(1.5, max_view_dc)
    
    ax_dc.legend(loc='upper left', frameon=True)
    # ax_dc.grid(True, alpha=0.3) # 关闭网格
    
    plt.subplots_adjust(left=0.12, right=0.82, top=0.95, bottom=0.1, hspace=0.15)
    
    output_path = Path(summary_path).parent / 'trends_tg_dc.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")

if __name__ == "__main__":
    visualize_trends('/root/code/polyga_project_mmpolymer/results_formal_run/summary.csv')
