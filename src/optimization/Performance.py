import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

# ==========================================
# 1. 假设你的 DataFrame 叫做 df
# 这里我们模拟一些数据来展示效果
# ==========================================
np.random.seed(42)
n_samples = 300
df = pd.DataFrame({
    # 生成模拟数据：介电常数 2.0 - 5.0
    'Epsilon': np.random.normal(3.2, 0.5, n_samples).clip(2.0, 5.0),
    # 生成模拟数据：Tg 100 - 400，假设稍微跟介电有点正相关（物理上的常见陷阱）
    'Tg': np.random.normal(200, 50, n_samples).clip(100, 400),
    # 生成模拟数据：SA Score 1 - 8
    'SA_Score': np.random.normal(3.5, 1.5, n_samples).clip(1.5, 8.5)
})
# 制造一点相关性：让高性能(左上角)的分子稍微难合成一点(红色)，以凸显筛选的价值
df['Tg'] += (df['Epsilon'] - 2.5) * 20
df['SA_Score'] += (df['Tg'] / 100) - (df['Epsilon'] / 2)

# ==========================================
# 1.1 读取真实数据（如果存在则覆盖上面的模拟数据）
# ==========================================
db_path: str = '/root/code/polyga_project_mmpolymer/results_formal_run/BRICSPolyPlanet/planetary_database.sqlite'
try:
    con = sqlite3.connect(db_path)
    df_real = pd.read_sql_query(
        'SELECT planetary_id, generation, smiles_string, properties FROM polymer',
        con,
    )
    con.close()

    def _parse_props(props: Any) -> tuple[float, float]:
        try:
            if isinstance(props, str):
                d = json.loads(props)
            elif isinstance(props, dict):
                d = props
            else:
                d = {}
            tg = float(d.get('Tg')) if d.get('Tg') is not None else float('nan')
            dc = float(d.get('DC')) if d.get('DC') is not None else float('nan')
            return tg, dc
        except Exception:
            return float('nan'), float('nan')

    parsed = df_real['properties'].apply(_parse_props)
    df_real[['Tg', 'Epsilon']] = pd.DataFrame(parsed.tolist(), index=df_real.index)
    df = df_real.dropna(subset=['Epsilon', 'Tg']).copy()

    script_dir = Path(__file__).resolve().parent
    smi2pos_dir = script_dir / 'smi2pos'
    if smi2pos_dir.exists():
        sys.path.insert(0, str(smi2pos_dir))

    try:
        import sascorer  # type: ignore
        from rdkit import Chem  # type: ignore

        def _get_sa_score(smiles: str) -> float:
            if not smiles:
                return 10.0
            clean_smi = smiles.replace('*', 'C')
            mol = Chem.MolFromSmiles(clean_smi)
            if mol is None:
                return 10.0
            try:
                score = sascorer.calculateScore(mol)
                return 10.0 if score is None else float(round(float(score), 3))
            except Exception:
                return 10.0

        df['SA_Score'] = df['smiles_string'].astype(str).apply(_get_sa_score)
    except Exception:
        pass

    if 'SA_Score' not in df.columns:
        df['SA_Score'] = 10.0

    weight_tg: float = 0.5
    weight_dc: float = 0.5
    tg_q_low: float = 0.10
    tg_q_high: float = 0.90
    dc_q_low: float = 0.10
    dc_q_high: float = 0.90
    dc_max_quantile: float = 0.80
    dc_penalty_factor: float = 0.10

    df_gen0 = df[df['generation'] == 0].copy() if 'generation' in df.columns else df.iloc[0:0].copy()
    if not df_gen0.empty:
        tg_L = float(df_gen0['Tg'].quantile(tg_q_low))
        tg_U = float(df_gen0['Tg'].quantile(tg_q_high))
        dc_L = float(df_gen0['Epsilon'].quantile(dc_q_low))
        dc_U = float(df_gen0['Epsilon'].quantile(dc_q_high))
        dc_th = float(df_gen0['Epsilon'].quantile(dc_max_quantile))

        tg_den = tg_U - tg_L
        dc_den = dc_U - dc_L
        tg_den = 1.0 if tg_den == 0.0 else tg_den
        dc_den = 1.0 if dc_den == 0.0 else dc_den

        tg_norm = ((df['Tg'] - tg_L) / tg_den).clip(0.0, 1.0)
        dc_norm = ((dc_U - df['Epsilon']) / dc_den).clip(0.0, 1.0)
        df['fitness'] = weight_tg * tg_norm + weight_dc * dc_norm
        penalized = df['Epsilon'] > dc_th
        df.loc[penalized, 'fitness'] = df.loc[penalized, 'fitness'] * dc_penalty_factor
except Exception:
    pass

# ==========================================
# 2. 绘图核心代码
# ==========================================
# 获取前五代和最后五代
has_gen = 'generation' in df.columns
if has_gen and not df.empty:
    first_gen = df['generation'].min()
    first_5_end = min(df['generation'].max(), first_gen + 4)
    last_gen = df['generation'].max()
    last_5_start = max(0, last_gen - 4)
    df_first = df[(df['generation'] >= first_gen) & (df['generation'] <= first_5_end)].copy()
    df_last = df[df['generation'] >= last_5_start].copy()
else:
    # Fallback to simulated data or single generation logic
    first_gen = 0
    first_5_end = 0
    last_gen = 0
    last_5_start = 0
    df_first = df.copy()
    df_last = df.copy()

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
    plt.rcParams['axes.labelsize'] = 25
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
    
fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=600, sharex=True, sharey=True)

# 定义目标阈值及显示文案
target_eps = 2.8
target_tg = 260
target_label = r'Design target: $\epsilon \leq 2.8$, $T_g \geq 260$ °C'

# 设置全局坐标轴范围
eps_min = float(df['Epsilon'].min())
eps_max = float(df['Epsilon'].max())
tg_min = float(df['Tg'].min())
tg_max = float(df['Tg'].max())

xlim_min = 1.8
xlim_max = 3.2
ylim_min = min(tg_min - 10.0, target_tg - 20.0)
ylim_max = tg_max + 20.0

scatter_kwargs = dict(
    s=100,              # 增大散点尺寸
    cmap='viridis_r',   # 蓝/深紫=易合成, 黄绿=难合成
    alpha=0.65,         # 增加透明度防止重叠遮挡
    edgecolors='none',  # 取消边框，高密度区更突出
    vmin=df['SA_Score'].min(),
    vmax=df['SA_Score'].max()
)

def plot_subplot(ax, df_sub, title):
    # 计算 target 区域内的比例
    in_target = df_sub[(df_sub['Epsilon'] <= target_eps) & (df_sub['Tg'] >= target_tg)]
    frac = len(in_target) / len(df_sub) * 100 if len(df_sub) > 0 else 0
    print(f"{title}: {frac:.1f}% in target")

    # 画出 Target 区域 (浅灰绿)
    ax.fill_between([xlim_min - 0.5, target_eps], 
                     target_tg, 
                     ylim_max + 50, 
                     color='#8FBC8F', alpha=0.15, label=target_label)

    # 细虚线辅助线
    ax.axvline(x=target_eps, color='#2E8B57', linestyle='--', alpha=0.4, linewidth=1.0)
    ax.axhline(y=target_tg, color='#2E8B57', linestyle='--', alpha=0.4, linewidth=1.0)

    # 绘制气泡散点图
    scatter = ax.scatter(
        df_sub['Epsilon'], 
        df_sub['Tg'], 
        c=df_sub['SA_Score'], 
        **scatter_kwargs
    )
    
    ax.set_ylabel(r'Glass transition temperature, $T_g$ (°C)', fontweight='bold', fontsize=15)
    ax.set_title(title, pad=10, fontweight='bold', fontsize=15)
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)
    
    # 图例
    ax.legend(loc='lower right', frameon=True)
    return scatter

# Plot Top (First 5 Gens)
if has_gen and not df.empty:
    top_title = f'Generations {int(first_gen)}–{int(first_5_end)}'
else:
    top_title = 'Initial Population'
scatter_fig = plot_subplot(axes[0], df_first, top_title)

# Plot Bottom (Last 5 Gens)
if has_gen and not df.empty:
    bottom_title = f'Generations {int(last_5_start)}–{int(last_gen)}'
else:
    bottom_title = 'Final Population'
_ = plot_subplot(axes[1], df_last, bottom_title)
axes[1].set_xlabel(r'Dielectric constant, $\epsilon$ (–)', fontweight='bold', fontsize=15)

# --- D. 美化 ---
fig.suptitle(r'Figure X. Evolution of synthesizable low-$\epsilon$ polymers under multi-objective optimization', fontweight='bold', y=0.96)

# 添加全局 Colorbar
cbar = fig.colorbar(scatter_fig, ax=axes.ravel().tolist(), fraction=0.03, pad=0.04)
cbar.set_label('Synthetic accessibility score')

output_path: str = '/root/code/polyga_project_mmpolymer/results_formal_run/Sweet_Spot_Map_Evolution.png'
plt.savefig(output_path, dpi=600, bbox_inches='tight')
backend = str(plt.get_backend()).lower()
if 'agg' not in backend:
    plt.show() # 如果在 Jupyter 中运行，请使用 plt.show()