import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def load_data(db_path):
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT generation, properties FROM polymer", con)
    con.close()
    
    # 解析 JSON properties
    def parse_props(x):
        try:
            p = json.loads(x)
            return p.get('Tg', np.nan), p.get('DC', np.nan)
        except:
            return np.nan, np.nan
            
    df[['Tg', 'DC']] = df['properties'].apply(lambda x: pd.Series(parse_props(x)))
    return df

def calculate_fitness(df, w_tg=0.5, w_dc=0.5):
    # 为了可视化进化趋势，我们需要一个统一的评价标准。
    # 我们使用全局数据的分位数作为固定的归一化边界。
    # 这样，随着代数增加，如果个体变得更好，Fitness 会明显上升。
    
    tg_L = df['Tg'].quantile(0.10)
    tg_U = df['Tg'].quantile(0.90)
    dc_L = df['DC'].quantile(0.10)
    dc_U = df['DC'].quantile(0.90)
    
    print(f"Normalization Bounds (Global):")
    print(f"Tg: [{tg_L:.2f}, {tg_U:.2f}] (Larger is better)")
    print(f"DC: [{dc_L:.2f}, {dc_U:.2f}] (Smaller is better)")
    
    def calc_row(row):
        tg = row['Tg']
        dc = row['DC']
        if pd.isna(tg) or pd.isna(dc):
            return 0.0
            
        # Tg Normalization (Maximize)
        if tg_U - tg_L == 0:
            tg_norm = 0.0
        else:
            tg_norm = (tg - tg_L) / (tg_U - tg_L)
        tg_norm = max(0.0, min(1.0, tg_norm))
        
        # DC Normalization (Minimize)
        if dc_U - dc_L == 0:
            dc_norm = 0.0
        else:
            dc_norm = (dc_U - dc) / (dc_U - dc_L)
        dc_norm = max(0.0, min(1.0, dc_norm))
        
        return w_tg * tg_norm + w_dc * dc_norm

    df['fitness'] = df.apply(calc_row, axis=1)
    return df

def plot_fitness_distribution(df, out_path):
    # 设置风格
    sns.set_style("white")
    
    # 定义分段 (Segments)
    # 既然总共 50 代，我们可以选 3 个代表性区间
    segments = [
        (0, 9),    # 初期
        (20, 29),  # 中期
        (40, 49)   # 晚期
    ]
    
    # 创建子图，共享 Y 轴
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1, 1]})
    plt.subplots_adjust(wspace=0.05)  # 调整子图间距
    
    generations = sorted(df['generation'].unique())
    
    from matplotlib.patches import Polygon
    
    # 遍历每个子图和分段
    for ax_idx, (ax, (seg_start, seg_end)) in enumerate(zip(axes, segments)):
        # 1. 绘制背景阴影 (Genetic Transfer Flow) - 使用平滑插值
        for i in range(seg_start, seg_end):
            gen_curr = i
            gen_next = i + 1
            
            if gen_next > 49: break 
            
            data_curr = df[df['generation'] == gen_curr]
            data_next = df[df['generation'] == gen_next]
            
            if len(data_curr) == 0 or len(data_next) == 0:
                continue
                
            # Parents Range (Top 50%)
            n_parents = int(len(data_curr) * 0.5)
            if n_parents < 1: n_parents = 1
            parents_fitness = data_curr.nlargest(n_parents, 'fitness')['fitness']
            p_min = parents_fitness.min()
            p_max = parents_fitness.max()
            
            # Offspring Range (All)
            o_min = data_next['fitness'].min()
            o_max = data_next['fitness'].max()
            
            # 平滑插值逻辑 (Cosine Interpolation)
            # t 从 0 到 1
            num_points = 20
            t = np.linspace(0, 1, num_points)
            # 使用 Cosine 平滑: (1 - cos(t*pi)) / 2
            smooth_factor = (1 - np.cos(t * np.pi)) / 2
            
            # 生成 X 坐标
            x_interp = gen_curr + t * (gen_next - gen_curr)
            
            # 生成上边界曲线 (P_max -> O_max)
            top_curve = p_max + (o_max - p_max) * smooth_factor
            
            # 生成下边界曲线 (P_min -> O_min)
            bottom_curve = p_min + (o_min - p_min) * smooth_factor
            
            # 构建多边形顶点
            verts = []
            # 上边界点 (从左到右)
            for x, y in zip(x_interp, top_curve):
                verts.append((x, y))
            # 下边界点 (从右到左，闭合多边形)
            for x, y in zip(reversed(x_interp), reversed(bottom_curve)):
                verts.append((x, y))
            
            poly = Polygon(verts, facecolor='#e0e0e0', edgecolor='none', alpha=0.4, zorder=0)
            ax.add_patch(poly)
            
        # 2. 绘制散点
        seg_df = df[(df['generation'] >= seg_start) & (df['generation'] <= seg_end)]
        
        scatter = ax.scatter(
            seg_df['generation'], 
            seg_df['fitness'], 
            c=seg_df['fitness'], 
            cmap='viridis_r',
            s=20, 
            alpha=0.7, 
            edgecolor='none',
            zorder=10
        )
        
        # 3. 设置 X 轴 (隔代显示，减少拥挤)
        ax.set_xlim(seg_start - 0.5, seg_end + 0.5)
        # 策略：起点和终点必显，中间隔1个显示
        # 例如 0, 2, 4, 6, 8
        ticks = list(range(seg_start, seg_end + 1, 2))
        # 确保最后一个刻度被包含（如果是奇数长度可能漏掉，但这里seg是0-9(10个), 20-29(10个) -> 0,2,4,6,8 OK）
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=9)
        
        # 4. 标注红色箭头 (更美观的红色书签风格)
        highlight_gens = [0, 10, 20, 30, 40, 49]
        for gen in highlight_gens:
            if seg_start <= gen <= seg_end:
                # 1. 在轴下方画一个红色小三角 (Marker)
                ax.plot(gen, -0.08, marker='^', color='#d62728', markersize=8, 
                       transform=ax.get_xaxis_transform(), clip_on=False, zorder=20)
                
                # 2. 在三角下方画红色圆角标签 (Text with Bbox)
                ax.text(
                    gen, -0.12, 
                    str(gen),
                    ha='center', va='top',
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#d62728", ec="none"),
                    transform=ax.get_xaxis_transform(),
                    zorder=20
                )

        # 5. 处理断轴样式 (Broken Axis) & Open Axes
        ax.set_ylim(-0.05, 1.05)
        
        # 通用设置：移除顶部和右侧脊柱 (Open Axes)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 针对不同子图的特殊设置
        if ax_idx == 0: # 第一个图
            # 保留左侧 Y 轴
            ax.spines['left'].set_visible(True)
            ax.yaxis.tick_left()
        else: # 中间和最后一个图
            # 隐藏左侧脊柱和刻度
            ax.spines['left'].set_visible(False)
            ax.tick_params(labelleft=False, left=False)
            
        # 绘制斜线 (//) - 仅在底部 X 轴绘制
        d = .015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        
        # 这里的逻辑是：
        # ax_idx=0 (左图): 右下角断开
        # ax_idx=1 (中图): 左下角断开 + 右下角断开
        # ax_idx=2 (右图): 左下角断开
        
        if ax_idx == 0:
            # 右下断口
            ax.plot((1-d, 1+d), (-d, +d), **kwargs)
            # 添加 Y 轴箭头 (顶部)
            ax.plot(0, 1, marker='^', color='black', markersize=6, 
                   transform=ax.transAxes, clip_on=False, zorder=100)
        elif ax_idx == len(segments) - 1:
            # 左下断口
            ax.plot((-d, +d), (-d, +d), **kwargs)
            # 添加 X 轴箭头 (右端)
            ax.plot(1, 0, marker='>', color='black', markersize=6, 
                   transform=ax.transAxes, clip_on=False, zorder=100)
        else:
            # 左下 + 右下
            ax.plot((-d, +d), (-d, +d), **kwargs)
            ax.plot((1-d, 1+d), (-d, +d), **kwargs)

    # 全局标签
    fig.text(0.5, 0.02, 'Generation', ha='center', fontsize=14, fontweight='bold')
    # 增加 labelpad (通过调整 x 位置或直接加 labelpad 参数，但 fig.text 没有 labelpad)
    # 这里我们通过调整位置 x=0.06 (原0.08) 来增加距离
    fig.text(0.06, 0.5, 'Fitness function', va='center', rotation='vertical', fontsize=14, fontweight='bold')
    
    # Colorbar (放在最右侧)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    # cbar.set_label('Fitness Value', rotation=270, labelpad=15)
    
    # 移除左上角图例代码
    
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
    
    plt.savefig(out_path, dpi=600)
    print(f"Figure saved to {out_path}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Fitness Distribution')
    parser.add_argument('--db', type=str, 
                        default="/root/code/polyga_project_mmpolymer/results_formal_run/BRICSPolyPlanet/planetary_database.sqlite",
                        help='Path to the sqlite database')
    parser.add_argument('--out', type=str, 
                        default="/root/code/polyga_project_mmpolymer/results_formal_run/figures/fitness_distribution.png",
                        help='Output path for the plot')
    args = parser.parse_args()
    
    db_path = args.db
    out_path = args.out
    
    print(f"Database path: {db_path}")
    print(f"Output path: {out_path}")
    
    print("Loading data...")
    df = load_data(db_path)
    print(f"Loaded {len(df)} records.")
    
    print("Calculating fitness...")
    df = calculate_fitness(df)
    
    print("Plotting...")
    plot_fitness_distribution(df, out_path)
