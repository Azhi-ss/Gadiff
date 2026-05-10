import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
import argparse

# --- Reuse functions from visualize_fitness_distribution.py ---
def load_data(db_path):
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT generation, properties FROM polymer", con)
    con.close()
    
    def parse_props(x):
        try:
            p = json.loads(x)
            return p.get('Tg', np.nan), p.get('DC', np.nan)
        except:
            return np.nan, np.nan
            
    df[['Tg', 'DC']] = df['properties'].apply(lambda x: pd.Series(parse_props(x)))
    return df

def calculate_fitness(df, w_tg=0.5, w_dc=0.5):
    tg_L = df['Tg'].quantile(0.10)
    tg_U = df['Tg'].quantile(0.90)
    dc_L = df['DC'].quantile(0.10)
    dc_U = df['DC'].quantile(0.90)
    
    def calc_row(row):
        tg = row['Tg']
        dc = row['DC']
        if pd.isna(tg) or pd.isna(dc):
            return 0.0
            
        if tg_U - tg_L == 0:
            tg_norm = 0.0
        else:
            tg_norm = (tg - tg_L) / (tg_U - tg_L)
        tg_norm = max(0.0, min(1.0, tg_norm))
        
        if dc_U - dc_L == 0:
            dc_norm = 0.0
        else:
            dc_norm = (dc_U - dc) / (dc_U - dc_L)
        dc_norm = max(0.0, min(1.0, dc_norm))
        
        return w_tg * tg_norm + w_dc * dc_norm

    df['fitness'] = df.apply(calc_row, axis=1)
    return df

# --- New Schematic Plotting Function ---
def plot_schematic(df, out_path):
    # Select data from Gen 0 and Gen 1 for the example
    gen_parents = 0
    gen_offspring = 1
    
    df_p = df[df['generation'] == gen_parents].copy()
    df_o = df[df['generation'] == gen_offspring].copy()
    
    # If empty (unlikely), fallback to dummy
    if len(df_p) == 0:
        print("Warning: No data for Gen 0/1, using dummy.")
        return

    # Setup vertical layout
    # Y axis: Top = Parents (y=1.5), Bottom = Offspring (y=0.5)
    # X axis: Fitness
    
    fig, ax = plt.subplots(figsize=(5, 6))
    sns.set_style("white")
    
    # 1. Draw "Genetic Transfer" Polygon (Vertical Flow)
    # Logic: Connect Top Parents (Top 50% fitness) to Bottom Offspring (Range)
    
    n_parents = int(len(df_p) * 0.5)
    if n_parents < 1: n_parents = 1
    parents_fitness = df_p.nlargest(n_parents, 'fitness')['fitness']
    p_min = parents_fitness.min()
    p_max = parents_fitness.max()
    
    o_min = df_o['fitness'].min()
    o_max = df_o['fitness'].max()
    
    # Interpolation (Vertical)
    # y goes from 1 (Top) to 0 (Bottom)
    num_points = 50
    t = np.linspace(0, 1, num_points) # t=0 at top, t=1 at bottom
    
    # Y coordinates for the flow (e.g., from y=1.0 down to y=0.0)
    y_top_base = 1.0
    y_bottom_base = 0.0
    y_coords = y_top_base - t * (y_top_base - y_bottom_base)
    
    # Smooth factor
    smooth_factor = (1 - np.cos(t * np.pi)) / 2
    
    # Left curve (Low fitness side)
    left_curve = p_min + (o_min - p_min) * smooth_factor
    # Right curve (High fitness side)
    right_curve = p_max + (o_max - p_max) * smooth_factor
    
    verts = []
    # Left side (Top to Bottom)
    for x, y in zip(left_curve, y_coords):
        verts.append((x, y))
    # Right side (Bottom to Top)
    for x, y in zip(reversed(right_curve), reversed(y_coords)):
        verts.append((x, y))
        
    poly = Polygon(verts, facecolor='#e0e0e0', edgecolor='none', alpha=0.5, zorder=0)
    ax.add_patch(poly)
    
    # 2. Scatter Points
    # Downsample points for clarity and reduced crowding
    n_p_show = 5
    n_o_show = 10
    
    # Helper to sample uniformly sorted by fitness to avoid crowding
    def get_uniform_sample(d, n):
        if len(d) <= n:
            return d
        d_sorted = d.sort_values('fitness')
        indices = np.linspace(0, len(d)-1, n, dtype=int)
        return d_sorted.iloc[indices]
        
    df_p_viz = get_uniform_sample(df_p, n_p_show)
    df_o_viz = get_uniform_sample(df_o, n_o_show)

    # Function to add vertical jitter
    def add_jitter(y_center, n, scale=0.0):
        return np.full(n, y_center)
    
    # Parents
    ax.scatter(
        df_p_viz['fitness'],
        add_jitter(1.0, len(df_p_viz)),
        c=df_p_viz['fitness'],
        cmap='viridis_r',
        s=100, alpha=0.8, edgecolor='white', linewidth=0.5, zorder=10
    )
    
    # Offspring
    ax.scatter(
        df_o_viz['fitness'],
        add_jitter(0.0, len(df_o_viz)),
        c=df_o_viz['fitness'],
        cmap='viridis_r',
        s=100, alpha=0.8, edgecolor='white', linewidth=0.5, zorder=10
    )
    
    # 3. Annotations (mimicking the user's image)
    
    # Parent Label (Top)
    p_center = (p_min + p_max) / 2
    ax.text(p_center, 1.1, "Parent polymers", ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Offspring Label (Bottom)
    o_center = (o_min + o_max) / 2
    ax.text(o_center, -0.1, "Offspring polymers", ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Arrow and text for flow (Left side)
    # Place text to the left of the flow
    mid_y = 0.5
    # Use the left edge of the flow at the middle for reference
    mid_x_left = left_curve[25] 
    
    ax.annotate("Fragment transfer to\nthe next generation", 
                xy=(mid_x_left, mid_y), 
                xytext=(mid_x_left - 0.2, mid_y),
                ha='right', va='center', fontsize=14,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Formatting
    # Adjust limits to fit the text
    ax.set_xlim(-0.8, 1.5)
    ax.set_ylim(-0.3, 1.4)
    ax.axis('off') # Turn off axis for schematic look
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Schematic saved to {out_path}")

if __name__ == "__main__":
    db_path = "/root/code/polyga_project_mmpolymer/results_formal_run/BRICSPolyPlanet/planetary_database.sqlite"
    out_path = "/root/code/polyga_project_mmpolymer/results_formal_run/figures/fitness_schematic_vertical.png"
    
    df = load_data(db_path)
    df = calculate_fitness(df)
    plot_schematic(df, out_path)
