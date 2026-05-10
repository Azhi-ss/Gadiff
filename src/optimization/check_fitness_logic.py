import pandas as pd
import numpy as np

# 模拟 run_optimization.py 中的归一化逻辑
def calculate_fitness(row, tg_L, tg_U, dc_L, dc_U, w_tg=0.5, w_dc=0.5):
    tg = row['best_tg']
    dc = row['best_dc']
    
    # Tg 归一化 (越大越好)
    if tg_U - tg_L == 0:
        tg_norm = 0.0
    else:
        tg_norm = (tg - tg_L) / (tg_U - tg_L)
    tg_norm = max(0.0, min(1.0, tg_norm))
    
    # DC 归一化 (越小越好)
    if dc_U - dc_L == 0:
        dc_norm = 0.0
    else:
        dc_norm = (dc_U - dc) / (dc_U - dc_L)
    dc_norm = max(0.0, min(1.0, dc_norm))
    
    return w_tg * tg_norm + w_dc * dc_norm

# 读取数据
df = pd.read_csv('/root/code/polyga_project_mmpolymer/results_formal_run/summary.csv')

print("=== PolyGA Formal Run Analysis (50 Generations) ===")
print(f"Total Generations: {len(df)}")

# 1. Fitness 趋势
print("\n--- Fitness Trend ---")
first_gen = df.iloc[0]
last_gen = df.iloc[-1]
best_gen = df.loc[df['best_fitness'].idxmax()]

print(f"Gen 0: Max Fitness={first_gen['fitness_max']:.4f}, Mean={first_gen['fitness_mean']:.4f}")
print(f"Final: Max Fitness={last_gen['fitness_max']:.4f}, Mean={last_gen['fitness_mean']:.4f}")
print(f"Best Ever: Gen {int(best_gen['generation'])} (Fitness={best_gen['best_fitness']:.4f})")

# 2. Properties 趋势
print("\n--- Properties Trend (Best Individual) ---")
print(f"Tg: {first_gen['best_tg']:.2f} -> {last_gen['best_tg']:.2f} (Max: {df['best_tg'].max():.2f})")
print(f"DC: {first_gen['best_dc']:.2f} -> {last_gen['best_dc']:.2f} (Min: {df['best_dc'].min():.2f})")

# 3. Diversity (Tanimoto)
print("\n--- Diversity ---")
print(f"Initial Diversity: {first_gen['tanimoto_mean']:.4f}")
print(f"Final Diversity: {last_gen['tanimoto_mean']:.4f}")

# 4. Top 3 Molecules
print("\n--- Top 3 Molecules by Fitness ---")
top3 = df.sort_values('best_fitness', ascending=False).head(3)
for idx, row in top3.iterrows():
    print(f"Gen {int(row['generation'])} | Fit: {row['best_fitness']:.4f} | Tg: {row['best_tg']:.1f} | DC: {row['best_dc']:.2f}")
    print(f"SMILES: {row['best_smiles']}")
    print("-" * 50)

# 5. Convergence Check (Last 5 gens)
print("\n--- Convergence Check (Last 5 Generations) ---")
print(df[['generation', 'fitness_max', 'fitness_mean', 'best_tg', 'best_dc']].tail(5).to_string(index=False))

