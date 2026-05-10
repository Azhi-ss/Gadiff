#!/usr/bin/env python3
"""
PolyGA + MMPolymer 融合运行脚本

整合:
1. PolyGA 的遗传算法框架
2. MMPolymer 的性质预测模型
3. 自定义适应度函数
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加路径
sys.path.insert(0, '/root/code/polyga')
sys.path.insert(0, '/root/code/MMPolymer')
sys.path.insert(0, '/root/code/MMga')

from polyga.polygod import PolyPlanet, PolyLand, PolyNation
from polyga.utils import chromosome_ids_to_smiles
from polyga_predict import create_mmpolymer_predictor


# ============================================================================
# 1. 染色体生成函数
# ============================================================================

def custom_generative_function(chromosome_ids, chromosomes, rng, **kwargs):
    """
    染色体组合函数
    
    使用 PolyGA 的标准实现将染色体组合成聚合物 SMILES
    
    Args:
        chromosome_ids: 染色体 ID 列表
        chromosomes: {id: smiles} 字典
        rng: numpy 随机数生成器
        **kwargs: 额外参数
        
    Returns:
        聚合物 SMILES 字符串
    """
    return chromosome_ids_to_smiles(chromosome_ids, chromosomes, rng, **kwargs)


# ============================================================================
# 2. 指纹函数
# ============================================================================

def simple_fingerprint_function(population_df):
    """
    简单指纹函数 - 基于染色体组成
    
    Args:
        population_df: 包含聚合物的 DataFrame
        
    Returns:
        (population_df, fp_headers): 添加了指纹的 DataFrame 和指纹列名
    """
    fp_headers = []
    
    # 统计每个染色体出现的频率
    all_chromosome_ids = set()
    for index, row in population_df.iterrows():
        chromosome_ids = row['chromosome_ids']
        all_chromosome_ids.update(chromosome_ids)
    
    # 为每个染色体创建计数列
    for cid in sorted(all_chromosome_ids):
        col_name = f'chr_{cid}_count'
        fp_headers.append(col_name)
        
        counts = []
        for index, row in population_df.iterrows():
            chromosome_ids = row['chromosome_ids']
            counts.append(chromosome_ids.count(cid))
        
        population_df[col_name] = counts
    
    return population_df, fp_headers


# ============================================================================
# 3. 适应度函数
# ============================================================================

def create_fitness_function(target_tg=None, target_dc=None, 
                           weight_tg=1.0, weight_dc=1.0,
                           penalty_invalid=True):
    """
    创建适应度函数
    
    Args:
        target_tg: 目标玻璃化转变温度 (None 表示最大化)
        target_dc: 目标介电常数 (None 表示最大化)
        weight_tg: Tg 的权重
        weight_dc: DC 的权重
        penalty_invalid: 是否惩罚无效预测
        
    Returns:
        fitness_function
    """
    def fitness_function(population_df, fp_headers):
        """
        基于 MMPolymer 预测结果计算适应度
        
        Args:
            population_df: 包含预测性质的 DataFrame
            fp_headers: 指纹列名
            
        Returns:
            添加了 fitness 列的 DataFrame
        """
        fitness_scores = []
        
        for index, row in population_df.iterrows():
            score = 0.0
            valid_predictions = 0
            
            # Tg 适应度
            if 'Tg' in row and not pd.isna(row['Tg']):
                if target_tg is None:
                    # 最大化 Tg (归一化到 0-100 范围)
                    score += weight_tg * (row['Tg'] / 10.0)
                else:
                    # 接近目标 Tg
                    score -= weight_tg * abs(row['Tg'] - target_tg) / 10.0
                valid_predictions += 1
            
            # DC 适应度
            if 'DC' in row and not pd.isna(row['DC']):
                if target_dc is None:
                    # 最大化 DC (归一化)
                    score += weight_dc * row['DC'] * 10.0
                else:
                    # 接近目标 DC
                    score -= weight_dc * abs(row['DC'] - target_dc) * 10.0
                valid_predictions += 1
            
            # Eb 适应度（如果有）
            if 'Eb' in row and not pd.isna(row['Eb']):
                score += 0.1 * (row['Eb'] / 10.0)
                valid_predictions += 1
            
            # 惩罚无效预测
            if penalty_invalid and valid_predictions == 0:
                score = -1000.0
            
            fitness_scores.append(score)
        
        population_df['fitness'] = fitness_scores
        return population_df
    
    return fitness_function


# ============================================================================
# 4. 主运行函数
# ============================================================================

def run_polyga_mmpolymer(
    # 数据路径
    dna_file='/root/code/MMga/DNA/dna.csv',
    initial_population_file=None,  # None 表示随机生成
    
    # MMPolymer 配置
    mmpolymer_path='/root/code/MMPolymer',
    weight_dir='/internfs/Zy/dataset/finetune_data',
    properties=['Tg', 'DC'],
    
    # PolyGA 配置
    planet_name='MMPolymer_Planet',
    land_name='MaterialLand',
    nation_name='PolymerNation',
    num_generations=10,
    population_size=50,
    num_chromosomes_initial=4,
    num_families=20,
    num_parents_per_family=2,
    num_children_per_family=2,
    
    # 遗传操作参数
    mutation_rate=0.3,
    mutation_add_block=0.1,
    crossover_position='relative_center',
    crossover_sigma_offset=1.0,
    
    # 适应度目标
    target_tg=None,  # None 表示最大化
    target_dc=None,  # None 表示最大化
    weight_tg=1.0,
    weight_dc=1.0,
    
    # 其他
    random_seed=42,
    save_folder='/root/code/MMga/results',
    num_cpus=1
):
    """
    运行 PolyGA + MMPolymer 融合遗传算法
    
    Args:
        dna_file: DNA 文件路径
        initial_population_file: 初始种群文件（None 表示随机生成）
        weight_dir: MMPolymer 权重目录
        properties: 要预测的性质列表
        num_generations: 进化代数
        population_size: 种群大小（仅当随机生成时使用）
        num_chromosomes_initial: 初始染色体数量（仅当随机生成时使用）
        其他参数见上方注释
        
    Returns:
        (planet, land, nation, results_df): PolyGA 对象和结果 DataFrame
    """
    
    # 确保保存目录存在
    os.makedirs(save_folder, exist_ok=True)
    
    print("=" * 80)
    print("🚀 启动 PolyGA + MMPolymer 融合遗传算法")
    print("=" * 80)
    
    # ------------------------------------------------------------------------
    # 创建 MMPolymer 预测器
    # ------------------------------------------------------------------------
    print("\n📊 步骤 1: 初始化 MMPolymer 预测器...")
    predict_fn = create_mmpolymer_predictor(
        weight_dir=weight_dir,
        properties=properties,
        cache_dir=os.path.join(save_folder, 'cache')
    )
    
    # ------------------------------------------------------------------------
    # 创建适应度函数
    # ------------------------------------------------------------------------
    print("\n🎯 步骤 2: 创建适应度函数...")
    fitness_fn = create_fitness_function(
        target_tg=target_tg,
        target_dc=target_dc,
        weight_tg=weight_tg,
        weight_dc=weight_dc
    )
    
    optimization_mode = []
    if target_tg is None:
        optimization_mode.append("最大化 Tg")
    else:
        optimization_mode.append(f"目标 Tg = {target_tg}")
    
    if target_dc is None:
        optimization_mode.append("最大化 DC")
    else:
        optimization_mode.append(f"目标 DC = {target_dc}")
    
    print(f"  优化目标: {', '.join(optimization_mode)}")
    print(f"  权重: Tg={weight_tg}, DC={weight_dc}")
    
    # ------------------------------------------------------------------------
    # 创建 PolyPlanet
    # ------------------------------------------------------------------------
    print("\n🌍 步骤 3: 创建 PolyPlanet...")
    planet = PolyPlanet(
        name=planet_name,
        predict_function=predict_fn,
        fingerprint_function=simple_fingerprint_function,
        models=None,  # 模型已封装在 predict_fn 中
        num_cpus=num_cpus,
        random_seed=random_seed,
        path_to_dna=dna_file,
        save_folder=save_folder,
        species='polymers'
    )
    
    print(f"  Planet 名称: {planet_name}")
    print(f"  DNA 文件: {dna_file}")
    print(f"  染色体数量: {len(planet.chromosomes)}")
    
    # ------------------------------------------------------------------------
    # 创建 PolyLand
    # ------------------------------------------------------------------------
    print("\n🏞️  步骤 4: 创建 PolyLand...")
    from polyga.polygod import PolyLand
    land = PolyLand(
        name=land_name,
        planet=planet,
        fitness_function=fitness_fn,
        generative_function=custom_generative_function,
        crossover_position=crossover_position,
        crossover_sigma_offset=crossover_sigma_offset,
        fraction_mutate_additional_block=mutation_add_block,
        fraction_mutation=mutation_rate
    )
    
    print(f"  Land 名称: {land_name}")
    print(f"  变异率: {mutation_rate}")
    print(f"  添加块变异率: {mutation_add_block}")
    print(f"  交叉位置: {crossover_position}")
    
    # ------------------------------------------------------------------------
    # 创建 PolyNation
    # ------------------------------------------------------------------------
    print("\n🏛️  步骤 5: 创建 PolyNation...")
    
    if initial_population_file and os.path.exists(initial_population_file):
        print(f"  使用初始种群文件: {initial_population_file}")
        nation = PolyNation(
            name=nation_name,
            land=land,
            initial_population_file=initial_population_file,
            num_families=num_families,
            num_parents_per_family=num_parents_per_family,
            num_children_per_family=num_children_per_family,
            partner_selection='random',
            emigration_rate=0.0  # 单个 nation 不需要移民
        )
    else:
        print(f"  生成随机初始种群...")
        print(f"    种群大小: {population_size}")
        print(f"    初始染色体数: {num_chromosomes_initial}")
        nation = PolyNation(
            name=nation_name,
            land=land,
            num_population_initial=population_size,
            num_chromosomes_initial=num_chromosomes_initial,
            num_families=num_families,
            num_parents_per_family=num_parents_per_family,
            num_children_per_family=num_children_per_family,
            partner_selection='random',
            emigration_rate=0.0  # 单个 nation 不需要移民
        )
    
    print(f"  Nation 名称: {nation_name}")
    print(f"  初始种群大小: {len(nation.population)}")
    print(f"  家族数: {num_families}")
    print(f"  每家族父母数: {num_parents_per_family}")
    print(f"  每家族子女数: {num_children_per_family}")
    
    # ------------------------------------------------------------------------
    # 运行进化
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🧬 步骤 6: 开始遗传算法进化")
    print("=" * 80)
    
    for generation in range(num_generations):
        print(f"\n{'=' * 80}")
        print(f"🔄 第 {generation + 1}/{num_generations} 代")
        print(f"{'=' * 80}")
        
        # 评分和预测
        print(f"\n⏳ 正在进行性质预测和适应度评估...")
        nation.score_and_emigrate(narrate=True)
        
        # 显示当前最佳
        if 'fitness' in nation.population.columns:
            best_idx = nation.population['fitness'].idxmax()
            best_polymer = nation.population.loc[best_idx]
            
            print(f"\n✨ 当前最佳聚合物:")
            print(f"   ID: {best_polymer['planetary_id']}")
            print(f"   适应度: {best_polymer['fitness']:.4f}")
            
            if 'Tg' in best_polymer:
                print(f"   Tg: {best_polymer['Tg']:.2f} K")
            if 'DC' in best_polymer:
                print(f"   DC: {best_polymer['DC']:.4f}")
            if 'Eb' in best_polymer:
                print(f"   Eb: {best_polymer['Eb']:.2f}")
            
            print(f"   染色体数: {best_polymer['num_chromosomes']}")
            print(f"   SMILES: {best_polymer['smiles_string'][:60]}...")
            
            # 统计信息
            valid_fitness = nation.population[nation.population['fitness'] > -500]
            if len(valid_fitness) > 0:
                print(f"\n📊 种群统计:")
                print(f"   有效个体: {len(valid_fitness)}/{len(nation.population)}")
                print(f"   平均适应度: {valid_fitness['fitness'].mean():.4f}")
                print(f"   适应度标准差: {valid_fitness['fitness'].std():.4f}")
        
        # 繁殖下一代
        if generation < num_generations - 1:
            print(f"\n👶 正在繁殖第 {generation + 2} 代...")
            nation.propagate_species(narrate=True)
    
    # ------------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("💾 步骤 7: 保存结果")
    print("=" * 80)
    
    # 从数据库读取所有聚合物
    import sqlite3
    import json
    db_path = os.path.join(save_folder, planet_name, 'planetary_database.sqlite')
    conn = sqlite3.connect(db_path)
    results_df = pd.read_sql_query("SELECT * FROM polymer", conn)
    conn.close()
    
    # 解析 JSON 字段
    if 'properties' in results_df.columns:
        # 解析 properties JSON 字段
        props_list = []
        for idx, row in results_df.iterrows():
            try:
                props = json.loads(row['properties'])
                props_list.append(props)
            except:
                props_list.append({})
        
        # 将 properties 字典展开为列
        props_df = pd.DataFrame(props_list)
        results_df = pd.concat([results_df, props_df], axis=1)
    
    # 计算适应度（重新计算以便排序）
    if 'Tg' in results_df.columns:
        fitness_scores = []
        for idx, row in results_df.iterrows():
            score = 0.0
            if 'Tg' in row and not pd.isna(row['Tg']):
                score += weight_tg * (row['Tg'] / 10.0)
            if 'DC' in row and not pd.isna(row['DC']):
                score += weight_dc * row['DC'] * 10.0
            fitness_scores.append(score)
        results_df['fitness'] = fitness_scores
    
    results_csv = os.path.join(save_folder, f'{planet_name}_all_polymers.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ 所有聚合物已保存: {results_csv}")
    print(f"  总数: {len(results_df)}")
    
    # 保存最佳聚合物
    if 'fitness' in results_df.columns:
        # 过滤有效个体
        valid_results = results_df[results_df['fitness'] > -500]
        
        if len(valid_results) > 0:
            top_n = min(20, len(valid_results))
            top_polymers = valid_results.nlargest(top_n, 'fitness')
            
            top_csv = os.path.join(save_folder, f'{planet_name}_top{top_n}.csv')
            top_polymers.to_csv(top_csv, index=False)
            print(f"✓ Top {top_n} 聚合物已保存: {top_csv}")
            
            # 显示 Top 5
            print(f"\n🏆 Top 5 聚合物:")
            for i, (idx, row) in enumerate(top_polymers.head(5).iterrows(), 1):
                print(f"\n  {i}. ID={row['planetary_id']}, 适应度={row['fitness']:.4f}")
                if 'Tg' in row:
                    print(f"     Tg={row['Tg']:.2f} K", end='')
                if 'DC' in row:
                    print(f", DC={row['DC']:.4f}", end='')
                print()
                print(f"     {row['smiles_string'][:60]}...")
    
    print("\n" + "=" * 80)
    print("✅ 遗传算法运行完成！")
    print("=" * 80)
    
    return planet, land, nation, results_df


# ============================================================================
# 5. 命令行入口
# ============================================================================

if __name__ == '__main__':
    # 默认配置运行
    planet, land, nation, results = run_polyga_mmpolymer(
        dna_file='/root/code/MMga/DNA/dna.csv',
        initial_population_file=None,  # 随机生成
        weight_dir='/internfs/Zy/dataset/finetune_data',
        properties=['Tg', 'DC'],
        num_generations=10,
        population_size=30,
        num_chromosomes_initial=4,
        num_families=10,
        num_parents_per_family=2,
        num_children_per_family=2,
        mutation_rate=0.3,
        mutation_add_block=0.1,
        crossover_position='relative_center',
        target_tg=None,  # 最大化
        target_dc=None,  # 最大化
        weight_tg=1.0,
        weight_dc=0.5,
        random_seed=42,
        save_folder='/root/code/MMga/results',
        num_cpus=1
    )

