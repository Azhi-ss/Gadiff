#!/usr/bin/env python3
"""
简化启动脚本 - PolyGA + MMPolymer 融合系统

运行配置:
- 初始种群: 随机从 DNA 生成
- 进化代数: 10 代
- 种群大小: 30
"""

import sys
sys.path.insert(0, '/root/code/MMga')

from polyga_mmpolymer_runner import run_polyga_mmpolymer

if __name__ == '__main__':
    print("\n" + "🎬 " * 20)
    print("启动 PolyGA + MMPolymer 融合遗传算法")
    print("🎬 " * 20 + "\n")
    
    # 运行参数
    config = {
        # 数据文件
        'dna_file': '/root/code/MMga/DNA/dna.csv',
        'initial_population_file': None,  # None = 随机生成初始种群
        
        # MMPolymer 模型配置
        'weight_dir': '/internfs/Zy/dataset/finetune_data',
        'properties': ['Tg', 'DC'],  # 预测 Tg 和 DC
        
        # 进化参数
        'num_generations': 10,  # 10 代
        'population_size': 30,  # 初始种群 30 个
        'num_chromosomes_initial': 4,  # 每个聚合物 4 个染色体块
        
        # 遗传操作
        'num_families': 10,  # 10 个家族
        'num_parents_per_family': 2,  # 每个家族 2 个父母
        'num_children_per_family': 2,  # 每个家族 2 个孩子
        'mutation_rate': 0.3,  # 30% 染色体块变异率
        'mutation_add_block': 0.1,  # 10% 概率添加新染色体块
        'crossover_position': 'relative_center',  # 交叉位置策略
        
        # 优化目标
        'target_tg': None,  # None = 最大化 Tg
        'target_dc': None,  # None = 最大化 DC
        'weight_tg': 1.0,  # Tg 权重
        'weight_dc': 0.5,  # DC 权重
        
        # 其他设置
        'planet_name': 'MMPolymer_Planet',
        'land_name': 'MaterialLand',
        'nation_name': 'PolymerNation',
        'random_seed': 42,
        'save_folder': '/root/code/MMga/results',
        'num_cpus': 1
    }
    
    print("📋 运行配置:")
    print(f"  • 初始种群: 随机生成 {config['population_size']} 个")
    print(f"  • 每个聚合物: {config['num_chromosomes_initial']} 个染色体块")
    print(f"  • 进化代数: {config['num_generations']} 代")
    print(f"  • 变异率: {config['mutation_rate'] * 100}%")
    print(f"  • 优化目标: 最大化 Tg 和 DC")
    print(f"  • 结果保存: {config['save_folder']}")
    print()
    
    # 运行
    try:
        planet, land, nation, results = run_polyga_mmpolymer(**config)
        
        print("\n" + "🎉 " * 20)
        print("运行成功完成！")
        print("🎉 " * 20 + "\n")
        
        print("📂 结果文件:")
        print(f"  • 所有聚合物: {config['save_folder']}/{config['planet_name']}_all_polymers.csv")
        print(f"  • Top 聚合物: {config['save_folder']}/{config['planet_name']}_top20.csv")
        print(f"  • 数据库: {config['save_folder']}/{config['planet_name']}/planetary_database.sqlite")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

