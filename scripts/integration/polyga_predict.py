#!/usr/bin/env python3
"""
MMPolymer 预测器适配模块（简化版）

直接使用 /root/code/MMPolymer/polyga_predict.py 中的真实 MMPolymer 预测器
如果无法加载，则回退到模拟预测

配置：
  - USE_REAL_PREDICTION = True:  使用真实 MMPolymer 预测
  - USE_REAL_PREDICTION = False: 使用模拟预测（快速测试）
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List

# ============================================================================
# 配置开关：控制是否使用真实预测
# ============================================================================
USE_REAL_PREDICTION = True  # 设置为 True 启用真实预测，False 使用模拟
# ============================================================================

# 尝试导入真实的 MMPolymer 预测器
# 注意：必须使用绝对路径导入避免循环引用
MMPolymer_available = False
create_real_predictor = None

if USE_REAL_PREDICTION:
    try:
        # 将 MMPolymer 添加到路径
        mmpolymer_path = '/root/code/MMPolymer'
        if mmpolymer_path not in sys.path:
            sys.path.insert(0, mmpolymer_path)
        
        # 导入时明确指定来自 MMPolymer 目录的模块
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mmpolymer_predictor",
            "/root/code/MMPolymer/polyga_predict.py"
        )
        mmpolymer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mmpolymer_module)
        
        create_real_predictor = mmpolymer_module.create_mmpolymer_predictor
        MMPolymer_available = True
        print("✓ 成功加载 MMPolymer 真实预测模块")
    except Exception as e:
        print(f"⚠ 警告: 无法导入 MMPolymer predict 模块，将使用模拟预测")
        print(f"  错误详情: {e}")
        MMPolymer_available = False
        create_real_predictor = None
else:
    print("ℹ 配置：使用模拟预测模式（USE_REAL_PREDICTION = False）")


def create_mmpolymer_predictor(weight_dir: str, 
                               properties: List[str],
                               cache_dir: str = None):
    """
    创建 PolyGA 兼容的预测函数
    
    Args:
        weight_dir: MMPolymer 模型权重目录
        properties: 要预测的性质列表，如 ['Tg', 'DC']
        cache_dir: 缓存目录（可选，仅在模拟模式下使用）
        
    Returns:
        predict_function: PolyGA 兼容的预测函数
    """
    
    # 如果 MMPolymer 可用，使用真实预测
    if MMPolymer_available:
        try:
            print(f"🔬 初始化真实 MMPolymer 预测器...")
            print(f"  权重目录: {weight_dir}")
            print(f"  预测性质: {properties}")
            
            # 调用真实的 MMPolymer 预测器工厂函数
            predict_fn = create_real_predictor(
                weight_dir=weight_dir,
                properties=properties,
                batch_size=32,  # 可以根据 GPU 内存调整
                use_gpu=True,   # 使用 GPU 加速
                cleanup_cache=False  # 保留缓存加速预测
            )
            
            print(f"✅ 真实 MMPolymer 预测器已就绪")
            return predict_fn
            
        except Exception as e:
            print(f"❌ 初始化真实预测器失败: {e}")
            print(f"  回退到模拟预测模式")
            # 继续到模拟预测
    
    # 模拟预测模式
    print(f"🎲 使用模拟预测模式")
    print(f"  性质: {properties}")
    print(f"  注意: 这些是随机值，不是真实预测！")
    
    # 创建简单的缓存
    cache = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'mock_prediction_cache.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                print(f"  已加载模拟缓存: {len(cache)} 条")
            except:
                pass
    else:
        cache_file = None
    
    def mock_predict_function(population_df: pd.DataFrame, 
                             fp_headers: List[str],
                             models: dict = None) -> pd.DataFrame:
        """
        模拟预测函数（仅用于测试）
        
        返回随机值，不是真实预测！
        """
        if 'smiles_string' not in population_df.columns:
            print("⚠ 警告: population_df 中没有 smiles_string 列")
            return population_df
        
        # 为每个 SMILES 生成模拟预测值
        for prop in properties:
            values = []
            for smiles in population_df['smiles_string']:
                # 使用缓存
                cache_key = f"{smiles}_{prop}"
                if cache_key in cache:
                    values.append(cache[cache_key])
                else:
                    # 生成模拟值
                    if prop == 'Tg':
                        val = np.random.uniform(250, 450)
                    elif prop == 'DC':
                        val = np.random.uniform(2.0, 5.0)
                    elif prop == 'Eb':
                        val = np.random.uniform(300, 500)
                    else:
                        val = np.random.uniform(0, 100)
                    
                    cache[cache_key] = val
                    values.append(val)
            
            population_df[prop] = values
        
        # 保存缓存
        if cache_file and len(cache) % 100 == 0:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)
            except:
                pass
        
        return population_df
    
    return mock_predict_function


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("测试 MMPolymer 预测器适配模块")
    print("="*60)
    
    # 创建预测函数
    predict_fn = create_mmpolymer_predictor(
        weight_dir='/internfs/Zy/dataset/finetune_data',
        properties=['Tg', 'DC'],
        cache_dir='/tmp/mmpolymer_test_cache'
    )
    
    # 创建测试 DataFrame
    test_df = pd.DataFrame({
        'smiles_string': [
            '*CC(C)(C(=O)OC)CC(C)(C)*',
            '*CC(C)CC(C)*',
            '*c1ccc(cc1)C(C)(C)*'
        ],
        'planetary_id': [1, 2, 3]
    })
    
    print("\n" + "="*60)
    print("测试 DataFrame:")
    print("="*60)
    print(test_df)
    
    # 运行预测
    print("\n" + "="*60)
    print("运行预测...")
    print("="*60)
    result_df = predict_fn(test_df, [], None)
    
    print("\n" + "="*60)
    print("预测结果:")
    print("="*60)
    print(result_df[['smiles_string', 'Tg', 'DC']])
    
    print("\n✅ 测试完成！")
