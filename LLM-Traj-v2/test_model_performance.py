#!/usr/bin/env python3
"""
测试LLM-Traj模型的推理时间和模型大小
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from models.llm_traj_model import LLMTrajModel
from data.waymo_loader import WaymoDataModule

def get_model_size(model):
    """计算模型大小（MB）"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def create_dummy_batch(batch_size=1, device='cuda'):
    """创建虚拟输入数据"""
    batch = {
        'image': torch.randn(batch_size, 3, 224, 224, device=device),
        'history_trajectory': torch.randn(batch_size, 10, 2, device=device),
        'future_trajectory': torch.randn(batch_size, 80, 2, device=device)
    }
    return batch

def test_inference_time(model, batch, num_runs=100, warmup_runs=10):
    """测试推理时间"""
    model.eval()
    
    # 预热
    print(f"预热 {warmup_runs} 次...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(batch)
    
    # 同步GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 测试推理时间
    print(f"测试推理时间 {num_runs} 次...")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            outputs = model(batch)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"  完成 {i + 1}/{num_runs}")
    
    return times

def main():
    """主测试函数"""
    print("=" * 60)
    print("LLM-Traj 模型性能测试")
    print("=" * 60)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # LLM模型路径
    llm_model_path = "/root/autodl-tmp/models/qwen2.5-0.5b"
    
    print("\n" + "=" * 60)
    print("测试不同配置的模型")
    print("=" * 60)
    
    configs = [
        {
            'name': '完整模型 (LLM + 反馈 + 示例)',
            'use_natural_language': True,
            'use_online_feedback': True,
            'use_dynamic_examples': True
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n测试配置: {config['name']}")
        print("-" * 40)
        
        try:
            # 创建模型
            model = LLMTrajModel(
                llm_model_path=llm_model_path,
                use_natural_language=config['use_natural_language'],
                use_online_feedback=config['use_online_feedback'],
                use_dynamic_examples=config['use_dynamic_examples'],
                learning_rate=1e-4,
                history_length=10,
                future_length=80,
                trajectory_scale=0.6
            ).to(device)
            
            # 计算模型大小和参数
            model_size_mb = get_model_size(model)
            total_params, trainable_params = count_parameters(model)
            
            print(f"模型大小: {model_size_mb:.2f} MB")
            print(f"总参数数: {total_params:,}")
            print(f"可训练参数: {trainable_params:,}")
            
            # 创建测试数据
            batch = create_dummy_batch(batch_size=1, device=device)
            
            # 测试推理时间
            times = test_inference_time(model, batch, num_runs=50, warmup_runs=5)
            
            # 计算统计信息
            mean_time = np.mean(times) * 1000  # 转换为毫秒
            std_time = np.std(times) * 1000
            min_time = np.min(times) * 1000
            max_time = np.max(times) * 1000
            
            print(f"推理时间统计 (ms):")
            print(f"  平均: {mean_time:.2f} ± {std_time:.2f}")
            print(f"  最小: {min_time:.2f}")
            print(f"  最大: {max_time:.2f}")
            print(f"  FPS: {1000/mean_time:.2f}")
            
            results.append({
                'config': config['name'],
                'model_size_mb': model_size_mb,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'mean_time_ms': mean_time,
                'std_time_ms': std_time,
                'fps': 1000/mean_time
            })
            
            # 清理内存
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"测试失败: {e}")
            continue
    
    # 输出汇总结果
    print("\n" + "=" * 80)
    print("性能汇总")
    print("=" * 80)
    
    print(f"{'配置':<25} {'大小(MB)':<10} {'参数数':<12} {'推理时间(ms)':<15} {'FPS':<8}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['config']:<25} "
              f"{result['model_size_mb']:<10.2f} "
              f"{result['total_params']:<12,} "
              f"{result['mean_time_ms']:<15.2f} "
              f"{result['fps']:<8.2f}")
    
    # 保存结果到文件
    with open('/root/autodl-tmp/LLM-Traj-New-copy/model_performance_results.txt', 'w', encoding='utf-8') as f:
        f.write("LLM-Traj 模型性能测试结果\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            f.write(f"配置: {result['config']}\n")
            f.write(f"  模型大小: {result['model_size_mb']:.2f} MB\n")
            f.write(f"  总参数数: {result['total_params']:,}\n")
            f.write(f"  可训练参数: {result['trainable_params']:,}\n")
            f.write(f"  平均推理时间: {result['mean_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms\n")
            f.write(f"  FPS: {result['fps']:.2f}\n\n")
    
    print(f"\n结果已保存到: /root/autodl-tmp/LLM-Traj-New-copy/model_performance_results.txt")

if __name__ == "__main__":
    main()
