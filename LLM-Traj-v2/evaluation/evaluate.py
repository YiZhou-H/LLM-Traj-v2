"""
独立评估脚本
支持对训练好的模型进行全面评估
"""
import os
import sys
import argparse
from pathlib import Path
import json
from typing import Dict, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.waymo_loader import WaymoDataModule
from models.llm_traj_model import LLMTrajModel
from evaluation.metrics import TrajectoryMetrics, MetricsVisualizer
from configs.config import config

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Evaluate LLM-Traj Model')
    
    # 基础参数
    parser.add_argument('--dataset', type=str, choices=['waymo'], 
                       default='waymo', help='Dataset to evaluate on (Waymo only)')
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default=None, help='Data root directory')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Data split to evaluate')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--save_predictions', action='store_true', 
                       help='Save prediction results')
    parser.add_argument('--visualize', action='store_true', 
                       help='Generate visualization plots')
    parser.add_argument('--num_vis_samples', type=int, default=10,
                       help='Number of samples to visualize')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    return parser.parse_args()

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(
        self,
        model: LLMTrajModel,
        dataset: str,
        device: str = 'cuda'
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        
        # 设置模型为评估模式
        self.model.eval()
        self.model.to(device)
        
        # 初始化评估指标
        self.metrics = TrajectoryMetrics(dataset=dataset)
        self.visualizer = MetricsVisualizer()
        
    def evaluate_dataloader(
        self, 
        dataloader: torch.utils.data.DataLoader,
        save_predictions: bool = False,
        output_dir: str = None
    ) -> Dict:
        """评估数据加载器"""
        all_predictions = []
        all_ground_truth = []
        all_history = []
        all_metrics = []
        
        print(f"开始评估 {len(dataloader)} 个批次...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                # 将数据移到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # 前向传播
                try:
                    outputs = self.model(batch)
                    pred_traj = outputs['predicted_trajectory']
                except Exception as e:
                    print(f"批次 {batch_idx} 评估失败: {e}")
                    continue
                
                true_traj = batch['future_trajectory']
                hist_traj = batch['history_trajectory']
                
                # 计算批次指标
                batch_metrics = self.metrics.compute_metrics(pred_traj, true_traj)
                all_metrics.append(batch_metrics)
                
                # 保存预测结果
                if save_predictions:
                    all_predictions.append(pred_traj.cpu().numpy())
                    all_ground_truth.append(true_traj.cpu().numpy())
                    all_history.append(hist_traj.cpu().numpy())
        
        # 计算总体指标
        overall_metrics = self._aggregate_metrics(all_metrics)
        
        # 保存结果
        results = {
            'metrics': overall_metrics,
            'num_samples': len(dataloader.dataset),
            'dataset': self.dataset
        }
        
        if save_predictions and output_dir:
            self._save_predictions(
                all_predictions, all_ground_truth, all_history, 
                overall_metrics, output_dir
            )
        
        return results
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """聚合批次指标"""
        if not metrics_list:
            return {}
        
        # 获取所有指标名称
        metric_names = metrics_list[0].keys()
        aggregated = {}
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            if values:
                aggregated[metric_name] = np.mean(values)
        
        return aggregated
    
    def _save_predictions(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray],
        history: List[np.ndarray],
        metrics: Dict,
        output_dir: str
    ):
        """保存预测结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 合并所有批次的数据
        all_pred = np.concatenate(predictions, axis=0)
        all_true = np.concatenate(ground_truth, axis=0)
        all_hist = np.concatenate(history, axis=0)
        
        # 保存为numpy文件
        np.save(os.path.join(output_dir, 'predictions.npy'), all_pred)
        np.save(os.path.join(output_dir, 'ground_truth.npy'), all_true)
        np.save(os.path.join(output_dir, 'history.npy'), all_hist)
        
        # 保存指标
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"预测结果已保存到: {output_dir}")
    
    def visualize_results(
        self,
        dataloader: torch.utils.data.DataLoader,
        output_dir: str,
        num_samples: int = 10
    ):
        """可视化结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if sample_count >= num_samples:
                    break
                
                # 将数据移到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # 前向传播
                try:
                    outputs = self.model(batch)
                    pred_traj = outputs['predicted_trajectory']
                except Exception as e:
                    continue
                
                true_traj = batch['future_trajectory']
                hist_traj = batch['history_trajectory']
                
                # 可视化每个样本
                batch_size = pred_traj.size(0)
                for i in range(min(batch_size, num_samples - sample_count)):
                    pred_np = pred_traj[i].cpu().numpy()
                    true_np = true_traj[i].cpu().numpy()
                    hist_np = hist_traj[i].cpu().numpy()
                    
                    # 生成可视化
                    save_path = os.path.join(output_dir, f'trajectory_{sample_count:03d}.png')
                    self.visualizer.plot_trajectory_comparison(
                        pred_np, true_np, hist_np, save_path
                    )
                    
                    sample_count += 1
                    if sample_count >= num_samples:
                        break
        
        print(f"可视化结果已保存到: {output_dir}")

def main():
    """主评估函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 配置输出目录
    output_dir = args.output_dir or os.path.join(config.output_dir, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"LLM-Traj 模型评估 - {args.dataset.upper()}")
    print("=" * 60)
    print(f"检查点路径: {args.checkpoint_path}")
    print(f"数据集: {args.dataset}")
    print(f"数据分割: {args.split}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 加载模型
    print("加载模型...")
    try:
        # 只支持Waymo训练模块
        from training.train_waymo import WaymoTrainingModule
        model = WaymoTrainingModule.load_from_checkpoint(args.checkpoint_path)
        
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 准备数据
    print("准备数据...")
    data_root = args.data_root or config.data.waymo_root
    
    # 只支持Waymo数据集
    data_module = WaymoDataModule(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    data_module.setup('test')
    
    # 获取数据加载器
    if args.split == 'train':
        dataloader = data_module.train_dataloader()
    elif args.split == 'val':
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()
    
    print(f"数据准备完成! 样本数量: {len(dataloader.dataset)}")
    
    # 创建评估器
    evaluator = ModelEvaluator(model, args.dataset, device)
    
    # 执行评估
    print("\n开始评估...")
    results = evaluator.evaluate_dataloader(
        dataloader,
        save_predictions=args.save_predictions,
        output_dir=output_dir if args.save_predictions else None
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    evaluator.metrics.print_metrics(results['metrics'], f"{args.dataset.upper()} 评估结果")
    
    # 保存评估报告
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n评估报告已保存到: {report_path}")
    
    # 生成可视化
    if args.visualize:
        print(f"\n生成可视化结果 (前{args.num_vis_samples}个样本)...")
        vis_dir = os.path.join(output_dir, 'visualizations')
        evaluator.visualize_results(dataloader, vis_dir, args.num_vis_samples)
    
    print("\n评估完成!")

if __name__ == '__main__':
    main()
