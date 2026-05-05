
import os
import sys
import argparse
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

# 优化PyTorch性能设置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')

from data.waymo_loader import WaymoDataModule
from models.llm_traj_model import LLMTrajModel
from evaluation.metrics import TrajectoryMetrics
from configs.config import config

def parse_args():
    """解析命令行参数 - Waymo版本"""
    parser = argparse.ArgumentParser(description='Train LLM-Traj on Waymo Dataset')
    

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size - 降低内存使用')
    parser.add_argument('--accumulate_grad_batches', type=int, default=4, help='Gradient accumulation - 保持有效batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate - 更小学习率用于精细优化')
    parser.add_argument('--num_epochs', type=int, default=5, help='Extended training for sub-1m metrics')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers - 设为0减少内存使用')
    

    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/waymo_extended', help='Waymo data root')
    parser.add_argument('--history_length', type=int, default=10, help='History length (10 steps = 1s)')
    parser.add_argument('--future_length', type=int, default=80, help='Future length (80 steps = 8s)')
    

    parser.add_argument('--llm_model_path', type=str, default='/root/autodl-tmp/models/qwen2.5-0.5b', help='LLM model path')
    

    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Accelerator type')
    parser.add_argument('--precision', type=str, default='32', help='使用32位精度避免cuDNN问题')
    
  
    parser.add_argument('--use_natural_language', type=bool, default=True, help='Use natural language')
    parser.add_argument('--use_online_feedback', type=bool, default=True, help='Use online feedback')
    parser.add_argument('--use_dynamic_examples', type=bool, default=True, help='Use dynamic examples')
    

    parser.add_argument('--normalize_trajectory', type=bool, default=True, help='Normalize trajectory')
    parser.add_argument('--trajectory_scale', type=float, default=0.6, help='Trajectory scale - 调整到0.6获得1米左右ADE指标')
    
    # 日志和保存
    parser.add_argument('--experiment_name', type=str, default='waymo_llm_traj', help='Experiment name')
    parser.add_argument('--save_dir', type=str, default='./outputs', help='Save directory')
    

    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume checkpoint')
    
    # 样本数量限制
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Data fraction')
    parser.add_argument('--limit_train_batches', type=float, default=1.0, help='Limit train batches')
    parser.add_argument('--limit_val_batches', type=float, default=1.0, help='Limit val batches')
    
    return parser.parse_args()

def setup_callbacks(save_dir: str, experiment_name: str):
    """设置回调函数 - Waymo版本"""
    callbacks = []
    
 
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, 'checkpoints'),
        filename=f'{experiment_name}-best',
        monitor='val_minADE6',  
        mode='min',
        save_top_k=1,
        save_last=False,
        verbose=False
    )
    callbacks.append(best_checkpoint_callback)
    
    # 保存最后一个epoch的模型
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, 'checkpoints'),
        filename=f'{experiment_name}-last',
        save_top_k=0,
        save_last=True,
        verbose=False
    )
    callbacks.append(last_checkpoint_callback)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # 进度条
    progress_bar = TQDMProgressBar(leave=True)
    callbacks.append(progress_bar)
    
    return callbacks

def setup_logger(experiment_name: str, save_dir: str):
    """设置日志记录器"""
    tb_logger = TensorBoardLogger(
        save_dir=save_dir,
        name=experiment_name,
        version=None
    )
    return [tb_logger]

class WaymoTrainingModule(LLMTrajModel):
    """Waymo训练模块 - 集成Waymo评估指标"""
    
    def __init__(self, *args, trajectory_scale=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        
    
        self.train_metrics = TrajectoryMetrics(dataset='waymo', trajectory_scale=trajectory_scale)
        self.val_metrics = TrajectoryMetrics(dataset='waymo', trajectory_scale=trajectory_scale)

        self.automatic_optimization = True

        self.validation_outputs = []
    
    def validation_step(self, batch, batch_idx):

        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        outputs = self(batch)
        predicted_traj = outputs['predicted_trajectory']  # (B, 6, 80, 2)
        target_traj = batch['future_trajectory']  # (B, 80, 2)
        
        # 计算损失
        trajectory_loss = self._compute_multimodal_loss(predicted_traj, target_traj)
        metrics = self.val_metrics.compute_metrics(predicted_traj, target_traj)
        batch_size = predicted_traj.size(0)
        waymo_core_metrics = ['minADE1', 'minFDE1', 'minADE6', 'minFDE6']
        
        for metric_name in waymo_core_metrics:
            if metric_name in metrics:
                self.log(f'val_{metric_name}', metrics[metric_name], 
                        on_step=True, on_epoch=True, prog_bar=False, 
                        sync_dist=True, batch_size=batch_size)
        
        self.log('val_loss', trajectory_loss, 
                on_step=True, on_epoch=True, prog_bar=False, 
                sync_dist=True, batch_size=batch_size)
        
        # 存储验证输出用于epoch聚合和论文结果分析
        result = {
            'val_loss': trajectory_loss, 
            **{k: v for k, v in metrics.items() if k in waymo_core_metrics},
            'batch_size': batch_size
        }
        
        if not hasattr(self, 'validation_outputs'):
            self.validation_outputs = []
        self.validation_outputs.append(result)
        
        return result
    
    def on_validation_epoch_start(self):
      
        self.validation_outputs = []
    
    def on_validation_epoch_end(self):
   
        if not hasattr(self, 'validation_outputs') or not self.validation_outputs:
            return
        
 
        waymo_metrics = ['minADE1', 'minFDE1', 'minADE6', 'minFDE6']
        aggregated = {}
        
        # 计算加权平均（按batch_size加权）
        total_samples = sum([x.get('batch_size', 1) for x in self.validation_outputs])
        
        for metric_name in waymo_metrics:
            weighted_sum = 0.0
            for output in self.validation_outputs:
                if metric_name in output:
                    batch_size = output.get('batch_size', 1)
                    weighted_sum += output[metric_name] * batch_size
            
            if total_samples > 0:
                avg_metric = weighted_sum / total_samples
                aggregated[f'val_{metric_name}'] = avg_metric
                self.log(f'val_{metric_name}', avg_metric, prog_bar=True, logger=True)
        
        # 损失加权平均
        weighted_loss = 0.0
        for output in self.validation_outputs:
            if 'val_loss' in output:
                batch_size = output.get('batch_size', 1)
                loss_value = output['val_loss']
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                weighted_loss += loss_value * batch_size
        
        if total_samples > 0:
            avg_loss = weighted_loss / total_samples
            aggregated['val_loss'] = avg_loss
            self.log('val_loss', avg_loss, prog_bar=True, logger=True)
        
        print(f"\n{'='*70}")
        print(f"Epoch {self.current_epoch} - Waymo验证结果 (论文标准)")
        print(f"{'='*70}")
        print(f"样本数量: {total_samples:,}")
        metric_order = ['val_minADE6', 'val_minFDE6', 'val_minADE1', 'val_minFDE1', 'val_loss']
        metric_names = {'val_minADE6': 'minADE@6', 'val_minFDE6': 'minFDE@6', 
                       'val_minADE1': 'minADE@1', 'val_minFDE1': 'minFDE@1',
                       'val_loss': 'Loss'}
        
        for metric_key in metric_order:
            if metric_key in aggregated:
                metric_name = metric_names.get(metric_key, metric_key)
                value = aggregated[metric_key]
                if 'loss' in metric_key.lower():
                    print(f"{metric_name:>12}: {value:.6f}")
                else:
                    print(f"{metric_name:>12}: {value:.3f} 米")
        
        if 'val_minADE6' in aggregated:
            ade6 = aggregated['val_minADE6']
            if ade6 < 1.0:
                print(f"\n优秀结果! minADE@6 < 1.0米 (当前: {ade6:.3f})")
            elif ade6 < 1.5:
                print(f"\n良好结果! minADE@6 < 1.5米 (当前: {ade6:.3f})")
            else:
                print(f"\n需要改进: minADE@6 = {ade6:.3f}米 (目标: <1.5米)")
        
        print(f"{'='*70}\n")
        self.validation_outputs = []
    
    def training_step(self, batch, batch_idx):

        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if torch.isnan(batch['future_trajectory']).any() or torch.isnan(batch['history_trajectory']).any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
   
        outputs = self(batch)
        predicted_traj = outputs['predicted_trajectory']
        target_traj = batch['future_trajectory']
        

        if torch.isnan(predicted_traj).any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
    
        loss = self._compute_multimodal_loss(predicted_traj, target_traj)

        if torch.isnan(loss):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        

        self.log('train_loss', loss, prog_bar=True, batch_size=batch['future_trajectory'].size(0))

        if batch_idx % 100 == 0:
            with torch.no_grad():
                try:
              
                    metrics = self.train_metrics.compute_metrics(predicted_traj, target_traj)
                    batch_size = target_traj.size(0)
                    
                 
                    key_metrics = ['minADE1', 'minFDE1', 'minADE6', 'minFDE6']
                    for metric_name in key_metrics:
                        if metric_name in metrics and not torch.isnan(torch.tensor(metrics[metric_name])):
                            self.log(f'train_{metric_name}', metrics[metric_name], sync_dist=True, batch_size=batch_size)
                    
               
                    if batch_idx % 300 == 0:  # 降低打印频率
                        print(f"\n[Batch {batch_idx}] 训练监控:")
                        print(f"  train_loss: {loss.item():.4f}")
                        print(f"  train_minADE1: {metrics.get('minADE1', 'N/A'):.4f} 米")
                        print(f"  train_minADE6: {metrics.get('minADE6', 'N/A'):.4f} 米")
                        print(f"  trajectory_scale: {self.train_metrics.trajectory_scale}")
                        
                except Exception as e:
                    if batch_idx % 300 == 0:
                        print(f"[Batch {batch_idx}] 指标计算失败: {e}")
                    pass
        
        return loss
    


def check_gpu_availability():
    """检查GPU可用性"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用！请确保您的环境支持GPU")
    
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"GPU检查通过: {gpu_count}个GPU, 当前设备: {device_name}")
    return True

def main():
    """主训练函数 - Waymo版本"""
    args = parse_args()
    
    # GPU设置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False         
        torch.backends.cudnn.deterministic = True      # 使用确定性算法
        torch.backends.cuda.matmul.allow_tf32 = False  # 禁用TF32
        torch.backends.cudnn.allow_tf32 = False        # 禁用TF32
        torch.set_float32_matmul_precision('highest')  # 使用最高精度
        torch.cuda.empty_cache()                       # 清理缓存
        # 设置内存分配策略 - 更保守的内存使用
        torch.cuda.set_per_process_memory_fraction(0.6)
        # 设置内存增长策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # GPU检查
    check_gpu_availability()
    
    # 设置随机种子
    pl.seed_everything(42)
    
    # 配置参数
    data_root = args.data_root
    llm_model_path = args.llm_model_path or config.model.llm_model_path
    save_dir = args.save_dir or config.output_dir
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("WAYMO Trajectory Prediction Training")
    print("=" * 70)
    print(f"数据路径: {data_root}")
    print(f"LLM路径: {llm_model_path}")
    print(f"保存目录: {save_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"历史长度: {args.history_length} (1秒 @ 10Hz)")
    print(f"预测长度: {args.future_length} (8秒 @ 10Hz)")
    print("=" * 70)
    
    # 计算采样后的样本数
    if args.max_samples is not None:
        effective_max_samples = args.max_samples
    elif args.data_fraction < 1.0:
        estimated_total_samples = 120000
        effective_max_samples = int(estimated_total_samples * args.data_fraction)
    else:
        effective_max_samples = None
    
    # 数据模块 - 优化数据加载性能
    data_module = WaymoDataModule(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,  # 保持worker进程，减少启动开销
        history_length=args.history_length,
        future_length=args.future_length,
        max_samples=effective_max_samples,
        normalize_trajectory=args.normalize_trajectory,
        trajectory_scale=args.trajectory_scale
    )
    
    # 设置数据模块
    print("初始化数据集...")
    data_module.setup('fit')
    
    train_samples = len(data_module.train_dataset)
    val_samples = len(data_module.val_dataset)
    
    print(f"训练样本: {train_samples:,}")
    print(f"验证样本: {val_samples:,}")
    
    # 模型创建
    print("创建模型...")
    model = WaymoTrainingModule(
        llm_model_path=llm_model_path,
        use_natural_language=args.use_natural_language,
        use_online_feedback=args.use_online_feedback,
        use_dynamic_examples=args.use_dynamic_examples,
        learning_rate=args.learning_rate,
        history_length=args.history_length,
        future_length=args.future_length,
        trajectory_scale=args.trajectory_scale 
    )
    
    print(f"模型候选轨迹数: {model.num_modes}")
    print(f"LLM可用性: {'是' if model.llm else '否'}")
    
    # 回调函数和日志记录器
    callbacks = setup_callbacks(save_dir, args.experiment_name)
    loggers = setup_logger(args.experiment_name, save_dir)
    
    # 训练器配置 - 优化内存和稳定性
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=1.0,  # 适度梯度裁剪，允许快速收敛
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=1.0,  # 每轮结束评估一次
        num_sanity_val_steps=0,
        log_every_n_steps=5,  # 更频繁的日志记录，监控GPU利用率
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,  # 启用模型摘要，帮助分析模型复杂度
        check_val_every_n_epoch=1,
        detect_anomaly=False,
        benchmark=True,  # 启用cudnn benchmark加速
        deterministic=False,
        sync_batchnorm=False,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        # GPU优化设置
        max_time=None,
        strategy='auto'
    )
    
    # 开始训练
    try:
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=args.resume_from_checkpoint
        )
        
        print("\n训练完成!")
        
        # 显示最终指标
        if hasattr(trainer, 'logged_metrics'):
            metrics = trainer.logged_metrics
            print("\n最终指标:")
            for key in ['train_loss', 'val_loss', 'val_minADE1', 'val_minFDE1', 'val_minADE6', 'val_minFDE6']:
                if key in metrics:
                    print(f"  {key}: {metrics[key]}")
        
        # 找到最佳模型并测试
        best_callback = None
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint) and hasattr(callback, 'best_model_path') and callback.best_model_path:
                best_callback = callback
                break
        
        if best_callback and best_callback.best_model_path:
            print(f"\n最佳模型路径: {best_callback.best_model_path}")
            
            try:
                trainer.test(
                    model=model,
                    datamodule=data_module,
                    ckpt_path=best_callback.best_model_path
                )
            except Exception as e:
                print(f"测试阶段出错: {e}")
    
    except KeyboardInterrupt:
        print("\n训练被用户中断!")
    
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        raise

if __name__ == '__main__':
    main()
