"""
配置文件 - LLM-Traj项目 (Waymo专用版本)
"""
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class ModelConfig:
    """模型配置"""
    # 预训练模型路径
    llm_model_path: str = "/root/autodl-tmp/models/qwen2.5-0.5b"  # 用户已下载的模型路径
    vision_encoder: str = "resnet50"
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # 轨迹相关 - Waymo标准
    history_length: int = 10   # Waymo标准：1秒历史 @ 10Hz = 10步
    future_length: int = 80    # Waymo标准：8秒预测 @ 10Hz = 80步
    coordinate_dim: int = 2    # (x, y)
    num_modes: int = 6         # Waymo标准：6个候选轨迹

@dataclass
class DataConfig:
    """数据配置 - Waymo专用"""
    # 数据路径
    waymo_root: str = "/root/autodl-tmp/waymo_extended"
    
    # 数据处理
    image_size: tuple = (224, 224)
    sample_rate: float = 10.0  # Hz - Waymo标准
    max_samples: Optional[int] = None  # 样本数量限制
    
    # 数据标准化  
    normalize_trajectory: bool = True
    trajectory_scale: float = 1.0  # 修复：使用1.0避免不必要的轨迹缩放
    
    # 数据增强
    use_augmentation: bool = True
    rotation_range: float = 0.05
    translation_range: float = 1.0

@dataclass
class TrainingConfig:
    """训练配置 - Waymo专用"""
    # 基础训练参数
    batch_size: int = 64
    accumulate_grad_batches: int = 2  # 梯度累积
    learning_rate: float = 1e-4
    num_epochs: int = 15
    num_workers: int = 16
    weight_decay: float = 0.01
    
    # PyTorch Lightning
    accelerator: str = "gpu"
    devices: int = 1
    precision: str = "16-mixed"
    
    # 验证和保存 - Waymo标准指标
    val_check_interval: float = 0.5
    save_top_k: int = 3
    monitor: str = "val_minADE6"  # 监控Waymo标准指标
    mode: str = "min"
    
    # 数据加载优化
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    
    # 消融实验开关
    use_natural_language: bool = True      # 是否使用自然语言转换
    use_online_feedback: bool = True       # 是否使用在线反馈
    use_dynamic_examples: bool = True      # 是否动态更新示例

@dataclass
class EvaluationConfig:
    """评估配置 - Waymo专用"""
    # Waymo评估指标  
    waymo_metrics: List[str] = None
    
    def __post_init__(self):
        if self.waymo_metrics is None:
            self.waymo_metrics = [
                "minADE1", "minFDE1", "minADE6", "minFDE6", 
                "MR1", "MR6", "EPA", "avgADE", "avgFDE"
            ]

@dataclass
class Config:
    """总配置类 - Waymo专用"""
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    
    # 项目路径
    project_root: str = "/root/autodl-tmp/LLM-Traj-New-Waymo"
    output_dir: str = "/root/autodl-tmp/LLM-Traj-New-Waymo/outputs"
    log_dir: str = "/root/autodl-tmp/LLM-Traj-New-Waymo/logs"
    
    def __post_init__(self):
        # 创建必要的目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

# 全局配置实例
config = Config()
