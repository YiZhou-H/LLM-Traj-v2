"""
LLM-Traj模型实现 - Waymo专用版本
基于ResNet50+ViT架构的视觉-语言轨迹预测模型
Waymo标准: 10步历史(1秒) + 80步预测(8秒) @ 10Hz
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import re
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        BlipProcessor, BlipForConditionalGeneration,
        BitsAndBytesConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    TaskType = None
import pytorch_lightning as pl
import timm

from configs.config import config

class LightweightViT(nn.Module):
    """轻量级ViT - 只有1层transformer，不影响训练速度"""
    
    def __init__(self, input_dim=2048, num_heads=8, mlp_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # 简单的多头注意力 - 只有1层
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # 轻量级MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, input_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2048) - ResNet50输出
        Returns:
            (B, 2048) - ViT处理后的特征
        """
        # 添加序列维度 (B, 1, 2048)
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
      
        return x.squeeze(1)

class VisionEncoder(nn.Module):

    
    def __init__(self):
        super().__init__()
        # 使用ResNet50作为特征提取器
        self.backbone = timm.create_model('resnet50', pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features  # 2048
        
        # ViT进行特征增强
        self.vit_layer = LightweightViT(input_dim=self.feature_dim)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) 
        Returns:
            visual_features: (B, 2048) - 视觉特征用于后续融合
        """
        # ResNet50特征提取
        cnn_features = self.backbone(images)  # (B, 2048)
        
        # ViT特征增强
        enhanced_features = self.vit_layer(cnn_features)  # (B, 2048)
        
        return enhanced_features

class TrajectoryTextConverter:

    
    @staticmethod
    def trajectory_to_text(trajectory) -> str:
        """
        将轨迹转换为文本描述
        按论文公式(4): TS = F(S) = "在t1至tT内，目标位置为(x1,y1) → ··· → (xT,yT)"
        
        Args:
            trajectory: numpy数组或PyTorch张量，形状为(T, 2)或(B, T, 2)
        Returns:
            str: 单个轨迹的文本描述
        """
        # 处理输入类型
        if isinstance(trajectory, torch.Tensor):
            if len(trajectory.shape) == 3:
                # 批量输入，只处理第一个
                traj = trajectory[0].detach().cpu().numpy()
            else:
                traj = trajectory.detach().cpu().numpy()
        else:
            # 已经是numpy数组
            if len(trajectory.shape) == 3:
                traj = trajectory[0]
            else:
                traj = trajectory
        
        seq_len = traj.shape[0]
        coord_parts = []
        for t in range(seq_len):
            x, y = traj[t]
            coord_parts.append(f"({x:.2f},{y:.2f})")

        text = f"在t1至t{seq_len}内，目标位置为" + " → ".join(coord_parts)
        
        return text
    
    @staticmethod
    def text_to_trajectory(text: str, seq_len: int, device: torch.device) -> torch.Tensor:
        """将文本描述解析为轨迹坐标"""
        pattern = r'\((-?\d+\.?\d*),(-?\d+\.?\d*)\)'
        matches = re.findall(pattern, text)
        
        trajectory = torch.zeros(seq_len, 2, device=device)
        
        for i, (x_str, y_str) in enumerate(matches[:seq_len]):
            try:
                trajectory[i, 0] = float(x_str)
                trajectory[i, 1] = float(y_str)
            except ValueError:
                continue
        
        return trajectory

class LLMTrajModel(pl.LightningModule):
    """论文方法：LLM-Traj主模型 - Waymo专用版本"""
    
    def __init__(
        self,
        llm_model_path: str,
        use_natural_language: bool = True,
        use_online_feedback: bool = True, 
        use_dynamic_examples: bool = True,
        learning_rate: float = 1e-4,
        history_length: int = 10,  # Waymo标准：1秒历史 @ 10Hz
        future_length: int = 80,   # Waymo标准：8秒预测 @ 10Hz
        trajectory_scale: float = 1.0,  # 轨迹缩放参数
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 消融实验开关
        self.use_natural_language = use_natural_language
        self.use_online_feedback = use_online_feedback
        self.use_dynamic_examples = use_dynamic_examples
        

        self.trajectory_scale = trajectory_scale
        self.vision_encoder = VisionEncoder()

        self.text_converter = TrajectoryTextConverter()
        self.tokenizer = None
        self.llm = None
        if TRANSFORMERS_AVAILABLE and self.use_natural_language:
            try:
                # 使用轻量级LLM进行语义增强
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
                self.llm = AutoModelForCausalLM.from_pretrained(
                    llm_model_path, 
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("LLM组件已启用，提供语义增强功能")
            except Exception as e:
                print(f"LLM加载失败，使用纯视觉模式: {e}")
                self.tokenizer = None
                self.llm = None
        else:
            print("LLM组件已禁用，使用纯视觉-轨迹特征融合")
        

        visual_dim = 2048  # VisionEncoder的ResNet50特征维度
        history_dim = history_length * 2  # 使用实际的历史轨迹维度

        self.num_modes = 6  # Waymo标准：6个候选轨迹用于minADE6评估

        llm_dim = 128 if (self.llm is not None and self.use_natural_language) else 0
        fusion_dim = visual_dim + history_dim + llm_dim  # 2048 + 20 + 128 = 2196

        self.visual_projector = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)  # 增加dropout
        )
        
        self.history_projector = nn.Sequential(
            nn.Linear(history_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)  # 增加dropout
        )
        
        # LLM语义特征投影器

        self.llm_projector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        
        fused_dim = 256 + 64 + 64  # 384 (visual + history + llm)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 增加dropout
            nn.Linear(512, 512),  # 保持特征维度
            nn.Dropout(0.2)   # 添加额外dropout
        )
        
      
        self.mode_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),   # 适度dropout防止过拟合
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(inplace=True),  
                nn.Dropout(0.15),
                nn.Linear(128, future_length * 2),  # 输出轨迹坐标
                nn.Tanh()  
            ) for _ in range(self.num_modes)
        ])
        
        self.mode_confidence = nn.Sequential(
            nn.Linear(512, 64),    # 减小置信度网络，专注轨迹回归
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),      # 降低dropout
            nn.Linear(64, self.num_modes)
        )
        
   
        self.visual_bn = nn.LayerNorm(visual_dim) 
        self.history_bn = nn.LayerNorm(history_dim)
        
   
        self._init_trajectory_predictor()

        self.example_bank = []
        self.max_examples = 50  # 减少示例数量
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
    
    def _init_trajectory_predictor(self):

        feature_components = [self.visual_projector, self.history_projector, self.fusion_layer, self.mode_confidence]
        
        for module in feature_components:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
       
        for mode_gen in self.mode_generators:
            for i, layer in enumerate(mode_gen):
                if isinstance(layer, nn.Linear):
                    if i == len(mode_gen) - 1:  
                       
                        nn.init.normal_(layer.weight, mean=0.0, std=0.001)  
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                    else: 
                        nn.init.xavier_normal_(layer.weight, gain=0.3)  
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
        
    def construct_prompt(self, history_trajectory: torch.Tensor) -> List[str]:
        """构建提示 """
        batch_size = history_trajectory.size(0)
        prompts = []
        
        for b in range(batch_size):
            # H: 任务描述
            task_description = "任务：轨迹预测。根据历史轨迹预测未来轨迹。\n"
            
            # TS: 历史轨迹文本化
            history_text = self.text_converter.trajectory_to_text(
                history_trajectory[b].cpu().numpy()
            )
            
            # C: 上下文示例（简化）
            example_text = ""
            if self.use_dynamic_examples and len(self.example_bank) > 0:
                best_example = max(self.example_bank, key=lambda x: x['reward'])
                example_text = f"\n示例：{best_example['input']} → {best_example['output']}\n"
            
            # 完整prompt
            prompt = f"{task_description}{history_text}{example_text}\n预测未来轨迹："
            prompts.append(prompt)
        
        return prompts
    
    def _compute_multimodal_loss(self, pred_traj: torch.Tensor, true_traj: torch.Tensor) -> torch.Tensor:

 
        batch_size, num_modes, seq_len = pred_traj.shape[:3]
        
       
        true_expanded = true_traj.unsqueeze(1).expand(-1, num_modes, -1, -1)
        
     
        diff = pred_traj - true_expanded  # (B, K, T, 2)
        l2_loss = (diff ** 2).sum(dim=-1)  # (B, K, T) - 欧氏距离平方
        

        ade_per_mode = torch.sqrt(l2_loss + 1e-8).mean(dim=2)  # (B, K) - 欧氏距离
        best_ade_loss = torch.min(ade_per_mode, dim=1)[0].mean()  # 最佳ADE

        final_l2 = ((pred_traj[:, :, -1, :] - true_traj[:, -1, :].unsqueeze(1)) ** 2).sum(dim=-1)
        fde_per_mode = torch.sqrt(final_l2 + 1e-8)  # (B, K)
        best_fde_loss = torch.min(fde_per_mode, dim=1)[0].mean()  # 最佳FDE
 
        velocity = diff[:, :, 1:] - diff[:, :, :-1]  # (B, K, T-1, 2)
        smoothness_loss = (velocity ** 2).mean()
        
        # 添加多样性损失 - 确保不同模态产生不同预测
        diversity_loss = 0.0
        if num_modes > 1:
            # 计算不同模态之间的距离
            for i in range(num_modes):
                for j in range(i+1, num_modes):
                    mode_diff = pred_traj[:, i] - pred_traj[:, j]  # (B, T, 2)
                    mode_distance = torch.norm(mode_diff, dim=-1).mean()  # 平均距离
                    # 鼓励模态间有足够的差异，如果差异太小就惩罚
                    diversity_loss += torch.exp(-mode_distance * 10.0)  # 差异越小损失越大

        ade_weight = 2.0      # 适度权重
        fde_weight = 1.8      # 适度权重  
        smooth_weight = 0.2   # 轻微平滑约束
        diversity_weight = 0.1 # 轻微多样性约束
        
        total_loss = (ade_weight * best_ade_loss + 
                     fde_weight * best_fde_loss + 
                     smooth_weight * smoothness_loss + 
                     diversity_weight * diversity_loss)
        
        # 数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(1.0, device=pred_traj.device, requires_grad=True)
        
        return total_loss
    
    def _parse_trajectory_text(self, generated_texts: List[str]) -> torch.Tensor:
        """从LLM生成的文本中解析轨迹坐标"""
        trajectories = []
        seq_len = self.hparams.future_length 
        
        for text in generated_texts:
            # 使用正则表达式提取坐标模式: (x,y) 或 (x, y)
            pattern = r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'
            matches = re.findall(pattern, text)
            
            # 修复设备问题：确保张量在正确的设备上
            traj = torch.zeros(seq_len, 2, device=self.device)
            
            if matches:
                for i, (x_str, y_str) in enumerate(matches[:seq_len]):
                    try:
                        traj[i, 0] = float(x_str)
                        traj[i, 1] = float(y_str)
                    except ValueError:
                        continue
            else:
                # 如果没有找到坐标，尝试其他模式
                numbers = re.findall(r'-?\d+\.?\d*', text)
                if len(numbers) >= 2:
                    for i in range(0, min(len(numbers), seq_len * 2), 2):
                        try:
                            if i + 1 < len(numbers):
                                traj[i//2, 0] = float(numbers[i])
                                traj[i//2, 1] = float(numbers[i+1])
                        except (ValueError, IndexError):
                            continue
            
            trajectories.append(traj)
        
        return torch.stack(trajectories, dim=0)
    
    def _extract_llm_features(self, history_trajectory: torch.Tensor) -> torch.Tensor:
        """使用LLM提取轨迹的语义特征"""
        batch_size = history_trajectory.size(0)
        
        if self.llm is None or not self.use_natural_language:
            return torch.zeros(batch_size, 128, device=history_trajectory.device)
        
        try:
            # 使用完整的提示构建（包含动态示例）
            trajectory_texts = self.construct_prompt(history_trajectory)
            
            # 使用LLM编码文本特征
            with torch.no_grad():
               
                inputs = self.tokenizer(
                    trajectory_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=64  # 限制长度减少计算
                ).to(history_trajectory.device)
                
                # 获取LLM的隐藏状态
                outputs = self.llm(**inputs, output_hidden_states=True)
                # 使用最后一层的平均池化作为特征
                hidden_states = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)
                llm_features = hidden_states.mean(dim=1)  # (B, hidden_dim)
                
                # 投影到固定维度
                if llm_features.size(-1) != 128:
                    if not hasattr(self, 'llm_feature_proj'):
                        self.llm_feature_proj = nn.Linear(llm_features.size(-1), 128).to(
                            device=history_trajectory.device, 
                            dtype=llm_features.dtype
                        )
                    # 确保数据类型匹配
                    llm_features = self.llm_feature_proj(llm_features.to(self.llm_feature_proj.weight.dtype))
                
                # 确保返回float32类型以与模型其他部分兼容
                return llm_features.float()
                
        except Exception as e:
            # 如果LLM处理失败，返回零特征
            print(f"LLM特征提取失败: {e}")
            return torch.zeros(batch_size, 128, device=history_trajectory.device, dtype=torch.float32)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """优化前向传播 - 强化回归预测，降低LLM依赖"""
        images = batch['image']  # (B, C, H, W)
        history_trajectory = batch['history_trajectory']  # (B, T, 2)
        batch_size = history_trajectory.size(0)
        
        # 特征提取
        visual_features = self.vision_encoder(images)  # (B, 2048)
        visual_features = self.visual_bn(visual_features)
        visual_projected = self.visual_projector(visual_features)  # (B, 256)
        
        history_features = history_trajectory.view(batch_size, -1)  # (B, T*2)
        history_features = self.history_bn(history_features)
        history_projected = self.history_projector(history_features)  # (B, 64)
        
      
        if self.llm is not None and self.use_natural_language:
            llm_features = self._extract_llm_features(history_trajectory)  # (B, 128)
            llm_projected = self.llm_projector(llm_features)  # (B, 64)
        else:
            # 如果LLM被禁用，提供零特征以保持维度一致性
            llm_projected = torch.zeros(batch_size, 64, device=history_trajectory.device)
        
        # 多模态特征融合 - 始终使用相同的维度
        combined_features = torch.cat([visual_projected, history_projected, llm_projected], dim=-1)  # (B, 384)
        
        fused_features = self.fusion_layer(combined_features)  # (B, 512)
        

        predicted_trajectories = []
        
        for i in range(self.num_modes):
            mode_gen = self.mode_generators[i]
            
            # 基础特征准备
            mode_features = fused_features
            
            # 适度多样性机制 - 确保6个模态有合理差异
            if i > 0:  # 第0个候选保持原始预测
                # 适度的多样性噪声
                noise_scale = 0.08 * i  # 适度噪声强度
                diversity_noise = torch.randn_like(mode_features) * noise_scale
                mode_features = mode_features + diversity_noise
                
                # 添加轻微的结构化多样性偏置
                if i == 1:  # 略保守的预测
                    mode_features = mode_features * 0.9
                elif i == 2:  # 略激进的预测
                    mode_features = mode_features * 1.1
                elif i >= 3:  # 其他模态使用轻微偏置
                    bias_noise = torch.randn(1, mode_features.size(1), device=mode_features.device) * 0.05
                    mode_features = mode_features + bias_noise
            
      
            trajectory_output = mode_gen(mode_features)  # (B, future_length * 2) 范围[-1,1]
            trajectory = trajectory_output.view(batch_size, self.hparams.future_length, 2)

            trajectory = trajectory * 5.0
            
            predicted_trajectories.append(trajectory)
        
        # 生成模态置信度评估
        confidence_scores = self.mode_confidence(fused_features)  # (B, num_modes)
        
        # 堆叠所有候选轨迹: (B, K, T, 2)
        predicted_trajectory = torch.stack(predicted_trajectories, dim=1)
        
        return {
            'predicted_trajectory': predicted_trajectory  # (B, K, T, 2)
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """多模态训练步骤"""
        outputs = self(batch)
        
        predicted_traj = outputs['predicted_trajectory']  # (B, K, T, 2)
        target_traj = batch['future_trajectory']  # (B, T, 2)
        
        # 多模态损失：最佳候选损失 (Best-of-N loss)
        trajectory_loss = self._compute_multimodal_loss(predicted_traj, target_traj)
        
        self.log('train_loss', trajectory_loss, prog_bar=True, batch_size=batch['future_trajectory'].size(0))
        
        # 在线反馈更新示例库
        if self.use_online_feedback:
            self._update_example_bank(batch, outputs)
        
        return trajectory_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """多模态验证步骤"""
        outputs = self(batch)
        
        # 使用多模态损失
        trajectory_loss = self._compute_multimodal_loss(
            outputs['predicted_trajectory'],  # (B, K, T, 2)
            batch['future_trajectory']       # (B, T, 2)
        )
        
        self.log('val_loss', trajectory_loss, prog_bar=True, batch_size=batch['future_trajectory'].size(0))
        
        return {
            'predicted_trajectory': outputs['predicted_trajectory'],  # (B, K, T, 2)
            'true_trajectory': batch['future_trajectory']           # (B, T, 2)
        }
    
    def _update_example_bank(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]):
        """简化的示例库更新"""
        if not self.use_online_feedback:
            return
        
        pred_traj = outputs['predicted_trajectory']  # (B, K, T, 2)
        true_traj = batch['future_trajectory']  # (B, T, 2)
        
        # 选择每个样本的最佳候选轨迹计算误差
        batch_size, num_modes = pred_traj.shape[:2]
        true_expanded = true_traj.unsqueeze(1).expand(-1, num_modes, -1, -1)
        candidate_errors = torch.norm(pred_traj - true_expanded, dim=-1).mean(dim=-1)  # (B, K)
        errors, best_indices = torch.min(candidate_errors, dim=1)  # (B,)
        
        for b in range(pred_traj.size(0)):
            error = errors[b].item()
            reward = -error
            
            hist_text = self.text_converter.trajectory_to_text(
                batch['history_trajectory'][b:b+1]
            )[0]
            
            # 使用该样本的最佳候选轨迹
            best_pred_traj = pred_traj[b, best_indices[b]].unsqueeze(0)  # (1, T, 2)
            pred_text = self.text_converter.trajectory_to_text(
                best_pred_traj
            )[0].replace("在t1至", "预测t1至")
            
            example = {
                'input': hist_text,
                'output': pred_text,
                'reward': reward
            }
            
            self.example_bank.append(example)
            
            if len(self.example_bank) > self.max_examples:
                self.example_bank.sort(key=lambda x: x['reward'], reverse=True)
                self.example_bank = self.example_bank[:self.max_examples]
    
    def configure_optimizers(self):
        """合理的优化器配置 - 目标1-1.5米精度"""
        # 使用简单的AdamW优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # 使用简单的StepLR调度器
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,   # 每10个epoch衰减一次
            gamma=0.7      # 适度衰减
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
