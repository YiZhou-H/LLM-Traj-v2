# LLM-Traj-v2: 基于大语言模型的轨迹预测

论文架构：视觉编码器 → 轨迹提取 → 文本化 → LLM推理，专注于NuScenes, Waymo Open数据集。

## 项目结构

```
LLM-Traj-New/
├── configs/                 # 配置文件
│   └── config.py           # 主配置文件
├── data/                   # 数据加载器
│   └── nuscenes_loader.py  # NuScenes数据加载器
├── models/                 # 模型定义
│   └── llm_traj_model.py   # 主模型文件（论文架构）
├── training/               # 训练脚本
│   └── train_nuscenes.py   # NuScenes训练脚本
├── evaluation/             # 评估代码
│   ├── metrics.py          # 简化评估指标
│   └── evaluate.py         # 评估脚本
├── utils/                  # 工具函数
│   └── logger.py           # 日志工具
└── requirements.txt        # 依赖包
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

### NuScenes数据集

1. 下载NuScenes数据集
2. 设置环境变量：

```bash
export NUSCENES_ROOT=/path/to/nuscenes/dataset
```

## 模型架构

论文方法的LLM-Traj模型包含以下核心组件：

1. **视觉编码器**: ResNet50直接提取视觉特征并映射到轨迹坐标
2. **轨迹文本转换器**: 将轨迹坐标转换为自然语言描述
3. **大语言模型**: 使用Qwen2.5-0.5B进行文本推理和轨迹预测
4. **LoRA微调**: 使用低秩适应技术高效微调LLM

**核心流程**: 图像 → 视觉编码器 → 轨迹坐标 → 文本化 → LLM推理 → 预测结果

## 训练

### NuScenes数据集训练

```bash
# 基础训练
python training/train_nuscenes.py \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 20 \
    --devices 1 \
    --precision 16-mixed

# 多GPU训练
python training/train_nuscenes.py \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 20 \
    --devices 2 \
    --precision 16-mixed \
    --accelerator gpu
```

## 评估

```bash
python evaluation/evaluate.py \
    --checkpoint_path outputs/checkpoints/best.ckpt \
    --dataset nuscenes
```

## 评估指标

专注于NuScenes核心指标：

- **ADE** (Average Displacement Error): 平均轨迹误差
- **FDE** (Final Displacement Error): 最终位置误差
- **minADE@1**: 单模态最小平均位移误差
- **minFDE@1**: 单模态最小最终位移误差
- **minADE@5**: 多模态前5最小平均位移误差
- **minFDE@5**: 多模态前5最小最终位移误差

## 配置

主要配置参数在`configs/config.py`中：

```python
# 模型配置
model:
  history_length: 20    # 历史轨迹长度
  future_length: 30     # 预测轨迹长度
  
# 数据配置  
data:
  batch_size: 16
  image_size: (224, 224)
  
# 训练配置
training:
  learning_rate: 1e-4
  num_epochs: 20
```

## 技术细节

### 论文方法实现

按照论文公式实现：

1. **公式(1)**: 视觉特征提取 `fv = VisionEncoder(I)`
2. **公式(2)**: 轨迹坐标映射 `(xt, yt) = A * fv + b`
3. **公式(4)**: 轨迹文本化 `TS = F(S)`
4. **公式(5)**: 提示构建 `Prompt = H ⊕ TS ⊕ C`

5. 
### Online Inference
nuScence 

<p align="center">
  <img src="nuScene demo 1.gif" alt="Demo 1" width="45%">
  <img src="nuScene demo 2.gif" alt="Demo 2" width="45%">
</p>

Waymo open

*Note: We use the official 5 Hz sampling from Waymo for visualization, and will later update to a smoother 20 Hz version.*
<p align="center">
  <img src="./Waymo_12s.gif" alt="Waymo Demo" width="600"/>
</p>

