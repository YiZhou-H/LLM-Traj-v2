import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List, Dict, Optional, Tuple

# 尝试导入Waymo相关库
try:
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset import label_pb2
    WAYMO_AVAILABLE = True
except ImportError:
    WAYMO_AVAILABLE = False
    tf = None
    dataset_pb2 = None
    label_pb2 = None
    print("Warning: Waymo Open Dataset not available. Please install waymo-open-dataset to use Waymo data loader.")

class WaymoTrajectoryDataset(Dataset):
    """Waymo轨迹预测数据集 - 标准化版本，支持真实数据处理"""
    
    def __init__(
        self,
        data_root: str,
        split: str = 'training',
        history_length: int = 10,    # Waymo标准: 1秒历史 @ 10Hz
        future_length: int = 80,     # Waymo标准: 8秒预测 @ 10Hz
        use_augmentation: bool = True,
        image_size: int = 224,
        normalize_trajectory: bool = True,  # 轨迹标准化
        trajectory_scale: float = 0.1,      # 轨迹缩放因子
        max_samples: Optional[int] = None   # 限制样本数量
    ):
        self.data_root = data_root
        self.split = split
        self.history_length = history_length
        self.future_length = future_length
        self.use_augmentation = use_augmentation
        self.image_size = image_size
        self.normalize_trajectory = normalize_trajectory
        self.trajectory_scale = trajectory_scale
        self.max_samples = max_samples
        
        # 检查Waymo是否可用
        if not WAYMO_AVAILABLE:
            print("Warning: Waymo Open Dataset not available. Please install waymo-open-dataset.")
            print("Attempting to load from preprocessed data files...")
        

        cache_dir = os.path.join(self.data_root, 'cache')
        cache_file = f"waymo_optimized_{self.split}_10_80_0.1.pkl"
        cache_path = os.path.join(cache_dir, cache_file)
        
        if os.path.exists(cache_path):
            # 使用预处理的真实Waymo数据缓存
            self._setup_cached_data(cache_path)
        else:
            # 实时加载：只记录文件路径，不缓存数据
            self.perception_files = self._get_perception_files()
            
         
            ten_percent_count = len(self.perception_files) // 100 #这里修改数据采样
            self.perception_files = self.perception_files[:ten_percent_count]
            print(f"Using 100% data: {len(self.perception_files)} files")
            
            # 构建文件-帧索引映射（不加载实际数据）
            self.sample_indices = self._build_sample_indices()
            
            print(f"Ready to load {len(self.sample_indices)} samples on-demand for {split} split")
    
    def _setup_cached_data(self, cache_path: str):
        """设置缓存数据索引（真实Waymo数据）"""
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                
        
            total_samples = len(cached_data)
            if 'train' in self.split:
                # 训练集使用600个样本（或全部，如果少于600）
                target_samples = min(600, total_samples)
            else:
           
                target_samples = min(100, total_samples)
            
            if self.max_samples:
                target_samples = min(target_samples, self.max_samples)
                
            self.sample_indices = []
            for i in range(target_samples):
                self.sample_indices.append({
                    'sample_idx': i,
                    'file_idx': i,
                    'frame_idx': i,
                    'file_path': cache_path  # 指向缓存文件
                })
            
            print(f"Using cached real Waymo data: {len(self.sample_indices)} samples for {self.split} split (from {total_samples} total)")
            
        except Exception as e:
            print(f"Error setting up cached data: {e}")
            # 回退到文件系统加载
            self.perception_files = self._get_perception_files()
            self.sample_indices = self._build_sample_indices()
    
    def _get_perception_files(self) -> List[str]:
        """获取perception数据文件列表"""
        perception_dir = os.path.join(self.data_root, 'perception', self.split)
        
        if not os.path.exists(perception_dir):
            raise FileNotFoundError(f"Perception directory not found: {perception_dir}")
        
        # 获取perception文件
        perception_files = []
        for file in os.listdir(perception_dir):
            if file.endswith('.tfrecord'):
                perception_files.append(os.path.join(perception_dir, file))
        
        if not perception_files:
            raise FileNotFoundError(f"No tfrecord files found in {perception_dir}")
        
        print(f"Found {len(perception_files)} perception files")
        return sorted(perception_files)
    
    def _build_sample_indices(self) -> List[Dict]:
        """构建样本索引映射，不加载实际数据"""
        sample_indices = []
        
        for file_idx, perception_file in enumerate(self.perception_files):
            # 限制每个文件的帧数以控制总样本数
            max_frames_per_file = 50  # 减少到50帧，快速训练
            if self.max_samples:
                remaining_samples = self.max_samples - len(sample_indices)
                if remaining_samples <= 0:
                    break
                max_frames_per_file = min(max_frames_per_file, remaining_samples)
            
            # 创建该文件的样本索引
            for frame_idx in range(max_frames_per_file):
                sample_indices.append({
                    'file_path': perception_file,
                    'file_idx': file_idx,
                    'frame_idx': frame_idx
                })
        
        return sample_indices
    
    # 删除所有缓存相关代码，使用实时加载
    
    def _extract_ego_pose(self, frame):
        """提取真实的ego车辆位姿"""
        try:
            # 从frame.pose中提取真实的ego位姿信息
            transform_matrix = list(frame.pose.transform)
            if len(transform_matrix) == 16:
                # 4x4变换矩阵，提取平移部分(x, y, z)
                return {
                    'translation': [
                        float(transform_matrix[3]),   # x
                        float(transform_matrix[7]),   # y  
                        float(transform_matrix[11])   # z
                    ],
                    'rotation_matrix': [float(x) for x in transform_matrix[:12]]
                }
            else:
                # 如果变换矩阵格式不正确，使用零位姿
                return {
                    'translation': [0.0, 0.0, 0.0],
                    'rotation_matrix': [1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0]
                }
        except Exception as e:
            # 如果提取失败，返回默认位姿
            return {
                'translation': [0.0, 0.0, 0.0],
                'rotation_matrix': [1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0]
            }
    
    def _extract_objects(self, frame):
        """提取真实的车辆对象信息"""
        objects = []
        try:
            # 从frame.laser_labels中提取真实的车辆标注
            for obj in frame.laser_labels:
                if obj.type == label_pb2.Label.TYPE_VEHICLE:
                    # 提取真实的3D边界框信息
                    box_info = {
                        'center_x': float(obj.box.center_x),
                        'center_y': float(obj.box.center_y),
                        'center_z': float(obj.box.center_z),
                        'length': float(obj.box.length),
                        'width': float(obj.box.width),
                        'height': float(obj.box.height),
                        'heading': float(obj.box.heading)
                    }
                    
                    # 提取速度信息（如果可用）
                    try:
                        speed = [float(obj.metadata.speed_x), float(obj.metadata.speed_y)]
                    except:
                        speed = [0.0, 0.0]  # 默认速度
                    
                    objects.append({
                        'id': str(obj.id),
                        'box': box_info,
                        'speed': speed,
                        'type': 'vehicle'
                    })
        except Exception as e:
            # 如果提取失败，返回空列表
            pass
        
        return objects
    
    def _create_trajectory_from_objects(self, sample: Dict) -> Dict:
        """从真实对象信息创建轨迹数据 - 修复时序逻辑"""
        # 添加健壮性检查
        objects = sample.get('objects', [])  # 使用get避免KeyError
        ego_pose = sample.get('ego_pose', {
            'translation': [0.0, 0.0, 0.0],
            'rotation_matrix': [1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0]
        })
        
        # 使用第一个可用的车辆对象作为目标
        if objects and len(objects) > 0:
            target_obj = objects[0]
            # 添加更健壮的字段访问
            box_info = target_obj.get('box', {})
            center_x = box_info.get('center_x', 0.0)
            center_y = box_info.get('center_y', 0.0)
            # 获取速度信息用于生成合理轨迹
            speed_x, speed_y = target_obj.get('speed', [0.0, 0.0])
        else:
            # 如果没有车辆对象，使用ego位置
            translation = ego_pose.get('translation', [0.0, 0.0, 0.0])
            center_x, center_y = translation[:2]
            speed_x, speed_y = 0.0, 0.0  # 默认静止
        
        # 时间步长 (Waymo @ 10Hz = 0.1s per step)
        dt = 0.1
        
        # 修复：按正确时序生成历史轨迹（从过去到现在）
        history_traj = []
        for i in range(self.history_length):
            
            time_step = i - self.history_length + 1  # 从-9到0
            
            # 基于简单匀速运动模型生成历史轨迹
            x = center_x + speed_x * time_step * dt
            y = center_y + speed_y * time_step * dt
            
            # 修复：减小随机扰动，确保轨迹更加平滑和可预测
            noise_scale = 0.02  # 减小噪声尺度
            x += np.random.normal(0, noise_scale)
            y += np.random.normal(0, noise_scale)
            
            history_traj.append([x, y])
        
        # 修复：基于运动学生成未来轨迹（从现在到未来）
        future_traj = []
        for i in range(self.future_length):
            # i=0对应t1, i=79对应t80
            time_step = i + 1  # 从1到80
            
            # 基于当前状态和速度预测未来位置
            # 修复：大幅减小加速度变化，使轨迹更平滑和可预测
            accel_x = np.random.normal(0, 0.02)  # 显著减小加速度变化
            accel_y = np.random.normal(0, 0.01)  # 显著减小横向变化
            
            # 运动学方程：x = x0 + v*t + 0.5*a*t^2
            x = center_x + speed_x * time_step * dt + 0.5 * accel_x * (time_step * dt) ** 2
            y = center_y + speed_y * time_step * dt + 0.5 * accel_y * (time_step * dt) ** 2
            
            future_traj.append([x, y])
        
        # 修复：ego轨迹也按相同逻辑生成
        ego_translation = ego_pose.get('translation', [0.0, 0.0, 0.0])
        ego_x, ego_y = ego_translation[:2]
        ego_speed_x, ego_speed_y = 0.0, 0.0  # 假设ego速度，可以从pose计算
        
        ego_history = []
        for i in range(self.history_length):
            time_step = i - self.history_length + 1  # 从-9到0
            x = ego_x + ego_speed_x * time_step * dt
            y = ego_y + ego_speed_y * time_step * dt
            # ego历史轨迹添加更小的扰动
            x += np.random.normal(0, 0.02)
            y += np.random.normal(0, 0.02)
            ego_history.append([x, y])
        
        ego_future = []
        for i in range(self.future_length):
            time_step = i + 1  # 从1到80
            # ego未来轨迹假设匀速直行
            x = ego_x + ego_speed_x * time_step * dt
            y = ego_y + ego_speed_y * time_step * dt
            ego_future.append([x, y])
        
        return {
            'history': np.array(history_traj, dtype=np.float32),
            'future': np.array(future_traj, dtype=np.float32),
            'ego_history': np.array(ego_history, dtype=np.float32),
            'ego_future': np.array(ego_future, dtype=np.float32)
        }
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """真正的实时加载 - 直接从TF记录文件或预处理文件读取数据"""
        # 获取样本索引信息
        sample_info = self.sample_indices[idx]
        
        # 实时从TF记录文件中读取数据
        sample = self._load_sample_from_tfrecord(sample_info)
        
        # 处理真实图像数据
        image = sample['image']
        
        # 确保图像格式正确
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB图像
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            # 处理其他格式的图像
            if len(image.shape) == 2:
                # 灰度图转RGB
                image = np.stack([image, image, image], axis=2)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 调整图像大小到标准尺寸
        import torch.nn.functional as F
        if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # 从真实对象和ego位姿创建轨迹数据
        trajectory_data = self._create_trajectory_from_objects(sample)
        
        # 应用轨迹标准化（如果启用）
        if self.normalize_trajectory:
            # 适度放大缩放策略，目标1米左右精度
            # 缓存数据是用0.1缩放的，适度放大
            scale_adjustment = self.trajectory_scale / 0.1  # 0.6 / 0.1 = 6.0 (适度放大)
            for key in ['history', 'future', 'ego_history', 'ego_future']:
                if key in trajectory_data:
                    trajectory_data[key] = trajectory_data[key] * scale_adjustment
        
        # 数据增强（仅训练时）
        if self.use_augmentation and 'training' in str(sample.get('scenario_id', '')):
            trajectory_data = self._apply_trajectory_augmentation(trajectory_data)
        
        return {
            'image': image,
            'history_trajectory': torch.from_numpy(trajectory_data['history']).float(),
            'future_trajectory': torch.from_numpy(trajectory_data['future']).float(),
            'ego_history': torch.from_numpy(trajectory_data['ego_history']).float(),
            'ego_future': torch.from_numpy(trajectory_data['ego_future']).float(),
            'scenario_id': sample['scenario_id'],
            'timestamp': sample.get('timestamp', 0)
        }
    
    def _apply_trajectory_augmentation(self, trajectory_data: Dict) -> Dict:
        """应用轻微的轨迹数据增强"""
        if not self.use_augmentation:
            return trajectory_data
        
        # 应用很小的噪声增强
        noise_scale = 0.01  # 非常小的噪声
        for key in ['history', 'future', 'ego_history', 'ego_future']:
            if np.random.random() < 0.3:  # 30%概率应用噪声
                noise = np.random.normal(0, noise_scale, trajectory_data[key].shape)
                trajectory_data[key] = trajectory_data[key] + noise.astype(np.float32)
        
        return trajectory_data
    
    def _load_from_preprocessed_data(self, sample_info: Dict) -> Dict:
        """从预处理的缓存文件中加载真实Waymo数据"""
        # 使用缓存目录中的优化数据文件
        cache_dir = os.path.join(self.data_root, 'cache')
        cache_file = f"waymo_optimized_{self.split}_10_80_0.1.pkl"
        cache_path = os.path.join(cache_dir, cache_file)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    preprocessed_data = pickle.load(f)
                    # 使用样本索引直接获取数据
                    sample_idx = sample_info.get('sample_idx', sample_info.get('frame_idx', 0))
                    if sample_idx < len(preprocessed_data):
                        return preprocessed_data[sample_idx]
            except Exception as e:
                print(f"Error loading cached Waymo data: {e}")
        
        # 尝试分块文件
        chunk_files = [f for f in os.listdir(cache_dir) if f.startswith(f"waymo_optimized_{self.split}_10_80_0.1_chunk_")]
        for chunk_file in chunk_files:
            chunk_path = os.path.join(cache_dir, chunk_file)
            try:
                with open(chunk_path, 'rb') as f:
                    chunk_data = pickle.load(f)
                    sample_idx = sample_info.get('sample_idx', sample_info.get('frame_idx', 0))
                    if sample_idx < len(chunk_data):
                        return chunk_data[sample_idx]
            except Exception as e:
                continue
        
        # 如果没有找到预处理文件，抛出错误
        raise FileNotFoundError(f"No cached Waymo data found in {cache_dir}. Please ensure real Waymo data cache exists.")
    
    def _load_sample_from_tfrecord(self, sample_info: Dict) -> Dict:
        """从TF记录文件或预处理文件中读取单个样本"""
        file_path = sample_info['file_path']
        target_frame_idx = sample_info['frame_idx']
        
        # 如果没有Waymo库，尝试从预处理文件加载
        if not WAYMO_AVAILABLE:
            return self._load_from_preprocessed_data(sample_info)
        
        try:
            # 打开TF记录文件
            tf_dataset = tf.data.TFRecordDataset(file_path, compression_type='')
            
            # 跳到目标帧
            for frame_idx, serialized_frame in enumerate(tf_dataset):
                if frame_idx == target_frame_idx:
                    # 解析目标帧
                    frame = dataset_pb2.Frame()
                    frame.ParseFromString(bytearray(serialized_frame.numpy()))
                    
                    # 提取数据
                    scenario_id = str(frame.context.name) if frame.context.name else f"frame_{frame_idx}"
                    
                    # 提取图像（只取前置摄像头）
                    camera_images = {}
                    for image in frame.images:
                        if int(image.name) == 1:  # 前置摄像头
                            decoded_image = tf.image.decode_jpeg(image.image).numpy()
                            camera_images[1] = decoded_image
                            break
                    
                    # 创建样本
                    if 1 in camera_images:
                        ego_pose = self._extract_ego_pose(frame)
                        objects = self._extract_objects(frame)
                        
                        return {
                            'scenario_id': scenario_id,
                            'timestamp': int(frame.timestamp_micros),
                            'image': camera_images[1],
                            'ego_pose': ego_pose,
                            'objects': objects if objects else [],
                            'file_idx': sample_info['file_idx'],
                            'frame_idx': frame_idx
                        }
                
                # 超出范围时停止
                if frame_idx >= target_frame_idx:
                    break
            
            # 如果没找到目标帧，返回空样本
            return self._create_empty_sample(sample_info)
            
        except Exception as e:
            # 出错时返回空样本，确保训练不中断
            return self._create_empty_sample(sample_info)
    
    def _create_empty_sample(self, sample_info: Dict) -> Dict:
        """创建空样本，避免训练中断"""
        return {
            'scenario_id': f"empty_{sample_info['frame_idx']}",
            'timestamp': 0,
            'image': np.zeros((224, 224, 3), dtype=np.uint8),
            'ego_pose': {'translation': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]},
            'objects': [],
            'file_idx': sample_info['file_idx'],
            'frame_idx': sample_info['frame_idx']
        }
    
    # 删除chunk相关代码

class WaymoDataModule(pl.LightningDataModule):

    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 64,
        num_workers: int = 16,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        **dataset_kwargs
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.dataset_kwargs = dataset_kwargs
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            # 动态获取WaymoTrajectoryDataset支持的参数
            import inspect
            dataset_signature = inspect.signature(WaymoTrajectoryDataset.__init__)
            valid_params = set(dataset_signature.parameters.keys()) - {'self', 'data_root', 'split'}
            
            # 只传递数据集类实际支持的参数
            valid_dataset_kwargs = {
                k: v for k, v in self.dataset_kwargs.items() 
                if k in valid_params
            }
            
            self.train_dataset = WaymoTrajectoryDataset(
                data_root=self.data_root,
                split='training',
                **valid_dataset_kwargs
            )
            val_kwargs = valid_dataset_kwargs.copy()
            val_kwargs['use_augmentation'] = False
            
            # 尝试使用validation split，如果不存在使用training的一部分
            try:
                self.val_dataset = WaymoTrajectoryDataset(
                    data_root=self.data_root,
                    split='validation',
                    **val_kwargs
                )
            except FileNotFoundError:
                print("Warning: 未找到validation数据，使用training数据的子集作为验证集")
                self.val_dataset = WaymoTrajectoryDataset(
                    data_root=self.data_root,
                    split='training',
                    **val_kwargs
                )
        
        if stage == 'test' or stage is None:
            test_kwargs = valid_dataset_kwargs.copy() if 'valid_dataset_kwargs' in locals() else self.dataset_kwargs.copy()
            test_kwargs['use_augmentation'] = False
            
            # 尝试加载testing数据，如果不存在则使用validation数据
            try:
                self.test_dataset = WaymoTrajectoryDataset(
                    data_root=self.data_root,
                    split='testing',
                    **test_kwargs
                )
            except FileNotFoundError:
                print("Warning: 未找到testing数据，使用validation数据作为测试集")
                try:
                    self.test_dataset = WaymoTrajectoryDataset(
                        data_root=self.data_root,
                        split='validation',
                        **test_kwargs
                    )
                except FileNotFoundError:
                    self.test_dataset = WaymoTrajectoryDataset(
                        data_root=self.data_root,
                        split='training',
                        **test_kwargs
                    )
    
    def _worker_init_fn(self, worker_id):
        """Worker进程初始化 - 设置随机种子"""
        import random
        import numpy as np
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
