from .sparsedrive import SparseDrive  # 从 sparsedrive 模块导入 SparseDrive 模型
from .sparsedrive_head import SparseDriveHead  # 从 sparsedrive_head 模块导入 SparseDriveHead 模型头
from .blocks import (  # 从 blocks 模块导入特定的网络构建块
    DeformableFeatureAggregation,  # 可变形特征聚合模块
    DenseDepthNet,  # 密集深度网络模块
    AsymmetricFFN,  # 非对称前馈网络模块
)
from .instance_bank import InstanceBank  # 从 instance_bank 模块导入 InstanceBank，用于存储实例特征
from .detection3d import (  # 从 detection3d 模块导入3D检测相关的组件
    SparseBox3DDecoder,  # 稀疏3D边界框解码器
    SparseBox3DTarget,  # 稀疏3D边界框目标分配器
    SparseBox3DRefinementModule,  # 稀疏3D边界框优化模块
    SparseBox3DKeyPointsGenerator,  # 稀疏3D边界框关键点生成器
    SparseBox3DEncoder,  # 稀疏3D边界框编码器
)
from .map import *  # 从 map 模块导入所有内容 (通常是地图分割/检测相关的模型组件)
from .motion import *  # 从 motion 模块导入所有内容 (通常是运动预测相关的模型组件)


__all__ = [  # 定义公开接口，当使用 from .models import * 时，会导入这些名称
    "SparseDrive",  # SparseDrive 模型
    "SparseDriveHead",  # SparseDrive 模型头
    "DeformableFeatureAggregation",  # 可变形特征聚合模块
    "DenseDepthNet",  # 密集深度网络模块
    "AsymmetricFFN",  # 非对称前馈网络模块
    "InstanceBank",  # 实例特征库
    "SparseBox3DDecoder",  # 稀疏3D边界框解码器
    "SparseBox3DTarget",  # 稀疏3D边界框目标分配器
    "SparseBox3DRefinementModule",  # 稀疏3D边界框优化模块
    "SparseBox3DKeyPointsGenerator",  # 稀疏3D边界框关键点生成器
    "SparseBox3DEncoder",  # 稀疏3D边界框编码器
]
