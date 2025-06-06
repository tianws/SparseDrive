from .decoder import SparseBox3DDecoder  # 从 decoder.py 导入稀疏3D边界框解码器
from .target import SparseBox3DTarget  # 从 target.py 导入稀疏3D边界框的目标分配/生成器
from .detection3d_blocks import (  # 从 detection3d_blocks.py 导入特定的3D检测构建块
    SparseBox3DRefinementModule,  # 稀疏3D边界框优化模块
    SparseBox3DKeyPointsGenerator,  # 稀疏3D边界框关键点生成器
    SparseBox3DEncoder,  # 稀疏3D边界框编码器
)
from .losses import SparseBox3DLoss  # 从 losses.py 导入稀疏3D边界框相关的损失函数
from .detection3d_head import Sparse4DHead  # 从 detection3d_head.py 导入稀疏4D检测头 (可能结合了时序信息)
