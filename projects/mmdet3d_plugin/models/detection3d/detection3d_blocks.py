import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import numpy as np  # 导入NumPy库

from mmcv.cnn import Linear, Scale, bias_init_with_prob  # 从MMCV导入线性层、尺度层和带概率的偏置初始化函数
from mmcv.runner.base_module import Sequential, BaseModule  # 从MMCV导入Sequential容器和BaseModule基类
from mmcv.cnn import xavier_init  # 从MMCV导入Xavier参数初始化方法
from mmcv.cnn.bricks.registry import (  # 从MMCV的注册表导入
    PLUGIN_LAYERS,  # 插件层注册表
    POSITIONAL_ENCODING,  # 位置编码注册表
)

from projects.mmdet3d_plugin.core.box3d import *  # 从项目中导入3D边界框相关的常量定义 (X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ)
from ..blocks import linear_relu_ln  # 从上一级目录的blocks模块导入linear_relu_ln辅助函数

__all__ = [  # 定义当使用 from .detection3d_blocks import * 时，哪些名称会被导入
    "SparseBox3DRefinementModule",
    "SparseBox3DKeyPointsGenerator",
    "SparseBox3DEncoder",
]


@POSITIONAL_ENCODING.register_module()  # 将SparseBox3DEncoder注册到MMCV的POSITIONAL_ENCODING注册表中
class SparseBox3DEncoder(BaseModule):  # 定义稀疏3D边界框编码器类
    """
    将3D边界框的各项参数（位置、尺寸、偏航角、速度）编码为特征嵌入。
    这些嵌入可以用于后续的注意力机制或特征融合。
    """
    def __init__(
        self,
        embed_dims,  # 目标嵌入维度，可以是单个整数或包含各部分嵌入维度的列表/元组
        vel_dims=3,  # 速度维度 (例如2D速度用2，3D速度用3)
        mode="add",  # 特征融合模式: "add" (相加) 或 "cat" (拼接)
        output_fc=True,  # 是否在最终输出前添加一个额外的全连接层
        in_loops=1,  # embedding_layer中每个out_loop内部的线性层+ReLU数量
        out_loops=2,  # embedding_layer中外部循环（接LayerNorm）的数量
    ):
        super().__init__()
        assert mode in ["add", "cat"]  # 确保模式是"add"或"cat"
        self.embed_dims = embed_dims  # 整体嵌入维度或各部分嵌入维度列表
        self.vel_dims = vel_dims  # 速度的维度
        self.mode = mode  # 融合模式

        def embedding_layer(input_dims, output_dims): # 定义一个内部辅助函数来创建嵌入层序列
            return nn.Sequential(
                *linear_relu_ln(output_dims, in_loops, out_loops, input_dims) # 使用linear_relu_ln创建层
            )

        if not isinstance(embed_dims, (list, tuple)): # 如果embed_dims是单个整数
            # 将其扩展为包含5个相同值的列表，分别对应pos, size, yaw, vel(如果使用), output_fc的嵌入维度
            embed_dims = [embed_dims] * 5

        # 为位置、尺寸、偏航角分别创建嵌入层
        self.pos_fc = embedding_layer(3, embed_dims[0])  # 位置 (x,y,z) -> embed_dims[0]
        self.size_fc = embedding_layer(3, embed_dims[1]) # 尺寸 (w,l,h) -> embed_dims[1]
        self.yaw_fc = embedding_layer(2, embed_dims[2])  # 偏航角 (sin_yaw, cos_yaw) -> embed_dims[2]

        if vel_dims > 0: # 如果使用速度信息
            self.vel_fc = embedding_layer(self.vel_dims, embed_dims[3]) # 速度 -> embed_dims[3]

        if output_fc: # 如果需要在最终输出前添加全连接层
            # 如果模式是拼接，且各部分嵌入维度不同，这里的输入维度需要注意。
            # 但通常如果mode='cat'，embed_dims列表中的前几个值会用于各部分，最后一个值用于output_fc的输出。
            # 如果mode='add'，则embed_dims[0]到embed_dims[3]应该相同，且等于embed_dims[-1]。
            # 此处假设embed_dims[-1]是期望的融合后或add模式下的统一维度。
            input_dim_for_output_fc = embed_dims[-1] if mode == 'add' else sum(embed_dims[:(3 + (1 if vel_dims > 0 else 0))])
            if mode == 'cat': # 如果是拼接模式，最终输出层的输入维度是各部分嵌入维度之和
                 input_dim_for_output_fc = embed_dims[0]+embed_dims[1]+embed_dims[2]
                 if vel_dims > 0: input_dim_for_output_fc += embed_dims[3]

            self.output_fc = embedding_layer(input_dim_for_output_fc, embed_dims[-1])
        else:
            self.output_fc = None

    def forward(self, box_3d: torch.Tensor): # 前向传播函数
        """
        将输入的3D边界框参数编码为特征。

        Args:
            box_3d (torch.Tensor): 形状为 (..., num_box_params) 的3D边界框张量。
                                   其中num_box_params至少包含位置、尺寸、偏航角和速度（如果使用）。

        Returns:
            torch.Tensor: 编码后的边界框特征嵌入。
        """
        # 分别提取位置、尺寸、偏航角信息并进行编码
        # X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX 等是预定义的索引常量
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]])

        if self.mode == "add":  # 如果是相加模式
            # 要求pos_feat, size_feat, yaw_feat具有相同的维度
            output = pos_feat + size_feat + yaw_feat
        elif self.mode == "cat":  # 如果是拼接模式
            output = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1) #沿最后一个维度拼接

        if self.vel_dims > 0:  # 如果使用速度信息
            vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims]) # 编码速度特征
            if self.mode == "add":
                output = output + vel_feat # 相加模式
            elif self.mode == "cat":
                output = torch.cat([output, vel_feat], dim=-1) # 拼接模式

        if self.output_fc is not None: # 如果存在最终的全连接输出层
            output = self.output_fc(output) # 通过输出层
        return output


@PLUGIN_LAYERS.register_module() # 将SparseBox3DRefinementModule注册到MMCV的PLUGIN_LAYERS注册表中
class SparseBox3DRefinementModule(BaseModule): # 定义稀疏3D边界框优化模块
    """
    用于优化输入的实例特征和锚点（anchor），预测更精确的3D边界框参数、
    类别以及可选的质量得分。
    """
    def __init__(
        self,
        embed_dims=256,  # 输入特征的嵌入维度
        output_dim=11,  # 输出边界框参数的维度 (例如 x,y,z,w,l,h,sin_yaw,cos_yaw,vx,vy,vz)
        num_cls=10,  # 类别数量 (用于分类分支)
        normalize_yaw=False,  # 是否对预测的yaw角进行归一化 (sin/cos)
        refine_yaw=False,  # 是否优化yaw角 (如果False，则可能只优化位置和尺寸)
        with_cls_branch=True,  # 是否包含分类分支
        with_quality_estimation=False,  # 是否包含质量估计分支
    ):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw

        # 定义需要优化的状态量索引，基于box3d.py中的常量
        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw: # 如果优化yaw角，则加入SIN_YAW和COS_YAW的索引
            self.refine_state += [SIN_YAW, COS_YAW]

        # 边界框参数优化层：包含线性层、ReLU、LayerNorm，最后是一个线性和尺度层
        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2), # 2个(线性+ReLU) + LN，重复2次
            Linear(self.embed_dims, self.output_dim), # 输出层，维度为output_dim
            Scale([1.0] * self.output_dim), # 尺度层，可学习的尺度参数，初始化为1
        )
        self.with_cls_branch = with_cls_branch
        if with_cls_branch: # 如果包含分类分支
            self.cls_layers = nn.Sequential( # 分类层序列
                *linear_relu_ln(embed_dims, 1, 2), # 1个(线性+ReLU) + LN，重复2次
                Linear(self.embed_dims, self.num_cls), # 输出类别 logits
            )
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation: # 如果包含质量估计分支
            self.quality_layers = nn.Sequential( # 质量估计层序列
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, 2), # 通常输出2个值，如centerness和yawness，或iou和centerness
            )

    def init_weight(self): # 参数初始化方法
        if self.with_cls_branch: # 如果有分类分支
            bias_init = bias_init_with_prob(0.01) # 根据概率初始化偏置 (常用于分类层，避免初始输出过于集中)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init) # 初始化分类输出层的偏置

    def forward(
        self,
        instance_feature: torch.Tensor,  # 输入的实例特征
        anchor: torch.Tensor,  # 对应的锚点/参考框参数
        anchor_embed: torch.Tensor,  # 锚点的嵌入特征
        time_interval: torch.Tensor = 1.0,  # 时间间隔 (用于速度相关的预测)
        return_cls=True,  # 是否返回分类结果
    ):
        feature = instance_feature + anchor_embed  # 将实例特征与锚点嵌入相加作为输入
        output = self.layers(feature)  # 通过优化层得到边界框参数的预测（通常是残差）

        # 将预测的残差加到锚点的对应状态上
        output[..., self.refine_state] = (
            output[..., self.refine_state] + anchor[..., self.refine_state]
        )
        if self.normalize_yaw: # 如果需要归一化yaw角
            # 对sin_yaw和cos_yaw进行L2归一化，确保它们在单位圆上
            output[..., [SIN_YAW, COS_YAW]] = torch.nn.functional.normalize(
                output[..., [SIN_YAW, COS_YAW]], dim=-1
            )
        if self.output_dim > 8: # 如果输出维度大于8，通常表示包含了速度 (VX, VY, VZ等在索引8之后)
            if not isinstance(time_interval, torch.Tensor): # 确保time_interval是张量
                time_interval = instance_feature.new_tensor(time_interval)
            # 假设output中VX之后的是位移delta_translation
            # delta_translation = output[..., VX:]
            # velocity_residual = delta_translation / time_interval
            # output[..., VX:] = velocity_residual + anchor_velocity
            # 这里实现似乎是直接将预测的output[..., VX:]视为速度残差或绝对速度，然后加上锚点速度
            # 如果output[..., VX:]是位移，则除以时间间隔得到速度；如果是速度残差，则直接加。
            # 当前代码：output[..., VX:]是预测的位移，然后转换为速度，再加上锚点速度。
            translation_pred = output[..., VX:] # 假设这部分是预测的位移量
            # (C, B) / (B) -> (C, B) -> (B, C)
            velocity_pred = translation_pred.transpose(0, -1) / time_interval
            velocity_pred = velocity_pred.transpose(0,-1)
            output[..., VX:] = velocity_pred + anchor[..., VX:] # 加上锚点的原始速度

        cls = None # 初始化分类输出
        if return_cls: # 如果需要返回分类结果
            assert self.with_cls_branch, "Without classification layers !!!" # 确保分类分支存在
            cls = self.cls_layers(instance_feature) # 通过分类层得到类别 logits

        quality = None # 初始化质量输出
        if return_cls and self.with_quality_estimation: # 如果需要返回分类且启用了质量估计
            quality = self.quality_layers(feature) # 通过质量估计层得到质量得分

        return output, cls, quality # 返回优化后的框参数、类别 logits 和质量得分


@PLUGIN_LAYERS.register_module() # 将SparseBox3DKeyPointsGenerator注册到MMCV的PLUGIN_LAYERS注册表中
class SparseBox3DKeyPointsGenerator(BaseModule): # 稀疏3D边界框关键点生成器
    """
    根据给定的3D锚点（anchor box）生成一组关键点。
    这些关键点可以是固定的相对位置（基于fix_scale），也可以是通过网络学习的。
    同时支持将这些关键点投影到过去的时间步（如果提供了时序信息）。
    """
    def __init__(
        self,
        embed_dims=256,  # 实例特征的嵌入维度 (如果使用可学习关键点)
        num_learnable_pts=0,  # 可学习关键点的数量
        fix_scale=None,  # 固定关键点的相对尺度列表/元组，每个元素是(sx, sy, sz)表示相对尺寸的比例
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None: # 如果未提供固定尺度，则默认一个在中心点的关键点 (0,0,0)
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = nn.Parameter( # 将固定尺度注册为不可训练的参数
            torch.tensor(fix_scale, dtype=torch.float32), requires_grad=False
        )
        self.num_pts = len(self.fix_scale) + num_learnable_pts # 总的关键点数量

        if num_learnable_pts > 0: # 如果存在可学习的关键点
            # 定义一个线性层，从实例特征预测可学习关键点的3D偏移量
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3)

    def init_weight(self): # 参数初始化方法
        if self.num_learnable_pts > 0: # 如果存在可学习关键点
            # 使用Xavier均匀分布初始化learnable_fc的权重
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,  # 输入的锚点框参数 (bs, num_anchor, box_dim)
        instance_feature=None,  # 对应的实例特征 (bs, num_anchor, embed_dims)，用于生成可学习关键点
        T_cur2temp_list=None,  # 从当前帧到过去各时序帧的变换矩阵列表 (可选)
        cur_timestamp=None,  # 当前帧的时间戳 (可选)
        temp_timestamps=None,  # 过去各时序帧的时间戳列表 (可选)
    ):
        bs, num_anchor = anchor.shape[:2]  # 批量大小和锚点数量
        # 获取锚点尺寸 (W, L, H)，并取指数（如果锚点中的尺寸是对数形式存储的）
        # 假设W,L,H常量对应的是log-scale的尺寸索引，或者模型直接输出log-scale尺寸
        # 如果anchor中的尺寸已经是真实尺度，则不需要.exp()
        size = anchor[..., None, [W, L, H]] # (bs, num_anchor, 1, 3)
        # 根据固定尺度计算固定关键点的位置 (相对于锚点中心)
        key_points = self.fix_scale * size # (bs, num_anchor, num_fix_pts, 3)

        if self.num_learnable_pts > 0 and instance_feature is not None: # 如果使用可学习关键点
            # 通过线性层从实例特征预测可学习关键点的相对偏移尺度
            learnable_scale = (
                self.learnable_fc(instance_feature) # (bs, num_anchor, num_learnable_pts * 3)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3) # (bs, num_anchor, num_learnable_pts, 3)
                .sigmoid() - 0.5 # 缩放到[-0.5, 0.5]范围，表示相对锚点尺寸的比例偏移
            )
            # 将可学习关键点与固定关键点拼接
            key_points = torch.cat(
                [key_points, learnable_scale * size], dim=-2 # 维度-2是num_pts维度
            ) # (bs, num_anchor, num_total_pts, 3)

        # 构建旋转矩阵用于将关键点从锚点坐标系旋转到全局或相机坐标系（取决于锚点的定义）
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3]) # 初始化旋转矩阵
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1 # Z轴保持不变 (假设是BEV下的yaw角旋转)

        # 将关键点应用旋转
        # key_points (bs, num_anchor, num_pts, 3) -> (bs, num_anchor, num_pts, 3, 1)
        # rotation_mat (bs, num_anchor, 3, 3) -> (bs, num_anchor, 1, 3, 3)
        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None] # (bs, num_anchor, num_pts, 3, 1)
        ).squeeze(-1) # (bs, num_anchor, num_pts, 3)

        # 将旋转后的关键点平移到锚点中心
        key_points = key_points + anchor[..., None, [X, Y, Z]] # (bs, num_anchor, num_pts, 3)

        # 如果不需要处理时序信息，则直接返回当前帧的关键点
        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        # 处理时序关键点投影
        temp_key_points_list = [] # 存储投影到过去帧的关键点列表
        velocity = anchor[..., VX:] # 获取锚点的速度 (..., vel_dim)

        for i, t_time in enumerate(temp_timestamps): # 遍历每个过去的时间戳
            time_interval = cur_timestamp - t_time # 计算当前帧与过去帧的时间差
            # 计算由于速度产生的位移 translation = velocity * time_interval
            translation = (
                velocity # (bs, num_anchor, vel_dim)
                * time_interval.to(dtype=velocity.dtype)[:, None, None] # (bs, 1, 1) 扩展以匹配速度维度
            )
            # 从当前帧关键点减去位移，得到在过去时刻t_time时这些关键点的位置（在当前帧坐标系下）
            temp_key_points = key_points - translation[:, :, None, :velocity.shape[-1]] # 只取与速度维度匹配的平移分量

            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype) # 获取从当前帧到过去第i帧的变换矩阵
            # 将这些“过去位置”的关键点通过T_cur2temp变换到过去第i帧的坐标系下
            temp_key_points_homogeneous = torch.cat( # 转换为齐次坐标
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]), # 添加1作为齐次坐标的最后一维
                    ],
                    dim=-1,
                ).unsqueeze(-1) # (bs, num_anchor, num_pts, 4, 1)

            # T_cur2temp (bs, 4, 4) -> (bs, 1, 1, 4, 4)
            projected_temp_key_points = T_cur2temp[:, None, None] @ temp_key_points_homogeneous
            projected_temp_key_points = projected_temp_key_points.squeeze(-1)[...,:3] # 转回非齐次坐标并取前3维 (x,y,z)

            temp_key_points_list.append(projected_temp_key_points) # 添加到列表

        return key_points, temp_key_points_list # 返回当前帧关键点和投影到过去各帧的关键点列表

    @staticmethod # 静态方法
    def anchor_projection( # 锚点投影函数，将锚点从源时间戳/坐标系投影到目标时间戳/坐标系
        anchor, # 源锚点 (..., box_dim)
        T_src2dst_list, # 从源坐标系到目标坐标系的变换矩阵列表
        src_timestamp=None, # 源时间戳 (可选)
        dst_timestamps=None, # 目标时间戳列表 (可选)
        time_intervals=None, # 直接提供的时间间隔列表 (可选，优先于src/dst_timestamp计算)
    ):
        dst_anchors = [] # 存储投影后的锚点列表
        for i in range(len(T_src2dst_list)): # 遍历每个目标变换/时间戳
            vel = anchor[..., VX:] # 获取锚点速度
            vel_dim = vel.shape[-1] # 速度维度
            T_src2dst = torch.unsqueeze( # 获取对应的变换矩阵并增加一个维度以进行广播
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1 # (bs, 1, 4, 4) or (1, 4, 4)
            )

            center = anchor[..., [X, Y, Z]] # 获取锚点中心

            # 计算时间间隔并更新中心点位置 (如果提供了时间信息)
            current_time_interval = None
            if time_intervals is not None:
                current_time_interval = time_intervals[i]
            elif src_timestamp is not None and dst_timestamps is not None:
                current_time_interval = (src_timestamp - dst_timestamps[i]).to(dtype=vel.dtype)

            if current_time_interval is not None:
                # translation = vel * current_time_interval (需要注意维度匹配)
                # vel (bs, num_anchor, vel_dim), current_time_interval (bs) or scalar
                # translation (bs, num_anchor, vel_dim)
                translation = vel * current_time_interval.view(-1, *([1]*(vel.dim()-1))) # 调整time_interval形状以广播
                center = center - translation[..., :3] # 从中心点减去（或加上，取决于时间间隔定义）位移

            # 应用空间变换 (旋转和平移)
            center_transformed = (
                torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None] # 旋转
                ).squeeze(dim=-1)
                + T_src2dst[..., :3, 3] # 平移
            )
            size = anchor[..., [W, L, H]] # 尺寸通常在投影中保持不变

            # 变换偏航角 (sin/cos形式)
            # T_src2dst[..., :2, :2] 是2D旋转部分
            # anchor[..., [COS_YAW, SIN_YAW], None] 将(cos, sin)视为列向量
            yaw_transformed_vec = torch.matmul(
                T_src2dst[..., :2, :2],
                torch.stack([anchor[..., COS_YAW], anchor[..., SIN_YAW]], dim=-1).unsqueeze(-1),
            ).squeeze(-1) # (..., 2) 得到变换后的 (cos', sin') 向量
            # 注意：NuScenes中yaw是(sin, cos)顺序，这里如果是(cos,sin)输入，输出也是(cos',sin')
            # 如果原始是 (sin, cos)，则 yaw_transformed_vec 的顺序需要调整或确保与输入一致
            # 假设输出也是 (cos', sin')，如果需要 (sin', cos')，则 yaw_transformed_vec[..., [1,0]]
            # 这里保持原样，假设后续处理能正确解析

            # 变换速度
            vel_transformed = torch.matmul(
                T_src2dst[..., :vel_dim, :vel_dim], vel[..., None] # 只旋转速度向量
            ).squeeze(-1)

            # 拼接成新的锚点参数
            # 注意：如果yaw_transformed_vec是(cos', sin')，而box格式期望(sin, cos)，需要调整顺序
            # 假设输出的yaw部分仍然是 (sin_yaw_new, cos_yaw_new)
            # 如果原始anchor的yaw是(sin,cos)，则这里需要用yaw_transformed_vec[...,[1,0]]
            # 假设这里的 yaw_transformed_vec 是 (cos_new, sin_new), 为了匹配 (SIN_YAW, COS_YAW) 顺序，应该是 yaw_transformed_vec[..., [1,0]]
            # 但代码是 yaw = yaw[..., [1,0]]，这意味着输入是 (cos, sin)，输出是 (sin, cos)
            yaw_final = yaw_transformed_vec[..., [1,0]] # 假设 T_src2dst[..., :2, :2] @ [cos; sin] = [cos_new; sin_new], 那么 [sin_new; cos_new]

            dst_anchor = torch.cat([center_transformed, size, yaw_final, vel_transformed], dim=-1)
            dst_anchors.append(dst_anchor)
        return dst_anchors # 返回投影后的锚点列表

    @staticmethod # 静态方法
    def distance(anchor): # 计算锚点到原点(自车位置)的BEV距离
        return torch.norm(anchor[..., :2], p=2, dim=-1) # 计算x,y坐标的L2范数
