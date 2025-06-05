from typing import List, Optional, Tuple  # Python类型提示库，用于指定变量、函数参数和返回值的类型

import numpy as np  # 导入NumPy库，用于高效的数值计算
import torch  # 导入PyTorch库，主要的深度学习框架
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.cuda.amp.autocast_mode import autocast  # 从PyTorch的自动混合精度模块导入autocast，用于混合精度训练

from mmcv.cnn import Linear, build_activation_layer, build_norm_layer  # 从MMCV导入线性层和构建激活层、归一化层的函数
from mmcv.runner.base_module import Sequential, BaseModule  # 从MMCV导入Sequential容器和BaseModule基类
from mmcv.cnn.bricks.transformer import FFN  # 从MMCV导入FFN (FeedForward Network)
from mmcv.utils import build_from_cfg  # 从MMCV导入从配置字典构建对象的函数
from mmcv.cnn.bricks.drop import build_dropout  # 从MMCV导入构建dropout层的函数
from mmcv.cnn import xavier_init, constant_init  # 从MMCV导入参数初始化方法 (Xavier和常数初始化)
from mmcv.cnn.bricks.registry import (  # 从MMCV的注册表导入
    ATTENTION,  # 注意力机制注册表
    PLUGIN_LAYERS,  # 插件层注册表
    FEEDFORWARD_NETWORK,  # 前馈网络注册表
)

try:
    from ..ops import deformable_aggregation_function as DAF  # 尝试从相对路径导入自定义的deformable_aggregation_function
except:
    DAF = None  # 如果导入失败，则DAF为None

__all__ = [  # 定义当使用 from .blocks import * 时，哪些名称会被导入
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "AsymmetricFFN",
]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None): # 定义一个辅助函数，用于快速构建包含线性层、ReLU激活和LayerNorm的序列
    """
    创建一个包含多个线性层、ReLU激活和LayerNorm层的序列。

    Args:
        embed_dims (int): 线性层的输出维度和LayerNorm的维度。
        in_loops (int): 每个外部循环中，内部线性层+ReLU的数量。
        out_loops (int): 外部循环的数量，每个外部循环后接一个LayerNorm。
        input_dims (int, optional): 第一个线性层的输入维度。如果为None，则默认为embed_dims。

    Returns:
        list: 包含nn.Module的列表。
    """
    if input_dims is None:  # 如果未指定输入维度
        input_dims = embed_dims  # 则将其设置为embed_dims
    layers = []  # 初始化层列表
    for _ in range(out_loops):  # 外层循环
        for _ in range(in_loops):  # 内层循环
            layers.append(Linear(input_dims, embed_dims))  # 添加线性层
            layers.append(nn.ReLU(inplace=True))  # 添加ReLU激活层，inplace=True表示直接在原地修改输入
            input_dims = embed_dims  # 更新下一层的输入维度为当前层的输出维度
        layers.append(nn.LayerNorm(embed_dims))  # 添加LayerNorm层
    return layers  # 返回构建的层列表


@ATTENTION.register_module()  # 将DeformableFeatureAggregation注册到MMCV的ATTENTION注册表中
class DeformableFeatureAggregation(BaseModule):  # 可变形特征聚合模块
    """
    可变形特征聚合模块，用于融合来自不同相机视图、不同特征层级的特征。
    它通过生成关键点（key points）并从特征图谱中采样这些点的特征，
    然后使用可学习的权重聚合这些采样特征。
    """
    def __init__(
        self,
        embed_dims: int = 256,  # 嵌入维度 (输出特征维度)
        num_groups: int = 8,  # 分组数量，用于分组注意力或特征处理
        num_levels: int = 4,  # 特征层级的数量
        num_cams: int = 6,  # 相机视图的数量
        proj_drop: float = 0.0,  # 输出投影后的dropout概率
        attn_drop: float = 0.0,  # 注意力权重上的dropout概率
        kps_generator: dict = None,  # 关键点生成器的配置字典
        temporal_fusion_module=None,  # 时序融合模块的配置字典 (可选)
        use_temporal_anchor_embed=True,  # 是否使用时序锚点嵌入 (未使用在此代码片段中)
        use_deformable_func=False,  # 是否使用自定义的deformable_aggregation_function (DAF)
        use_camera_embed=False,  # 是否使用相机嵌入来调整权重
        residual_mode="add",  # 残差连接模式 ("add" 或 "cat")
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:  # 检查嵌入维度是否能被分组数整除
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)  # 每个组的维度
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:  # 如果使用自定义DAF函数
            assert DAF is not None, "deformable_aggregation needs to be set up." # 确保DAF已成功导入
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop  # 注意力dropout概率
        self.residual_mode = residual_mode  # 残差连接模式
        self.proj_drop = nn.Dropout(proj_drop)  # 输出dropout层

        kps_generator["embed_dims"] = embed_dims  # 将embed_dims添加到关键点生成器配置中
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)  # 从配置构建关键点生成器
        self.num_pts = self.kps_generator.num_pts  # 获取生成的关键点数量

        if temporal_fusion_module is not None:  # 如果配置了时序融合模块
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )  # 构建时序融合模块
        else:
            self.temp_module = None  # 否则时序融合模块为None

        self.output_proj = Linear(embed_dims, embed_dims)  # 输出线性投影层

        if use_camera_embed:  # 如果使用相机嵌入
            # 相机编码器，使用之前定义的linear_relu_ln辅助函数创建
            # 输入维度为12 (通常对应投影矩阵的前3行x4列，扁平化后为12)
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            # 用于生成权重的全连接层，输出维度对应 num_groups * num_levels * num_pts
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:  # 如果不使用相机嵌入
            self.camera_encoder = None
            # 权重全连接层的输出维度对应 num_groups * num_cams * num_levels * num_pts
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self): # 参数初始化方法
        constant_init(self.weights_fc, val=0.0, bias=0.0)  # 将权重全连接层初始化为0
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)  # 使用Xavier均匀分布初始化输出投影层

    def forward(
        self,
        instance_feature: torch.Tensor,  # 输入的实例特征 (查询特征)
        anchor: torch.Tensor,  # 锚点/参考点
        anchor_embed: torch.Tensor,  # 锚点嵌入
        feature_maps: List[torch.Tensor],  # 多层级的特征图谱列表
        metas: dict,  # 包含元信息的字典 (如投影矩阵、图像尺寸)
        **kwargs: dict,  # 其他关键字参数
    ):
        bs, num_anchor = instance_feature.shape[:2]  # 获取批量大小和锚点数量
        # 生成关键点 (通常是相对于锚点的偏移点或采样点)
        key_points = self.kps_generator(anchor, instance_feature)
        # 获取用于聚合特征的权重
        weights = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_deformable_func:  # 如果使用自定义的DAF函数
            # 将3D关键点投影到2D图像平面
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],  # 投影矩阵
                    metas.get("image_wh"),  # 图像宽高 (可选，用于归一化)
                )
                .permute(0, 2, 3, 1, 4)  # 调整维度顺序
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2) # 重塑形状
            )
            # 调整权重张量的形状以匹配DAF函数的输入要求
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            # 调用DAF函数进行特征聚合
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:  # 如果不使用自定义DAF函数，则使用标准的特征采样和融合流程
            # 从特征图谱中采样关键点处的特征
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            # 融合来自多视图和多层级的采样特征
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # 融合多点特征 (沿num_pts维度求和)

        output = self.proj_drop(self.output_proj(features))  # 应用输出投影和dropout

        # 残差连接
        if self.residual_mode == "add":  # 加法模式
            output = output + instance_feature
        elif self.residual_mode == "cat":  # 拼接模式
            output = torch.cat([output, instance_feature], dim=-1)
        return output  # 返回聚合后的特征

    def _get_weights(self, instance_feature, anchor_embed, metas=None): # 获取注意力权重的内部方法
        bs, num_anchor = instance_feature.shape[:2] # 批量大小和锚点数量
        # 结合实例特征和锚点嵌入作为生成权重的输入特征
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None: # 如果使用相机嵌入
            # 从投影矩阵中提取相机参数 (前3行，通常包含旋转和平移)，并reshape
            camera_params = metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1 # -1 表示自动计算该维度大小 (12)
                )
            camera_embed = self.camera_encoder(camera_params) # 通过相机编码器得到相机嵌入
            # 将实例/锚点特征与相机嵌入相加 (广播机制)
            feature = feature[:, :, None] + camera_embed[:, None]

        # 通过全连接层生成权重，并调整形状
        weights = (
            self.weights_fc(feature) # (bs, num_anchor, num_cams_or_1, C_out_weights_fc)
            .reshape(bs, num_anchor, -1, self.num_groups) # (bs, num_anchor, num_cams*num_levels*num_pts or num_levels*num_pts, num_groups)
            .softmax(dim=-2) # 在倒数第二个维度上进行softmax，使得每个group的权重和为1
            .reshape( # 重新调整为更结构化的形状
                bs,
                num_anchor,
                self.num_cams if self.camera_encoder is None else 1, # 如果没有相机编码器，则权重是针对每个相机的
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.camera_encoder is not None: # 如果使用了相机编码器，权重不区分相机，则复制到每个相机维度
             weights = weights.repeat(1,1,self.num_cams,1,1,1) # (bs, num_anchor, num_cams, num_levels, num_pts, num_groups)


        if self.training and self.attn_drop > 0: # 如果是训练模式且设置了注意力dropout
            # 创建一个随机掩码
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1 # 掩码在level和group维度是共享的
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype) # 将掩码移到与权重相同的设备和数据类型
            weights = ((mask > self.attn_drop) * weights) / ( # 应用dropout: 大于阈值的保留并进行缩放，小于等于阈值的置0
                1 - self.attn_drop # 除以 (1 - drop_prob) 以保持期望值不变
            )
        return weights # 返回计算得到的权重

    @staticmethod  # 静态方法，不需要访问类实例的属性
    def project_points(key_points, projection_mat, image_wh=None): # 将3D关键点投影到2D图像平面
        bs, num_anchor, num_pts = key_points.shape[:3] # 批量大小, 锚点数, 每锚点关键点数

        # 将3D关键点转换为齐次坐标 (x, y, z, 1)
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        # 使用投影矩阵进行投影: P @ K
        # projection_mat: (bs, num_cams, 4, 4)
        # pts_extend: (bs, num_anchor, num_pts, 4)
        # 调整维度以进行批量矩阵乘法:
        # projection_mat[:, :, None, None]: (bs, num_cams, 1, 1, 4, 4)
        # pts_extend[:, None, ..., None]: (bs, 1, num_anchor, num_pts, 4, 1)
        # 结果 points_2d: (bs, num_cams, num_anchor, num_pts, 4, 1) -> squeeze后 (bs, num_cams, num_anchor, num_pts, 4)
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)

        # 归一化，将投影后的坐标 (x', y', z') 转换为 (x'/z', y'/z')
        points_2d = points_2d[..., :2] / torch.clamp( # 取前两个维度 (x', y')
            points_2d[..., 2:3], min=1e-5 # 除以深度z' (裁剪以避免除零)
        )
        if image_wh is not None: # 如果提供了图像宽高，则进行归一化 (将坐标范围调整到 [0, 1] 或 [-1, 1] 取决于后续处理)
            # image_wh: (bs, num_cams, 2) -> (bs, num_cams, 1, 1, 2)
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d # 返回2D投影点

    @staticmethod # 静态方法
    def feature_sampling( # 从特征图谱中采样特征
        feature_maps: List[torch.Tensor],  # 多层级特征图列表，每个元素形状 (bs, num_cams, C, H_level, W_level)
        key_points: torch.Tensor,  # 3D关键点，形状 (bs, num_anchor, num_pts, 3)
        projection_mat: torch.Tensor,  # 投影矩阵，形状 (bs, num_cams, 4, 4)
        image_wh: Optional[torch.Tensor] = None,  # 图像宽高，形状 (bs, num_cams, 2)，可选
    ) -> torch.Tensor:
        num_levels = len(feature_maps)  # 特征层级数
        num_cams = feature_maps[0].shape[1]  # 相机数量
        bs, num_anchor, num_pts = key_points.shape[:3]  # 批量大小, 锚点数, 每锚点关键点数

        # 将3D关键点投影到2D，并归一化到[-1, 1]范围 (grid_sample要求)
        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        ) # (bs, num_cams, num_anchor, num_pts, 2)
        points_2d = points_2d * 2 - 1 # 将[0,1]范围 (如果image_wh提供了) 或其他范围的点映射到[-1,1]

        # grid_sample期望的输入是 (N, C, H_in, W_in) 和 grid (N, H_out, W_out, 2)
        # 这里将bs和num_cams合并到N维度，或者bs*num_anchor*num_pts作为采样点数
        # points_2d: (bs, num_cams, num_anchor, num_pts, 2) -> (bs * num_cams, num_anchor * num_pts, 1, 2) for grid_sample if H_out=num_anchor*num_pts, W_out=1
        # 或者更常见的做法是，将每个(cam, anchor, pt)组合视为一个独立的采样点
        # points_2d needs to be (N, n_points_to_sample, 2)
        # feature_maps[l] (bs, num_cams, C, H, W) -> (bs*num_cams, C, H, W)
        # points_2d (bs, num_cams, num_anchor, num_pts, 2) -> (bs*num_cams, num_anchor*num_pts, 2)
        points_2d = points_2d.permute(0,2,1,3,4).reshape(bs*num_anchor, num_cams*num_pts, 2) # (bs*num_anchor, num_cams*num_pts, 2)
        # 上述reshape可能不符合grid_sample的预期，grid_sample通常对每个图像进行采样
        # 正确的方式应该是将bs和num_cams合并，然后对每个 (bs*num_cams) 的图像使用其对应的点进行采样
        # points_2d (bs, num_cams, num_anchor, num_pts, 2) -> (bs*num_cams, num_anchor, num_pts, 2) after permute and reshape
        # then flatten num_anchor and num_pts for grid_sample: (bs*num_cams, num_anchor*num_pts, 2)
        points_2d_reshaped = points_2d.permute(0, 2, 1, 3, 4).reshape(bs * num_cams, num_anchor * num_pts, 2)


        features = []  # 存储从不同层级采样到的特征
        for fm_level_idx, fm in enumerate(feature_maps): # 遍历每个层级的特征图
            # fm: (bs, num_cams, C, H_level, W_level)
            # -> (bs * num_cams, C, H_level, W_level)
            fm_reshaped = fm.reshape(bs * num_cams, fm.shape[2], fm.shape[3], fm.shape[4])
            # points_2d_for_level: (bs, num_cams, num_anchor, num_pts, 2)
            # We need to select the right num_cams dimension for points_2d or expand fm
            # Here, it seems points_2d is (bs, num_cams, num_anchor, num_pts, 2)
            # grid_sample expects grid of shape (N, H_out, W_out, 2) or (N, n_points, 2)
            # Let's reshape points_2d to (bs * num_cams, num_anchor * num_pts, 2)
            # And fm to (bs * num_cams, C, H_level, W_level)

            # points_2d (bs, num_cams, num_anchor, num_pts, 2)
            # We need to get (bs * num_cams, num_anchor * num_pts, 2) for grid_sample
            # Each image (indexed by bs*num_cams) has num_anchor*num_pts points to sample
            current_level_points = points_2d.permute(0,2,1,3,4).reshape(bs, num_anchor, num_cams, num_pts, 2) # Back to original like structure to be sure
            current_level_points = current_level_points.permute(0,2,1,3,4).reshape(bs*num_cams, num_anchor*num_pts, 2)


            sampled_feat = torch.nn.functional.grid_sample(
                fm_reshaped, current_level_points.unsqueeze(1), padding_mode='zeros', align_corners=False # (N, C, H_in, W_in), (N, H_out, W_out, 2) -> (N, C, H_out, W_out)
            ) # (bs*num_cams, C, 1, num_anchor*num_pts)
            sampled_feat = sampled_feat.squeeze(2).reshape(bs, num_cams, fm.shape[2], num_anchor, num_pts) # (bs, num_cams, C, num_anchor, num_pts)
            features.append(sampled_feat.permute(0,3,1,4,2)) # (bs, num_anchor, num_cams, num_pts, C)

        features = torch.stack(features, dim=3)  # (bs, num_anchor, num_cams, num_levels, num_pts, C)
        # 之前的permute: features.reshape(bs, num_cams, num_levels, -1, num_anchor, num_pts).permute(0, 4, 1, 2, 5, 3)
        # (bs, num_anchor, num_cams, num_levels, num_pts, embed_dims)

        return features

    def multi_view_level_fusion( # 多视图、多层级特征融合
        self,
        features: torch.Tensor,  # 采样到的特征，形状 (bs, num_anchor, num_cams, num_levels, num_pts, embed_dims)
        weights: torch.Tensor,  # 对应的权重，形状 (bs, num_anchor, num_cams, num_levels, num_pts, num_groups)
    ):
        bs, num_anchor = weights.shape[:2]
        # 将特征按组划分，并与权重相乘
        # features reshaped: (bs, num_anchor, num_cams, num_levels, num_pts, num_groups, group_dims)
        # weights[..., None]: (bs, num_anchor, num_cams, num_levels, num_pts, num_groups, 1)
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims) # 在最后一维增加group_dims
        )
        # 融合相机视图和层级特征 (沿num_cams和num_levels维度加权求和，权重已经乘进去了，所以直接求和)
        features = features.sum(dim=2).sum(dim=2) # sum over num_cams, then sum over num_levels
        # 重塑形状为 (bs, num_anchor, num_pts, embed_dims)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features


@PLUGIN_LAYERS.register_module() # 将DenseDepthNet注册到MMCV的PLUGIN_LAYERS注册表中
class DenseDepthNet(BaseModule): # 密集深度预测网络
    """
    一个用于从特征图谱预测密集深度图的网络模块。
    它通常接收多层级的特征图作为输入，并为每个指定的层级预测一个深度图。
    """
    def __init__(
        self,
        embed_dims=256,  # 输入特征图的嵌入维度
        num_depth_layers=1,  # 用于预测深度的特征图层级数量
        equal_focal=100,  # 用于深度缩放的等效焦距参考值
        max_depth=60,  # 预测深度的最大值 (用于裁剪)
        loss_weight=1.0,  # 深度损失的权重
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()  # 用于存储深度预测卷积层的模块列表
        for i in range(num_depth_layers): # 为每个指定的深度层级创建一个1x1卷积
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0) # 输出通道为1，表示深度值
            )

    def forward(self, feature_maps, focal=None, gt_depths=None): # 前向传播
        """
        Args:
            feature_maps (list[Tensor]): 多层级特征图列表。
            focal (Tensor, optional): 当前图像的实际焦距，用于调整深度预测的尺度。
            gt_depths (list[Tensor], optional): 真实的深度图列表，用于计算损失 (仅在训练时提供)。

        Returns:
            list[Tensor] or Tensor: 如果是评估模式或gt_depths为None，返回预测的深度图列表。
                                   如果是训练模式且gt_depths不为None，返回计算得到的深度损失。
        """
        if focal is None: # 如果未提供实际焦距
            focal = self.equal_focal # 使用预设的等效焦距
        else:
            focal = focal.reshape(-1) # 展平焦距张量

        depths = [] # 存储预测的深度图
        # 遍历指定数量的特征图层级进行深度预测
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            # feat: (bs * num_cams, C, H, W)
            # depth_layers[i](feat.flatten(end_dim=1).float()) -> (bs * num_cams, 1, H, W)
            # .exp() 将预测值（通常是log-depth）转换为真实深度
            depth = self.depth_layers[i](feat.float()).exp() # 使用1x1卷积预测深度并取指数

            # 根据焦距进行尺度调整: depth_new = depth_pred * (focal_actual / focal_equal)
            # depth: (bs*num_cams, 1, H, W)
            # focal: (bs*num_cams)
            # depth.transpose(0, -1) -> (W, H, 1, bs*num_cams)
            # focal / self.equal_focal -> (bs*num_cams)
            # depth = depth.transpose(0, -1) * focal[:,None,None,None] / self.equal_focal # 调整维度以匹配
            depth = depth * focal.view(-1, 1, 1, 1) / self.equal_focal
            # depth = depth.transpose(0, -1) # 转置回来
            depths.append(depth) # 添加到列表

        if gt_depths is not None and self.training: # 如果提供了真实深度且是训练模式
            loss = self.loss(depths, gt_depths) # 计算损失
            return loss # 返回损失
        return depths # 否则返回预测的深度图列表

    def loss(self, depth_preds, gt_depths): # 计算深度损失
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths): # 遍历每个层级的预测深度和真实深度
            # pred: (bs*num_cams, 1, H, W), gt: (bs*num_cams, H, W)
            pred = pred.squeeze(1).permute(0, 2, 1).contiguous().reshape(-1) # (N_pixels)
            gt = gt.reshape(-1) # (N_pixels)

            # 创建有效像素的掩码 (真实深度大于0且预测深度不是NaN)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask] # 应用掩码
            pred = pred[fg_mask] # 应用掩码
            pred = torch.clip(pred, 0.0, self.max_depth) # 将预测深度裁剪到最大深度范围内

            with autocast(enabled=False): # 在计算损失时通常禁用自动混合精度，以保证数值稳定性
                error = torch.abs(pred - gt).sum() # 计算L1损失 (绝对误差和)
                _loss = (
                    error # 总误差
                    / max(1.0, len(gt) * len(depth_preds)) # 除以有效像素数量和预测层级数进行平均
                    * self.loss_weight # 乘以损失权重
                )
            loss = loss + _loss # 累加损失
        return loss


@FEEDFORWARD_NETWORK.register_module() # 将AsymmetricFFN注册到MMCV的FEEDFORWARD_NETWORK注册表中
class AsymmetricFFN(BaseModule): # 非对称前馈网络 (FFN)
    """
    一个非对称的前馈网络模块，允许输入和输出通道数不同，
    并且可以配置预归一化 (pre_norm) 和残差连接。
    """
    def __init__(
        self,
        in_channels=None, # 输入通道数，如果为None，则默认为embed_dims
        pre_norm=None, # 预归一化层的配置字典 (可选)
        embed_dims=256, # FFN内部的嵌入维度和输出维度
        feedforward_channels=1024, # FFN中间隐藏层的通道数
        num_fcs=2, # 全连接层的数量 (至少为2，一个扩展维度，一个压缩回embed_dims)
        act_cfg=dict(type="ReLU", inplace=True), # 激活函数配置
        ffn_drop=0.0, # FFN内部的dropout概率
        dropout_layer=None, # 残差连接前的dropout层配置 (可选)
        add_identity=True, # 是否添加恒等/残差连接
        init_cfg=None, # 初始化配置
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, ( # 确保至少有两个全连接层
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels if in_channels is not None else embed_dims # 设置实际输入通道数
        self.pre_norm = pre_norm # 是否使用预归一化
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg) # 构建激活层

        layers = [] # 初始化层列表
        current_in_channels = self.in_channels # 当前层的输入通道数
        if pre_norm is not None: # 如果配置了预归一化
            self.pre_norm_layer = build_norm_layer(pre_norm, current_in_channels)[1] # 构建归一化层

        for i in range(num_fcs - 1): # 构建除了最后一层之外的全连接层
            layers.append(
                Sequential( # 每个层是一个包含线性、激活和dropout的序列
                    Linear(current_in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            current_in_channels = feedforward_channels # 更新下一层的输入通道数
        layers.append(Linear(feedforward_channels, embed_dims)) # 添加最后一个线性层 (将维度映射回embed_dims)
        layers.append(nn.Dropout(ffn_drop)) # 添加最后的dropout
        self.layers = Sequential(*layers) # 将所有层构建为一个Sequential容器

        self.dropout_layer = ( # 构建残差连接前的dropout层
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity() # 如果未配置，则为恒等映射
        )
        self.add_identity = add_identity # 是否添加残差连接
        if self.add_identity: # 如果添加残差连接
            # 如果输入通道数与输出嵌入维度不同，则需要一个线性层来匹配维度
            self.identity_fc = (
                torch.nn.Identity() # 如果维度相同，则为恒等映射
                if self.in_channels == embed_dims
                else Linear(self.in_channels, embed_dims) # 否则使用线性层调整维度
            )

    def forward(self, x, identity=None): # 前向传播
        residual_x = x # 保存原始输入x，用于可能的残差连接
        if hasattr(self, 'pre_norm_layer') and self.pre_norm_layer is not None: # 如果存在预归一化层
            x = self.pre_norm_layer(x) # 应用预归一化

        out = self.layers(x) # 通过FFN层

        if not self.add_identity: # 如果不添加残差连接
            return self.dropout_layer(out) # 直接返回dropout后的输出

        if identity is None: # 如果未提供用于残差连接的identity张量
            identity = residual_x # 使用FFN的输入作为identity (或者预归一化前的输入，取决于具体设计)

        identity = self.identity_fc(identity) # 通过identity_fc调整identity的维度 (如果需要)
        return identity + self.dropout_layer(out) # 返回残差连接的结果 (identity + FFN输出)
