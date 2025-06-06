import torch # 导入PyTorch库
from torch.autograd.function import Function, once_differentiable # 从PyTorch的autograd模块导入Function基类和once_differentiable装饰器

from . import deformable_aggregation_ext # 从同级目录导入名为deformable_aggregation_ext的扩展模块（通常是C++/CUDA编译的）


class DeformableAggregationFunction(Function): # 定义可变形特征聚合的自定义PyTorch Function
    """
    可变形特征聚合函数的PyTorch Function接口。
    该类定义了自定义操作的前向和后向传播逻辑，底层实现通常在C++/CUDA扩展中。
    它允许从多相机、多尺度的特征图谱中，根据指定的采样点和权重来聚合特征。
    """
    @staticmethod # 标记为静态方法，因为Function的forward和backward是静态调用的
    def forward(
        ctx,  # 上下文对象，用于保存反向传播所需的信息
        mc_ms_feat,  # 多相机、多尺度特征图谱 (通常是经过feature_maps_format格式化后的柱状特征)
        spatial_shape,  # 原始特征图的空间形状信息 (num_cams, num_levels, 2)，告知每个层级的高和宽
        scale_start_index,  # 每个尺度特征在扁平化特征图中的起始索引 (num_cams, num_levels)
        sampling_location,  # 采样点的2D坐标 (bs, num_queries, num_groups, num_cams, num_levels, num_pts, 2)
                             # 这些坐标通常是归一化到[0,1]或[-1,1]范围的
        weights,  # 对应每个采样点的权重 (bs, num_queries, num_groups, num_cams, num_levels, num_pts)
    ):
        """
        可变形特征聚合的前向传播。

        Args:
            ctx: PyTorch自动求导上下文。
            mc_ms_feat (torch.Tensor): 格式化后的多相机多尺度特征图谱。
                                       形状通常是 (bs, total_num_pixels_across_all_selected_feature_maps, embed_dims)。
            spatial_shape (torch.Tensor): 每个特征图的空间H, W。
            scale_start_index (torch.Tensor): 每个特征图在mc_ms_feat中的起始索引。
            sampling_location (torch.Tensor): 采样点在对应特征图上的归一化坐标。
            weights (torch.Tensor): 采样点的权重。

        Returns:
            torch.Tensor: 聚合后的特征，形状通常是 (bs, num_queries, embed_dims)。
        """
        # 确保输入张量是连续的，并且是正确的类型，以供C++扩展使用
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        # 调用C++/CUDA扩展模块中的前向传播函数
        output = deformable_aggregation_ext.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )

        # 保存反向传播所需的张量到上下文中
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output # 返回聚合后的特征

    @staticmethod # 标记为静态方法
    @once_differentiable # 装饰器，表示这个backward函数只对输出进行一次微分（对于复杂图可能需要）
    def backward(ctx, grad_output): # 可变形特征聚合的反向传播
        """
        可变形特征聚合的反向传播。

        Args:
            ctx: PyTorch自动求导上下文，包含forward中保存的张量。
            grad_output (torch.Tensor): 输出特征的梯度。

        Returns:
            tuple: 对应forward函数输入的梯度。不需要梯度的输入项返回None。
                   返回 (grad_mc_ms_feat, None, None, grad_sampling_location, grad_weights)。
        """
        # 从上下文中获取forward过程中保存的张量
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors

        # 确保张量连续性和类型
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        # 初始化用于存储梯度的张量，形状与对应的输入相同
        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat) # 特征图谱的梯度
        grad_sampling_location = torch.zeros_like(sampling_location) # 采样点位置的梯度
        grad_weights = torch.zeros_like(weights) # 采样权重的梯度

        # 调用C++/CUDA扩展模块中的反向传播函数
        deformable_aggregation_ext.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(), # 输出的梯度作为输入
            grad_mc_ms_feat, # 待计算的特征图谱梯度
            grad_sampling_location, # 待计算的采样点位置梯度
            grad_weights, # 待计算的采样权重梯度
        )

        # 返回对应forward输入的梯度
        # spatial_shape 和 scale_start_index 通常不需要梯度，所以返回None
        return (
            grad_mc_ms_feat,
            None, # spatial_shape 的梯度
            None, # scale_start_index 的梯度
            grad_sampling_location,
            grad_weights,
        )
