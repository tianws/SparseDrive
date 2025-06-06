import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块

from mmcv.utils import build_from_cfg  # 从MMCV导入从配置字典构建对象的函数
from mmdet.models.builder import LOSSES  # 从MMDetection的模型构建器中导入LOSSES注册表
from mmdet.models.losses import l1_loss, smooth_l1_loss  # 从MMDetection导入L1损失和平滑L1损失函数


@LOSSES.register_module()  # 将LinesL1Loss注册到MMDetection的LOSSES注册表中
class LinesL1Loss(nn.Module):  # 定义线段L1损失类
    """
    用于线段预测的L1损失或平滑L1损失。
    这个损失在计算后会除以线段中的点数，以对不同长度的线段进行归一化。
    """

    def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
        """
        L1损失。与平滑L1损失类似。
        Args:
            reduction (str, optional): 损失规约方法。
                可选值为 "none", "mean" 和 "sum"。默认为 "mean"。
            loss_weight (float, optional): 损失的权重。默认为 1.0。
            beta (float, optional): 平滑L1损失的beta参数。如果beta > 0，则使用平滑L1损失，
                                  否则使用标准L1损失。默认为 0.5。
        """

        super().__init__()
        self.reduction = reduction  # 损失规约方法
        self.loss_weight = loss_weight  # 损失权重
        self.beta = beta  # 平滑L1损失的beta参数

    def forward(self,
                pred,  # 预测的线段坐标张量
                target,  # 真实的线段坐标张量
                weight=None,  # 样本权重 (可选)
                avg_factor=None,  # 用于平均损失的因子 (可选)
                reduction_override=None): # 覆盖默认规约方法的选项 (可选)
        """Forward function. # 前向传播函数
        Args:
            pred (torch.Tensor): 预测值，形状: [bs, num_lines, num_points_x_coords] 或类似。
            target (torch.Tensor): 学习目标，形状: [bs, num_lines, num_points_x_coords] 或类似。
            weight (torch.Tensor, optional): 每个预测的损失权重。默认为None。
                                           当预测并非全部有效时很有用。
            avg_factor (int, optional): 用于平均损失的平均因子。默认为None。
            reduction_override (str, optional): 用于覆盖损失原始规约方法的规约方法。
                                               默认为None。
        """
        assert reduction_override in (None, 'none', 'mean', 'sum') # 确保规约覆盖选项有效
        reduction = (
            reduction_override if reduction_override else self.reduction) # 确定最终使用的规约方法

        if self.beta > 0: # 如果beta大于0，使用平滑L1损失
            loss = smooth_l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
        
        else: # 否则，使用标准L1损失
            loss = l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        
        # 假设pred的最后一个维度是 num_points * 2 (x,y坐标)
        num_points = pred.shape[-1] // 2 # 计算每条线段的点数
        if num_points > 0:
            loss = loss / num_points # 将损失除以点数进行归一化

        return loss * self.loss_weight # 应用损失权重


@LOSSES.register_module() # 将SparseLineLoss注册到MMDetection的LOSSES注册表中
class SparseLineLoss(nn.Module): # 定义稀疏线段损失类
    """
    用于稀疏线段预测的组合损失。
    该损失首先对预测线段和目标线段进行归一化处理（平移和缩放），
    然后应用一个内部的线段损失（例如LinesL1Loss）。
    """
    def __init__(
        self,
        loss_line,  # 内部线段损失的配置字典 (例如LinesL1Loss)
        num_sample=20,  # 每条线段的采样点数 (用于归一化和reshape)
        roi_size=(30, 60),  # 感兴趣区域(ROI)的尺寸 (宽, 高)，用于归一化线段坐标
    ):
        super().__init__()

        def build(cfg, registry): # 内部辅助函数，用于构建损失模块
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.loss_line = build(loss_line, LOSSES)  # 构建内部线段损失模块
        self.num_sample = num_sample  # 每条线段的点数
        self.roi_size = roi_size  # ROI尺寸，格式为 (宽度, 高度)

    def forward(
        self,
        line,  # 预测的线段，形状 (bs, num_queries, num_sample * 2)
        line_target,  # 真实的目标线段，形状 (bs, num_queries, num_sample * 2)
        weight=None,  # 样本权重 (可选)
        avg_factor=None,  # 平均因子 (可选)
        prefix="",  # 损失名称前缀 (可选)
        suffix="",  # 损失名称后缀 (可选)
        **kwargs,  # 其他关键字参数
    ):
        """
        计算稀疏线段损失。

        Args:
            line (torch.Tensor): 预测的线段。
            line_target (torch.Tensor): 真实的目标线段。
            weight (torch.Tensor, optional): 每个样本的损失权重。
            avg_factor (int, optional): 用于归一化损失的平均因子。
            prefix (str, optional): 添加到损失名称字典键的前缀。
            suffix (str, optional): 添加到损失名称字典键的后缀。

        Returns:
            dict[str, torch.Tensor]: 包含计算得到的线段损失的字典。
        """

        output = {} # 初始化损失字典
        # 对预测线段和目标线段进行归一化处理
        line = self.normalize_line(line)
        line_target = self.normalize_line(line_target)

        # 计算归一化后的线段之间的损失
        line_loss = self.loss_line(
            line, line_target, weight=weight, avg_factor=avg_factor
        )
        output[f"{prefix}loss_line{suffix}"] = line_loss # 添加到损失字典

        return output

    def normalize_line(self, line): # 归一化线段坐标
        """
        将线段坐标归一化到ROI区域内。
        首先将坐标原点平移到ROI中心，然后根据ROI尺寸进行缩放。

        Args:
            line (torch.Tensor): 输入的线段张量，形状 (N, num_points * 2) 或 (B, N, num_points * 2)。

        Returns:
            torch.Tensor: 归一化后的线段张量。
        """
        if line.shape[0] == 0: # 如果没有线段，直接返回
            return line

        # 将线段的最后一个维度重塑为 (num_sample, 2) 或 (num_sample, coords_dim)
        # 这里假设coords_dim为2
        line = line.view(line.shape[:-1] + (self.num_sample, -1)) # (..., num_sample, 2)
        
        # 定义ROI的原点 (通常是ROI的左上角或中心，这里假设是中心，所以平移向量是负的ROI尺寸的一半)
        # roi_size 是 (宽度, 高度)，对应 (x, y)
        # 因此 origin 是 (-roi_width/2, -roi_height/2)
        origin = -line.new_tensor([self.roi_size[0]/2, self.roi_size[1]/2])
        line = line - origin # 将线段坐标的原点平移到ROI的中心

        # 使用ROI的尺寸对坐标进行缩放，使坐标范围大致在 [0, 1] 或 (-0.5, 0.5) 之间（取决于原点定义）
        # 此处 origin 是负值，line - origin 相当于将坐标移动到以 (roi_width/2, roi_height/2) 为原点的坐标系
        # 然后除以 roi_size，使得范围大致在 [0,1]
        eps = 1e-5 # 防止除零的小量
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps # 归一化因子 (宽度, 高度)
        line = line / norm # 缩放

        # 将点坐标重新展平
        line = line.flatten(-2, -1) # (..., num_sample * 2)

        return line
