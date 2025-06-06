import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块

from mmcv.utils import build_from_cfg  # 从MMCV导入从配置字典构建对象的函数
from mmdet.models.builder import LOSSES  # 从MMDetection的模型构建器中导入LOSSES注册表

from projects.mmdet3d_plugin.core.box3d import *  # 从项目中导入3D边界框相关的常量定义 (X, Y, Z, SIN_YAW, COS_YAW, CNS, YNS等)


@LOSSES.register_module()  # 将SparseBox3DLoss注册到MMDetection的LOSSES注册表中
class SparseBox3DLoss(nn.Module):  # 定义稀疏3D边界框损失类
    """
    用于稀疏3D目标检测的损失函数集合。
    它主要包含一个边界框回归损失，并可选地包含中心度损失（centerness loss）
    和偏航度损失（yawness loss），这些辅助损失通常用于提高定位和方向预测的质量。
    此外，该损失还处理了一些特定类别（如nuScenes中的barrier）偏航角180度模糊性的情况。
    """
    def __init__(
        self,
        loss_box,  # 边界框回归损失的配置字典
        loss_centerness=None,  # 中心度损失的配置字典 (可选)
        loss_yawness=None,  # 偏航度损失的配置字典 (可选)
        cls_allow_reverse=None,  # 允许偏航角反向180度的类别ID列表 (可选)
    ):
        super().__init__()

        def build(cfg, registry): # 定义一个内部辅助函数来构建损失模块
            if cfg is None: # 如果配置为None，则不构建
                return None
            return build_from_cfg(cfg, registry) # 从配置字典和注册表构建模块

        self.loss_box = build(loss_box, LOSSES)  # 构建边界框回归损失模块
        self.loss_cns = build(loss_centerness, LOSSES)  # 构建中心度损失模块 (如果配置了)
        self.loss_yns = build(loss_yawness, LOSSES)  # 构建偏航度损失模块 (如果配置了)
        self.cls_allow_reverse = cls_allow_reverse  # 存储允许偏航角反向的类别列表

    def forward(
        self,
        box,  # 预测的边界框参数 (..., D)，D通常包含x,y,z,w,l,h,sin_yaw,cos_yaw等
        box_target,  # 真实的边界框参数 (..., D)
        weight=None,  # 样本权重 (可选)
        avg_factor=None,  # 用于平均损失的因子 (可选)
        prefix="",  # 损失名称的前缀 (可选)
        suffix="",  # 损失名称的后缀 (可选)
        quality=None,  # 预测的质量参数，可能包含centerness和yawness的logits (可选)
        cls_target=None,  # 真实目标的类别标签 (可选，用于cls_allow_reverse)
        **kwargs,  # 其他关键字参数
    ):
        """
        计算损失。

        Args:
            box (torch.Tensor): 预测的边界框。
            box_target (torch.Tensor): 真实的边界框。
            weight (torch.Tensor, optional): 每个样本的损失权重。
            avg_factor (int, optional): 用于归一化损失的平均因子。
            prefix (str, optional): 添加到损失名称字典键的前缀。
            suffix (str, optional): 添加到损失名称字典键的后缀。
            quality (torch.Tensor, optional): 预测的质量得分，通常包含中心度和偏航度预测。
                                           形状应为 (..., Q)，Q >= 2。
            cls_target (torch.Tensor, optional): 真实目标的类别标签，用于处理允许方向反转的类别。

        Returns:
            dict[str, torch.Tensor]: 包含计算得到的各项损失的字典。
        """
        # 处理某些类别（如barrier）偏航角180度模糊性的问题
        # 如果一个类别的物体旋转180度后外观和功能不变，则其预测的偏航角与真实偏航角相差180度也应被认为是正确的。
        if self.cls_allow_reverse is not None and cls_target is not None:
            # 计算预测偏航角和目标偏航角之间的余弦相似度
            # box_target[..., [SIN_YAW, COS_YAW]] 提取 (sin, cos)
            # box[..., [SIN_YAW, COS_YAW]] 提取 (sin_pred, cos_pred)
            # 如果余弦相似度小于0，表示角度差大于90度（可能接近180度）
            if_reverse = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]], # 真实yaw向量
                    box[..., [SIN_YAW, COS_YAW]],       # 预测yaw向量
                    dim=-1,  # 沿最后一个维度计算余弦相似度
                )
                < 0  # 判断是否方向大致相反 (夹角大于90度)
            )
            # 检查当前目标的类别是否在允许反转的列表中，并且其方向确实大致相反
            if_reverse = (
                torch.isin( # 判断cls_target中的元素是否存在于cls_allow_reverse列表中
                    cls_target, cls_target.new_tensor(self.cls_allow_reverse)
                )
                & if_reverse # 逻辑与操作
            )
            # 如果需要反转，则将目标边界框的 (sin_yaw, cos_yaw) 取反，相当于将yaw角旋转180度
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                if_reverse[..., None],  # 条件张量，需要扩展维度以匹配 (sin, cos)
                -box_target[..., [SIN_YAW, COS_YAW]],  # 如果为True，取反
                box_target[..., [SIN_YAW, COS_YAW]],  # 如果为False，保持不变
            )

        output = {}  # 初始化损失字典
        # 计算主要的边界框回归损失
        box_loss = self.loss_box(
            box, box_target, weight=weight, avg_factor=avg_factor
        )
        output[f"{prefix}loss_box{suffix}"] = box_loss  # 添加到损失字典

        if quality is not None:  # 如果提供了质量预测 (通常包含centerness和yawness的logits)
            # CNS 和 YNS 是预定义的索引，指向quality张量中centerness和yawness预测值的位置
            cns_pred_logits = quality[..., CNS]  # 预测的中心度 logits
            yns_pred_prob = quality[..., YNS].sigmoid()  # 预测的偏航度 logits，经过sigmoid转换为概率

            # 计算中心度目标：基于预测框中心和目标框中心之间的距离
            # 距离越近，目标中心度越高 (通过指数衰减函数计算)
            # X, Y, Z 是预定义的索引
            cns_target = torch.norm(
                box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], p=2, dim=-1
            )
            cns_target = torch.exp(-cns_target)  # 目标中心度

            # 计算中心度损失 (通常是二值交叉熵或类似的损失)
            cns_loss = self.loss_cns(cns_pred_logits, cns_target, avg_factor=avg_factor)
            output[f"{prefix}loss_cns{suffix}"] = cns_loss  # 添加到损失字典

            # 计算偏航度目标：基于预测偏航角和目标偏航角（可能已反转）的余弦相似度
            # 如果余弦相似度大于0 (夹角小于90度)，则认为方向一致，目标为1，否则为0
            yns_target = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                > 0 # 方向是否一致
            )
            yns_target = yns_target.float()  # 转换为浮点型 (0.0 或 1.0)

            # 计算偏航度损失 (通常是二值交叉熵)
            yns_loss = self.loss_yns(yns_pred_prob, yns_target, avg_factor=avg_factor)
            output[f"{prefix}loss_yns{suffix}"] = yns_loss  # 添加到损失字典

        return output  # 返回包含各项损失的字典
