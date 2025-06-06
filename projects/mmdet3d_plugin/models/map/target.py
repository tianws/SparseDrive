import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.builder import (BBOX_SAMPLERS, BBOX_ASSIGNERS)
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core import (build_assigner, build_sampler)
from mmdet.core.bbox.assigners import (AssignResult, BaseAssigner)

from ..base_target import BaseTargetWithDenoising


@BBOX_SAMPLERS.register_module() # 将SparsePoint3DTarget注册到MMDetection的BBOX_SAMPLERS注册表中
class SparsePoint3DTarget(BaseTargetWithDenoising): # 定义稀疏3D点/线目标分配器类
    """
    用于稀疏3D点/线（例如地图元素）检测的目标分配器。
    它使用一个分配器（如匈牙利分配器）来匹配预测和真实目标，
    并对坐标进行归一化处理。同时，它继承了去噪相关的功能。
    """
    def __init__(
        self,
        assigner=None,  # 分配器配置字典 (例如HungarianLinesAssigner)
        num_dn_groups=0,  # 去噪组的数量 (从基类继承，但在此类中可能不直接使用，具体看get_dn_anchors等方法的实现)
        dn_noise_scale=0.5,  # 去噪时添加的噪声尺度 (从基类继承)
        max_dn_gt=32,  # 用于去噪的最大真实目标数量 (从基类继承)
        add_neg_dn=True,  # 是否为去噪添加负样本 (从基类继承)
        num_temp_dn_groups=0,  # 时序去噪组的数量 (从基类继承)
        num_cls=3,  # 地图元素的类别数量
        num_sample=20,  # 每条线段/点序列的采样点数
        roi_size=(30, 60),  # 感兴趣区域(ROI)的尺寸 (宽度, 高度)，单位米，用于归一化线段坐标
    ):
        """
        构造函数。

        Args:
            assigner (dict, optional): 用于将预测与真实目标进行匹配的分配器配置。
            num_dn_groups (int, optional): （继承）去噪组数量。
            dn_noise_scale (float, optional): （继承）去噪噪声尺度。
            max_dn_gt (int, optional): （继承）用于去噪的最大GT数量。
            add_neg_dn (bool, optional): （继承）是否添加负去噪样本。
            num_temp_dn_groups (int, optional): （继承）时序去噪组数量。
            num_cls (int): 地图元素的类别数量。
            num_sample (int): 每个地图元素（如线）表示的点数。
            roi_size (tuple[float, float]): 用于归一化坐标的ROI尺寸 (宽度, 高度)。
        """
        super(SparsePoint3DTarget, self).__init__( # 调用父类BaseTargetWithDenoising的初始化函数
            num_dn_groups, num_temp_dn_groups
        )
        self.assigner = build_assigner(assigner) # 构建分配器 (例如HungarianLinesAssigner)
        self.dn_noise_scale = dn_noise_scale # 存储去噪参数 (即使在此类中不直接使用，也由基类管理)
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn

        self.num_cls = num_cls # 地图元素的类别数
        self.num_sample = num_sample # 每条线的点数
        self.roi_size = roi_size # 归一化时使用的ROI尺寸 (宽度, 高度)

    def sample(
        self,
        cls_preds,
        pts_preds,
        cls_targets,
        pts_targets,
    ):
        pts_targets  = [x.flatten(2, 3) if len(x.shape)==4 else x for x in pts_targets]
        indices = []
        for(cls_pred, pts_pred, cls_target, pts_target) in zip(
            cls_preds, pts_preds, cls_targets, pts_targets
        ):
            # normalize to (0, 1)
            pts_pred = self.normalize_line(pts_pred)
            pts_target = self.normalize_line(pts_target)
            preds=dict(lines=pts_pred, scores=cls_pred)
            gts=dict(lines=pts_target, labels=cls_target)
            indice = self.assigner.assign(preds, gts)
            indices.append(indice)
        
        bs, num_pred, num_cls = cls_preds.shape
        output_cls_target = cls_targets[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        output_box_target = pts_preds.new_zeros(pts_preds.shape)
        output_reg_weights = pts_preds.new_zeros(pts_preds.shape)
        for i, (pred_idx, target_idx, gt_permute_index) in enumerate(indices):
            if len(cls_targets[i]) == 0:
                continue
            permute_idx = gt_permute_index[pred_idx, target_idx]
            output_cls_target[i, pred_idx] = cls_targets[i][target_idx]
            output_box_target[i, pred_idx] = pts_targets[i][target_idx, permute_idx]
            output_reg_weights[i, pred_idx] = 1

        return output_cls_target, output_box_target, output_reg_weights

    def normalize_line(self, line):
        if line.shape[0] == 0:
            return line
        
        line = line.view(line.shape[:-1] + (self.num_sample, -1))
        
        origin = -line.new_tensor([self.roi_size[0]/2, self.roi_size[1]/2])
        line = line - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps
        line = line / norm
        line = line.flatten(-2, -1)

        return line


@BBOX_ASSIGNERS.register_module()
class HungarianLinesAssigner(BaseAssigner):
    """
        Computes one-to-one matching between predictions and ground truth.
        This class computes an assignment between the targets and the predictions
        based on the costs. The costs are weighted sum of three components:
        classification cost and regression L1 cost. The
        targets don't include the no_object, so generally there are more
        predictions than targets. After the one-to-one matching, the un-matched
        are treated as backgrounds. Thus each query prediction will be assigned
        with `0` or a positive integer indicating the ground truth index:
        - 0: negative sample, no assigned gt
        - positive integer: positive sample, index (1-based) of assigned gt
        Args:
            cls_weight (int | float, optional): The scale factor for classification
                cost. Default 1.0.
            bbox_weight (int | float, optional): The scale factor for regression
                L1 cost. Default 1.0. # 回归L1代价的比例因子。默认为1.0。
    """

    def __init__(self, cost:dict, **kwargs): # 构造函数
        """
        Args:
            cost (dict): 包含各种匹配代价配置的字典。
                         例如: {'cls_cost': {'type':'FocalLossCost', ...},
                                'reg_cost': {'type':'LinesL1Cost', ...},
                                'iou_cost': {'type':'IoUCost', ...} (可选)
                               }
                         这些代价模块会被用于计算预测和真实目标之间的匹配代价矩阵。
        """
        self.cost = build_match_cost(cost) # 构建组合的匹配代价计算模块

    def assign(self, # 执行分配的核心方法
               preds: dict, # 模型的预测输出字典，通常包含 'scores' (分类分数) 和 'lines' (线段点坐标)
               gts: dict,   # 真实目标字典，通常包含 'labels' (类别标签) 和 'lines' (真实线段点坐标)
               ignore_cls_cost=False, # 是否在计算总代价时忽略分类代价（例如在某些去噪场景下）
               gt_bboxes_ignore=None, # 要忽略的真实边界框 (当前实现中明确断言其为None，表示不支持)
               eps=1e-7): # 用于数值稳定性的epsilon值 (当前实现中未使用，但保留参数)
        """
            Computes one-to-one matching based on the weighted costs. # 基于加权代价计算一对一匹配。
            This method assign each query prediction to a ground truth or # 此方法将每个查询预测分配给一个真实目标或背景。
            background. The `assigned_gt_inds` with -1 means don't care, # `assigned_gt_inds`为-1表示不关心（忽略），
            0 means negative sample, and positive number is the index (1-based) # 0表示负样本（背景），正数是分配的GT的索引（基于1）。
            of assigned gt.                                                    # (注意：此函数实际返回的是匹配上的pred索引和gt索引，后续处理会生成AssignResult对象)
            The assignment is done in the following steps, the order matters. # 分配按以下步骤完成，顺序很重要。
            1. assign every prediction to -1 # (此步骤不在此函数中，通常在调用此函数的外部逻辑中完成初始化)
            2. compute the weighted costs # 2. 计算加权代价 (通过self.cost模块)
            3. do Hungarian matching on CPU based on the costs # 3. 基于代价在CPU上进行匈牙利匹配
            4. assign all to 0 (background) first, then for each matched pair # (此步骤不在此函数中)
            between predictions and gts, treat this prediction as foreground
            and assign the corresponding gt index (plus 1) to it.
            Args: # 参数说明
                preds (dict): 包含预测的字典，例如:
                    lines (Tensor): predicted normalized lines: # 预测的归一化线段：
                        [num_query, num_points*2] # [查询数量, 点数*2]
                    scores (Tensor): Predicted classification logits, shape # 预测的分类logits，形状
                        [num_query, num_class]. # [查询数量, 类别数量]。

                gts (dict): 包含真实目标的字典，例如:
                    lines (Tensor): Ground truth lines # 真实的线段
                        [num_gt, num_points*2] or [num_gt, num_perms, num_points*2].
                    labels (Tensor): Label of `gt_lines`, shape (num_gt,). # `gt_lines`的标签，形状 (GT数量,)。
                gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are # 被标记为`ignored`的真实边界框。
                    labelled as `ignored`. Default None. # 默认为None。
                eps (int | float, optional): A value added to the denominator for # 为数值稳定性添加到分母的值。
                    numerical stability. Default 1e-7. # 默认为1e-7。
            Returns: # 返回值
                tuple: (matched_row_inds, matched_col_inds, gt_permute_idx_for_matched_rows)
                       分别是匹配上的预测的行索引 (对应preds中的索引)、
                       匹配上的真实目标的列索引 (对应gts中的索引)，
                       以及真实目标的最佳排列的索引（如果self.cost.reg_cost.permute为True）。
                       如果无法匹配 (例如num_gts=0或num_preds=0)，则返回 (None, None, None)。
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.' # 明确断言不支持忽略GT边界框的情况
        
        num_gts, num_preds = gts['lines'].size(0), preds['lines'].size(0) # 获取真实目标数量和预测数量

        # 如果没有真实目标或没有预测，则无法进行匹配
        if num_gts == 0 or num_preds == 0:
            return None, None, None # 返回None表示没有有效的匹配结果

        # 计算加权代价矩阵
        gt_permute_idx_all = None # 初始化GT排列索引为None (形状是 num_preds, num_gts)

        # self.cost 是一个组合代价对象 (例如 MapQueriesCost)
        # 它会内部调用其包含的 cls_cost, reg_cost 等来计算总代价矩阵
        cost_val = self.cost(preds, gts, ignore_cls_cost)

        # 检查回归代价是否处理了排列 (例如，LinesL1Cost中permute=True)
        # self.cost.reg_cost 访问的是组合代价对象内部的回归代价模块
        if hasattr(self.cost, 'reg_cost') and hasattr(self.cost.reg_cost, 'permute') and self.cost.reg_cost.permute:
            cost, gt_permute_idx_all = cost_val # 如果处理排列，cost_val会返回 (cost_matrix, gt_permute_indices_matrix)
        else:
            cost = cost_val # 否则，cost_val只包含代价矩阵

        # 使用匈牙利算法 (linear_sum_assignment) 在CPU上进行匹配
        # linear_sum_assignment 找到使总代价最小的一对一分配方案
        cost_np = cost.detach().cpu().numpy() # 将代价张量转移到CPU并转为NumPy数组
        # matched_row_inds: 匹配上的预测的索引 (在cost矩阵的行索引，对应preds中的索引)
        # matched_col_inds: 匹配上的GT的索引 (在cost矩阵的列索引，对应gts中的索引)
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost_np)

        # 如果处理了排列，提取匹配上的GT的最佳排列索引
        gt_permute_idx_for_matched_rows = None
        if gt_permute_idx_all is not None and len(matched_row_inds) > 0:
            # gt_permute_idx_all 的形状是 (num_preds, num_gts)
            # 我们需要根据匹配结果 (matched_row_inds, matched_col_inds) 来选取对应的排列索引
            gt_permute_idx_for_matched_rows = gt_permute_idx_all[matched_row_inds, matched_col_inds]

        return matched_row_inds, matched_col_inds, gt_permute_idx_for_matched_rows