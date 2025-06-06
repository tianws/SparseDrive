import torch # 导入PyTorch库
from mmdet.core.bbox.match_costs.builder import MATCH_COST # 从MMDetection导入MATCH_COST注册表
from mmdet.core.bbox.match_costs import build_match_cost # 从MMDetection导入构建匹配代价的函数
from torch.nn.functional import smooth_l1_loss # 从PyTorch导入smooth_l1_loss函数


@MATCH_COST.register_module() # 将LinesL1Cost注册到MMDetection的MATCH_COST注册表中
class LinesL1Cost(object): # 定义线段L1代价计算类
    """LinesL1Cost. # 线段L1代价。
     Args: # 参数说明
         weight (int | float, optional): loss_weight # 代价权重，默认为1.0。
    """

    def __init__(self, weight=1.0, beta=0.0, permute=False):
        """
        构造函数。

        Args:
            weight (float, optional): 此代价项的权重。默认为1.0。
            beta (float, optional): smooth L1 loss的beta参数。如果大于0，则使用smooth L1，
                                  否则使用标准L1。默认为0.0 (即标准L1)。
            permute (bool, optional): GT线段是否包含多种排列方式。
                                    如果为True，gt_lines的形状应为 [num_gt, num_permute, 2*num_points]，
                                    代价计算会考虑所有排列并取最小值。默认为False。
        """
        self.weight = weight # 代价权重
        self.permute = permute # 是否处理GT线段的排列
        self.beta = beta # smooth L1 loss的beta参数

    def __call__(self, lines_pred, gt_lines, **kwargs): # 使类实例可调用
        """
        计算预测线段和真实线段之间的L1或Smooth L1代价。

        Args:
            lines_pred (Tensor): 预测的归一化线段，形状: [num_query, 2*num_points]。
                                 num_query是预测数量，2*num_points是线段点坐标展平后的维度。
            gt_lines (Tensor): 真实的线段。
                               形状: [num_gt, 2*num_points] (如果permute=False)
                               或 [num_gt, num_permute, 2*num_points] (如果permute=True)。
        Returns:
            torch.Tensor: 带权重的回归代价矩阵，形状 [num_query, num_gt]。
                          如果permute=True，还会返回一个gt_permute_index张量。
        """        
        if self.permute: # 如果处理排列
            assert len(gt_lines.shape) == 3, "gt_lines should be [num_gt, num_permute, dim] if permute is True"
        else:
            assert len(gt_lines.shape) == 2, "gt_lines should be [num_gt, dim] if permute is False"

        num_pred, num_gt = len(lines_pred), len(gt_lines) # 获取预测数量和真实GT数量

        if self.permute:
            # permute-invarint labels # 处理排列不变的标签
            # 将gt_lines展平为 (num_gt*num_permute, 2*num_pts)，以便后续计算所有排列的代价
            gt_lines_flat = gt_lines.flatten(0, 1)
        else:
            gt_lines_flat = gt_lines

        num_pts = lines_pred.shape[-1] // 2 # 每条线段的点数

        if self.beta > 0: # 如果使用Smooth L1 Loss
            # lines_pred扩展为 (num_pred, 1, dim)，gt_lines_flat扩展为 (1, num_gt_flat, dim)
            # 以便计算所有预测和所有（可能是排列后的）GT之间的smooth L1距离
            lines_pred_expanded = lines_pred.unsqueeze(1).repeat(1, len(gt_lines_flat), 1)
            gt_lines_expanded = gt_lines_flat.unsqueeze(0).repeat(num_pred, 1, 1)
            # 计算smooth L1损失，reduction='none'表示不进行规约，保留每个元素对的损失
            # .sum(-1) 沿最后一个维度（点坐标维度）求和，得到每个(pred, gt_perm)对的线段总损失
            dist_mat = smooth_l1_loss(lines_pred_expanded, gt_lines_expanded, reduction='none', beta=self.beta).sum(-1)

        else: # 如果使用标准L1 Loss
            # torch.cdist计算所有行向量对之间的p范数距离，p=1表示L1距离（曼哈顿距离）
            dist_mat = torch.cdist(lines_pred, gt_lines_flat, p=1)

        if num_pts > 0 :
            dist_mat = dist_mat / num_pts # 按点数归一化代价

        if self.permute:
            # dist_mat: (num_pred, num_gt*num_permute)
            # 将代价矩阵重塑为 (num_pred, num_gt, num_permute) 以便找到每个(pred, gt)对的最佳排列
            dist_mat = dist_mat.view(num_pred, num_gt, -1)
            # 沿排列维度取最小值，得到每个(pred, gt)对的最小代价和对应的排列索引
            dist_mat, gt_permute_index = torch.min(dist_mat, 2)
            return dist_mat * self.weight, gt_permute_index # 返回带权重的代价和最佳排列索引
        
        return dist_mat * self.weight # 返回带权重的代价矩阵


@MATCH_COST.register_module() # 将MapQueriesCost注册到MMDetection的MATCH_COST注册表中
class MapQueriesCost(object): # 定义地图查询代价计算类，用于组合多种代价
    """
    计算地图元素查询（通常是线段或点序列）的匹配代价。
    这个代价是分类代价、回归代价（例如LinesL1Cost）以及可选的IoU代价的加权和。
    """
    def __init__(self, cls_cost, reg_cost, iou_cost=None):
        """
        构造函数。

        Args:
            cls_cost (dict): 分类代价的配置字典。
            reg_cost (dict): 回归代价的配置字典。
            iou_cost (dict, optional): IoU代价的配置字典。默认为None。
        """
        self.cls_cost = build_match_cost(cls_cost) # 构建分类代价计算模块
        self.reg_cost = build_match_cost(reg_cost) # 构建回归代价计算模块

        self.iou_cost = None # 初始化IoU代价为None
        if iou_cost is not None: # 如果配置了IoU代价
            self.iou_cost = build_match_cost(iou_cost) # 构建IoU代价计算模块

    def __call__(self, preds: dict, gts: dict, ignore_cls_cost: bool = False):
        """
        计算预测和真实目标之间的总匹配代价。

        Args:
            preds (dict): 包含模型预测的字典，期望键如 'scores' (分类分数) 和 'lines' (线段预测)。
            gts (dict): 包含真实目标的字典，期望键如 'labels' (类别标签) 和 'lines' (真实线段)。
            ignore_cls_cost (bool): 是否忽略分类代价。在某些阶段（如去噪）可能为True。

        Returns:
            torch.Tensor or tuple: 总代价矩阵。如果回归代价处理排列，则返回 (cost, gt_permute_idx)。
        """

        # Classification cost. # 分类代价。
        cls_cost = self.cls_cost(preds['scores'], gts['labels']) # 计算分类代价

        # Regression cost. # 回归代价。
        regkwargs = {} # 初始化回归代价的额外参数
        # DynamicLinesCost 可能需要额外的掩码信息，这里暂时注释掉了
        # if 'masks' in preds and 'masks' in gts:
        #     assert isinstance(self.reg_cost, DynamicLinesCost), ' Issues!!'
        #     regkwargs = {
        #         'masks_pred': preds['masks'],
        #         'masks_gt': gts['masks'],
        #     }

        reg_cost_val = self.reg_cost(preds['lines'], gts['lines'], **regkwargs) # 计算回归代价
        gt_permute_idx = None # 初始化排列索引为None
        if self.reg_cost.permute: # 如果回归代价处理了排列
            reg_cost, gt_permute_idx = reg_cost_val # 解包代价和排列索引
        else:
            reg_cost = reg_cost_val

        # Weighted sum of above costs. # 上述代价的加权和。
        if ignore_cls_cost: # 如果忽略分类代价 (例如在去噪任务中，类别是给定的)
            cost = reg_cost
        else:
            cost = cls_cost + reg_cost # 总代价 = 分类代价 + 回归代价

        # IoU cost. # IoU代价。
        if self.iou_cost is not None: # 如果配置了IoU代价
            # 注意：计算线段的IoU可能不直接，这里可能指代的是基于线段生成的掩码或包围框的IoU
            iou_cost_val = self.iou_cost(preds['lines'],gts['lines'])
            cost += iou_cost_val # 将IoU代价加到总代价中
        
        if self.reg_cost.permute and gt_permute_idx is not None: # 如果处理了排列
            return cost, gt_permute_idx # 返回总代价和排列索引
        return cost # 返回总代价
