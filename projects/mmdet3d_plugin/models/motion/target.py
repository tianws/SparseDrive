import torch # 导入PyTorch库

from mmdet.core.bbox.builder import BBOX_SAMPLERS # 从MMDetection导入BBOX_SAMPLERS注册表

__all__ = ["MotionTarget", "PlanningTarget"] # 定义当使用 from .target import * 时，哪些名称会被导入


def get_cls_target( # 根据回归目标确定最佳的分类目标（即最佳模态）
    reg_preds,  # 预测的回归轨迹 (bs, num_pred, num_modes, fut_ts, coords_dim)
    reg_target, # 真实的回归轨迹 (bs, num_pred, fut_ts, coords_dim) - 注意：已经根据匹配分配给每个pred
    reg_weight, # 回归目标的权重 (bs, num_pred, fut_ts) - 标记有效时间步
):
    """
    根据预测的多模态轨迹与目标轨迹之间的距离，确定每个预测的最佳模态。

    Args:
        reg_preds (torch.Tensor): 预测的多模态轨迹，通常是相对位移。
        reg_target (torch.Tensor): 真实的轨迹，通常是相对位移。
        reg_weight (torch.Tensor): 真实轨迹的有效性权重/掩码。

    Returns:
        torch.Tensor: 每个预测的最佳模态索引 (bs, num_pred)。
    """
    bs, num_pred, mode, ts, d = reg_preds.shape # 获取维度信息

    # 将相对位移累加成绝对（或相对于共同起点的）轨迹
    reg_preds_cum = reg_preds.cumsum(dim=-2) # (bs, num_pred, num_modes, fut_ts, coords_dim)
    reg_target_cum = reg_target.cumsum(dim=-2) # (bs, num_pred, fut_ts, coords_dim)

    # 计算每个预测模态轨迹与目标轨迹之间的L2距离
    # reg_target_cum.unsqueeze(2): (bs, num_pred, 1, fut_ts, coords_dim)
    # dist: (bs, num_pred, num_modes, fut_ts)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)

    # 应用权重，只考虑有效时间步的距离
    # reg_weight.unsqueeze(2): (bs, num_pred, 1, fut_ts)
    dist = dist * reg_weight.unsqueeze(2)

    # 计算每个模态轨迹在所有有效时间步上的平均距离
    dist = dist.mean(dim=-1) # (bs, num_pred, num_modes)

    # 为每个预测选择平均距离最小的模态作为分类目标
    mode_idx = torch.argmin(dist, dim=-1) # (bs, num_pred)
    return mode_idx

def get_best_reg( # 根据最佳模态索引，从多模态预测中选出最佳的回归轨迹
    reg_preds,  # 预测的回归轨迹 (bs, num_pred, num_modes, fut_ts, coords_dim)
    reg_target, # 真实的回归轨迹 (bs, num_pred, fut_ts, coords_dim)
    reg_weight, # 回归目标的权重 (bs, num_pred, fut_ts)
):
    """
    首先确定最佳模态（同get_cls_target），然后从预测中选出该模态对应的轨迹。

    Args:
        reg_preds (torch.Tensor): 预测的多模态轨迹。
        reg_target (torch.Tensor): 真实的轨迹。
        reg_weight (torch.Tensor): 真实轨迹的有效性权重/掩码。

    Returns:
        torch.Tensor: 每个预测对应的最佳模态的轨迹 (bs, num_pred, fut_ts, coords_dim)。
    """
    bs, num_pred, mode, ts, d = reg_preds.shape
    # --- 与 get_cls_target 中相同的逻辑来找到最佳模态索引 ---
    reg_preds_cum = reg_preds.cumsum(dim=-2)
    reg_target_cum = reg_target.cumsum(dim=-2)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
    dist = dist * reg_weight.unsqueeze(2)
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1) # (bs, num_pred)
    # --- 结束寻找最佳模态索引 ---

    # 将mode_idx扩展维度以便于使用torch.gather从reg_preds中选取对应的模态
    # mode_idx: (bs, num_pred) -> (bs, num_pred, 1, 1, 1)
    # reg_preds: (bs, num_pred, num_modes, fut_ts, coords_dim)
    # mode_idx_expanded: (bs, num_pred, 1, fut_ts, coords_dim)
    mode_idx_expanded = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)

    # 使用gather沿模态维度(dim=2)选取最佳模态的轨迹
    best_reg = torch.gather(reg_preds, 2, mode_idx_expanded).squeeze(2) # (bs, num_pred, fut_ts, coords_dim)
    return best_reg


@BBOX_SAMPLERS.register_module() # 将MotionTarget注册到MMDetection的BBOX_SAMPLERS注册表中
class MotionTarget(): # 运动预测的目标分配和采样器
    """
    为其他智能体的运动预测任务生成目标。
    它接收检测头（或更早阶段的assigner）给出的匹配结果（indices），
    然后根据这些匹配为预测的agent轨迹分配真实的未来轨迹，并确定最佳的预测模态。
    """
    def __init__(
        self,
        # 此类目前没有自定义参数，但保留构造函数以备将来扩展
    ):
        super(MotionTarget, self).__init__()

    def sample(
        self,
        reg_pred,  # 模型预测的多模态轨迹 (bs, num_anchor, num_modes, fut_ts, coords_dim)
        gt_reg_target,  # 真实的未来轨迹列表，每个元素形状 (num_gt_i, fut_ts, coords_dim)
        gt_reg_mask,  # 真实未来轨迹的有效性掩码列表，每个元素形状 (num_gt_i, fut_ts)
        motion_loss_cache,  # 包含匹配索引的缓存字典，通常来自检测头的assigner结果
                            # motion_loss_cache['indices'] 是一个列表，每个元素是元组 (pred_idx, target_idx)
    ):
        """
        为运动预测任务采样和分配目标。

        Args:
            reg_pred (torch.Tensor): 预测的多模态轨迹。
            gt_reg_target (list[torch.Tensor]): 真实未来轨迹列表。
            gt_reg_mask (list[torch.Tensor]): 真实未来轨迹掩码列表。
            motion_loss_cache (dict): 包含先前阶段匹配结果的缓存。

        Returns:
            tuple:
                - cls_target (torch.Tensor): 分类目标（最佳模态索引）(bs, num_anchor)。
                - cls_weight (torch.Tensor): 分类损失权重 (bs, num_anchor)。
                - best_reg (torch.Tensor): 最佳模态的预测轨迹 (bs, num_anchor, fut_ts, coords_dim)。
                - reg_target_assigned (torch.Tensor): 分配给每个预测的真实轨迹 (bs, num_anchor, fut_ts, coords_dim)。
                - reg_weight_assigned (torch.Tensor): 分配给每个预测的真实轨迹掩码 (bs, num_anchor, fut_ts)。
                - num_pos (torch.Tensor): 正样本的数量。
        """
        bs, num_anchor, mode, ts, d = reg_pred.shape # 获取维度信息

        # 初始化目标张量
        reg_target_assigned = reg_pred.new_zeros((bs, num_anchor, ts, d)) # 分配给每个预测的GT轨迹
        reg_weight_assigned = reg_pred.new_zeros((bs, num_anchor, ts)) # 分配给每个预测的GT掩码

        indices = motion_loss_cache['indices'] # 获取检测阶段的匹配结果
        num_pos = reg_pred.new_tensor([0], dtype=torch.float) # 初始化正样本计数器

        for i, (pred_idx, target_idx) in enumerate(indices): # 遍历每个样本的匹配结果
            # pred_idx: 当前样本中，匹配上GT的预测query的索引
            # target_idx: 当前样本中，被匹配上的GT的索引
            if pred_idx is None or len(gt_reg_target[i]) == 0: # 如果没有匹配或没有GT
                continue
            # 将匹配上的GT轨迹和掩码分配给对应的预测query
            reg_target_assigned[i, pred_idx] = gt_reg_target[i][target_idx]
            reg_weight_assigned[i, pred_idx] = gt_reg_mask[i][target_idx]
            num_pos += len(pred_idx) # 累加正样本数量
        
        # 根据分配的GT轨迹和多模态预测，确定分类目标（最佳模态）
        cls_target = get_cls_target(reg_pred, reg_target_assigned, reg_weight_assigned)
        # 分类权重：如果一个预测query至少有一个有效的时间步的GT轨迹，则其分类权重为1
        cls_weight = reg_weight_assigned.any(dim=-1)
        # 获取最佳模态对应的预测轨迹
        best_reg_pred = get_best_reg(reg_pred, reg_target_assigned, reg_weight_assigned)

        return cls_target, cls_weight, best_reg_pred, reg_target_assigned, reg_weight_assigned, num_pos


@BBOX_SAMPLERS.register_module() # 将PlanningTarget注册到MMDetection的BBOX_SAMPLERS注册表中
class PlanningTarget(): # 自车规划的目标分配和采样器
    """
    为自车（ego）路径规划任务生成目标。
    它根据给定的高层命令（如左转、直行、右转）选择对应的预测轨迹模态，
    并与真实的自车未来轨迹进行比较以生成损失计算所需的目标。
    """
    def __init__(
        self,
        ego_fut_ts, # 自车规划的未来时间步数量
        ego_fut_mode, # 每个命令下的自车规划轨迹模态数量
    ):
        super(PlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

    def sample(
        self,
        cls_pred,  # 规划的分类预测 (bs, num_plan_queries, num_cmds * ego_fut_mode) 或 (bs, num_cmds, ego_fut_mode)
        reg_pred,  # 规划的轨迹预测 (bs, num_plan_queries, num_cmds * ego_fut_mode, ego_fut_ts, 2) 或 (bs, num_cmds, ego_fut_mode, ego_fut_ts, 2)
        gt_reg_target,  # 真实的自车未来轨迹 (bs, ego_fut_ts, 2)
        gt_reg_mask,  # 真实自车未来轨迹的掩码 (bs, ego_fut_ts)
        data,  # 输入数据字典，包含 'gt_ego_fut_cmd' (bs, num_cmds)
    ):
        """
        为规划任务采样和分配目标。

        Args:
            cls_pred (torch.Tensor): 规划的分类/模态选择 logits。
            reg_pred (torch.Tensor): 预测的多模态规划轨迹。
            gt_reg_target (torch.Tensor): 真实的自车未来轨迹。
            gt_reg_mask (torch.Tensor): 真实自车未来轨迹的掩码。
            data (dict): 包含真实高级命令 'gt_ego_fut_cmd' 的字典。

        Returns:
            tuple:
                - cls_pred_selected_cmd (torch.Tensor): 根据真实命令选择后的分类 logits (bs, 1, ego_fut_mode)。
                - cls_target (torch.Tensor): 分类目标（最佳模态索引）(bs, 1)。
                - cls_weight (torch.Tensor): 分类损失权重 (bs, 1)。
                - best_reg_pred (torch.Tensor): 最佳模态的预测轨迹 (bs, 1, ego_fut_ts, 2)。
                - gt_reg_target_expanded (torch.Tensor): 扩展后的真实轨迹 (bs, 1, ego_fut_ts, 2)。
                - gt_reg_mask_expanded (torch.Tensor): 扩展后的真实轨迹掩码 (bs, 1, ego_fut_ts)。
        """
        # 扩展GT轨迹和掩码的维度，以匹配多模态预测的形状 (增加一个模态维度)
        gt_reg_target_expanded = gt_reg_target.unsqueeze(1) # (bs, 1, ego_fut_ts, 2)
        gt_reg_mask_expanded = gt_reg_mask.unsqueeze(1)   # (bs, 1, ego_fut_ts)

        bs = reg_pred.shape[0] # 批量大小
        bs_indices = torch.arange(bs, device=reg_pred.device) # 批次索引 [0, 1, ..., bs-1]

        # 从data中获取真实的高级驾驶命令 (例如，左转、直行、右转)
        # gt_ego_fut_cmd 通常是one-hot编码或概率分布 (bs, num_cmds)
        cmd_idx = data['gt_ego_fut_cmd'].argmax(dim=-1) # 选择概率最大的命令索引 (bs)

        # 根据真实命令索引，从预测中选择对应的分类分数和轨迹
        # 假设 cls_pred 和 reg_pred 的形状是 (bs, num_plan_queries, num_cmds, ego_fut_mode, ...)
        # 或者，如果 num_plan_queries=1, 则是 (bs, num_cmds, ego_fut_mode, ...)
        # 这里假设 num_plan_queries = 1 (即一个ego query对应所有命令和模态)
        # 并且 cls_pred/reg_pred 的第二维是命令维度 (num_cmds, 通常为3)

        # Reshape cls_pred: (bs, num_total_modes) -> (bs, num_cmds, ego_fut_mode)
        # Reshape reg_pred: (bs, num_total_modes, ts, d) -> (bs, num_cmds, ego_fut_mode, ts, d)
        # num_total_modes = num_cmds * ego_fut_mode
        # 这里的reshape依赖于输入cls_pred和reg_pred的原始形状，假设它们已经扁平化了命令和模态维度。
        # 如果输入已经是 (bs, num_cmds, ego_fut_mode, ...)，则不需要第一步的reshape。
        # 假设 cls_pred (bs, C), reg_pred (bs, C, ts, d) where C = num_cmds * ego_fut_mode
        num_cmds = data['gt_ego_fut_cmd'].shape[1] # 通常为3

        cls_pred_reshaped = cls_pred.reshape(bs, num_cmds, self.ego_fut_mode)
        reg_pred_reshaped = reg_pred.reshape(bs, num_cmds, self.ego_fut_mode, self.ego_fut_ts, 2)

        # 根据cmd_idx选取对应命令的预测
        cls_pred_selected_cmd = cls_pred_reshaped[bs_indices, cmd_idx] # (bs, ego_fut_mode)
        reg_pred_selected_cmd = reg_pred_reshaped[bs_indices, cmd_idx] # (bs, ego_fut_mode, ego_fut_ts, 2)

        # 在选定命令的模态中，找到与GT最匹配的模态作为分类目标
        # gt_reg_target_expanded: (bs, 1, ts, d) -> 需要扩展到 (bs, ego_fut_mode, ts, d) 或调整reg_pred_selected_cmd
        # 这里假设 get_cls_target 的 reg_target 参数期望 (bs, num_pred_queries_for_target, ts, d)
        # 当前 reg_pred_selected_cmd 是 (bs, ego_fut_mode, ts, d)
        # gt_reg_target_expanded 是 (bs, 1, ts, d)
        # cls_target: (bs, ego_fut_mode) -> (bs) after argmin
        # 这里将num_pred视为1 (因为是自车规划)，num_modes视为ego_fut_mode
        cls_target = get_cls_target(reg_pred_selected_cmd.unsqueeze(1), # (bs, 1, ego_fut_mode, ts, d)
                                    gt_reg_target_expanded,             # (bs, 1, ts, d)
                                    gt_reg_mask_expanded)               # (bs, 1, ts)
        cls_target = cls_target.squeeze(1) # (bs) -> 最佳模态索引

        # 分类权重：如果GT轨迹至少有一个有效时间步，则权重为1
        cls_weight = gt_reg_mask_expanded.any(dim=-1) # (bs, 1)

        # 获取最佳模态对应的预测轨迹
        best_reg_pred = get_best_reg(reg_pred_selected_cmd.unsqueeze(1),
                                     gt_reg_target_expanded,
                                     gt_reg_mask_expanded)
        best_reg_pred = best_reg_pred.squeeze(1) # (bs, ego_fut_ts, 2)

        # 返回的目标张量需要与损失函数期望的输入形状对齐
        # cls_pred_selected_cmd: (bs, ego_fut_mode) - 用于计算分类损失的预测logits
        # cls_target: (bs) - 最佳模态的索引，作为分类目标
        # cls_weight: (bs, 1) - 分类损失的权重
        # best_reg_pred: (bs, ego_fut_ts, 2) - 最佳模态的预测轨迹，用于与GT比较计算回归损失
        # gt_reg_target_expanded.squeeze(1): (bs, ego_fut_ts, 2) - 真实的轨迹，用于回归损失
        # gt_reg_mask_expanded.squeeze(1): (bs, ego_fut_ts) - 真实轨迹的掩码，用于回归损失
        return (cls_pred_selected_cmd, # (bs, ego_fut_mode)
                cls_target,            # (bs)
                cls_weight.squeeze(1), # (bs)
                best_reg_pred,         # (bs, ego_fut_ts, 2)
                gt_reg_target_expanded.squeeze(1), # (bs, ego_fut_ts, 2)
                gt_reg_mask_expanded.squeeze(1))   # (bs, ego_fut_ts)
