from typing import Optional

import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.detection3d.decoder import *
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners


@BBOX_CODERS.register_module() # 将此类注册到MMDetection的BBOX_CODERS注册表中
class SparseBox3DMotionDecoder(SparseBox3DDecoder): # 定义稀疏3D框运动解码器，继承自基础的SparseBox3DDecoder
    """
    用于解码包含运动轨迹的稀疏3D边界框的解码器。
    它扩展了父类 SparseBox3DDecoder 的功能，以额外处理和输出轨迹预测信息。
    这个解码器在目标检测的基础上，增加了对每个检测到的目标未来运动轨迹的解码。
    """
        此运动解码器类重用这些参数，并专注于在其 decode 方法中添加运动相关的解码逻辑。
        """
        super(SparseBox3DMotionDecoder, self).__init__() # 调用父类SparseBox3DDecoder的初始化函数

    def decode(
        self,
        cls_scores,  # 类别分数张量列表 (通常对应解码器不同层级的输出)
        box_preds,  # 边界框预测张量列表
        instance_id=None,  # 实例ID张量 (可选)
        quality=None,  # 质量预测张量列表 (可选)
        motion_output=None,  # 包含运动预测相关信息的字典
        output_idx=-1,  # 指定使用哪一层级的输出进行解码
    ):
        """
        解码模型输出，生成包含3D边界框、运动轨迹等的最终结果。
        此方法覆盖并扩展了父类`SparseBox3DDecoder`的`decode`方法，
        在基础的目标检测结果上增加了运动轨迹和相关历史信息的解码。

        Args:
            cls_scores (list[torch.Tensor]): 类别分数。每个元素形状 (bs, num_query, num_classes)。
            box_preds (list[torch.Tensor]): 边界框预测。每个元素形状 (bs, num_query, box_dim)。
            instance_id (torch.Tensor, optional): 实例ID，形状 (bs, num_query)。
            quality (list[torch.Tensor], optional): 质量预测。每个元素形状 (bs, num_query, num_quality_metrics)。
            motion_output (dict, optional): 包含运动预测特定输出的字典，例如:
                                           'prediction': 预测的轨迹 (list[Tensor])
                                           'classification': 轨迹的分类/置信度 (list[Tensor])
                                           'anchor_queue': 历史锚点队列 (list[Tensor] or Tensor)
                                           'period': 历史锚点的周期/时间间隔 (list[Tensor] or Tensor)
            output_idx (int): 使用哪个解码器层级的输出。默认为-1 (最后一层)。

        Returns:
            list[dict]: 每个样本的解码结果列表。每个字典除了父类可能输出的
                        'boxes_3d', 'scores_3d', 'labels_3d', 'cls_scores', 'instance_ids' 外，
                        还包含:
                        'trajs_3d': 解码后的未来轨迹 (torch.Tensor, CPU)。
                        'trajs_score': 轨迹的置信度/概率 (torch.Tensor, CPU)。
                        'anchor_queue': 解码后的历史锚点队列 (torch.Tensor, CPU)。
                        'period': 历史锚点的周期 (torch.Tensor, CPU)。
        """
        # --- 开始：与父类SparseBox3DDecoder中类似的初始解码逻辑 ---
        squeeze_cls = instance_id is not None # 如果提供了instance_id，则后续处理类别ID时可能需要特殊处理

        cls_scores_selected = cls_scores[output_idx].sigmoid() # 选择指定层输出并应用sigmoid

        cls_ids_topk_original = None # 用于存储在squeeze_cls=True情况下的原始类别ID
        if squeeze_cls: # 如果每个query只输出一个类别
            cls_scores_max, cls_ids_topk_original = cls_scores_selected.max(dim=-1) # 取最大类别分数和ID
            cls_scores_processed = cls_scores_max.unsqueeze(dim=-1) # (bs, num_query, 1)
        else:
            cls_scores_processed = cls_scores_selected # (bs, num_query, num_classes)

        box_preds_selected = box_preds[output_idx] # 选择指定层输出的框预测
        bs, num_pred_queries, num_cls_effective = cls_scores_processed.shape # 批量大小, query数量, 有效类别数 (1或num_classes)

        # 从所有(query, class)组合中选择top-k (self.num_output是父类定义的输出数量上限)
        cls_scores_flat = cls_scores_processed.flatten(start_dim=1) # (bs, num_query * num_cls_effective)

        current_k = min(self.num_output, cls_scores_flat.shape[1])
        if current_k == 0:
             return [{"trajs_3d": torch.empty(0), "trajs_score": torch.empty(0),
                      "anchor_queue": torch.empty(0), "period": torch.empty(0)} for _ in range(bs)]


        cls_scores_topk, indices_topk = cls_scores_flat.topk(
            current_k, dim=1, sorted=self.sorted
        )

        query_indices_topk: torch.Tensor # (bs, current_k) 每个topk结果对应的原始query索引
        cls_ids_topk: torch.Tensor       # (bs, current_k) 每个topk结果对应的类别ID

        if not squeeze_cls: # 如果每个query对所有类别都有分数
            query_indices_topk = indices_topk // num_cls_effective # 通过整除恢复query索引
            cls_ids_topk = indices_topk % num_cls_effective      # 通过取模恢复类别ID
        else: # 如果已经squeeze了，indices_topk直接对应query索引
            query_indices_topk = indices_topk
            cls_ids_topk = torch.gather(cls_ids_topk_original, 1, query_indices_topk) # 用query索引从原始ID中gather

        mask_thresh = None # 分数阈值掩码
        if self.score_threshold is not None:
            mask_thresh = cls_scores_topk >= self.score_threshold

        # 处理质量得分 (如centerness)
        quality_selected = None
        if quality is not None and output_idx < len(quality) and quality[output_idx] is not None:
            quality_selected = quality[output_idx]

        cls_scores_final = cls_scores_topk
        cls_scores_origin_topk = cls_scores_topk.clone() # 保留原始topk分数

        if quality_selected is not None:
            centerness = quality_selected[..., CNS] # (bs, num_pred_queries)
            centerness_topk = torch.gather(centerness, 1, query_indices_topk) # 获取topk query对应的centerness

            cls_scores_final = cls_scores_topk * centerness_topk.sigmoid() # 类别分数乘以centerness

            cls_scores_final, idx_after_quality = torch.sort(cls_scores_final, dim=1, descending=True) # 重排序
            cls_ids_topk = torch.gather(cls_ids_topk, 1, idx_after_quality)
            query_indices_topk = torch.gather(query_indices_topk, 1, idx_after_quality)
            cls_scores_origin_topk = torch.gather(cls_scores_origin_topk, 1, idx_after_quality)

            if mask_thresh is not None:
                mask_thresh = torch.gather(mask_thresh, 1, idx_after_quality)
        # --- 结束：与父类SparseBox3DDecoder中类似的初始解码逻辑 ---

        output = [] # 初始化输出列表
        # 从motion_output字典中提取运动预测和历史信息
        # anchor_queue可能是列表或已堆叠的Tensor，确保它是Tensor
        anchor_queue_raw = motion_output["anchor_queue"]
        if isinstance(anchor_queue_raw, list):
            anchor_queue_tensor = torch.stack(anchor_queue_raw, dim=2) # (bs, num_query, queue_len, box_dim)
        else:
            anchor_queue_tensor = anchor_queue_raw
        period_tensor = motion_output["period"] # (bs, num_query)

        for i in range(bs): # 遍历每个样本
            # 获取当前样本经过topk和quality调整后的类别ID、分数和对应的原始query索引
            category_ids_sample = cls_ids_topk[i]
            scores_sample = cls_scores_final[i]
            current_query_indices_filtered = query_indices_topk[i] # 这些是经过排序和筛选前的query索引

            # 根据选中的query索引获取框预测
            box_sample_pred = box_preds_selected[i, current_query_indices_filtered]

            # 应用分数阈值
            mask_sample_final = None
            if self.score_threshold is not None and mask_thresh is not None:
                mask_sample_final = mask_thresh[i]
                category_ids_sample = category_ids_sample[mask_sample_final]
                scores_sample = scores_sample[mask_sample_final]
                box_sample_pred = box_sample_pred[mask_sample_final]
                current_query_indices_filtered = current_query_indices_filtered[mask_sample_final] # 更新query索引以反映阈值过滤

            # 如果过滤后没有检测结果，则添加空字典或特定标记
            if box_sample_pred.shape[0] == 0:
                output.append({
                    "boxes_3d": torch.empty(0, box_preds_selected.shape[-1] - 2 + 1 + (box_preds_selected.shape[-1]-7), device='cpu'), # 调整维度以匹配decode_box输出
                    "scores_3d": torch.empty(0, device='cpu'),
                    "labels_3d": torch.empty(0, dtype=torch.long, device='cpu'),
                    "trajs_3d": torch.empty(0, device='cpu'),
                    "trajs_score": torch.empty(0, device='cpu'),
                    "anchor_queue": torch.empty(0, device='cpu'),
                    "period": torch.empty(0, device='cpu')
                })
                if quality_selected is not None: output[-1]["cls_scores"] = torch.empty(0, device='cpu')
                if instance_id is not None: output[-1]["instance_ids"] = torch.empty(0, dtype=torch.long, device='cpu')
                continue

            cls_scores_origin_sample = None
            if quality_selected is not None: # 如果使用了质量进行分数调整
                cls_scores_origin_sample = cls_scores_origin_topk[i] # 获取调整前的原始分数
                if self.score_threshold is not None and mask_sample_final is not None:
                    cls_scores_origin_sample = cls_scores_origin_sample[mask_sample_final]

            box_decoded_sample = decode_box(box_sample_pred) # 解码边界框参数 (x,y,z,w,l,h,yaw_rad,vx,vy,vz)

            # 初始化当前样本的输出字典
            sample_output_dict = {
                "boxes_3d": box_decoded_sample.cpu(),
                "scores_3d": scores_sample.cpu(),
                "labels_3d": category_ids_sample.cpu(),
            }
            if cls_scores_origin_sample is not None:
                sample_output_dict["cls_scores"] = cls_scores_origin_sample.cpu()

            if instance_id is not None: # 如果有实例ID
                ids_sample = instance_id[i, current_query_indices_filtered] # 根据最终筛选的query索引提取ID
                sample_output_dict["instance_ids"] = ids_sample.cpu()

            # --- 开始：运动相关的解码 ---
            if motion_output is not None:
                trajs_pred_allmodes = motion_output["prediction"][output_idx] # (bs, num_all_queries, fut_mode, fut_ts, 2)
                traj_cls_pred_allmodes = motion_output["classification"][output_idx].sigmoid() # (bs, num_all_queries, fut_mode)

                # 根据选中的query索引提取对应的轨迹和轨迹置信度
                traj_sample_modes = trajs_pred_allmodes[i, current_query_indices_filtered] # (num_filtered_queries, fut_mode, fut_ts, 2)
                traj_cls_sample_modes = traj_cls_pred_allmodes[i, current_query_indices_filtered] # (num_filtered_queries, fut_mode)

                # 轨迹通常是相对于当前时刻box中心点的偏移，需要累加并加上当前框的中心点
                # traj_sample_modes: (num_filtered_queries, fut_mode, fut_ts, 2)
                # box_decoded_sample: (num_filtered_queries, box_dim), 取其中心点 (x,y)
                # box_decoded_sample[:, None, None, :2] -> (num_filtered_queries, 1, 1, 2) 用于广播
                traj_accumulated = traj_sample_modes.cumsum(dim=-2) # 沿时间步维度(-2)累加偏移
                traj_world = traj_accumulated + box_decoded_sample[:, None, None, :2] # 加到当前框的中心点 (x,y)

                sample_output_dict.update({
                    "trajs_3d": traj_world.cpu(), # 预测的未来轨迹
                    "trajs_score": traj_cls_sample_modes.cpu() # 轨迹的置信度/概率 (每个模态一个)
                })

                # 处理历史锚点队列和周期信息
                # anchor_queue_tensor: (bs, num_all_queries, queue_len, box_dim)
                # period_tensor: (bs, num_all_queries)
                temp_anchor_sample = anchor_queue_tensor[i, current_query_indices_filtered] # (num_filtered_queries, queue_len, box_dim)
                temp_period_sample = period_tensor[i, current_query_indices_filtered]       # (num_filtered_queries)

                num_pred_filtered, queue_len = temp_anchor_sample.shape[:2]
                if num_pred_filtered > 0 : # 确保有有效的预测框
                    temp_anchor_flat = temp_anchor_sample.flatten(0, 1) # (num_filtered_queries * queue_len, box_dim)
                    temp_anchor_decoded = decode_box(temp_anchor_flat) # 解码历史锚点
                    temp_anchor_reshaped = temp_anchor_decoded.reshape([num_pred_filtered, queue_len, box_decoded_sample.shape[-1]])
                    sample_output_dict['anchor_queue'] = temp_anchor_reshaped.cpu()
                else: # 如果没有有效的预测框，则anchor_queue为空
                    sample_output_dict['anchor_queue'] = torch.empty(0, queue_len, box_decoded_sample.shape[-1], device='cpu') if queue_len > 0 else torch.empty(0,0,box_decoded_sample.shape[-1], device='cpu')

                sample_output_dict['period'] = temp_period_sample.cpu()
            # --- 结束：运动相关的解码 ---
            output.append(sample_output_dict) # 添加当前样本结果到列表
        
        return output # 返回解码后的结果列表


@BBOX_CODERS.register_module() # 将HierarchicalPlanningDecoder注册到BBOX_CODERS注册表中 (尽管它不是典型的bbox coder)
class HierarchicalPlanningDecoder(object): # 定义层级式规划解码器类
    """
    用于解码层级式规划结果的解码器。
    它处理规划模块输出的分类分数（例如不同驾驶命令或模式的置信度）和
    回归轨迹，并根据检测和运动预测的结果进行选择或重估（rescore）。
    """
    def __init__(
        self,
        ego_fut_ts, # 自车规划的未来时间步数量
        ego_fut_mode, # 自车规划的每个命令下的轨迹模态数量
        use_rescore=False, # 是否使用重估逻辑来调整规划轨迹的得分
    ):
        """
        构造函数。

        Args:
            ego_fut_ts (int): 自车规划的未来轨迹时间步长。
            ego_fut_mode (int): 自车规划的每个高级命令下的轨迹模态数量。
            use_rescore (bool, optional): 是否启用基于碰撞等因素的轨迹重估逻辑。默认为False。
        """
        super(HierarchicalPlanningDecoder, self).__init__()
        self.ego_fut_ts = ego_fut_ts # 存储自车未来时间步数量
        self.ego_fut_mode = ego_fut_mode # 存储自车每个命令下的轨迹模态数量
        self.use_rescore = use_rescore # 是否启用重估分数
    
    def decode(
        self, 
        det_output, # 检测模块的输出 (用于重估)
        motion_output, # 运动预测模块的输出 (用于重估)
        planning_output, # 规划模块的原始输出
        data, # 输入数据，包含元信息或真实GT (例如真实驾驶命令 'gt_ego_fut_cmd')
    ):
        """
        解码规划模块的输出，选择或重估最终的规划轨迹。

        Args:
            det_output (dict): 检测模块的输出。
            motion_output (dict): 运动预测模块的输出。
            planning_output (dict): 规划模块的原始输出，通常包含:
                                   'classification': (list[Tensor]) 不同层级规划模态的分类 logits。
                                   'prediction': (list[Tensor]) 不同层级规划模态的轨迹预测 (相对偏移)。
                                   'anchor_queue': (list[Tensor] or Tensor) 自车历史锚点队列。
                                   'period': (list[Tensor] or Tensor) 自车历史锚点周期。
            data (dict): 输入数据，必须包含 'gt_ego_fut_cmd' 用于选择高层命令对应的轨迹簇。

        Returns:
            list[dict]: 每个样本的解码规划结果列表。每个字典包含:
                        'planning_score': 所有命令下所有模态的规划得分 (bs, num_cmds, ego_fut_mode)。
                        'planning': 所有命令下所有模态的规划轨迹 (bs, num_cmds, ego_fut_mode, ego_fut_ts, 2)。
                        'final_planning': 根据命令选择和(可选的)重估后，最终选定的单条规划轨迹 (bs, ego_fut_ts, 2)。
                        'ego_period': 自车相关的周期信息。
                        'ego_anchor_queue': 解码后的自车相关的历史锚点队列。
        """
        # 选择最后一层解码器的输出进行处理
        classification_raw = planning_output['classification'][-1] # (bs, num_plan_queries, num_cmds*ego_fut_mode)
        prediction_raw = planning_output['prediction'][-1]     # (bs, num_plan_queries, num_cmds*ego_fut_mode*ego_fut_ts*2)
        bs = classification_raw.shape[0] # 批量大小

        # 假设 num_plan_queries = 1 (一个自车查询对应所有命令和模态)
        # 将分类分数和预测轨迹重塑为按命令和模态组织的形状
        # num_cmds 通常为3 (例如：左转、直行、右转)
        num_cmds = data.get('gt_ego_fut_cmd', torch.empty(bs,3).to(classification_raw.device)).shape[1] # 安全获取命令数量

        # classification: (bs, num_cmds, ego_fut_mode)
        classification_reshaped = classification_raw.reshape(bs, num_cmds, self.ego_fut_mode)
        # prediction: (bs, num_cmds, ego_fut_mode, ego_fut_ts, 2)
        # 轨迹是相对偏移量，需要沿时间步维度累加得到绝对（或相对于当前ego）坐标
        prediction_reshaped = prediction_raw.reshape(bs, num_cmds, self.ego_fut_mode, self.ego_fut_ts, 2).cumsum(dim=-2)

        # 根据真实命令选择轨迹簇，并可选地进行重估，最终选择一条轨迹
        classification_final_scores, final_planning_trajectory = self.select(
            det_output, motion_output, classification_reshaped, prediction_reshaped, data
        )

        # 处理与自车相关的历史锚点队列和周期信息
        anchor_queue_raw = planning_output["anchor_queue"]
        # anchor_queue_raw可能是列表(多层输出)或直接是Tensor。确保取最后一层并处理。
        current_anchor_queue = anchor_queue_raw[-1] if isinstance(anchor_queue_raw, list) else anchor_queue_raw
        if isinstance(current_anchor_queue, list): # 如果仍然是列表 (例如每个batch item一个Tensor)
             current_anchor_queue = torch.stack(current_anchor_queue, dim=0) # (bs, num_ego_queries, queue_len, box_dim)
        # 假设 current_anchor_queue 是 (bs, num_ego_queries, queue_len, box_dim)
        # 且 num_ego_queries = 1

        current_period = planning_output["period"]
        if isinstance(current_period, list):
            current_period = torch.stack(current_period, dim=0) if current_period else torch.empty(bs,0).to(bs_indices.device)


        output = [] # 初始化输出列表
        for i in range(bs): # 遍历每个样本
            # 解码历史锚点队列 (假设每个样本只有一个ego query)
            # anchor_queue_sample: (queue_len, box_dim)
            anchor_queue_sample = current_anchor_queue[i].squeeze(0) if current_anchor_queue.numel() > 0 else current_anchor_queue[i]
            decoded_anchor_queue_sample = decode_box(anchor_queue_sample) if anchor_queue_sample.numel() > 0 else anchor_queue_sample

            output.append(
                {
                    "planning_score": classification_final_scores[i].sigmoid().cpu(), # (num_cmds, ego_fut_mode) 最终规划分数(所有模态)
                    "planning": prediction_reshaped[i].cpu(), # (num_cmds, ego_fut_mode, ego_fut_ts, 2) 所有预测的规划轨迹
                    "final_planning": final_planning_trajectory[i].cpu(), # (ego_fut_ts, 2) 最终选择的单条规划轨迹
                    "ego_period": current_period[i].cpu() if current_period.numel() > 0 else current_period.new_empty(0).cpu(), # 自车周期信息
                    "ego_anchor_queue": decoded_anchor_queue_sample.cpu(), # 解码后的自车历史锚点
                }
            )

        return output # 返回解码后的规划结果列表

    def select(
        self,
        det_output,
        motion_output,
        plan_cls,
        plan_reg,
        data,
    ):
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1]
        
        # cmd select
        bs = motion_cls.shape[0]
        bs_indices = torch.arange(bs, device=motion_cls.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        plan_cls_full = plan_cls.detach().clone()
        plan_cls = plan_cls[bs_indices, cmd]
        plan_reg = plan_reg[bs_indices, cmd]

        # rescore
        if self.use_rescore:
            plan_cls = self.rescore(
                plan_cls,
                plan_reg, 
                motion_cls,
                motion_reg, 
                det_anchors,
                det_confidence,
            )
        plan_cls_full[bs_indices, cmd] = plan_cls
        mode_idx = plan_cls.argmax(dim=-1)
        final_planning = plan_reg[bs_indices, mode_idx]
        return plan_cls_full, final_planning

    def rescore(
        self, 
        plan_cls,
        plan_reg, 
        motion_cls,
        motion_reg, 
        det_anchors,
        det_confidence,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
    ):
        
        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat
        
        def get_yaw(traj, start_yaw=np.pi/2):
            yaw = traj.new_zeros(traj.shape[:-1])
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            yaw[..., 0] = start_yaw
            # for static object, estimated future yaw would be unstable
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            start_yaw = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(
                mask.unsqueeze(-1),
                start_yaw,
                yaw,
            )
            return yaw.unsqueeze(-1)
        
        ## ego
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, self.ego_fut_mode, self.ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        ## motion
        motion_reg = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2)
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, self.ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].exp()
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

        ego_box[0] += offset * torch.cos(ego_box[6])
        ego_box[1] += offset * torch.sin(ego_box[6])
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False # for case that all modes collide, no need to rescore
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset
        return plan_cls


def check_collision(boxes1, boxes2):
    '''
        A rough check for collision detection: 
            check if any corner point of boxes1 is inside boxes2 and vice versa.
        
        boxes1: tensor with shape [N, 7], [x, y, z, w, l, h, yaw]
        boxes2: tensor with shape [N, 7]
    '''
    col_1 = corners_in_box(boxes1.clone(), boxes2.clone())
    col_2 = corners_in_box(boxes2.clone(), boxes1.clone())
    collision = torch.logical_or(col_1, col_2)

    return collision

def corners_in_box(boxes1, boxes2):
    if  boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return False

    boxes1_yaw = boxes1[:, 6].clone()
    boxes1_loc = boxes1[:, :3].clone()
    cos_yaw = torch.cos(-boxes1_yaw)
    sin_yaw = torch.sin(-boxes1_yaw)
    rot_mat_T = torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw]),
            torch.stack([-sin_yaw, cos_yaw]),
        ]
    )
    # translate and rotate boxes
    boxes1[:, :3] = boxes1[:, :3] - boxes1_loc
    boxes1[:, :2] = torch.einsum('ij,jki->ik', boxes1[:, :2], rot_mat_T)
    boxes1[:, 6] = boxes1[:, 6] - boxes1_yaw

    boxes2[:, :3] = boxes2[:, :3] - boxes1_loc
    boxes2[:, :2] = torch.einsum('ij,jki->ik', boxes2[:, :2], rot_mat_T)
    boxes2[:, 6] = boxes2[:, 6] - boxes1_yaw

    corners_box2 = box3d_to_corners(boxes2)[:, [0, 3, 7, 4], :2]
    corners_box2 = torch.from_numpy(corners_box2).to(boxes2.device)
    H = boxes1[:, [3]]
    W = boxes1[:, [4]]

    collision = torch.logical_and(
        torch.logical_and(corners_box2[..., 0] <= H / 2, corners_box2[..., 0] >= -H / 2),
        torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1] >= -W / 2),
    )
    collision = collision.any(dim=-1)

    return collision