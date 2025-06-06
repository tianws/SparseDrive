from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk


@HEADS.register_module() # 将MotionPlanningHead注册到MMDetection的HEADS注册表中
class MotionPlanningHead(BaseModule): # 定义运动规划头
    """
    用于运动预测和路径规划的头部模块。
    该模块集成了多个组件，包括实例队列（用于时序信息）、图神经网络（用于交互建模）、
    以及特定的采样器、损失函数和解码器，以实现对场景中其他智能体的运动预测
    和自车（ego）的路径规划。
    """
    def __init__(
        self,
        fut_ts=12,  # (Agent)未来轨迹的时间步长 (例如，预测未来12帧的轨迹)
        fut_mode=6,  # (Agent)未来轨迹的模态数量 (例如，预测6种可能的未来轨迹)
        ego_fut_ts=6,  # (Ego)自车规划的未来时间步长
        ego_fut_mode=3,  # (Ego)自车规划的每个命令下的轨迹模态数量 (配置文件中可能是6)
        motion_anchor=None,  # 运动锚点文件路径 (.npy) 或已加载的numpy数组 (预定义的轨迹原型)
        plan_anchor=None,  # 规划锚点文件路径 (.npy) 或已加载的numpy数组 (预定义的ego轨迹原型)
        embed_dims=256,  # 特征嵌入维度
        decouple_attn=False,  # 是否在某些注意力机制中解耦内容和位置嵌入
        instance_queue=None,  # 实例队列的配置字典 (用于存储和管理历史实例信息)
        operation_order=None,  # 定义模块内部特征处理层操作顺序的列表 (例如 ['temp_gnn', 'gnn', 'refine'])
        temp_graph_model=None,  # 时序图模型的配置字典 (用于处理实例间的时序关系)
        graph_model=None,  # 交互图模型的配置字典 (用于处理当前帧实例间的关系)
        cross_graph_model=None,  # 跨类型图模型的配置字典 (例如，agent与map元素间的交互)
        norm_layer=None,  # 归一化层的配置字典
        ffn=None,  # 前馈网络的配置字典
        refine_layer=None,  # 优化层的配置字典 (例如 MotionPlanningRefinementModule)
        motion_sampler=None,  # 运动预测的目标采样器/分配器配置
        motion_loss_cls=None,  # 运动预测的分类损失配置 (例如，轨迹模态分类)
        motion_loss_reg=None,  # 运动预测的回归损失配置 (轨迹点回归)
        planning_sampler=None,  # 规划的目标采样器/分配器配置
        plan_loss_cls=None,  # 规划的分类损失配置 (例如，规划模态分类)
        plan_loss_reg=None,  # 规划的回归损失配置 (轨迹点回归)
        plan_loss_status=None,  # 规划的状态损失配置 (例如，速度、加速度等ego状态)
        motion_decoder=None,  # 运动预测解码器配置 (用于后处理生成最终结果)
        planning_decoder=None,  # 规划解码器配置
        num_det=50,  # 从检测结果中选取的top-k物体实例数量，用于后续的交互和运动预测
        num_map=10,  # 从地图元素检测结果中选取的top-k元素数量，用于后续的交互
    ):
        super(MotionPlanningHead, self).__init__() # 调用父类BaseModule的初始化
        self.fut_ts = fut_ts # agent未来轨迹长度 (时间步数)
        self.fut_mode = fut_mode # agent未来轨迹的模态数
        self.ego_fut_ts = ego_fut_ts # ego未来轨迹长度
        self.ego_fut_mode = ego_fut_mode # ego每个命令下的轨迹模态数

        self.decouple_attn = decouple_attn # 是否在图模型注意力中解耦内容和位置
        self.operation_order = operation_order # 内部各操作层的执行顺序列表

        # =========== 构建各个子模块 ===========
        def build(cfg, registry): # 内部辅助函数，用于从配置字典和注册表构建模块
            if cfg is None: # 如果配置为None，则不构建，返回None
                return None
            return build_from_cfg(cfg, registry) # 使用MMCV的工具函数构建
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS) # 构建实例队列模块
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS) # 构建运动目标采样/分配器
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS) # 构建规划目标采样/分配器
        self.motion_decoder = build(motion_decoder, BBOX_CODERS) # 构建运动解码器
        self.planning_decoder = build(planning_decoder, BBOX_CODERS) # 构建规划解码器

        # op_config_map 将操作名称映射到其配置和对应的MMCV注册表
        # 这样可以根据 operation_order 列表灵活构建处理流程
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],      # 时序图模型 (通常是某种注意力机制)
            "gnn": [graph_model, ATTENTION],               # 场景内交互图模型
            "cross_gnn": [cross_graph_model, ATTENTION],   # 跨模态交互图模型
            "norm": [norm_layer, NORM_LAYERS],             # 归一化层
            "ffn": [ffn, FEEDFORWARD_NETWORK],             # 前馈网络
            "refine": [refine_layer, PLUGIN_LAYERS],       # 优化/预测层
        }
        self.layers = nn.ModuleList( # 根据operation_order构建一个包含多个操作层的ModuleList
            [
                build(*self.op_config_map.get(op, [None, None])) # 如果操作名不在map中，则构建为None
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims # 存储嵌入维度

        if self.decouple_attn: # 如果使用解耦注意力，定义额外的线性层用于拼接或分离特征和位置编码
            self.fc_before = nn.Linear( # 注意力计算前，可能用于将拼接的(特征+位置)映射回原维度或扩展维度
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear( # 注意力计算后，可能用于将注意力输出映射回原维度
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else: # 如果不解耦，则使用恒等映射
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        # 构建各项损失函数
        self.motion_loss_cls = build_loss(motion_loss_cls) # 运动模态分类损失
        self.motion_loss_reg = build_loss(motion_loss_reg) # 运动轨迹回归损失
        self.plan_loss_cls = build_loss(plan_loss_cls)     # 规划模态分类损失
        self.plan_loss_reg = build_loss(plan_loss_reg)     # 规划轨迹回归损失
        self.plan_loss_status = build_loss(plan_loss_status) # 规划状态损失

        # 初始化运动锚点 (预定义的典型轨迹原型)
        if motion_anchor is not None:
            motion_anchor_data = np.load(motion_anchor) if isinstance(motion_anchor, str) else motion_anchor
            self.motion_anchor = nn.Parameter( # (num_motion_classes, fut_mode, fut_ts, 2)
                torch.tensor(motion_anchor_data, dtype=torch.float32),
                requires_grad=False, # 锚点通常是固定的，不参与训练
            )
            self.motion_anchor_encoder = nn.Sequential( # 对运动锚点（通常是其末端点或整体形状）进行编码的网络
                *linear_relu_ln(embed_dims, 1, 1), # 1个(线性+ReLU)+LN
                Linear(embed_dims, embed_dims),
            )
        else:
            self.motion_anchor = None
            self.motion_anchor_encoder = None


        # 初始化规划锚点 (预定义的典型自车轨迹原型，可能对应不同命令或意图)
        if plan_anchor is not None:
            plan_anchor_data = np.load(plan_anchor) if isinstance(plan_anchor, str) else plan_anchor
            self.plan_anchor = nn.Parameter( # (num_plan_cmds, ego_fut_mode, ego_fut_ts, 2)
                torch.tensor(plan_anchor_data, dtype=torch.float32),
                requires_grad=False,
            )
            self.plan_anchor_encoder = nn.Sequential( # 对规划锚点进行编码的网络
                *linear_relu_ln(embed_dims, 1, 1),
                Linear(embed_dims, embed_dims),
            )
        else:
            self.plan_anchor = None
            self.plan_anchor_encoder = None

        self.num_det = num_det # 从检测头选取的top-k物体数量
        self.num_map = num_map # 从地图头选取的top-k地图元素数量

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def get_motion_anchor(
        self, 
        classification, 
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self, 
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):   
        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
        )

        map_instance_feature = map_output["instance_feature"]
        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )

        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, dim = instance_feature.shape
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,
        )
        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        plan_anchor = torch.tile(
            self.plan_anchor[None], (bs, 1, 1, 1, 1)
        )

        # =========== mode query init ===========
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2) 
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_reg,
                    plan_status,
                ) = self.layers[i](
                    motion_query,
                    plan_query,
                    instance_feature[:, num_anchor:],
                    anchor_embed[:, num_anchor:],
                )
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)
        
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        return motion_output, planning_output
    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache
    ):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output, 
            data,
        )

        return motion_result, planning_result