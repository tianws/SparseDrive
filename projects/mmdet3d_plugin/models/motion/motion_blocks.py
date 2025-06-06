import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
)

from projects.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln


@PLUGIN_LAYERS.register_module()
class MotionPlanningRefinementModule(BaseModule): # 定义运动规划优化模块
    """
    运动规划优化模块。
    该模块接收运动查询、规划查询以及自车（ego）的特征和锚点嵌入作为输入，
    并分别输出运动预测（分类和轨迹回归）、规划预测（分类和轨迹回归）以及规划状态。
    """
    def __init__(
        self,
        embed_dims=256,  # 特征嵌入的维度
        fut_ts=12,  # 运动预测的未来时间步数量 (例如，预测未来12帧的轨迹)
        fut_mode=6,  # 运动预测的模态数量 (例如，预测6种可能的未来轨迹)
        ego_fut_ts=6,  # 自车规划的未来时间步数量
        ego_fut_mode=3,  # 自车规划的模态数量 (注意：配置文件中可能是6，这里是3，需确认)
    ):
        """
        构造函数。

        Args:
            embed_dims (int, optional): 输入特征和查询的嵌入维度。默认为256。
            fut_ts (int, optional): 其他智能体未来轨迹预测的时间步长。默认为12。
            fut_mode (int, optional): 其他智能体未来轨迹预测的模态数量。默认为6。
            ego_fut_ts (int, optional): 自车规划轨迹的时间步长。默认为6。
            ego_fut_mode (int, optional): 自车规划轨迹的模态数量。默认为3。
        """
        super(MotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims # 嵌入维度
        self.fut_ts = fut_ts # Agent未来轨迹长度
        self.fut_mode = fut_mode # Agent未来轨迹模态数
        self.ego_fut_ts = ego_fut_ts # Ego未来轨迹长度
        self.ego_fut_mode = ego_fut_mode # Ego未来轨迹模态数

        # 运动预测分类分支：判断运动预测是否有效或运动的类别/模态置信度
        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2), # 1个(线性+ReLU)+LN，重复2次 (构建MLP)
            Linear(embed_dims, 1), # 输出一个logit值 (每个模态一个logit，或者是一个总的有效性logit)
                                    # 如果是每个模态一个logit，输出维度应为 fut_mode
                                    # 从forward看，输出是(bs, num_anchor)，所以这里是每个query输出一个logit
        )
        # 运动预测回归分支：预测未来轨迹的(x,y)偏移量
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2), # 每个时间步预测2个坐标(x,y)，总共 fut_ts*2 个值
                                               # 这个输出是针对单个模态的，如果有多模态，会在forward中reshape
        )
        # 规划分类分支：判断规划轨迹的类别或模态置信度
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1), # 输出一个logit值 (类似motion_cls_branch)
                                    # 如果是每个模态一个logit，输出维度应为 ego_fut_mode
        )
        # 规划回归分支：预测自车未来轨迹的(x,y)偏移量
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2), # 每个时间步预测2个坐标(x,y)
        )
        # 规划状态分支：预测自车的驾驶状态（例如，速度、加速度、命令执行状态等）
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10), # 输出10个状态值 (具体含义需查阅模型设计文档)
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        bias_init = bias_init_with_prob(0.01) # 使用较小的概率初始化偏置，避免初始阶段输出过于集中
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init) # 初始化运动分类分支最后一个线性层的偏置
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init) # 初始化规划分类分支最后一个线性层的偏置

    def forward(
        self,
        motion_query,  # 运动查询张量 (bs, num_motion_queries, embed_dims) 或 (bs*num_agents, fut_mode, embed_dims)
        plan_query,  # 规划查询张量 (bs, num_plan_queries_total_modes, embed_dims)
        ego_feature,  # 自车特征张量 (bs, embed_dims)
        ego_anchor_embed,  # 自车锚点嵌入 (bs, embed_dims)
    ):
        """
        前向传播函数。

        Args:
            motion_query (torch.Tensor): 用于其他智能体运动预测的查询。
                                         形状可以是 (bs, num_agents, fut_mode, embed_dims) 或经过flatten的。
            plan_query (torch.Tensor): 用于自车规划的查询。
                                       形状可以是 (bs, num_plan_modes, embed_dims) 或经过flatten的。
            ego_feature (torch.Tensor): 自车的特征。
            ego_anchor_embed (torch.Tensor): 自车锚点的嵌入。

        Returns:
            tuple: 包含以下元素的元组:
                - motion_cls (torch.Tensor): 运动分类 logits (bs, num_motion_queries)。
                                             (如果输入是 (bs*num_agents, fut_mode, ...), reshape前是 (bs*num_agents, fut_mode))
                - motion_reg (torch.Tensor): 运动轨迹回归预测
                                             (bs, num_motion_queries, fut_mode, fut_ts, 2)。
                - plan_cls (torch.Tensor): 规划分类 logits (bs, num_plan_queries_total_modes)。
                - plan_reg (torch.Tensor): 规划轨迹回归预测
                                           (bs, num_plan_queries_or_1, num_total_planning_modes_ανα_query, ego_fut_ts, 2)。
                                           reshape后的具体形状取决于plan_query的输入形状和ego_fut_mode的定义。
                                           原始代码为: (bs, 1, 3 * self.ego_fut_mode, self.ego_fut_ts, 2)
                                           这暗示plan_query可能是(bs, 1*num_modes_total, dim)或者 (bs,1,dim)然后预测所有模式。
                - planning_status (torch.Tensor): 规划状态预测 (bs, 10)。
        """
        bs, num_anchor = motion_query.shape[:2] # 获取motion_query的批量大小和第二维度（可能是num_agents*fut_mode或num_queries）
                                            # 假设motion_query是 (bs, num_total_motion_queries, dim)
                                            # 这里的num_anchor对应于展平后的查询数量

        # 运动预测分支
        # motion_cls_branch的输出是 (bs, num_total_motion_queries, 1)，squeeze后是 (bs, num_total_motion_queries)
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        # motion_reg_branch的输出是 (bs, num_total_motion_queries, fut_ts * 2)
        # reshape后是 (bs, num_original_anchors, fut_mode, fut_ts, 2)
        # 这要求 num_total_motion_queries == num_original_anchors * fut_mode
        # 或者，如果motion_query是 (bs, num_original_anchors, dim)，则reshape前是 (bs, num_original_anchors, fut_ts*2)
        # 这里的num_anchor是从motion_query.shape[:2]来的，即num_total_motion_queries
        # 所以reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2) 要求 num_anchor能被fut_mode整除，
        # 且 num_anchor / fut_mode 是原始的agent数量或query数量。
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor // self.fut_mode if self.fut_mode > 0 else num_anchor, self.fut_mode, self.fut_ts, 2)

        # 规划预测分支
        # plan_cls_branch的输出是 (bs, num_total_plan_queries, 1)，squeeze后是 (bs, num_total_plan_queries)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        # plan_reg_branch的输出是 (bs, num_total_plan_queries, ego_fut_ts * 2)
        # reshape后是 (bs, num_original_plan_queries, num_modes_per_original_query, ego_fut_ts, 2)
        # 原始代码: reshape(bs, 1, 3 * self.ego_fut_mode, self.ego_fut_ts, 2)
        # 这暗示num_total_plan_queries = 1 * (3 * self.ego_fut_mode)
        # 即plan_query可能是(bs, 3*self.ego_fut_mode, dim)
        # 假设plan_query是(bs, N_plan_queries, dim)
        num_plan_queries_input = plan_query.shape[1]
        plan_reg = self.plan_reg_branch(plan_query).reshape(bs, num_plan_queries_input // self.ego_fut_mode if self.ego_fut_mode > 0 else num_plan_queries_input, self.ego_fut_mode, self.ego_fut_ts, 2)

        # 规划状态分支，输入是自车特征和锚点嵌入的和
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed) # (bs, 10)

        return motion_cls, motion_reg, plan_cls, plan_reg, planning_status