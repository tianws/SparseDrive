import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@PLUGIN_LAYERS.register_module()
class InstanceBank(nn.Module):  # 定义InstanceBank类，用于存储和管理实例（或锚点）的特征和状态
    """
    InstanceBank用于存储和更新一组“实例”或“锚点”的特征和状态。
    这些实例可以包括预定义的锚点、先前帧中检测到的对象（用于时序处理），
    以及用于去噪训练的加噪样本。
    它支持：
    - 初始化一组固定的锚点。
    - 缓存和更新来自先前时间步的实例特征和锚点。
    - 根据置信度选择和更新实例。
    - 管理实例ID（用于跟踪）。
    """
    def __init__(
        self,
        num_anchor,  # 基础锚点的数量 (要从加载的锚点中选择多少个)
        embed_dims,  # 实例特征的嵌入维度
        anchor,  # 锚点数据，可以是文件路径(str)指向.npy文件，列表/元组，或numpy数组
        anchor_handler=None,  # 锚点处理器配置字典，用于锚点投影等操作 (可选)
        num_temp_instances=0,  # 用于时序处理的缓存实例数量 (从前一帧传递到当前帧)
        default_time_interval=0.5,  # 默认时间间隔 (秒)，当实际时间间隔不可用或为0时使用
        confidence_decay=0.6,  # 缓存实例置信度的衰减因子 (每帧衰减)
        anchor_grad=True,  # 锚点参数是否参与梯度更新 (是否可学习)
        feat_grad=True,  # 实例特征参数是否参与梯度更新 (是否可学习)
        max_time_interval=2,  # 允许的最大时间间隔（秒），超出此间隔的缓存实例可能被视为无效或被大力衰减
    ):
        super(InstanceBank, self).__init__() # 调用父类nn.Module的构造函数
        self.embed_dims = embed_dims # 实例特征维度
        self.num_temp_instances = num_temp_instances  # 时序实例（来自过去的帧）的数量
        self.default_time_interval = default_time_interval # 默认时间间隔
        self.confidence_decay = confidence_decay # 置信度衰减因子
        self.max_time_interval = max_time_interval # 最大有效时间间隔

        if anchor_handler is not None:  # 如果配置了锚点处理器
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)  # 从配置构建锚点处理器
            assert hasattr(anchor_handler, "anchor_projection")  # 确保处理器有anchor_projection方法
        self.anchor_handler = anchor_handler # 存储锚点处理器

        # 加载和初始化锚点数据
        if isinstance(anchor, str):  # 如果anchor是字符串，则假定为numpy文件路径
            anchor = np.load(anchor) # 从.npy文件加载
        elif isinstance(anchor, (list, tuple)):  # 如果是列表或元组，转换为numpy数组
            anchor = np.array(anchor)

        # 特殊处理地图锚点，其形状可能为 (num_anchor_types, num_points_per_type, dim_per_point)
        # 将其reshape为 (num_anchor_types, num_points_per_type * dim_per_point)
        if len(anchor.shape) == 3: # for map (例如，线段锚点)
            anchor = anchor.reshape(anchor.shape[0], -1) # 将后两维展平

        self.num_anchor = min(len(anchor), num_anchor)  # 最终使用的锚点数量取加载的锚点数和配置的num_anchor中的较小值
        anchor = anchor[:self.num_anchor] # 截取所需数量的锚点

        # 将锚点注册为PyTorch的参数 (nn.Parameter)，使其可以被模型学习（如果requires_grad=True）
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32), # 转换为PyTorch张量
            requires_grad=anchor_grad, # 设置是否需要梯度
        )
        self.anchor_init = anchor  # 保存初始锚点值，用于可能的重置 (例如在init_weight中)

        # 初始化实例特征参数，初始值为全0
        # 这些特征与上面的锚点一一对应，可以看作是每个锚点的初始可学习特征嵌入
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]), # 形状为 (num_anchor, embed_dims)
            requires_grad=feat_grad, # 设置是否需要梯度
        )
        self.reset() # 初始化或重置InstanceBank的缓存状态

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1) # 使用Xavier均匀分布初始化实例特征

    def reset(self): # 重置InstanceBank的状态，清空所有缓存信息
        """
        重置所有与时序缓存和实例ID追踪相关的状态变量。
        这个方法通常在每个新的评估序列开始时，或者在训练开始前被调用，
        以确保不同序列或训练阶段之间没有状态泄漏。
        """
        self.cached_feature = None  # 缓存的实例特征 (来自上一有效时间步的top-k实例)
        self.cached_anchor = None  # 缓存的实例锚点 (与cached_feature对应)
        self.metas = None  # 缓存的元数据 (与cached_feature/anchor对应的时间戳、变换矩阵等)
        self.mask = None  # 缓存实例的有效性掩码 (基于时间间隔判断是否有效)
        self.confidence = None  # 缓存实例的置信度 (经过衰减和更新)
        self.temp_confidence = None # 在cache方法中计算的当前帧所有实例的临时置信度 (用于选择下一帧的缓存)
        self.instance_id = None  # 缓存实例的ID (用于跟踪)
        self.prev_id = 0  # 用于生成新的实例ID的计数器，每次reset后从0开始

    def get(self, batch_size, metas=None, dn_metas=None):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        self.confidence = torch.where(
            self.mask[:, None],
            self.confidence,
            self.confidence.new_tensor(0)
        )
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )