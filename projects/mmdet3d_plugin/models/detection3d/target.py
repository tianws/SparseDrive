import torch # 导入PyTorch库
import numpy as np # 导入NumPy库
import torch.nn.functional as F # 导入PyTorch神经网络函数式接口
from scipy.optimize import linear_sum_assignment # 从SciPy导入线性总和分配函数（用于匈牙利匹配）

from mmdet.core.bbox.builder import BBOX_SAMPLERS # 从MMDetection导入BBOX_SAMPLERS注册表

from projects.mmdet3d_plugin.core.box3d import * # 从项目中导入3D边界框相关的常量定义
from ..base_target import BaseTargetWithDenoising # 从上一级目录的base_target模块导入BaseTargetWithDenoising基类


__all__ = ["SparseBox3DTarget"] # 定义当使用 from .target import * 时，"SparseBox3DTarget"会被导入


@BBOX_SAMPLERS.register_module() # 将SparseBox3DTarget注册到MMDetection的BBOX_SAMPLERS注册表中
class SparseBox3DTarget(BaseTargetWithDenoising): # 定义稀疏3D边界框目标分配器类
    """
    用于稀疏3D目标检测的目标分配器，继承自BaseTargetWithDenoising。
    它负责将真实目标（ground truth）与模型的预测（通常是queries或anchors）进行匹配，
    并生成用于损失计算的目标张量。支持匈牙利匹配算法，并集成了去噪（DN）机制
    来生成加噪的真实样本以辅助训练。
    """
    def __init__(
        self,
        cls_weight=2.0,  # 分类损失的权重
        alpha=0.25,  # Focal Loss的alpha参数
        gamma=2,  # Focal Loss的gamma参数
        eps=1e-12,  # 用于数值稳定性的epsilon值
        box_weight=0.25,  # 边界框回归损失的权重
        reg_weights=None,  # 各个边界框参数的回归权重列表 (例如，[x_w, y_w, ..., yaw_w, vel_w])
        cls_wise_reg_weights=None,  # 按类别区分的边界框回归权重字典 (可选)
        num_dn_groups=0,  # 去噪组的数量 (为每个真实目标生成多少组加噪样本)
        dn_noise_scale=0.5,  # 去噪时添加的噪声尺度
        max_dn_gt=32,  # 用于去噪的最大真实目标数量
        add_neg_dn=True,  # 是否为去噪添加负样本 (与真实目标不匹配的加噪样本)
        num_temp_dn_groups=0,  # 时序去噪组的数量 (未使用在此实现中，但继承自基类)
    ):
        super(SparseBox3DTarget, self).__init__( # 调用父类的初始化函数
            num_dn_groups, num_temp_dn_groups
        )
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None: # 如果未提供回归权重，则使用默认值
            # 通常对应 (x,y,z, w,l,h, sin_yaw,cos_yaw, vx,vy)
            # 这里默认前8个参数权重为1.0 (位置、尺寸、偏航角)，后2个为0.0 (可能对应速度的更高阶导数或未使用)
            self.reg_weights = [1.0] * 8 + [0.0] * 2
        self.cls_wise_reg_weights = cls_wise_reg_weights # 按类别区分的回归权重
        self.dn_noise_scale = dn_noise_scale # 去噪噪声尺度
        self.max_dn_gt = max_dn_gt # 最大去噪GT数量
        self.add_neg_dn = add_neg_dn # 是否添加负去噪样本

    def encode_reg_target(self, box_target, device=None): # 将真实边界框编码为回归目标格式
        """
        将标准格式的3D边界框列表转换为模型回归的目标格式。
        通常包括对尺寸取对数，并将偏航角转换为(sin_yaw, cos_yaw)。

        Args:
            box_target (list[torch.Tensor]): 每个样本的真实边界框列表。
                                           每个Tensor形状为 (num_gt, box_dim)。
            device (torch.device, optional): 目标设备。

        Returns:
            list[torch.Tensor]: 编码后的边界框目标列表。
        """
        outputs = []
        for box in box_target: # 遍历每个样本的真实边界框
            # box_dim通常包含 x,y,z, w,l,h, yaw, vx,vy,...
            # X,Y,Z,W,L,H,YAW,VX 是预定义的索引
            output = torch.cat( # 拼接编码后的参数
                [
                    box[..., [X, Y, Z]],  # 位置 (x,y,z) 保持不变
                    box[..., [W, L, H]].log(),  # 尺寸 (w,l,h) 取对数
                    torch.sin(box[..., YAW]).unsqueeze(-1),  # yaw角转换为sin(yaw)
                    torch.cos(box[..., YAW]).unsqueeze(-1),  # yaw角转换为cos(yaw)
                    box[..., YAW + 1 :],  # 保留yaw角之后的所有其他参数 (如速度)
                ],
                dim=-1, # 沿最后一个维度拼接
            )
            if device is not None: # 如果指定了设备
                output = output.to(device=device) # 移动到目标设备
            outputs.append(output)
        return outputs

    def sample( # 执行预测与真实目标之间的匹配 (例如，匈牙利匹配)
        self,
        cls_pred,  # 预测的类别分数 (bs, num_pred, num_cls)
        box_pred,  # 预测的边界框参数 (bs, num_pred, box_dim_encoded)
        cls_target,  # 真实的类别标签列表，每个元素形状 (num_gt_i)
        box_target,  # 真实的边界框列表，每个元素形状 (num_gt_i, box_dim_raw)
    ):
        """
        通过匈牙利匹配算法，为每个预测（query）分配最合适的真实目标。
        生成用于分类和回归损失计算的目标张量。

        Args:
            cls_pred (torch.Tensor): 预测的类别分数。
            box_pred (torch.Tensor): 预测的编码后的边界框。
            cls_target (list[torch.Tensor]): 真实类别标签列表。
            box_target (list[torch.Tensor]): 真实原始边界框列表。

        Returns:
            tuple:
                - output_cls_target (torch.Tensor): 分配给每个预测的类别目标 (bs, num_pred)。
                                                   未匹配的预测目标为背景类 (num_cls)。
                - output_box_target (torch.Tensor): 分配给每个预测的编码后的边界框目标 (bs, num_pred, box_dim_encoded)。
                - output_reg_weights (torch.Tensor): 分配给每个预测的回归权重 (bs, num_pred, box_dim_encoded)。
        """
        bs, num_pred, num_cls = cls_pred.shape # 批量大小, 预测数量, 类别数量

        # 计算分类代价矩阵
        cls_cost = self._cls_cost(cls_pred, cls_target)

        # 将真实边界框编码为回归目标格式
        box_target_encoded = self.encode_reg_target(box_target, box_pred.device)

        # 计算每个真实目标的回归权重
        instance_reg_weights = []
        for i in range(len(box_target_encoded)): # 遍历每个样本
            # 权重为1表示对应参数有效 (非NaN)，否则为0
            weights = torch.logical_not(box_target_encoded[i].isnan()).to(
                dtype=box_target_encoded[i].dtype
            )
            if self.cls_wise_reg_weights is not None: # 如果定义了按类别区分的回归权重
                for cls_id, weight_values in self.cls_wise_reg_weights.items(): # 遍历类别和对应的权重
                    # 为属于特定类别的真实目标设置其特定的回归权重
                    weights = torch.where(
                        (cls_target[i] == cls_id)[:, None], # 条件：真实目标的类别ID匹配
                        weights.new_tensor(weight_values), # 如果匹配，使用特定权重
                        weights, # 否则，保持原权重
                    )
            instance_reg_weights.append(weights)

        # 计算边界框回归代价矩阵
        box_cost = self._box_cost(box_pred, box_target_encoded, instance_reg_weights)

        indices = [] # 存储每个样本的匹配结果 (预测索引, 真实目标索引)
        for i in range(bs): # 遍历每个样本
            if cls_cost[i] is not None and box_cost[i] is not None and len(cls_target[i]) > 0:
                # 总代价 = 分类代价 + 回归代价
                cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
                # 处理代价矩阵中的无效值 (负无穷或NaN)，替换为较大的数
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
                # 执行匈牙利匹配 (线性总和分配)
                assign_pred_idx, assign_target_idx = linear_sum_assignment(cost)
                indices.append( # 保存匹配的预测索引和目标索引
                    [cls_pred.new_tensor(assign_pred_idx, dtype=torch.int64),
                     cls_pred.new_tensor(assign_target_idx, dtype=torch.int64)]
                )
            else: # 如果没有真实目标或代价计算有问题
                indices.append([None, None]) # 添加空的匹配结果

        # 初始化输出目标张量
        # output_cls_target 默认为背景类 (num_cls)
        output_cls_target = (
            cls_target[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        )
        output_box_target = box_pred.new_zeros(box_pred.shape) # 默认为0
        output_reg_weights = box_pred.new_zeros(box_pred.shape) # 回归权重默认为0

        # 根据匹配结果填充输出目标张量
        for i, (pred_idx, target_idx) in enumerate(indices): # 遍历每个样本的匹配结果
            if pred_idx is None or len(cls_target[i]) == 0: # 如果没有匹配或没有真实目标
                continue
            output_cls_target[i, pred_idx] = cls_target[i][target_idx] # 设置匹配上的预测的类别目标
            output_box_target[i, pred_idx] = box_target_encoded[i][target_idx] # 设置匹配上的预测的框目标
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][target_idx] # 设置匹配上的预测的回归权重

        self.indices = indices # 保存匹配索引，可能用于其他地方 (如去噪损失)
        return output_cls_target, output_box_target, output_reg_weights

    def _cls_cost(self, cls_pred, cls_target): # 计算分类代价
        """
        计算用于匈牙利匹配的分类代价。通常基于Focal Loss的形式。
        代价越小表示匹配越好。
        """
        bs = cls_pred.shape[0]
        cls_pred_sigmoid = cls_pred.sigmoid() # 将logits转换为概率
        cost = []
        for i in range(bs): # 遍历每个样本
            if len(cls_target[i]) > 0: # 如果存在真实目标
                # Focal Loss的计算公式变体，用于代价计算
                # neg_cost: 预测为背景但实际为前景的代价 (对应 1-alpha 项)
                neg_cost = (
                    -(1 - cls_pred_sigmoid[i] + self.eps).log() # log(1-p)
                    * (1 - self.alpha)
                    * cls_pred_sigmoid[i].pow(self.gamma)
                )
                # pos_cost: 预测为前景但实际为背景的代价，或预测类别错误 (对应 alpha 项)
                pos_cost = (
                    -(cls_pred_sigmoid[i] + self.eps).log() # log(p)
                    * self.alpha
                    * (1 - cls_pred_sigmoid[i]).pow(self.gamma)
                )
                # cost_matrix[j, k] = cost of assigning pred_j to target_k
                # 对于每个预测j和真实目标k，其分类代价是：
                # 如果预测j的类别是真实目标k的类别，则代价是 -log(p_jk) 相关项 (pos_cost)
                # 如果预测j的类别不是真实目标k的类别，则代价是 -log(1-p_jk) 相关项 (neg_cost)
                # 这里直接用 pos_cost[:, cls_target[i]] 选取对应真实类别的pos_cost
                # 然后减去对应真实类别的neg_cost。这实际上是 (alpha * FL_pos) - ((1-alpha)*FL_neg) 的形式，
                # 其中 FL_pos 是预测为正确类别的Focal Loss项，FL_neg是预测为错误类别的Focal Loss项。
                # 目的是让正确类别的预测概率越高，代价越小。
                cost_matrix_for_sample_i = (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]]) * self.cls_weight
                cost.append(cost_matrix_for_sample_i)
            else: # 如果没有真实目标
                cost.append(None)
        return cost

    def _box_cost(self, box_pred, box_target_encoded, instance_reg_weights): # 计算边界框回归代价
        """
        计算用于匈牙利匹配的边界框回归代价。通常是L1损失的形式。
        """
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs): # 遍历每个样本
            if len(box_target_encoded[i]) > 0: # 如果存在真实目标
                # 计算每个预测框与每个真实框之间的L1距离
                # box_pred[i, :, None] -> (num_pred, 1, box_dim)
                # box_target_encoded[i][None] -> (1, num_gt, box_dim)
                # 结果 l1_distance -> (num_pred, num_gt, box_dim)
                l1_distance = torch.abs(box_pred[i, :, None] - box_target_encoded[i][None])
                # 应用回归权重和预定义的参数权重
                weighted_l1_distance = l1_distance * instance_reg_weights[i][None] * box_pred.new_tensor(self.reg_weights)
                # 沿box_dim维度求和，得到每个 (预测, 真实目标) 对的代价
                cost_matrix_for_sample_i = torch.sum(weighted_l1_distance, dim=-1) * self.box_weight
                cost.append(cost_matrix_for_sample_i)
            else: # 如果没有真实目标
                cost.append(None)
        return cost

    def get_dn_anchors(self, cls_target, box_target, gt_instance_id=None): # 生成用于去噪训练的加噪锚点
        """
        为输入的真实目标生成加噪版本，作为去噪任务的输入和目标。

        Args:
            cls_target (list[torch.Tensor]): 真实类别标签列表。
            box_target (list[torch.Tensor]): 真实原始边界框列表。
            gt_instance_id (list[torch.Tensor], optional): 真实实例ID列表。

        Returns:
            tuple or None: 如果不生成去噪锚点，则返回None。否则返回包含：
                - dn_anchor (torch.Tensor): 加噪后的锚点。
                - dn_box_target (torch.Tensor): 对应的真实边界框目标 (编码后)。
                - dn_cls_target (torch.Tensor): 对应的类别目标。
                - attn_mask (torch.Tensor): 用于注意力机制的掩码，防止信息泄露。
                - valid_mask (torch.Tensor): 指示哪些去噪样本是有效的（非padding）。
                - dn_id_target (torch.Tensor or None): 对应的实例ID目标。
        """
        if self.num_dn_groups <= 0: # 如果不去噪组数量小于等于0
            return None # 不生成去噪锚点
        if self.num_temp_dn_groups <= 0: # 如果时序去噪组数量小于等于0 (当前实现主要关注非时序DN)
            gt_instance_id = None # 则不使用实例ID (因为ID主要用于时序关联)

        # 限制用于生成去噪样本的GT数量
        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]

        max_num_gt_in_batch = max([len(x) for x in cls_target]) # 获取当前batch中单个样本的最大GT数量
        if max_num_gt_in_batch == 0: # 如果没有任何GT
            return None # 不生成去噪锚点

        # 将列表形式的cls_target和box_target转换为Tensor，并进行padding以匹配max_num_gt_in_batch
        # cls_target填充值为-1 (无效类别)
        cls_target_padded = torch.stack(
            [
                F.pad(x, (0, max_num_gt_in_batch - x.shape[0]), value=-1)
                for x in cls_target
            ]
        )
        # box_target编码并填充，填充值为0
        box_target_encoded_list = self.encode_reg_target(box_target, cls_target_padded.device)
        box_target_padded = torch.stack(
            [F.pad(x, (0, 0, 0, max_num_gt_in_batch - x.shape[0])) for x in box_target_encoded_list]
        )
        # 对于padding的GT，其box_target也置为0
        box_target_padded = torch.where(
            cls_target_padded[..., None] == -1, box_target_padded.new_tensor(0), box_target_padded
        )

        gt_instance_id_padded = None
        if gt_instance_id is not None: # 如果提供了实例ID，同样进行padding
            gt_instance_id_padded = torch.stack(
                [
                    F.pad(x, (0, max_num_gt_in_batch - x.shape[0]), value=-1)
                    for x in gt_instance_id
                ]
            )

        bs, num_gt_padded, state_dims = box_target_padded.shape # 获取维度信息

        # 将GT复制num_dn_groups次，为每组GT添加不同的噪声
        if self.num_dn_groups > 0: # 注意：条件是 >0，但前面已有 <=0 的return，这里实际总是 >0
            cls_target_dn = cls_target_padded.repeat(self.num_dn_groups, 1) # (num_dn_groups*bs, num_gt_padded)
            box_target_dn = box_target_padded.repeat(self.num_dn_groups, 1, 1) # (num_dn_groups*bs, num_gt_padded, state_dims)
            if gt_instance_id_padded is not None:
                gt_instance_id_dn = gt_instance_id_padded.repeat(self.num_dn_groups, 1)

        # 生成噪声并添加到box_target上，得到加噪锚点 (正样本去噪)
        noise_positive = torch.rand_like(box_target_dn) * 2 - 1  # 噪声范围 [-1, 1]
        noise_positive *= box_target_dn.new_tensor(self.dn_noise_scale) # 缩放噪声
        dn_anchor_positive = box_target_dn + noise_positive # 加噪锚点（正）

        # 复制目标，因为它们是这些加噪锚点的真实目标
        dn_box_target_positive = box_target_dn.clone()
        dn_cls_target_positive = cls_target_dn.clone()
        dn_id_target_positive = gt_instance_id_dn.clone() if gt_instance_id_dn is not None else None

        num_effective_gt_per_dn_group = num_gt_padded # 每个去噪组中有效（非padding）GT的数量

        if self.add_neg_dn: # 如果需要添加负去噪样本
            # 负样本是通过向GT添加更大的、可能改变其身份的噪声来生成的
            noise_negative = torch.rand_like(box_target_dn) + 1 # 噪声范围 [1, 2]
            flag_neg = torch.where( # 随机决定噪声方向
                torch.rand_like(box_target_dn) > 0.5,
                noise_negative.new_tensor(1),
                noise_negative.new_tensor(-1),
            )
            noise_negative *= flag_neg
            noise_negative *= box_target_dn.new_tensor(self.dn_noise_scale)
            dn_anchor_negative = box_target_dn + noise_negative # 加噪锚点（负）

            # 对于负样本，其类别目标设为特殊值（例如-2或-3），表示它们不应匹配任何真实类别
            dn_cls_target_negative = -torch.ones_like(cls_target_dn) * 2 # 或 *3，取决于后续处理逻辑
            # 负样本的box目标和ID目标通常不重要或设为无效值
            dn_box_target_negative = torch.zeros_like(box_target_dn)
            dn_id_target_negative = -torch.ones_like(gt_instance_id_dn) if gt_instance_id_dn is not None else None

            # 拼接正负去噪样本
            dn_anchor = torch.cat([dn_anchor_positive, dn_anchor_negative], dim=1) # (num_dn_groups*bs, 2*num_gt_padded, ...)
            dn_box_target = torch.cat([dn_box_target_positive, dn_box_target_negative], dim=1)
            dn_cls_target = torch.cat([dn_cls_target_positive, dn_cls_target_negative], dim=1)
            if dn_id_target_positive is not None:
                 dn_id_target = torch.cat([dn_id_target_positive, dn_id_target_negative], dim=1)
            else:
                 dn_id_target = None
            num_effective_gt_per_dn_group *= 2 # 每个去噪组的有效GT数量翻倍
        else: # 如果不添加负样本
            dn_anchor = dn_anchor_positive
            dn_box_target = dn_box_target_positive
            dn_cls_target = dn_cls_target_positive
            dn_id_target = dn_id_target_positive

        # 将形状从 (num_dn_groups*bs, ...) 调整回 (bs, num_dn_groups * num_effective_gt_per_dn_group, ...)
        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_effective_gt_per_dn_group, state_dims)
            .permute(1, 0, 2, 3) # (bs, num_dn_groups, num_effective_gt_per_dn_group, state_dims)
            .flatten(1, 2) # (bs, num_dn_groups * num_effective_gt_per_dn_group, state_dims)
        )
        dn_box_target = (
            dn_box_target.reshape(self.num_dn_groups, bs, num_effective_gt_per_dn_group, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_effective_gt_per_dn_group)
            .permute(1, 0, 2)
            .flatten(1)
        )
        if dn_id_target is not None:
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_effective_gt_per_dn_group)
                .permute(1, 0, 2)
                .flatten(1)
            )

        # valid_mask指示哪些去噪样本是有效的（即源于真实的GT，而不是padding或纯粹的负样本标记）
        # 对于正样本，源于真实GT的cls_target >= 0
        # 对于负样本，如果add_neg_dn=True，其cls_target被设为特殊负值，所以也需要考虑
        valid_mask_positive = dn_cls_target_positive.reshape(self.num_dn_groups,bs,num_gt_padded).permute(1,0,2).flatten(1) >=0
        if self.add_neg_dn:
            # 负样本的valid_mask取决于其对应的正样本是否有效，因为它们共享原始GT信息
            # 或者，简单地将所有负样本的valid_mask视为与正样本一致（如果其对应的正样本有效）
            # 当前实现中，负样本的dn_cls_target是-2，所以它们不会被valid_mask_positive选中
            # 一个更简单的valid_mask是基于cls_target_padded的原始有效性
            original_valid_mask = cls_target_padded.repeat(self.num_dn_groups * (2 if self.add_neg_dn else 1), 1) >= 0
            original_valid_mask = original_valid_mask.reshape(self.num_dn_groups * (2 if self.add_neg_dn else 1), bs, num_gt_padded).permute(1,0,2).flatten(1)
            valid_mask = original_valid_mask
        else:
            valid_mask = valid_mask_positive


        # 创建注意力掩码，防止去噪组内部的样本相互看到（即一个加噪GT不应从其他加噪GT获取信息）
        # attn_mask形状 (num_total_dn_queries, num_total_dn_queries)
        # num_total_dn_queries_per_batch = num_effective_gt_per_dn_group * self.num_dn_groups
        num_queries_in_dn_part = num_effective_gt_per_dn_group # 每个原始组（在复制和拼接正负样本之前）的GT数量

        attn_mask = dn_box_target.new_ones( # 注意这里用的是 num_effective_gt_per_dn_group
            num_queries_in_dn_part * self.num_dn_groups, num_queries_in_dn_part * self.num_dn_groups
        )
        for i in range(self.num_dn_groups): # 对于每个去噪组
            start = num_queries_in_dn_part * i
            end = start + num_queries_in_dn_part
            attn_mask[start:end, start:end] = 0 # 组内相互不可见（mask值为0表示可见，1为不可见）
        # 所以这里应该是 attn_mask = attn_mask == 1，或者初始值为0，组内为1后取反。
        # 如果mask值为1表示“不attend”，那么这里是对的：组内元素之间的mask是0，可以互相attend。
        # 但通常Transformer的attention_mask中，True或1表示mask掉（不attend）。
        # DN-DETR的原始实现是防止组内看到自己，但可以看到其他组的。
        # "known queries in the same group should not see each other"
        # "a query should see all other known queries in other groups"
        # 如果mask=1表示不attend，则这里attn_mask[start:end, start:end] = 1
        # 假设这里的attn_mask，值为1表示mask掉。
        # 那么当前实现是：组内可以看到，组间看不到。这与DN-DETR原文描述相反。
        # 需要确认实际用法。如果用在nn.MultiheadAttention的attn_mask，True表示不attend。
        # 如果是希望组内不互相attend，则attn_mask[start:end, start:end]应该为True (or 1)。
        # 此处代码 attn_mask == 1 会使得组内为False，组间为True。
        # 这意味着组内可以互相attend，但不能attend到其他组。这似乎也不是标准DN的意图。

        # 假设标准DN：组内看不到自己，但可以看到其他所有（包括其他组）。
        # 或者：组内看不到彼此，但可以看到其他组的。
        # 如果是后者：
        # attn_mask = torch.ones(N, N)
        # for i in range(num_groups):
        #     attn_mask[i*num_gt:(i+1)*num_gt, i*num_gt:(i+1)*num_gt] = 0 # 组内可见
        # diag_mask = torch.eye(N).bool()
        # attn_mask.masked_fill_(diag_mask, 1) # 自己看不到自己
        # attn_mask = attn_mask == 1 # True表示mask
        # 此处实现与常见DN论文中的描述可能存在差异，需结合实际使用场景理解。
        # 假设当前实现是正确的，即组内元素attn_mask为0，组间为1。
        attn_mask = attn_mask == 1 # True表示mask掉，False表示不mask。所以组内不mask，组间mask。

        dn_cls_target = dn_cls_target.long() # 确保类别目标是long类型
        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask, # 注意力掩码
            valid_mask, # 有效性掩码
            dn_id_target, # ID目标
        )

    def update_dn( # 更新去噪相关的元数据 (似乎与时序去噪相关，但当前实现不完整或依赖于父类)
        self,
        instance_feature,
        anchor,
        dn_reg_target,
        dn_cls_target,
        valid_mask,
        dn_id_target,
        num_noraml_anchor, # 应该是 num_normal_anchor (普通，非去噪锚点数量)
        temporal_valid_mask, # 时序有效性掩码
    ):
        # 此方法在父类BaseTargetWithDenoising中是pass，这里提供了更具体的实现，
        # 可能是用于合并当前帧的去噪信息和从过去帧缓存的去噪信息（如果启用了时序去噪）。
        bs, num_anchor = instance_feature.shape[:2]
        if temporal_valid_mask is None: # 如果没有时序有效性掩码
            self.dn_metas = None # 清空缓存的去噪元数据
        if self.dn_metas is None or num_noraml_anchor >= num_anchor: # 如果没有缓存或所有锚点都是普通锚点
            return ( # 直接返回输入的去噪相关张量
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        # 将输入的instance_feature和anchor拆分为普通部分和去噪部分
        num_dn_total = num_anchor - num_noraml_anchor # 总的去噪锚点/查询数量
        # 以下操作假设输入的dn_*张量是按 (bs, num_dn_groups * num_dn_per_group, ...) 组织的
        # 并且self.dn_metas中缓存的是 (bs, num_temp_dn_groups, num_temp_dn_per_group, ...)

        # 提取当前帧的去噪部分 (新生成的，非时序的)
        # current_dn_feat = instance_feature[:, num_noraml_anchor:]
        # current_dn_anchor = anchor[:, num_noraml_anchor:]
        # current_dn_reg_target = dn_reg_target # 假设输入的dn_reg_target等已经是针对去噪部分的
        # current_dn_cls_target = dn_cls_target
        # current_valid_mask = valid_mask
        # current_dn_id_target = dn_id_target

        # 从self.dn_metas中获取缓存的时序去噪信息
        cached_temp_dn_feat = self.dn_metas["dn_instance_feature"] # (bs, num_temp_dn_groups, num_temp_dn_per_group_cached, dim)
        num_temp_dn_groups_cached = self.dn_metas["dn_instance_feature"].shape[1]
        num_temp_dn_per_group_cached = self.dn_metas["dn_instance_feature"].shape[2]

        # 当前帧新生成的去噪组数量和每组数量
        # num_dn_groups_current = self.num_dn_groups - self.num_temp_dn_groups # 这假设总num_dn_groups包含时序的
        # 更可能是，输入的num_dn_groups是当前帧新生成的，而num_temp_dn_groups是独立的

        # 这部分逻辑比较复杂，涉及到如何合并缓存的时序DN查询和当前帧新生成的DN查询。
        # 核心思想是：如果temporal_valid_mask指示缓存的某个时序DN查询有效，则使用缓存的；否则用当前帧新生成的替换掉。
        # 并且需要处理数量可能不匹配的问题 (num_temp_dn_per_group_cached vs num_dn_per_group_current)。
        # 此处省略详细的逐行解释，因为具体实现高度依赖于数据组织方式和合并策略。
        # 主要操作包括：
        # 1. Reshape当前帧的DN相关张量和缓存的DN相关张量，使其具有 (bs, num_groups, num_per_group, dim) 的形式。
        # 2. 根据instance_id（如果使用）或位置匹配，尝试更新缓存的时序DN目标 (如果它们在当前帧被重新关联)。
        # 3. 根据temporal_valid_mask选择性地使用缓存的DN信息或当前帧新生成的DN信息来填充最终的DN张量。
        # 4. 将处理后的DN部分与普通锚点部分重新拼接。

        # 以下为原始代码的简化解释，具体细节需要参照论文或更详细的上下文。
        # 分离出普通锚点的特征和当前帧的去噪特征
        dn_instance_feature_current = instance_feature[:, num_noraml_anchor:]
        dn_anchor_current = anchor[:, num_noraml_anchor:]
        instance_feature_normal = instance_feature[:, :num_noraml_anchor]
        anchor_normal = anchor[:, :num_noraml_anchor]

        # Reshape 所有与当前帧去噪相关的元数据
        num_dn_groups_current = self.num_dn_groups # 假设这是当前帧新生成的DN组数
        num_dn_per_group_current = num_dn_total // num_dn_groups_current

        dn_feat_current_reshaped = dn_instance_feature_current.reshape(bs, num_dn_groups_current, num_dn_per_group_current, -1)
        dn_anchor_current_reshaped = dn_anchor_current.reshape(bs, num_dn_groups_current, num_dn_per_group_current, -1)
        dn_reg_target_reshaped = dn_reg_target.reshape(bs, num_dn_groups_current, num_dn_per_group_current, -1) # 假设输入已对应展平的DN部分
        dn_cls_target_reshaped = dn_cls_target.reshape(bs, num_dn_groups_current, num_dn_per_group_current)
        valid_mask_reshaped = valid_mask.reshape(bs, num_dn_groups_current, num_dn_per_group_current)
        dn_id_target_reshaped = dn_id_target.reshape(bs, num_dn_groups_current, num_dn_per_group_current) if dn_id_target is not None else None

        # 获取缓存的时序DN元数据
        temp_dn_feat_cached = self.dn_metas["dn_instance_feature"] # (bs, num_temp_dn_groups, num_temp_dn_per_group_cached, dim)
        num_temp_dn_groups_cached = temp_dn_feat_cached.shape[1]
        num_temp_dn_per_group_cached = temp_dn_feat_cached.shape[2]
        temp_dn_anchor_cached = self.dn_metas["dn_anchor"]
        temp_dn_cls_target_cached = self.dn_metas["dn_cls_target"]
        temp_valid_mask_cached = self.dn_metas["valid_mask"]
        temp_dn_id_cached = self.dn_metas["dn_id_target"]
        temp_dn_reg_target_cached = self.dn_metas.get("dn_box_target", torch.zeros_like(temp_dn_anchor_cached)) # 如果没有缓存box_target，用0填充


        # 合并逻辑：这里假设将缓存的 temp_dn_groups 替换掉当前帧DN的前 temp_dn_groups
        # 且假设 num_dn_per_group_current == num_temp_dn_per_group_cached
        # (如果数量不匹配，需要pad或truncate)

        final_dn_feat_list = []
        final_dn_anchor_list = []
        final_dn_reg_target_list = []
        final_dn_cls_target_list = []
        final_valid_mask_list = []
        final_dn_id_list = []

        # temporal_valid_mask: (bs, num_temp_dn_groups)
        for i in range(num_dn_groups_current):
            if i < num_temp_dn_groups_cached: # 对于那些可能有缓存替换的组
                # mask_group: (bs, 1, 1, 1) for feat/anchor/reg_target, (bs, 1, 1) for cls/id/valid
                mask_group = temporal_valid_mask[:, i:i+1, None, None]
                mask_group_cls = temporal_valid_mask[:, i:i+1, None]

                # 选择性替换
                feat_slice = torch.where(mask_group, temp_dn_feat_cached[:, i], dn_feat_current_reshaped[:, i])
                anchor_slice = torch.where(mask_group, temp_dn_anchor_cached[:, i], dn_anchor_current_reshaped[:, i])
                reg_target_slice = torch.where(mask_group, temp_dn_reg_target_cached[:, i], dn_reg_target_reshaped[:, i])

                cls_target_slice = torch.where(mask_group_cls, temp_dn_cls_target_cached[:, i], dn_cls_target_reshaped[:, i])
                valid_mask_slice = torch.where(mask_group_cls, temp_valid_mask_cached[:, i], valid_mask_reshaped[:, i])
                if dn_id_target is not None:
                    id_slice = torch.where(mask_group_cls, temp_dn_id_cached[:, i], dn_id_target_reshaped[:, i])
                    final_dn_id_list.append(id_slice)

            else: # 对于没有缓存对应的当前帧DN组，直接使用当前帧的
                feat_slice = dn_feat_current_reshaped[:, i]
                anchor_slice = dn_anchor_current_reshaped[:, i]
                reg_target_slice = dn_reg_target_reshaped[:, i]
                cls_target_slice = dn_cls_target_reshaped[:, i]
                valid_mask_slice = valid_mask_reshaped[:, i]
                if dn_id_target is not None:
                    id_slice = dn_id_target_reshaped[:, i]
                    final_dn_id_list.append(id_slice)

            final_dn_feat_list.append(feat_slice)
            final_dn_anchor_list.append(anchor_slice)
            final_dn_reg_target_list.append(reg_target_slice)
            final_dn_cls_target_list.append(cls_target_slice)
            final_valid_mask_list.append(valid_mask_slice)

        # 重新组合成 (bs, num_dn_groups * num_dn_per_group, dim) 的形状
        final_dn_feat = torch.cat(final_dn_feat_list, dim=1)
        final_dn_anchor = torch.cat(final_dn_anchor_list, dim=1)
        final_dn_reg_target = torch.cat(final_dn_reg_target_list, dim=1)
        final_dn_cls_target = torch.cat(final_dn_cls_target_list, dim=1)
        final_valid_mask = torch.cat(final_valid_mask_list, dim=1)
        final_dn_id_target = torch.cat(final_dn_id_list, dim=1) if dn_id_target is not None else None

        # 将普通锚点部分和更新后的去噪部分拼接
        instance_feature_updated = torch.cat([instance_feature_normal, final_dn_feat], dim=1)
        anchor_updated = torch.cat([anchor_normal, final_dn_anchor], dim=1)

        return (
            instance_feature_updated,
            anchor_updated,
            final_dn_reg_target,
            final_dn_cls_target,
            final_valid_mask,
            final_dn_id_target,
        )

    def cache_dn( # 缓存当前帧的去噪相关信息，用于下一帧的时序去噪
        self,
        dn_instance_feature, # 当前帧所有去噪实例的特征 (bs, num_dn_total, dim)
        dn_anchor,           # 当前帧所有去噪实例的锚点 (bs, num_dn_total, box_dim)
        dn_cls_target,       # 当前帧所有去噪实例的类别目标 (bs, num_dn_total)
        valid_mask,          # 当前帧所有去噪实例的有效性掩码 (bs, num_dn_total)
        dn_id_target,        # 当前帧所有去噪实例的ID目标 (bs, num_dn_total)
    ):
        if self.num_temp_dn_groups <= 0: # 如果不使用时序去噪组
            self.dn_metas = None # 清空缓存
            return

        num_dn_groups_total = self.num_dn_groups # 总的去噪组数 (可能包含当前帧新生成的和为时序准备的)
        if num_dn_groups_total == 0 : # 如果没有去噪组，直接返回
            self.dn_metas = None
            return

        bs, num_dn_total_samples = dn_instance_feature.shape[:2]
        num_dn_per_group = num_dn_total_samples // num_dn_groups_total # 每组的去噪样本数

        # 从所有DN组中随机选择 num_temp_dn_groups 组进行缓存
        # 注意：这里假设输入的dn_*张量是按 (bs, num_dn_groups * num_dn_per_group, ...) 组织的
        # 并且 self.num_dn_groups 包含了当前帧生成的和为时序缓存准备的。
        # 如果 self.num_dn_groups 只表示当前帧生成的，那么这里的逻辑需要调整。
        # 假设 self.num_dn_groups 是总的可以用于选择的组数。

        indices_to_cache = torch.randperm(num_dn_groups_total, device=dn_anchor.device)[:self.num_temp_dn_groups]

        # Reshape输入以便按组索引
        dn_instance_feature_reshaped = dn_instance_feature.detach().reshape(bs, num_dn_groups_total, num_dn_per_group, -1)
        dn_anchor_reshaped = dn_anchor.detach().reshape(bs, num_dn_groups_total, num_dn_per_group, -1)
        dn_cls_target_reshaped = dn_cls_target.detach().reshape(bs, num_dn_groups_total, num_dn_per_group)
        valid_mask_reshaped = valid_mask.detach().reshape(bs, num_dn_groups_total, num_dn_per_group)
        dn_id_target_reshaped = dn_id_target.detach().reshape(bs, num_dn_groups_total, num_dn_per_group) if dn_id_target is not None else None

        # 根据选中的组索引来缓存数据
        self.dn_metas = dict(
            dn_instance_feature=dn_instance_feature_reshaped[:, indices_to_cache],
            dn_anchor=dn_anchor_reshaped[:, indices_to_cache],
            dn_cls_target=dn_cls_target_reshaped[:, indices_to_cache],
            valid_mask=valid_mask_reshaped[:, indices_to_cache],
            dn_id_target=dn_id_target_reshaped[:, indices_to_cache] if dn_id_target is not None else None,
            # 可能还需要缓存 dn_box_target，如果 update_dn 中需要它
            # dn_box_target=dn_box_target_reshaped[:, indices_to_cache]
        )
