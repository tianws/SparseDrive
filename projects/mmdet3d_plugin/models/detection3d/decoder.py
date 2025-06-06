from typing import Optional # 导入类型提示Optional，用于可能为None的类型

import torch # 导入PyTorch库

from mmdet.core.bbox.builder import BBOX_CODERS # 从MMDetection的核心包中导入BBOX_CODERS注册表

from projects.mmdet3d_plugin.core.box3d import * # 从项目中导入3D边界框相关的常量定义 (X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX等)

def decode_box(box): # 定义一个函数，用于解码单个或一批边界框
    """
    将包含sin/cos形式偏航角和指数形式尺寸的边界框表示转换为标准形式。
    标准形式：(x, y, z, w_decoded, l_decoded, h_decoded, yaw_radian, vx, ...)。

    Args:
        box (torch.Tensor): 输入的边界框张量，形状为 (..., D)，
                            其中D至少包含X,Y,Z,W,L,H,SIN_YAW,COS_YAW以及可选的速度VX等。
                            W,L,H通常是以对数尺度存储的，需要取指数。

    Returns:
        torch.Tensor: 解码后的边界框张量。
    """
    # 从sin_yaw和cos_yaw计算偏航角 (弧度)
    # box[..., SIN_YAW] 和 box[..., COS_YAW] 分别是偏航角的正弦和余弦值
    yaw = torch.atan2(box[..., SIN_YAW], box[..., COS_YAW])
    # 构建解码后的边界框张量
    box = torch.cat( # 沿最后一个维度拼接
        [
            box[..., [X, Y, Z]],  # 中心点坐标 (x, y, z)
            box[..., [W, L, H]].exp(),  # 尺寸 (w, l, h)，取指数恢复到真实尺度
            yaw[..., None],  # 计算得到的偏航角 (弧度)，增加一个维度以匹配拼接
            box[..., VX:],  # 速度分量 (vx, vy, vz等，如果有的话)
        ],
        dim=-1, # 沿最后一个维度拼接
    )
    return box


@BBOX_CODERS.register_module() # 将SparseBox3DDecoder注册到MMDetection的BBOX_CODERS注册表中
class SparseBox3DDecoder(object): # 定义稀疏3D边界框解码器类
    """
    用于稀疏3D目标检测模型的解码器。
    它处理模型原始输出的类别分数、边界框预测等，
    执行top-k选择、分数阈值过滤，并结合可选的质量度量（如中心度）来调整分数，
    最终输出标准格式的检测结果。
    """
    def __init__(
        self,
        num_output: int = 300,  # 每个样本输出的最大检测框数量
        score_threshold: Optional[float] = None,  # 分数阈值，低于此阈值的检测结果将被过滤 (可选)
        sorted: bool = True,  # 是否对输出的检测结果按分数排序
    ):
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output # 输出检测框数量上限
        self.score_threshold = score_threshold # 分数阈值
        self.sorted = sorted # 是否排序

    def decode(
        self,
        cls_scores,  # 类别分数张量列表 (通常对应解码器不同层级的输出)
        box_preds,  # 边界框预测张量列表 (通常对应解码器不同层级的输出)
        instance_id=None,  # 实例ID张量 (可选，用于跟踪或关联)
        quality=None,  # 质量预测张量列表 (可选，例如中心度、偏航度等)
        output_idx=-1,  # 指定使用哪一层级的输出进行解码，默认为最后一层
    ):
        """
        解码模型输出，生成最终的3D检测结果。

        Args:
            cls_scores (list[torch.Tensor]): 每个解码器层级的类别分数。
                                           每个张量形状 (bs, num_query, num_classes)。
            box_preds (list[torch.Tensor]): 每个解码器层级的边界框预测。
                                          每个张量形状 (bs, num_query, box_dim)。
            instance_id (torch.Tensor, optional): 实例ID，形状 (bs, num_query)。
            quality (list[torch.Tensor], optional): 每个解码器层级的质量预测。
                                                  每个张量形状 (bs, num_query, num_quality_metrics)。
            output_idx (int): 使用哪个解码器层级的输出。默认为-1 (最后一层)。

        Returns:
            list[dict]: 每个样本的检测结果列表。每个字典包含:
                        'boxes_3d': 解码后的3D边界框 (torch.Tensor, CPU)。
                        'scores_3d': 最终的检测分数 (torch.Tensor, CPU)。
                        'labels_3d': 类别ID (torch.Tensor, CPU)。
                        'cls_scores' (optional): 原始分类分数 (如果使用了quality调整)。
                        'instance_ids' (optional): 实例ID。
        """
        squeeze_cls = instance_id is not None # 如果提供了instance_id，则后续处理类别ID时可能需要特殊处理（例如，如果ID已经包含了类别信息）

        cls_scores = cls_scores[output_idx].sigmoid()  # 获取指定层级的类别分数并应用sigmoid转换为概率

        if squeeze_cls: # 如果为True，意味着每个query只预测一个主要类别
            cls_scores, cls_ids = cls_scores.max(dim=-1)  # 取每个query的最大类别分数和对应的类别ID
            cls_scores = cls_scores.unsqueeze(dim=-1)  # 恢复维度以与其他情况统一处理 (bs, num_query, 1)

        box_preds = box_preds[output_idx]  # 获取指定层级的边界框预测
        bs, num_pred, num_cls = cls_scores.shape  # 批量大小, 预测数量 (query数), 类别数

        # 将类别分数展平并通过topk选取前num_output个最高分
        # indices 是展平后分数张量中的索引
        cls_scores_flat = cls_scores.flatten(start_dim=1) # (bs, num_pred * num_cls)
        if cls_scores_flat.shape[1] < self.num_output : # 如果总预测数小于num_output
             # 在这种情况下，topk会返回所有元素，我们不需要改变num_output
             current_k = cls_scores_flat.shape[1]
        else:
             current_k = self.num_output

        cls_scores_topk, indices = cls_scores_flat.topk(
            current_k, dim=1, sorted=self.sorted
        )

        if not squeeze_cls: # 如果不是每个query只预测一个类别 (即每个query对所有类别都有分数)
            cls_ids = indices % num_cls # 通过取模运算从展平索引中恢复类别ID

        mask = None # 初始化掩码
        if self.score_threshold is not None: # 如果设置了分数阈值
            mask = cls_scores_topk >= self.score_threshold # 创建基于阈值的掩码

        current_quality = None
        if quality is not None and quality[output_idx] is not None:
             current_quality = quality[output_idx]

        if current_quality is not None: # 如果提供了质量预测 (如centerness)
            # CNS 是centerness在quality张量中的索引
            centerness = current_quality[..., CNS] # (bs, num_query)
            # 根据topk选择的索引，获取对应的centerness值
            # indices // num_cls 得到的是原始query的索引
            centerness = torch.gather(centerness, 1, indices // num_cls if not squeeze_cls else indices)

            cls_scores_origin = cls_scores_topk.clone() # 保存原始的topk类别分数
            cls_scores_topk *= centerness.sigmoid() # 将类别分数与sigmoid后的centerness相乘，作为新的分数

            # 根据新的分数重新排序
            cls_scores_topk, idx_after_quality = torch.sort(cls_scores_topk, dim=1, descending=True)
            if not squeeze_cls: # 如果需要，同步更新cls_ids的顺序
                cls_ids = torch.gather(cls_ids, 1, idx_after_quality)
            if self.score_threshold is not None and mask is not None: # 如果需要，同步更新mask的顺序
                mask = torch.gather(mask, 1, idx_after_quality)
            indices = torch.gather(indices, 1, idx_after_quality) # 同步更新原始索引的顺序

        output = [] # 初始化输出列表，用于存储每个样本的检测结果
        for i in range(bs): # 遍历每个样本
            category_ids_sample = cls_ids[i] if not squeeze_cls else cls_ids[indices[i] // num_cls] # 获取类别ID
            scores_sample = cls_scores_topk[i] # 获取最终分数
            # 根据展平后的索引恢复原始query的索引，并选取对应的边界框预测
            box_sample = box_preds[i, indices[i] // num_cls if not squeeze_cls else indices[i]]

            mask_sample = None
            if self.score_threshold is not None and mask is not None: # 如果应用了分数阈值
                mask_sample = mask[i]
                category_ids_sample = category_ids_sample[mask_sample] # 应用掩码
                scores_sample = scores_sample[mask_sample] # 应用掩码
                box_sample = box_sample[mask_sample] # 应用掩码

            cls_scores_origin_sample = None
            if current_quality is not None: # 如果使用了质量调整分数
                cls_scores_origin_sample = cls_scores_origin[i] # 获取调整前的分数
                if self.score_threshold is not None and mask_sample is not None:
                    cls_scores_origin_sample = cls_scores_origin_sample[mask_sample] # 应用掩码

            box_decoded_sample = decode_box(box_sample) # 解码边界框参数

            sample_output = { # 构建当前样本的输出字典
                "boxes_3d": box_decoded_sample.cpu(), # 3D边界框 (转到CPU)
                "scores_3d": scores_sample.cpu(), # 最终分数 (转到CPU)
                "labels_3d": category_ids_sample.cpu(), # 类别ID (转到CPU)
            }
            if cls_scores_origin_sample is not None: # 如果有原始分类分数
                sample_output["cls_scores"] = cls_scores_origin_sample.cpu()

            if instance_id is not None: # 如果有实例ID
                # 根据展平后的索引恢复原始query的索引，并选取对应的实例ID
                ids_sample = instance_id[i, indices[i] // num_cls if not squeeze_cls else indices[i]]
                if self.score_threshold is not None and mask_sample is not None:
                    ids_sample = ids_sample[mask_sample] # 应用掩码
                sample_output["instance_ids"] = ids_sample.cpu() # 添加实例ID (转到CPU)
            output.append(sample_output) # 将当前样本的结果添加到输出列表

        return output # 返回所有样本的检测结果列表
