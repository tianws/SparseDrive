from typing import Optional, List

import torch

from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module() # 将SparsePoint3DDecoder注册到BBOX_CODERS注册表中
class SparsePoint3DDecoder(object): # 定义稀疏3D点解码器类
    """
    用于解码稀疏3D点预测结果的解码器。
    这类预测通常用于表示地图元素，例如车道线、路沿等，其中每个元素由一系列点构成。
    解码器负责处理模型输出的类别分数和点坐标，进行筛选和格式化。
    """
    def __init__(
        self,
        coords_dim: int = 2,  # 每个点的坐标维度 (例如2D地图点为2, 3D点为3)
        score_threshold: Optional[float] = None,  # 分数阈值，低于此阈值的预测将被过滤 (可选)
    ):
        """
        构造函数。

        Args:
            coords_dim (int, optional): 每个预测点的坐标维度。默认为2 (例如x, y)。
            score_threshold (float, optional): 用于过滤低置信度预测的分数阈值。
                                               如果为None，则不进行基于阈值的过滤。默认为None。
        """
        super(SparsePoint3DDecoder, self).__init__()
        self.score_threshold = score_threshold # 存储分数阈值
        self.coords_dim = coords_dim # 存储坐标维度

    def decode(
        self,
        cls_scores,  # 类别分数张量列表 (通常对应解码器不同层级的输出)
        pts_preds,  # 点坐标预测张量列表 (通常对应解码器不同层级的输出)
        instance_id=None,  # 实例ID张量 (可选，当前实现中未使用)
        quality=None,  # 质量预测张量 (可选，当前实现中未使用)
        output_idx=-1,  # 指定使用哪一层级的输出进行解码，默认为最后一层
    ):
        """
        解码模型输出，生成最终的稀疏点/线向量表示。

        Args:
            cls_scores (list[torch.Tensor]): 每个解码器层级的类别分数。
                                           每个张量形状 (bs, num_query, num_classes)。
            pts_preds (list[torch.Tensor]): 每个解码器层级的点坐标预测。
                                          每个张量形状 (bs, num_query, num_points_per_instance * coords_dim)。
            instance_id (torch.Tensor, optional): 实例ID。 (未使用在此实现中)
            quality (torch.Tensor, optional): 质量预测。 (未使用在此实现中)
            output_idx (int): 使用哪个解码器层级的输出。默认为-1 (最后一层)。

        Returns:
            list[dict]: 每个样本的解码结果列表。每个字典包含:
                        'vectors': 点序列列表，每个点序列是一个numpy数组。
                        'scores': 检测分数 (numpy.array)。
                        'labels': 类别ID (numpy.array)。
        """
        # 获取指定解码层级的输出
        # cls_scores[output_idx] 的形状: (bs, num_pred, num_cls)
        # pts_preds[output_idx] 的形状: (bs, num_pred, num_points_per_instance * coords_dim)
        bs, num_pred, num_cls = cls_scores[output_idx].shape # 批量大小, 预测数量(query数), 类别数量
        cls_scores_selected = cls_scores[output_idx].sigmoid()  # 对类别分数应用sigmoid转换为概率

        # 将点预测重塑为 (bs, num_pred, num_points_per_instance, coords_dim)
        # -1 会自动推断出 num_points_per_instance
        pts_preds_selected = pts_preds[output_idx].reshape(bs, num_pred, -1, self.coords_dim)

        # 选取每个预测(query)中分数最高的类别作为该预测的类别。
        # 或者更准确地说，是从所有 (query, class) 组合中选出分数最高的 num_pred 个。
        # cls_scores_selected.flatten(start_dim=1) 将 (bs, num_pred, num_cls) 变为 (bs, num_pred * num_cls)
        cls_scores_flat = cls_scores_selected.flatten(start_dim=1)

        # 为了安全，确保topk的k值不超过实际存在的元素数量
        current_k = min(num_pred, cls_scores_flat.shape[1])
        if current_k == 0: # 如果没有可供选择的预测（例如空输入）
            # 为每个batch item返回空结果
            return [{"vectors": [], "scores": np.array([]), "labels": np.array([])} for _ in range(bs)]

        # 选取top-k (或current_k) 的分数和它们在展平张量中的索引
        cls_scores_topk, indices = cls_scores_flat.topk(
            current_k, dim=1
        )
        # 通过取模运算从展平索引中恢复类别ID
        cls_ids_topk = indices % num_cls
        # 通过整除运算从展平索引中恢复原始query的索引
        query_indices = indices // num_cls

        mask_thresh = None # 初始化分数阈值掩码
        if self.score_threshold is not None: # 如果设置了分数阈值
            mask_thresh = cls_scores_topk >= self.score_threshold # 创建基于阈值的掩码

        output = [] # 初始化输出列表，用于存储每个样本的解码结果
        for i in range(bs): # 遍历每个样本
            category_ids_sample = cls_ids_topk[i] # 当前样本的类别ID (top-k选择后)
            scores_sample = cls_scores_topk[i] # 当前样本的分数 (top-k选择后)
            # 根据选中的原始query索引，从点预测中选取对应的点坐标序列
            pts_sample = pts_preds_selected[i, query_indices[i]]

            if self.score_threshold is not None and mask_thresh is not None: # 如果应用了分数阈值
                current_mask_sample = mask_thresh[i] # 获取当前样本的掩码
                category_ids_sample = category_ids_sample[current_mask_sample] # 应用掩码
                scores_sample = scores_sample[current_mask_sample] # 应用掩码
                pts_sample = pts_sample[current_mask_sample] # 应用掩码

            # 如果经过筛选后没有剩余的预测，则添加空结果
            if pts_sample.shape[0] == 0:
                output.append({"vectors": [], "scores": np.array([]), "labels": np.array([])})
                continue

            output.append( # 构建当前样本的输出字典
                {
                    # 将每个预测的点序列转换为numpy数组，并存入列表
                    "vectors": [vec.detach().cpu().numpy() for vec in pts_sample],
                    "scores": scores_sample.detach().cpu().numpy(), # 分数转numpy
                    "labels": category_ids_sample.detach().cpu().numpy(), # 类别ID转numpy
                }
            )
        return output # 返回所有样本的解码结果列表