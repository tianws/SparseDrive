import torch # 导入PyTorch库

from .deformable_aggregation import DeformableAggregationFunction # 从同级目录的deformable_aggregation模块导入DeformableAggregationFunction类


def deformable_aggregation_function( # 定义一个可变形特征聚合的函数接口
    feature_maps,  # 输入的特征图谱 (通常是经过feature_maps_format格式化后的)
    spatial_shape,  # 特征图谱的空间形状信息
    scale_start_index,  # 每个尺度特征在扁平化特征图中的起始索引
    sampling_location,  # 采样点的坐标
    weights,  # 对应每个采样点的权重
):
    """
    可变形特征聚合函数的包装器。
    调用底层实现的 DeformableAggregationFunction.apply 方法。

    Args:
        feature_maps (torch.Tensor): 格式化后的特征图谱。
        spatial_shape (torch.Tensor): 原始特征图的空间形状。
        scale_start_index (torch.Tensor): 各尺度特征在扁平化特征图中的起始索引。
        sampling_location (torch.Tensor): 采样位置。
        weights (torch.Tensor): 采样权重。

    Returns:
        torch.Tensor: 聚合后的特征。
    """
    return DeformableAggregationFunction.apply( # 调用自定义CUDA操作的apply方法
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )


def feature_maps_format(feature_maps, inverse=False): # 定义特征图谱格式化函数
    """
    在两种特征图谱表示格式之间进行转换：
    1. 列表格式：一个列表，其中每个元素代表一个相机视图，该元素本身又是一个列表，
                 包含该视图下不同层级的特征图张量 (bs, C, H_level, W_level)。
                 例如: [[cam1_lvl1, cam1_lvl2], [cam2_lvl1, cam2_lvl2], ...]
    2. 柱状格式（Columnar Format）：一种更紧凑的表示，将所有特征图谱数据合并到一个大的张量中，
                               并辅以形状和索引信息。通常用于自定义CUDA操作的输入。
                               返回一个包含 [col_feats, spatial_shape, scale_start_index] 的列表。

    Args:
        feature_maps (list or tuple): 输入的特征图谱。
                                     如果 inverse=False，则期望为列表格式。
                                     如果 inverse=True，则期望为柱状格式。
        inverse (bool): 是否执行逆向转换（从柱状格式转回列表格式）。默认为False。

    Returns:
        list or list[list[torch.Tensor]]: 格式化后的特征图谱。
                                          如果 inverse=False，返回柱状格式。
                                          如果 inverse=True，返回列表格式。
    """
    if inverse: # 从柱状格式转回列表格式
        col_feats, spatial_shape, scale_start_index = feature_maps # 解包柱状格式的三个组件
        num_cams, num_levels = spatial_shape.shape[:2] # 获取相机数量和层级数量

        # split_size 计算每个相机每个层级特征图展平后的大小 (H_level * W_level)
        split_size_per_level = spatial_shape[..., 0] * spatial_shape[..., 1] # (num_cams, num_levels)
        split_size_per_cam = split_size_per_level.sum(dim=1).cpu().numpy().tolist() # 每个相机所有层级展平后的总大小

        # cam_split 和 cam_split_size_actual 用于处理一种特殊情况：
        # 如果不同相机的特征图谱组合方式不同（例如，某些相机组的特征在col_feats中是连续存储的），
        # 这里似乎假设所有相机的层级结构是相同的，或者col_feats是按相机主要顺序然后是层级拼接的。
        # 简化理解：将col_feats按每个相机应有的大小进行切分。
        # 这里的cam_split和cam_split_size_actual的逻辑比较复杂，似乎是为了处理更一般化的特征组合情况，
        # 但在典型用法中，通常每个相机有相同数量和结构的层级。
        # 假设col_feats是 (bs, sum_all_cam_level_sizes, C)
        # 我们需要将其拆分为 (bs, num_cams, num_levels, H, W, C) -> (bs, num_cams, C, H, W)

        # 重新组织以支持更灵活的相机-层级分组。
        # cam_split_sizes_for_split_op: 记录col_feats中每个主要块的大小，每个块可能包含一个或多个相机的特征。
        cam_block_splits = []
        current_block_size = 0
        # spatial_shape (num_total_cam_entries, num_levels_per_entry, 2)
        # scale_start_index (num_total_cam_entries, num_levels_per_entry)
        # 这里的逻辑似乎是处理当feature_maps输入到feature_maps_format(inverse=False)时，
        # 如果输入的是一个包含多个相机组的列表，例如 [[cam0-3_feats], [cam4-5_feats]]
        # 那么spatial_shape和scale_start_index会有多个对应的条目。
        # inverse=True时，需要根据这些信息正确地还原。

        # 假设 col_feats 是 (bs, total_pixels_all_cams_all_levels, C)
        # spatial_shape (total_num_cam_view_configs, num_levels, 2)
        # scale_start_index (total_num_cam_view_configs, num_levels)

        # 简化的逆向转换逻辑（假设所有相机视图具有相同的层级结构，并且在col_feats中是简单拼接的）：
        # C = col_feats.shape[-1]
        # mc_ms_feat = []
        # current_offset = 0
        # for cam_idx in range(num_cams):
        #     cam_feats_list = []
        #     for lvl_idx in range(num_levels):
        #         h, w = spatial_shape[cam_idx, lvl_idx, 0], spatial_shape[cam_idx, lvl_idx, 1]
        #         num_pixels_lvl = h * w
        #         feat_lvl = col_feats[:, current_offset : current_offset + num_pixels_lvl, :]
        #         feat_lvl = feat_lvl.reshape(col_feats.shape[0], h, w, C).permute(0, 3, 1, 2) # bs, C, H, W
        #         cam_feats_list.append(feat_lvl)
        #         current_offset += num_pixels_lvl
        #     mc_ms_feat.append(cam_feats_list)
        # return mc_ms_feat
        # 上述简化逻辑与原始代码中的复杂分组和split/unflatten不完全一致，原始代码支持更复杂的输入结构。
        # 以下保留原始代码的逻辑，并尝试解释。

        # 确定col_feats中每个主要相机组块的大小
        cam_split_sizes_for_split_op = []
        # spatial_shape_cpu (total_cam_entries, num_levels, 2)
        spatial_shape_cpu = spatial_shape.cpu().numpy()
        # split_size_cpu (total_cam_entries, num_levels)
        split_size_cpu = (spatial_shape_cpu[...,0] * spatial_shape_cpu[...,1])

        # cam_group_boundaries: 标记哪些相机配置属于同一个组（在col_feats中连续存储）
        # 例如，如果前3个相机配置在col_feats中是一块，后2个是另一块。
        # 这里的逻辑是基于相邻的spatial_shape是否完全相同来判断是否属于一个连续块。
        # 这部分在Deformable DETR等模型中，如果不同相机共享FPN，则它们的spatial_shape和scale_start_index会相同。
        # 但如果用了类似多相机独立FPN或不同相机组用不同FPN，则会不同。

        # 重新解析原始代码的逆向逻辑：
        # col_feats: (bs, total_flat_pixels, C)
        # spatial_shape: (num_cam_entries_in_col_feats, num_levels_per_entry, 2)
        # scale_start_index: (num_cam_entries_in_col_feats, num_levels_per_entry) - 这个在逆向时似乎没直接用，用spatial_shape计算split_size

        bs = col_feats.shape[0]
        C = col_feats.shape[-1]

        num_cam_entries = spatial_shape.shape[0] # 对应于最初调用format(inverse=False)时，feature_maps列表的长度

        reconstructed_feature_maps = []
        current_pixel_offset_in_col_feats = 0

        # 外层循环对应原始输入feature_maps的每个元素（每个元素可能是一个相机的多层特征，或一组相机的多层特征）
        for entry_idx in range(num_cam_entries):
            # 当前条目（例如一个相机或一个相机组）对应的所有层级的特征
            entry_features_per_level = []
            num_levels_this_entry = spatial_shape.shape[1]

            for lvl_idx in range(num_levels_this_entry):
                h = spatial_shape[entry_idx, lvl_idx, 0].item()
                w = spatial_shape[entry_idx, lvl_idx, 1].item()
                num_pixels_level = h * w

                # 从col_feats中切片出当前层级的扁平化特征
                flat_feat_level = col_feats[:, current_pixel_offset_in_col_feats : current_pixel_offset_in_col_feats + num_pixels_level, :]
                # Reshape回 (bs, C, H, W)
                feat_level_reshaped = flat_feat_level.reshape(bs, num_pixels_level, C).transpose(1,2).reshape(bs, C, h, w)
                entry_features_per_level.append(feat_level_reshaped)
                current_pixel_offset_in_col_feats += num_pixels_level

            # 在原始实现中，如果输入是多相机共享FPN，则一个entry_idx可能对应多个相机
            # 而这里的spatial_shape[entry_idx]是 (num_levels, 2)
            # Deformable DETR中，spatial_shapes是 (num_levels, 2)，level_start_index是 (num_levels)
            # 多相机版本中，这些会复制num_cams次，或者每个相机有自己的。
            # 这里的实现似乎假设输入的feature_maps最外层列表的每个元素对应一个"相机条目"，
            # 这个"相机条目"内部再包含多层级特征。
            # 如果原始输入是 List[List[Tensor]] (List[cam_feats_for_all_levels])
            # 那么 num_cam_entries 就是相机数。
            reconstructed_feature_maps.append(entry_features_per_level)
        return reconstructed_feature_maps


    # 处理列表格式输入 -> 柱状格式输出
    if isinstance(feature_maps[0], (list, tuple)): # 如果输入是嵌套列表 (例如，多个FPN输出的列表)
        # 递归调用feature_maps_format处理每个子列表
        formated = [feature_maps_format(x) for x in feature_maps]
        # 将每个子列表返回的柱状特征进行拼接
        col_feats = torch.cat([x[0] for x in formated], dim=1) # 拼接扁平化特征
        spatial_shape = torch.cat([x[1] for x in formated], dim=0) # 拼接空间形状信息
        # scale_start_index需要重新计算或调整，因为拼接改变了相对位置
        # 这里的实现是直接拼接，这可能要求后续使用者理解这种结构
        # 或者假设每个formated_x的scale_start_index是相对于其自身的col_feats的。
        # 如果col_feats是沿dim=1（像素维度）拼接，那么后续的scale_start_index需要加上前面块的总像素数。
        # 假设这里的拼接是针对更复杂的场景，例如将不同来源的特征图（已格式化为柱状）合并。
        # 对于标准的多相机单FPN输入，通常不会进入这个分支。
        # 重新审视：如果feature_maps是 List[fpn_output_for_cam_group_i]，每个fpn_output_for_cam_group_i是List[Tensor]
        # 那么递归调用后，formated的每个元素是 [col_feat_group_i, spatial_shape_group_i, scale_start_index_group_i]
        # col_feats拼接是合理的。spatial_shape拼接也是合理的，表示不同组的形状信息。
        # scale_start_index的拼接也是合理的，每个index是相对于其对应col_feat_group_i的。
        # 但如果最终要得到一个统一的scale_start_index用于整个拼接后的col_feats，则需要调整。
        # 此处代码似乎是直接拼接，意味着返回的scale_start_index也是一个分段的索引。
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        return [col_feats, spatial_shape, scale_start_index]

    # 处理标准的多相机单FPN输入: List[Tensor(bs, num_cams, C_lvl, H_lvl, W_lvl)]
    # 或者更常见的是 List[Tensor(bs*num_cams, C_lvl, H_lvl, W_lvl)] 或 List[Tensor(bs, C_lvl, H_lvl, W_lvl)] (如果num_cams=1)
    # 假设输入 feature_maps 是一个列表，每个元素是 (bs, num_cams, C_level, H_level, W_level)
    # 或者 (bs, C_level, H_level, W_level) 如果num_cams=1且已集成到bs或C
    # 这里的代码似乎期望 feature_maps[0] 的形状能直接给出 bs 和 num_cams

    # 假设输入 feature_maps 是一个列表，每个元素是特定层级的特征图:
    # feature_maps = [lvl0_feat, lvl1_feat, ...]
    # 每个 lvl_feat 的形状是 (bs, num_cams, C_lvl, H_lvl, W_lvl)

    bs, num_cams = feature_maps[0].shape[:2] # 从第一个层级的特征图获取bs和num_cams
    spatial_shapes_list_of_tuples = [] # 用于存储每个层级的 (H, W)

    col_feats_list = [] # 用于存储每个层级展平后的特征
    for i, feat_level in enumerate(feature_maps): # 遍历每个层级的特征图
        # feat_level: (bs, num_cams, C_lvl, H_lvl, W_lvl)
        spatial_shapes_list_of_tuples.append(feat_level.shape[-2:]) # 记录 (H_lvl, W_lvl)
        # 将特征展平并调整维度顺序
        # (bs, num_cams, C_lvl, H_lvl*W_lvl)
        col_feats_list.append(
            torch.reshape(feat_level, (bs, num_cams, feat_level.shape[2], -1))
        )

    # 将所有层级的展平特征沿最后一个维度（像素维度）拼接
    # col_feats_list 中每个元素 (bs, num_cams, C_lvl, H_lvl*W_lvl)
    # 注意：如果不同层级的C_lvl不同，则不能直接cat。这里假设C_lvl是相同的 (embed_dims)
    # 或者说，这里的C实际上是embed_dims，对于多尺度特征，通常channel数是一致的。
    # (bs, num_cams, embed_dims, total_pixels_across_levels)
    col_feats_combined_levels = torch.cat(col_feats_list, dim=-1)

    # 调整维度顺序为 (bs, num_cams, total_pixels_across_levels, embed_dims)
    # 然后将 num_cams 和 total_pixels_across_levels 维度展平
    # 最终得到 (bs, num_cams * total_pixels_across_levels, embed_dims)
    # 这是自定义CUDA核通常期望的格式之一：所有点（跨相机、跨层级）的特征向量列表
    col_feats_final = col_feats_combined_levels.permute(0, 1, 3, 2).flatten(1, 2)

    # 构建spatial_shape张量: (num_cams, num_levels, 2)
    # spatial_shapes_list_of_tuples: [(H0,W0), (H1,W1), ...]
    # 将其复制num_cams次，假设所有相机视图共享相同的FPN结构和特征图尺寸
    spatial_shape_tensor = [spatial_shapes_list_of_tuples] * num_cams
    spatial_shape_tensor = torch.tensor(
        spatial_shape_tensor,
        dtype=torch.int64,
        device=col_feats_final.device,
    ) # (num_cams, num_levels, 2)

    # 计算scale_start_index: 每个层级在展平后的特征中的起始索引
    # (num_cams, num_levels)
    scale_start_index_tensor = spatial_shape_tensor[..., 0] * spatial_shape_tensor[..., 1]
    # (num_cams * num_levels)
    scale_start_index_tensor = scale_start_index_tensor.flatten().cumsum(dim=0)
    # 插入0作为第一个层级的起始索引
    scale_start_index_tensor = torch.cat(
        [torch.tensor([0], device=scale_start_index_tensor.device), scale_start_index_tensor[:-1]]
    )
    # (num_cams, num_levels)
    scale_start_index_tensor = scale_start_index_tensor.reshape(num_cams, -1)

    # 组合成柱状格式输出
    formatted_feature_maps = [
        col_feats_final, # 展平并拼接的特征 (bs, N_total_pixels, C)
        spatial_shape_tensor,    # 各层级空间形状 (num_cams, num_levels, 2)
        scale_start_index_tensor, # 各层级起始索引 (num_cams, num_levels)
    ]
    return formatted_feature_maps
