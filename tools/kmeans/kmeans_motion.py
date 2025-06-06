import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

def lidar2agent(trajs_offset, boxes):
    origin = np.zeros((trajs_offset.shape[0], 1, 2), dtype=np.float32)
    trajs_offset = np.concatenate([origin, trajs_offset], axis=1)
    trajs = trajs_offset.cumsum(axis=1)
    yaws = - boxes[:, 6]
    rot_sin = np.sin(yaws)
    rot_cos = np.cos(yaws)
    rot_mat_T = np.stack(
        [
            np.stack([rot_cos, rot_sin]),
            np.stack([-rot_sin, rot_cos]),
        ]
    )
    trajs_new = np.einsum('aij,jka->aik', trajs, rot_mat_T)
    trajs_new = trajs_new[:, 1:]
    # 移除轨迹的第一个点（即agent自身当前位置的(0,0)点），只保留未来的相对轨迹点
    trajs_agent_local = trajs_agent_local[:, 1:] # (N, fut_ts, 2)
    return trajs_agent_local

# --- K-Means聚类参数 ---
K = 6 # 定义每个类别的运动意图聚类的簇数量 (即为每个类别生成K个典型的未来轨迹模式/锚点)
DIS_THRESH = 55 # 距离阈值 (米)，用于筛选有效的GT智能体，只考虑距离自车DIS_THRESH米范围内的智能体
# 注意: 假设轨迹长度(fut_ts)是固定的，例如12。如果不是，脚本中处理轨迹的部分 (如reshape(-1, 24) 或 reshape(-1, 12, 2)) 需要调整。
# 从 fut_masks_cls.sum(axis=1) == 12 的判断来看，脚本期望固定长度为12的轨迹。
EXPECTED_TRAJ_LEN = 12

# --- 数据加载和预处理 ---
# 指定NuScenes数据集信息文件的路径 (通常是训练集，因为它包含最丰富的轨迹信息)
fp = 'data/infos/nuscenes_infos_train.pkl'
data = mmcv.load(fp) # 使用mmcv加载.pkl文件中的数据
# 从加载的数据中提取'infos'字段，并按时间戳排序
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

intention_trajectories_by_class = dict() # 初始化字典，用于按类别存储所有提取并转换后的智能体未来轨迹
for i in range(len(CLASSES)): # 为每个类别初始化一个空列表
    intention_trajectories_by_class[i] = []

print("从数据集中提取、筛选并转换GT轨迹...")
for idx in tqdm(range(len(data_infos))): # 使用tqdm显示处理进度，遍历所有数据样本
    info = data_infos[idx] # 当前样本信息
    # 提取当前样本的GT信息
    boxes = info['gt_boxes'] # GT边界框 (x,y,z,w,l,h,yaw,...)
    names = info['gt_names'] # GT类别名称
    fut_masks = info['gt_agent_fut_masks'] # GT智能体未来轨迹的有效性掩码 (N, fut_ts)
    trajs_offsets = info['gt_agent_fut_trajs']     # GT智能体未来轨迹的相对偏移量 (N, fut_ts, 2)
    # velos = info['gt_velocity']            # GT智能体当前速度 (N, 2) (未使用在此脚本的后续逻辑中)

    # 将类别名称转换为预定义的CLASSES列表中的索引
    labels = []
    for cat_name in names:
        if cat_name in CLASSES:
            labels.append(CLASSES.index(cat_name))
        else:
            labels.append(-1) #不在关注列表中的类别标记为-1
    labels = np.array(labels)

    if len(boxes) == 0: # 如果当前样本没有GT框，则跳过
        continue    

    for class_idx in range(len(CLASSES)): # 遍历每个定义的类别
        cls_mask_current_sample = (labels == class_idx) # 获取当前样本中属于当前类别的实例的掩码
        if not np.any(cls_mask_current_sample): # 如果当前类别没有实例，则跳过
            continue

        # 提取当前类别实例的相关信息
        box_cls_current_sample = boxes[cls_mask_current_sample]
        fut_masks_cls_current_sample = fut_masks[cls_mask_current_sample]
        trajs_cls_current_sample = trajs_offsets[cls_mask_current_sample]

        # 筛选条件：
        # 1. 未来轨迹完全有效 (所有EXPECTED_TRAJ_LEN个时间步都有效)
        # 2. 当前智能体在自车附近 (距离小于DIS_THRESH)
        distance_to_ego_bev = np.linalg.norm(box_cls_current_sample[:, :2], axis=1) # 计算BEV平面距离
        filter_mask_valid_traj = np.logical_and(
            fut_masks_cls_current_sample.sum(axis=1) == EXPECTED_TRAJ_LEN, # 确保所有时间步都有效
            distance_to_ego_bev < DIS_THRESH, # 距离筛选
        )

        trajs_cls_filtered = trajs_cls_current_sample[filter_mask_valid_traj] # 筛选后的轨迹偏移量
        box_cls_filtered = box_cls_current_sample[filter_mask_valid_traj]     # 筛选后的对应GT框

        if trajs_cls_filtered.shape[0] == 0: # 如果筛选后没有有效轨迹，则跳过
            continue

        # 将筛选后的轨迹转换到各个智能体的局部坐标系 (以agent当前位置为原点，朝向为x或y轴)
        trajs_agent_local_frame = lidar2agent(trajs_cls_filtered, box_cls_filtered)
        if trajs_agent_local_frame.shape[0] == 0: # 再次检查是否有有效轨迹
            continue
        intention_trajectories_by_class[class_idx].append(trajs_agent_local_frame) # 将转换后的局部轨迹添加到对应类别的列表中

# --- 对每个类别的轨迹进行K-Means聚类 ---
clustered_prototypes_all_classes = [] # 用于存储所有类别聚类得到的运动锚点（原型轨迹）
print(f"为每个类别进行K-Means聚类 (K={K})...")
for class_idx in range(len(CLASSES)): # 遍历每个类别
    if not intention_trajectories_by_class[class_idx]: # 如果该类别没有收集到有效轨迹
        print(f"类别 '{CLASSES[class_idx]}' 没有轨迹数据进行聚类，将使用零轨迹作为占位符。")
        # 使用零轨迹作为占位符，以保持后续np.stack的形状一致
        placeholder_cluster = np.zeros((K, EXPECTED_TRAJ_LEN, 2))
        clustered_prototypes_all_classes.append(placeholder_cluster)
        continue

    # 将该类别的所有轨迹片段拼接起来，并reshape为 (num_total_trajs_for_cls, fut_ts * 2) 以便KMeans输入
    # fut_ts * 2 是因为每个时间步有2个坐标(x,y)
    concatenated_trajs_for_class = np.concatenate(intention_trajectories_by_class[class_idx], axis=0)
    # 确保轨迹长度维度存在且正确
    if concatenated_trajs_for_class.shape[1] != EXPECTED_TRAJ_LEN:
        print(f"警告: 类别 '{CLASSES[class_idx]}' 的轨迹长度不一致或不等于期望长度 {EXPECTED_TRAJ_LEN}。跳过此类。")
        placeholder_cluster = np.zeros((K, EXPECTED_TRAJ_LEN, 2))
        clustered_prototypes_all_classes.append(placeholder_cluster)
        continue

    trajs_flat_for_kmeans = concatenated_trajs_for_class.reshape(-1, EXPECTED_TRAJ_LEN * 2)
    print(f"类别 '{CLASSES[class_idx]}' 有 {trajs_flat_for_kmeans.shape[0]} 条轨迹用于聚类。")

    current_class_prototypes = None
    if trajs_flat_for_kmeans.shape[0] < K: # 如果轨迹数量少于K
        print(f"类别 '{CLASSES[class_idx]}' 的轨迹数量 ({trajs_flat_for_kmeans.shape[0]}) 少于K ({K})，将使用重复/零填充的轨迹作为锚点。")
        if trajs_flat_for_kmeans.shape[0] == 0:
            current_class_prototypes = np.zeros((K, EXPECTED_TRAJ_LEN, 2))
        else:
            num_repeats = K // trajs_flat_for_kmeans.shape[0] + 1
            cluster_trajs_cat_repeated = np.tile(trajs_flat_for_kmeans, (num_repeats, 1))[:K]
            current_class_prototypes = cluster_trajs_cat_repeated.reshape(-1, EXPECTED_TRAJ_LEN, 2)
    else: # 轨迹数量足够，执行K-Means聚类
        k_means_model_motion = KMeans(n_clusters=K, random_state=0, n_init='auto')
        cluster_centers_flat = k_means_model_motion.fit(trajs_flat_for_kmeans).cluster_centers_
        current_class_prototypes = cluster_centers_flat.reshape(-1, EXPECTED_TRAJ_LEN, 2)

    clustered_prototypes_all_classes.append(current_class_prototypes)

    # 可视化当前类别的K个原型轨迹
    plt.figure(figsize=(8,8))
    for j in range(K):
        if j < current_class_prototypes.shape[0]:
            plt.plot(current_class_prototypes[j, :, 0], current_class_prototypes[j, :, 1], marker='.', linestyle='-')
    plt.title(f'Motion Anchors for {CLASSES[class_idx]} (K={K})')
    plt.xlabel('X offset (m, agent local frame)')
    plt.ylabel('Y offset (m, agent local frame)')
    plt.grid(True)
    plt.axis('equal')
    output_visualization_path = f'vis/kmeans/motion_intention_{CLASSES[class_idx]}_{K}.png'
    plt.savefig(output_visualization_path, bbox_inches='tight')
    print(f"类别 '{CLASSES[class_idx]}' 的运动锚点可视化图像已保存至: {output_visualization_path}")
    plt.close()

if not clustered_prototypes_all_classes:
    print("未能成功为任何类别生成运动锚点。")
else:
    final_motion_anchors_np = np.stack(clustered_prototypes_all_classes, axis=0)
    output_npy_path = f'data/kmeans/kmeans_motion_{K}.npy'
    np.save(output_npy_path, final_motion_anchors_np)
    print(f"生成的运动锚点数据 (形状: {final_motion_anchors_np.shape}) 已保存至: {output_npy_path}")
# --- 主逻辑结束 ---