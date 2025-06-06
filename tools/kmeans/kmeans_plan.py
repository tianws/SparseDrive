import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

# K = 6 # 定义每个驾驶命令下，要聚类的簇数量 (即为每个命令生成K个典型的规划轨迹锚点)
# 注意: 脚本中的轨迹长度是硬编码的 (例如6个时间步，12个值)。如果数据集中的轨迹长度不同，需要调整。
EXPECTED_EGO_TRAJ_LEN = 6 # 期望的自车未来轨迹长度 (时间步数)

# --- 主逻辑开始 ---
# 指定NuScenes数据集信息文件的路径 (通常是训练集)
fp = 'data/infos/nuscenes_infos_train.pkl'
data = mmcv.load(fp) # 使用mmcv加载.pkl文件中的数据
# 从加载的数据中提取'infos'字段，并按时间戳排序
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

# 初始化列表，用于按驾驶命令存储自车未来轨迹
# 假设有3种命令 (例如：右转、左转、直行，对应索引0, 1, 2，这通常由数据集预处理决定)
# CMD_LIST = ['Turn Right', 'Turn Left', 'Go Straight'] (来自bev_render.py，这里假设命令索引与之对应)
num_commands = 3
navi_trajs_by_command = [[] for _ in range(num_commands)]

print("从数据集中提取、筛选并按命令分类自车GT轨迹...")
for idx in tqdm(range(len(data_infos))): # 使用tqdm显示处理进度，遍历所有数据样本
    info = data_infos[idx] # 当前样本信息

    # info['gt_ego_fut_trajs'] 存储的是相对于前一时刻的位移 (num_timesteps, 2)
    # .cumsum(axis=0) 将其转换为相对于当前时刻(0,0)的累积轨迹 (num_timesteps, 2)
    # 注意：axis=-2 对于 (T,2) 的数组，等同于 axis=0
    plan_traj_cumulative = info['gt_ego_fut_trajs'].cumsum(axis=0)
    plan_mask = info['gt_ego_fut_masks'] # 轨迹有效性掩码 (num_timesteps)

    # info['gt_ego_fut_cmd'] 通常是one-hot编码或概率分布 (num_cmds,)
    cmd_one_hot = info['gt_ego_fut_cmd'].astype(np.int32)
    cmd_idx = cmd_one_hot.argmax(axis=-1) # 获取概率最大的命令的索引

    # 只保留未来轨迹完整（例如，持续EXPECTED_EGO_TRAJ_LEN个时间步）的样本
    if not plan_mask.sum() == EXPECTED_EGO_TRAJ_LEN:
        continue # 跳过不完整的轨迹

    if cmd_idx < num_commands: # 确保命令索引有效
        navi_trajs_by_command[cmd_idx].append(plan_traj_cumulative) # 将有效的累积轨迹按命令分类存储
    else:
        print(f"警告: 样本 {idx} 的命令索引 {cmd_idx} 超出范围，已忽略。")


clustered_prototypes_all_commands = [] # 用于存储所有命令的聚类得到的规划锚点（原型轨迹）
print(f"为每个驾驶命令的轨迹进行K-Means聚类 (K={K})...")

# 遍历每个命令下的轨迹列表
# cmd_idx_enum: 0, 1, 2; trajs_for_this_command: 该命令下的所有轨迹列表
for cmd_idx_enum, trajs_for_this_command in enumerate(navi_trajs_by_command):
    command_name = CMD_LIST[cmd_idx_enum] if cmd_idx_enum < len(CMD_LIST) else f"Command_{cmd_idx_enum}"
    if not trajs_for_this_command: # 如果当前命令没有收集到轨迹数据
        print(f"命令 '{command_name}' 没有轨迹数据进行聚类，将使用零轨迹作为占位符。")
        # 使用零轨迹作为占位符，以保持后续np.stack的形状一致
        placeholder_cluster = np.zeros((K, EXPECTED_EGO_TRAJ_LEN, 2))
        clustered_prototypes_all_commands.append(placeholder_cluster)
        continue

    # 将该命令下的所有轨迹片段拼接起来 (N_trajs_cmd, EXPECTED_EGO_TRAJ_LEN, 2)
    # 然后reshape为 (N_trajs_cmd, EXPECTED_EGO_TRAJ_LEN * 2) 以便KMeans输入
    concatenated_trajs_for_cmd = np.concatenate(trajs_for_this_command, axis=0)
    trajs_flat_for_kmeans = concatenated_trajs_for_cmd.reshape(-1, EXPECTED_EGO_TRAJ_LEN * 2)
    print(f"命令 '{command_name}' 有 {trajs_flat_for_kmeans.shape[0]} 条轨迹用于聚类。")

    current_command_prototypes = None
    if trajs_flat_for_kmeans.shape[0] < K: # 如果轨迹数量少于K
        print(f"命令 '{command_name}' 的轨迹数量 ({trajs_flat_for_kmeans.shape[0]}) 少于K ({K})，将使用重复/零填充的轨迹作为锚点。")
        if trajs_flat_for_kmeans.shape[0] == 0:
            current_command_prototypes = np.zeros((K, EXPECTED_EGO_TRAJ_LEN, 2))
        else: # 重复已有轨迹或用0填充至K个
            num_repeats = K // trajs_flat_for_kmeans.shape[0] + 1
            cluster_trajs_cat_repeated = np.tile(trajs_flat_for_kmeans, (num_repeats, 1))[:K]
            current_command_prototypes = cluster_trajs_cat_repeated.reshape(-1, EXPECTED_EGO_TRAJ_LEN, 2)
    else: # 轨迹数量足够，执行K-Means聚类
        k_means_model_plan = KMeans(n_clusters=K, random_state=0, n_init='auto')
        cluster_centers_flat = k_means_model_plan.fit(trajs_flat_for_kmeans).cluster_centers_
        # 将扁平的簇中心reshape回轨迹形状 (K, EXPECTED_EGO_TRAJ_LEN, 2)
        current_command_prototypes = cluster_centers_flat.reshape(-1, EXPECTED_EGO_TRAJ_LEN, 2)

    clustered_prototypes_all_commands.append(current_command_prototypes)

    # 可视化当前命令的K个原型轨迹
    plt.figure(figsize=(8,8)) # 为每个命令创建一个新的图像
    for j in range(K): # 遍历K个原型轨迹
        if j < current_command_prototypes.shape[0]:
            plt.plot(current_command_prototypes[j, :, 0], current_command_prototypes[j, :, 1], marker='.', linestyle='-')
    plt.title(f'Planning Anchors for Command: {command_name} (K={K})')
    plt.xlabel('X offset (m, ego frame)')
    plt.ylabel('Y offset (m, ego frame)')
    plt.grid(True)
    plt.axis('equal') # 保持x,y轴等比例
    output_visualization_path = f'vis/kmeans/plan_cmd_{command_name.replace(" ", "_")}_anchor_{K}.png'
    plt.savefig(output_visualization_path, bbox_inches='tight')
    print(f"命令 '{command_name}' 的规划锚点可视化图像已保存至: {output_visualization_path}")
    plt.close() # 关闭当前图像，释放资源

if not clustered_prototypes_all_commands or len(clustered_prototypes_all_commands) != num_commands :
    print("未能成功为所有命令生成规划锚点，或部分命令数据不足。请检查输出。")
    # 如果列表为空或长度不匹配，创建一个占位符或根据需求处理错误
    if not clustered_prototypes_all_commands: # 如果完全为空
         final_planning_anchors_np = np.zeros((num_commands, K, EXPECTED_EGO_TRAJ_LEN, 2))
         print(f"生成了全零的规划锚点数据，形状: {final_planning_anchors_np.shape}")
    else: # 如果部分成功，尝试堆叠，但可能会因形状不一致而出错，除非都填充了
         try:
            final_planning_anchors_np = np.stack(clustered_prototypes_all_commands, axis=0)
         except ValueError as e:
            print(f"堆叠聚类结果时出错 (可能是由于某些命令没有足够的轨迹数据且未正确填充): {e}")
            print("将保存部分成功的结果（如果有）或一个空数组。")
            # 尝试保存第一个成功的结果作为示例，或者保存一个明确的错误标记文件
            if clustered_prototypes_all_commands[0] is not None:
                 np.save(f'data/kmeans/kmeans_plan_{K}_PARTIAL_ERROR.npy', clustered_prototypes_all_commands[0])
            final_planning_anchors_np = np.array([]) # 表示失败
else:
    # 将所有命令的原型轨迹堆叠成一个大的NumPy数组
    # 最终形状 (num_commands, K, EXPECTED_EGO_TRAJ_LEN, 2)
    final_planning_anchors_np = np.stack(clustered_prototypes_all_commands, axis=0)
    # 保存生成的规划锚点到.npy文件
    output_npy_path = f'data/kmeans/kmeans_plan_{K}.npy'
    np.save(output_npy_path, final_planning_anchors_np)
    print(f"生成的规划锚点数据 (形状: {final_planning_anchors_np.shape}) 已保存至: {output_npy_path}")
# --- 主逻辑结束 ---