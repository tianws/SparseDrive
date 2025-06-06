import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

# K = 100 # 定义聚类的簇数量 (即生成的线状锚点/查询原型的数量)
# num_sample = 20 # 每个生成的线状锚点包含的采样点数量

# --- 主逻辑开始 ---
# 指定NuScenes数据集信息文件的路径 (通常是训练集)
fp = 'data/infos/nuscenes_infos_train.pkl'
data = mmcv.load(fp) # 使用mmcv加载.pkl文件中的数据
# 从加载的数据中提取'infos'字段，并按时间戳排序
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

center_points_list = [] # 初始化列表，用于存储所有地图元素的中心点坐标
print("从数据集中提取地图元素的中心点...")
for idx in tqdm(range(len(data_infos))): # 使用tqdm显示处理进度，遍历所有数据样本
    # data_infos[idx]["map_annos"] 是一个字典，键是类别索引，值是该类别下的几何对象列表
    # 每个几何对象 (geom_points_raw) 是一个numpy数组，表示组成该线段/多边形的点的坐标序列
    for cls_idx, geoms_list_for_cls in data_infos[idx]["map_annos"].items(): # 遍历每个类别及其几何对象列表
        for geom_points_raw in geoms_list_for_cls:  # geom_points_raw 是一个 (num_points_in_geom, 2 or 3) 的数组
            if len(geom_points_raw) > 0: # 确保几何对象至少有一个点
                # 计算几何对象所有点的平均值，作为其中心点，只取x,y坐标
                center_points_list.append(geom_points_raw.mean(axis=0)[:2])

all_center_points = np.stack(center_points_list, axis=0) # 将所有中心点坐标合并成一个大的NumPy数组 (N, 2)
print(f"提取了 {all_center_points.shape[0]} 个地图元素的中心点。")
print(f"开始对中心点进行K-Means聚类 (K={K})，这可能需要一些时间...")

# 对提取的地图元素中心点进行K-Means聚类，得到K个簇中心
# 这些簇中心可以看作是地图元素的典型位置或锚点位置的起始点
k_means_model_map = KMeans(n_clusters=K, random_state=0, n_init='auto') # random_state保证结果可复现
generated_cluster_centers_bev = k_means_model_map.fit(all_center_points).cluster_centers_ # (K, 2)
print(f"聚类完成，生成了 {generated_cluster_centers_bev.shape[0]} 个地图锚点基础位置。")

# 基于每个聚类中心（锚点基础位置），生成一个默认的垂直线段作为初始的线状锚点/原型
# delta_y 定义了线段上采样点相对于中心点的y方向偏移量 (在BEV下，通常y轴向前)
# delta_x 定义了x方向偏移量 (在BEV下，通常x轴向左或右)
# 这里生成的是以聚类中心为几何中心，沿Y轴对称分布的直线段
delta_y_coords = np.linspace(-4, 4, num_sample) # 在[-4, 4]米范围内均匀采样num_sample个y偏移值
delta_x_coords = np.zeros([num_sample]) # x方向偏移量全部为0，所以是垂直于x轴（即沿y轴）的线段
# delta_offsets: (num_sample, 2) 组合成(dx, dy)偏移序列
delta_offsets = np.stack([delta_x_coords, delta_y_coords], axis=-1)

# 将偏移量序列加到每个聚类中心上，生成K个线状锚点
# generated_cluster_centers_bev[:, np.newaxis, :] -> (K, 1, 2)
# delta_offsets[np.newaxis, :, :] -> (1, num_sample, 2)
# 通过广播机制，得到 prototype_lines: (K, num_sample, 2)，即K条线，每条线num_sample个点，每个点2个(x,y)坐标
prototype_lines = generated_cluster_centers_bev[:, np.newaxis, :] + delta_offsets[np.newaxis, :, :]
print(f"生成的线状锚点数组形状: {prototype_lines.shape}")

# 可视化生成的K个线状锚点
plt.figure(figsize=(10,10)) # 创建一个新的图像
for i in range(K): # 遍历每个锚点线
    if i < prototype_lines.shape[0]: # 确保索引有效 (如果聚类结果少于K)
        x_coords = prototype_lines[i, :, 0] # 当前线的所有x坐标
        y_coords = prototype_lines[i, :, 1] # 当前线的所有y坐标
        plt.plot(x_coords, y_coords, linewidth=1, marker='o', linestyle='-', markersize=2) # 绘制线段
plt.title(f'K-Means Map Anchor Lines (K={K}, num_sample={num_sample})') # 设置标题
plt.xlabel('X coordinate (m)') # X轴标签
plt.ylabel('Y coordinate (m)') # Y轴标签
plt.grid(True) # 显示网格
plt.axis('equal') # 确保x,y轴等比例显示
output_visualization_path = f'vis/kmeans/map_anchor_lines_{K}.png' # 定义可视化图片的保存路径
plt.savefig(output_visualization_path, bbox_inches='tight') # 保存图像
print(f"地图锚点线可视化图像已保存至: {output_visualization_path}")
plt.close() # 关闭当前图像，释放资源

# 保存生成的线状锚点到.npy文件
# 最终保存的prototype_lines是 (K, num_sample, 2) 的形状
output_npy_path = f'data/kmeans/kmeans_map_{K}.npy'
np.save(output_npy_path, prototype_lines)
print(f"生成的地图锚点线数据已保存至: {output_npy_path}")
# --- 主逻辑结束 ---