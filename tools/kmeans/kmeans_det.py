import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

os.makedirs('data/kmeans', exist_ok=True)
os.makedirs('vis/kmeans', exist_ok=True)

# K = 900 # 定义聚类的簇数量 (即生成的锚点/查询点的数量)
# DIS_THRESH = 55 # 距离阈值 (米)，用于筛选有效的GT边界框中心点

# --- 主逻辑开始 ---
# 指定NuScenes数据集信息文件的路径 (通常是训练集)
fp = 'data/infos/nuscenes_infos_train.pkl'
data = mmcv.load(fp) # 使用mmcv加载.pkl文件中的数据
# 从加载的数据中提取'infos'字段，并按时间戳排序
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

center_coords_list = [] # 初始化列表，用于存储所有符合条件的GT边界框中心点坐标
print("从数据集中提取GT边界框中心点...")
for idx in tqdm(range(len(data_infos))): # 使用tqdm显示处理进度，遍历所有数据样本
    # 获取当前样本的GT边界框的前3列 (x, y, z中心点坐标)
    # gt_boxes 形状通常是 (num_boxes, 7+)，包含 x,y,z,w,l,h,yaw,...
    current_boxes_centers = data_infos[idx]['gt_boxes'][:,:3]
    if len(current_boxes_centers) == 0: # 如果当前样本没有GT框，则跳过
        continue
    # 计算每个GT框中心点在BEV平面上到原点(自车位置)的距离
    distance_bev = np.linalg.norm(current_boxes_centers[:, :2], axis=1) # 只考虑x,y坐标计算距离
    # 只保留距离小于DIS_THRESH的框的中心点
    center_coords_list.append(current_boxes_centers[distance_bev < DIS_THRESH])

# 将所有样本的有效中心点坐标合并成一个大的NumPy数组
all_center_coords = np.concatenate(center_coords_list, axis=0)
print(f"提取了 {all_center_coords.shape[0]} 个有效中心点。")
print(f"开始对这 {all_center_coords.shape[0]} 个3D中心点进行K-Means聚类 (K={K})，这可能需要几分钟...")

# 执行K-Means聚类
# n_clusters: 要形成的簇的数量 (K)
# .fit(all_center_coords): 用3D中心点数据进行聚类
# .cluster_centers_: 获取聚类后每个簇的中心点坐标 (K, 3)
k_means_model = KMeans(n_clusters=K, random_state=0, n_init='auto') # random_state保证结果可复现, n_init='auto'适应sklearn版本变化
generated_cluster_centers = k_means_model.fit(all_center_coords).cluster_centers_
print(f"聚类完成，生成了 {generated_cluster_centers.shape[0]} 个3D簇中心。")

# 可视化聚类中心的BEV分布 (x, y 坐标)
plt.figure(figsize=(10,10)) # 创建一个新的图像
plt.scatter(generated_cluster_centers[:,0], generated_cluster_centers[:,1], s=5) # s是点的大小
plt.title(f'K-Means Detection Anchor Locations (BEV projection, K={K})') # 设置标题
plt.xlabel('X coordinate (m)') # X轴标签
plt.ylabel('Y coordinate (m)') # Y轴标签
plt.grid(True) # 显示网格
plt.axis('equal') # 确保x,y轴等比例
output_visualization_path = f'vis/kmeans/det_anchor_bev_{K}.png' # 定义可视化图片的保存路径
plt.savefig(output_visualization_path, bbox_inches='tight') # 保存图像，bbox_inches='tight'确保内容不被裁剪
print(f"聚类中心BEV分布图已保存至: {output_visualization_path}")
plt.close() # 关闭当前图像，释放资源

# 为每个3D聚类中心附加其他默认参数，以形成完整的锚点/查询参数。
# 这些附加参数通常代表锚点的默认尺寸、方向和速度。
# 这里的8维常量 [1.,1.,1., 0.,1., 0.,0.,0.] 解释为：
# W=1.0, L=1.0, H=1.0 (默认尺寸)
# sin(yaw)=0.0, cos(yaw)=1.0 (表示yaw=0，即锚点方向与x轴正方向一致)
# vx=0.0, vy=0.0, vz=0.0 (默认速度为0)
# 最终形成的锚点参数顺序为: [cluster_x, cluster_y, cluster_z, W, L, H, sin(yaw), cos(yaw), vx, vy, vz] (11维)
# 注意：如果模型期望的是log(W),log(L),log(H)，则这里的1.0,1.0,1.0应对应log(desired_W),log(desired_L),log(desired_H)
default_other_params = np.array([1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 0.0, 0.0], dtype=np.float32)
# 将默认参数扩展为 (K, 8) 的形状，以便与 (K, 3) 的聚类中心拼接
repeated_other_params = np.tile(default_other_params[np.newaxis, :], (K, 1))

# 将3D聚类中心 (K, 3) 和附加的其他参数 (K, 8) 拼接起来
final_anchors = np.concatenate([generated_cluster_centers, repeated_other_params], axis=1) # (K, 11)
print(f"最终生成的锚点参数形状: {final_anchors.shape}")

# 保存生成的锚点/查询点到.npy文件
output_npy_path = f'data/kmeans/kmeans_det_{K}.npy'
np.save(output_npy_path, final_anchors)
print(f"生成的检测锚点参数已保存至: {output_npy_path}")
# --- 主逻辑结束 ---