import os # 导入os模块，用于与操作系统交互，如路径操作
import math # 导入math模块，用于数学运算，如atan2, asin
import copy # 导入copy模块，用于对象的复制
import argparse # 导入argparse模块，用于解析命令行参数
from os import path as osp # 从os.path导入path，并重命名为osp，方便路径操作
from collections import OrderedDict # 导入OrderedDict，有序字典
from typing import List, Tuple, Union # 导入类型提示相关的模块

import numpy as np # 导入NumPy库，用于数值计算
from pyquaternion import Quaternion # 导入pyquaternion库，用于处理四元数和旋转
from shapely.geometry import MultiPoint, box # 从shapely库导入几何对象处理（当前代码中未直接使用box和MultiPoint，但可能被依赖项使用）

import mmcv # 导入MMCV库，OpenMMLab计算机视觉基础库

from nuscenes.nuscenes import NuScenes # 从nuscenes-devkit导入NuScenes主类
from nuscenes.can_bus.can_bus_api import NuScenesCanBus # 从nuscenes-devkit导入CAN总线API类
from nuscenes.utils.geometry_utils import transform_matrix # 从nuscenes-devkit导入生成变换矩阵的函数
from nuscenes.utils.data_classes import Box as NuScenesBox # 从nuscenes-devkit导入Box类 (这里重命名为NuScenesBox以区分shapely.geometry.box)
from nuscenes.utils.geometry_utils import view_points # 从nuscenes-devkit导入将点从一个坐标系投影到另一个坐标系的函数
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global # 从nuscenes-devkit的预测模块导入帮助类和坐标转换函数

from projects.mmdet3d_plugin.datasets.map_utils.nuscmap_extractor import NuscMapExtractor # 从项目中导入地图提取工具类

# NuScenes类别名称到自定义类别名称的映射字典
NameMapping = {
    "movable_object.barrier": "barrier", # 可移动障碍物 -> barrier
    "vehicle.bicycle": "bicycle", # 自行车
    "vehicle.bus.bendy": "bus", # 铰接式公交车 -> bus
    "vehicle.bus.rigid": "bus", # 普通公交车 -> bus
    "vehicle.car": "car", # 小汽车
    "vehicle.construction": "construction_vehicle", # 工程车辆
    "vehicle.motorcycle": "motorcycle", # 摩托车
    "human.pedestrian.adult": "pedestrian", # 成人行人 -> pedestrian
    "human.pedestrian.child": "pedestrian", # 儿童行人 -> pedestrian
    "human.pedestrian.construction_worker": "pedestrian", # 建筑工人 -> pedestrian
    "human.pedestrian.police_officer": "pedestrian", # 警察 -> pedestrian
    "movable_object.trafficcone": "traffic_cone", # 交通锥
    "vehicle.trailer": "trailer", # 拖车
    "vehicle.truck": "truck", # 卡车
}

def quart_to_rpy(qua): # 将四元数转换为翻滚角(roll)、俯仰角(pitch)、偏航角(yaw)
    """
    将四元数表示的旋转转换为欧拉角（翻滚、俯仰、偏航）。
    Args:
        qua (tuple or list): 包含(x, y, z, w)的四元数。
    Returns:
        tuple: (roll, pitch, yaw) 欧拉角（弧度）。
    """
    x, y, z, w = qua # 解包四元数
    # 计算翻滚角 (绕x轴旋转)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    # 计算俯仰角 (绕y轴旋转)
    pitch = math.asin(2 * (w * y - x * z))
    # 计算偏航角 (绕z轴旋转)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def locate_message(utimes, utime): # 在有序的时间戳序列中定位最接近给定时间戳的消息索引
    """
    在已排序的时间戳数组 `utimes` 中找到与给定时间戳 `utime` 最接近的元素的索引。
    Args:
        utimes (np.ndarray): 已排序的时间戳数组 (微秒)。
        utime (int): 要查找的单个时间戳 (微秒)。
    Returns:
        int: 最接近 `utime` 的时间戳在 `utimes` 中的索引。
    """
    # searchsorted返回utime可以插入utimes以保持顺序的索引i
    # 这意味着utimes[i-1] < utime <= utimes[i]
    i = np.searchsorted(utimes, utime)
    if i == len(utimes): # 如果utime大于所有utimes中的时间戳
        i -= 1 # 选择最后一个元素
    elif i > 0 and utime - utimes[i-1] < utimes[i] - utime: # 如果utime更接近前一个时间戳
        i -= 1 # 选择前一个元素的索引
    return i

def geom2anno(map_geoms): # 将NuscMapExtractor提取的几何对象转换为标注格式
    """
    将NuscMapExtractor输出的几何对象字典转换为更适合存储的标注格式。
    主要提取特定类别的线段坐标。
    Args:
        map_geoms (dict):键为地图层名称，值为几何对象列表的字典。
    Returns:
        dict: 键为类别索引，值为包含线段坐标numpy数组的列表。
    """
    MAP_CLASSES = ( # 定义我们关心的地图元素类别
        'ped_crossing',  # 人行横道
        'divider',    # 车道分割线
        'boundary',   # 道路边界线
    )
    vectors = {} # 初始化结果字典
    for cls, geom_list in map_geoms.items(): # 遍历每个地图层及其几何对象列表
        if cls in MAP_CLASSES: # 如果是我们关心的类别
            label = MAP_CLASSES.index(cls) # 获取类别索引
            vectors[label] = [] # 为该类别初始化一个空列表
            for geom in geom_list: # 遍历该类别的每个几何对象 (通常是LineString)
                line = np.array(geom.coords) # 提取线段的坐标点
                vectors[label].append(line) # 添加到结果字典
    return vectors

def create_nuscenes_infos(root_path, # 创建NuScenes数据集信息文件的主函数
                          out_path, # 输出信息文件的路径
                          can_bus_root_path, # CAN总线数据根路径
                          info_prefix, # 信息文件的前缀名
                          version='v1.0-trainval', # NuScenes数据集版本
                          max_sweeps=10, # 每个关键帧最多包含的激光雷达扫描（sweep）数量
                          roi_size=(30, 60),): # 地图ROI大小 (宽度, 高度)，米
    """Create info file of nuscene dataset. # 创建NuScenes数据集的信息文件。

    Given the raw data, generate its related info file in pkl format. # 给定原始数据，生成相关的pkl格式信息文件。

    Args:
        root_path (str): Path of the data root. # 数据根目录的路径。
        info_prefix (str): Prefix of the info file to be generated. # 要生成的信息文件的前缀。
        version (str): Version of the data. # 数据版本。
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps. # 最大扫描次数。
            Default: 10
        can_bus_root_path (str): Path to CAN bus data. # CAN总线数据的路径。
        out_path (str): Directory to save the output .pkl file. # 保存输出.pkl文件的目录。
        roi_size (tuple): Size of the ROI for map extraction. # 地图提取的ROI大小。
    """
    print(f"处理 NuScenes 版本: {version}, 数据根目录: {root_path}")
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True) # 初始化NuScenes API实例
    nusc_map_extractor = NuscMapExtractor(root_path, roi_size) # 初始化地图提取器
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path) # 初始化CAN总线API实例

    from nuscenes.utils import splits # 导入NuScenes的预定义场景分割信息
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini'] # 支持的数据集版本
    assert version in available_vers # 确保所选版本受支持

    if version == 'v1.0-trainval':
        train_scenes = splits.train # 训练场景名称列表
        val_scenes = splits.val     # 验证场景名称列表
    elif version == 'v1.0-test':
        train_scenes = splits.test # 测试场景名称列表 (注意：测试集没有GT标注)
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train # Mini数据集的训练场景
        val_scenes = splits.mini_val   # Mini数据集的验证场景
        out_path = osp.join(out_path, 'mini') # Mini数据集的输出路径通常在子目录中
    else:
        raise ValueError(f'未知的NuScenes版本: {version}')

    os.makedirs(out_path, exist_ok=True) # 创建输出目录 (如果不存在)

    # 过滤掉数据路径中实际不存在的场景
    available_scenes = get_available_scenes(nusc) # 获取实际可用的场景列表
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    # 将场景名称转换为场景token
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    if test: # 如果是测试集模式
        print(f'测试集场景数量: {len(train_scenes)}')
    else:
        print(f'训练集场景数量: {len(train_scenes)}, 验证集场景数量: {len(val_scenes)}')

    # 调用核心函数填充训练和验证信息
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_map_extractor, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps,
        fut_ts=12, ego_fut_ts=6 # fut_ts 和 ego_fut_ts 硬编码，可考虑作为参数传入
    )

    metadata = dict(version=version) # 元数据，记录数据集版本
    if test: # 如果是测试集
        print(f'测试集样本数量: {len(train_nusc_infos)}')
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             f'{info_prefix}_infos_test.pkl') # 测试集信息文件名
        mmcv.dump(data, info_path) # 保存为pkl文件
    else: # 如果是训练/验证集
        print(f'训练集样本数量: {len(train_nusc_infos)}, 验证集样本数量: {len(val_nusc_infos)}')
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             f'{info_prefix}_infos_train.pkl') # 训练集信息文件名
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos # 替换为验证集信息
        info_val_path = osp.join(out_path,
                                 f'{info_prefix}_infos_val.pkl') # 验证集信息文件名
        mmcv.dump(data, info_val_path)

def get_available_scenes(nusc): # 获取实际存在数据的场景列表
    """Get available scenes from the input nuscenes class. # 从输入的NuScenes类中获取可用场景。

    Given the raw data, get the information of available scenes for # 给定原始数据，获取可用场景的信息以供后续信息生成。
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset. # NuScenes数据集中的数据集类。

    Returns:
        available_scenes (list[dict]): List of basic information for the # 可用场景的基本信息列表。
            available scenes.
    """
    available_scenes = []
    print(f'总场景数: {len(nusc.scene)}')
    for scene in nusc.scene: # 遍历NuScenes对象中的所有场景记录
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token) # 获取场景记录
        sample_rec = nusc.get('sample', scene_rec['first_sample_token']) # 获取场景的第一个样本记录
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP']) # 获取第一个样本的LIDAR_TOP传感器数据记录

        # 检查该场景的第一个LIDAR_TOP数据文件是否存在，以此判断该场景是否有效
        has_more_frames = True # (此变量名在此处略有误导，实际是检查当前sd_rec的文件是否存在)
        scene_not_exist = False
        # 这个while循环实际上只检查第一个LIDAR_TOP文件
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token']) # 获取激光雷达数据路径和关联的boxes
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path: # 如果路径是绝对路径且包含当前工作目录
                # 将其转换为相对于当前工作目录的相对路径
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not mmcv.is_filepath(lidar_path): # 检查转换后的路径是否是有效的文件路径格式
                scene_not_exist = True # 如果路径无效（例如文件不存在或格式错误），标记场景不存在
                break
            else: # 如果路径有效，则跳出循环 (因为只需要检查一个文件)
                break # Bug? 如果文件存在，这里应该检查文件实际是否存在 os.path.exists(lidar_path)
                      # 当前逻辑是只要路径字符串有效就认为场景存在。
                      # mmcv.is_filepath 只检查字符串是否像一个路径，不检查文件存在性。
                      # 正确的检查应该是 if not osp.exists(lidar_path): scene_not_exist = True
        if scene_not_exist: # 如果场景被标记为不存在
            continue # 跳过此场景
        available_scenes.append(scene) # 将有效场景添加到列表
    print(f'实际存在数据的场景数: {len(available_scenes)}')
    return available_scenes

def _fill_trainval_infos(nusc, # 核心函数，用于填充每个样本的详细信息
                         nusc_map_extractor, # 地图提取器实例
                         nusc_can_bus,       # CAN总线API实例
                         train_scenes,       # 训练场景token集合
                         val_scenes,         # 验证场景token集合
                         test=False,         # 是否为测试模式 (无标注)
                         max_sweeps=10,      # 最大激光雷达扫描数量
                         fut_ts=12,          # agent未来轨迹时间步
                         ego_fut_ts=6):      # ego未来轨迹时间步
    """Generate the train/val infos from the raw data. # 从原始数据生成训练/验证信息。 """
    train_nusc_infos = [] # 初始化训练信息列表
    val_nusc_infos = []   # 初始化验证信息列表

    # 创建类别名称到类别索引的映射
    cat2idx = {cat['name']: idx for idx, cat in enumerate(nusc.category)}

    predict_helper = PredictHelper(nusc) # NuScenes预测辅助工具，用于获取未来轨迹

    for sample in mmcv.track_iter_progress(nusc.sample): # 遍历NuScenes中的每个样本(关键帧)
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location'] # 获取样本所在地的地图位置名称
        lidar_token = sample['data']['LIDAR_TOP'] # 获取LIDAR_TOP传感器的token
        sd_rec = nusc.get('sample_data', lidar_token) # 获取LIDAR_TOP的样本数据记录
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token']) # 获取激光雷达的标定传感器记录
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token']) # 获取激光雷达数据对应的自车位姿记录
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token) # 获取激光雷达数据路径和当前帧的3D标注框
        mmcv.check_file_exist(lidar_path) # 确保激光雷达文件存在

        info = { # 初始化当前样本的信息字典
            'lidar_path': lidar_path, # 激光雷达点云文件路径
            'token': sample['token'], # 样本token
            'sweeps': [], # 存储历史扫描帧的信息列表
            'cams': dict(), # 存储相机相关信息的字典
            'scene_token': sample['scene_token'], # 场景token
            'lidar2ego_translation': cs_record['translation'], # 激光雷达到自车坐标系的平移
            'lidar2ego_rotation': cs_record['rotation'],    # 激光雷达到自车坐标系的旋转 (四元数)
            'ego2global_translation': pose_record['translation'], # 自车到全局坐标系的平移
            'ego2global_rotation': pose_record['rotation'],    # 自车到全局坐标系的旋转 (四元数)
            'timestamp': sample['timestamp'], # 样本时间戳 (微秒)
            'map_location': map_location, # 地图位置名称
        }

        # 获取激光雷达到自车(l2e)和自车到全局(e2g)的变换矩阵的旋转和平移分量
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix # 四元数转旋转矩阵
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # 提取地图矢量化标注
        lidar2ego = np.eye(4) # 构建激光雷达到自车的4x4变换矩阵
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4) # 构建自车到全局的4x4变换矩阵
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        lidar2global = ego2global @ lidar2ego # 计算激光雷达到全局的变换矩阵

        # 获取激光雷达在全局坐标系下的位姿，用于提取局部地图
        translation_global_lidar = list(lidar2global[:3, 3])
        rotation_global_lidar_quat = list(Quaternion(matrix=lidar2global).elements) # 使用elements确保w,x,y,z顺序
        # 提取以当前激光雷达位置为中心的ROI区域内的地图几何元素
        map_geoms = nusc_map_extractor.get_map_geom(map_location, translation_global_lidar, rotation_global_lidar_quat)
        map_annos = geom2anno(map_geoms) # 将几何对象转换为自定义的标注格式
        info['map_annos'] = map_annos # 存入信息字典

        # 获取6个相机的图像信息及变换矩阵
        camera_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
        ]
        for cam in camera_types: # 遍历每个相机类型
            cam_token = sample['data'][cam] # 获取相机数据token
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token) # 获取相机图像路径和内参
            # 计算从当前相机到LIDAR_TOP的变换信息
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic) # 添加相机内参到相机信息中
            info['cams'].update({cam: cam_info}) # 更新到总信息字典

        # 获取历史激光雷达扫描帧（sweeps）的信息
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP']) # 当前激光雷达数据记录
        sweeps = []
        while len(sweeps) < max_sweeps: # 最多获取max_sweeps帧历史扫描
            if not sd_rec['prev'] == '': # 如果存在前一帧
                # 计算从前一帧传感器到当前LIDAR_TOP的变换信息
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep) # 添加到sweeps列表
                sd_rec = nusc.get('sample_data', sd_rec['prev']) # 更新sd_rec到更早一帧
            else: # 没有更早的帧了
                break
        info['sweeps'] = sweeps # 存入信息字典

        # 获取标注信息 (如果不是测试模式)
        if not test:
            # 目标检测标注: boxes (位置, 尺寸, 偏航角, 速度), 类别名称, 有效性标志
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']] # 获取当前样本的所有标注记录

            # locs, dims, rots 直接从NuScenes的Box对象获取 (已经转换到LIDAR_TOP坐标系)
            locs = np.array([b.center for b in boxes]).reshape(-1, 3) # 中心点 (N, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)   # 尺寸 (N, 3)，顺序是w,l,h
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1) # 偏航角 (N, 1)

            # 获取速度 (全局坐标系下的2D速度)
            velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']]) # (N, 2)
            # 将速度从全局坐标系转换到当前激光雷达坐标系
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0]) # 扩展到3D速度 (z方向速度为0)
                # 全局速度 -> 当前自车坐标系速度 -> 当前激光雷达坐标系速度
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2] # 取转换后的x,y方向速度

            names = [b.name for b in boxes] # 获取原始类别名称
            for i in range(len(names)): # 将原始类别名称映射到自定义的类别名称
                if names[i] in NameMapping:
                    names[i] = NameMapping[names[i]]
            names = np.array(names)

            # 有效性标志：如果标注框内激光雷达点或雷达点数量大于0，则认为有效
            valid_flag = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                                   for anno in annotations], dtype=bool).reshape(-1)

            # 将边界框格式转换为 (x,y,z,l,w,h,rot) - 注意尺寸顺序从w,l,h变为l,w,h
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(annotations), f'{len(gt_boxes)}, {len(annotations)}' # 确保转换后数量一致
            
            # 目标跟踪标注: 实例ID
            instance_inds = [nusc.getind('instance', anno['instance_token']) for anno in annotations]

            # 运动预测标注: agent未来轨迹（相对于当前agent位置的偏移量）和有效性掩码
            num_box = len(boxes)
            gt_agent_fut_trajs = np.zeros((num_box, fut_ts, 2)) # (N, T_fut, 2)
            gt_agent_fut_masks = np.zeros((num_box, fut_ts))    # (N, T_fut)
            for i, anno in enumerate(annotations): # 遍历每个标注的agent
                instance_token = anno['instance_token']
                # 使用NuScenes的PredictHelper获取未来轨迹 (在agent的局部坐标系下，x轴指向前方)
                fut_traj_local = predict_helper.get_future_for_agent(
                    instance_token, sample['token'], seconds=fut_ts/2, in_agent_frame=True # seconds=fut_ts/2 (假设采样率为2Hz)
                )
                if fut_traj_local.shape[0] > 0: # 如果存在未来轨迹
                    box = boxes[i] # 当前agent的box对象
                    # 将局部轨迹转换到场景全局坐标系
                    fut_traj_scene = convert_local_coords_to_global(fut_traj_local, box.center, Quaternion(matrix=box.rotation_matrix))
                    valid_step = fut_traj_scene.shape[0] # 有效的未来时间步数量
                    # 计算相对于当前位置的第一个未来时间步的偏移
                    gt_agent_fut_trajs[i, 0] = fut_traj_scene[0] - box.center[:2]
                    # 计算后续时间步之间的相对偏移
                    if valid_step > 1 : gt_agent_fut_trajs[i, 1:valid_step] = fut_traj_scene[1:] - fut_traj_scene[:-1]
                    gt_agent_fut_masks[i, :valid_step] = 1 # 设置有效性掩码

            # 运动规划标注: 自车未来轨迹（相对于当前自车位置的偏移量），有效性掩码，驾驶命令
            ego_fut_trajs_abs_global = np.zeros((ego_fut_ts + 1, 3)) # 存储未来绝对全局坐标 (x,y,z)
            ego_fut_masks = np.zeros((ego_fut_ts + 1))
            sample_cur = sample # 当前样本
            current_ego_pose_glob = get_global_sensor_pose(sample_cur, nusc) # 获取当前自车在全局坐标系下的位姿

            for i in range(ego_fut_ts + 1): # 遍历未来时间步 (包括当前时刻 t=0)
                pose_mat_at_t = get_global_sensor_pose(sample_cur, nusc) # 获取在t时刻的自车全局位姿
                ego_fut_trajs_abs_global[i] = pose_mat_at_t[:3, 3] # 提取全局平移 (x,y,z)
                ego_fut_masks[i] = 1 # 标记为有效
                if sample_cur['next'] == '': # 如果没有下一帧了
                    ego_fut_trajs_abs_global[i+1:] = ego_fut_trajs_abs_global[i] # 用当前位置填充剩余的未来轨迹
                    ego_fut_masks[i+1:] = 0 # 标记为无效
                    break
                else:
                    sample_cur = nusc.get('sample', sample_cur['next']) # 移动到下一帧

            # 将全局绝对轨迹转换为相对于当前LIDAR_TOP坐标系的相对偏移轨迹
            # 1. 全局 -> 当前自车坐标系
            ego_fut_trajs_rel_ego = ego_fut_trajs_abs_global - np.array(pose_record['translation']) # 减去当前自车全局平移
            rot_mat_glob_to_ego = Quaternion(pose_record['rotation']).inverse.rotation_matrix # 全局到自车的旋转
            ego_fut_trajs_rel_ego = np.dot(rot_mat_glob_to_ego, ego_fut_trajs_rel_ego.T).T
            # 2. 当前自车 -> 当前激光雷达坐标系
            ego_fut_trajs_rel_lidar = ego_fut_trajs_rel_ego - np.array(cs_record['translation']) # 减去激光雷达相对自车的平移
            rot_mat_ego_to_lidar = Quaternion(cs_record['rotation']).inverse.rotation_matrix # 自车到激光雷达的旋转
            ego_fut_trajs_rel_lidar = np.dot(rot_mat_ego_to_lidar, ego_fut_trajs_rel_lidar.T).T

            # 根据最终未来位置判断驾驶命令 (右转、左转、直行)
            # ego_fut_trajs_rel_lidar[-1] 是最后一个未来时间步在当前激光雷达坐标系下的位置
            if ego_fut_trajs_rel_lidar[-1][0] >= 2: # x方向位移大于2米 (假设x轴向前，y轴向左) -> 右转
                command = np.array([1, 0, 0])  # Turn Right
            elif ego_fut_trajs_rel_lidar[-1][0] <= -2: # x方向位移小于-2米 -> 左转
                command = np.array([0, 1, 0])  # Turn Left
            else: # 否则认为是直行
                command = np.array([0, 0, 1])  # Go Straight

            # 计算相邻时间步之间的相对偏移量 (作为回归目标)
            ego_fut_trajs_offsets = ego_fut_trajs_rel_lidar[1:] - ego_fut_trajs_rel_lidar[:-1]

            # 获取自车状态 (速度，加速度等)
            ego_status = get_ego_status(nusc, nusc_can_bus, sample)

            # 存入info字典
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag
            info['instance_inds'] = instance_inds
            info['gt_agent_fut_trajs'] = gt_agent_fut_trajs.astype(np.float32)
            info['gt_agent_fut_masks'] = gt_agent_fut_masks.astype(np.float32)
            info['gt_ego_fut_trajs'] = ego_fut_trajs_offsets[:, :2].astype(np.float32) # 只取x,y方向的偏移
            info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32) # 掩码也对应偏移量
            info['gt_ego_fut_cmd'] = command.astype(np.float32) # 驾驶命令
            info['ego_status'] = ego_status # 自车状态

        if sample['scene_token'] in train_scenes: # 如果场景属于训练集
            train_nusc_infos.append(info) # 添加到训练信息列表
        else: # 否则属于验证集
            val_nusc_infos.append(info) # 添加到验证信息列表

    return train_nusc_infos, val_nusc_infos # 返回填充好的训练和验证信息列表

def get_ego_status(nusc, nusc_can_bus, sample): # 从CAN总线获取自车状态信息
    """
    获取给定样本时间戳下自车的状态信息（加速度、角速度、速度、方向盘转角）。
    """
    ego_status = []
    ref_scene = nusc.get("scene", sample['scene_token']) # 获取场景记录
    try:
        # 获取位姿和方向盘角度的CAN消息
        pose_msgs = nusc_can_bus.get_messages(ref_scene['name'],'pose')
        steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
        # 获取消息的时间戳列表
        pose_uts = [msg['utime'] for msg in pose_msgs]
        steer_uts = [msg['utime'] for msg in steer_msgs]
        ref_utime = sample['timestamp'] # 当前样本的时间戳

        # 找到最接近当前样本时间戳的CAN消息
        pose_index = locate_message(pose_uts, ref_utime)
        pose_data = pose_msgs[pose_index]
        steer_index = locate_message(steer_uts, ref_utime)
        steer_data = steer_msgs[steer_index]

        # 提取状态信息
        ego_status.extend(pose_data["accel"]) # 加速度 (ego系, m/s^2)
        ego_status.extend(pose_data["rotation_rate"]) # 角速度 (ego系, rad/s)
        ego_status.extend(pose_data["vel"]) # 速度 (ego系, m/s)
        ego_status.append(steer_data["value"]) # 方向盘转角 (正为左转, 负为右转)
    except Exception as e: # 如果获取CAN消息失败 (例如某些场景可能缺失CAN数据)
        # print(f"Warning: Failed to get CAN bus data for sample {sample['token']}: {e}")
        ego_status = [0.0] * 10 # 使用默认值0填充
    
    return np.array(ego_status).astype(np.float32)

def get_global_sensor_pose(rec, nusc): # 获取传感器在全局坐标系下的位姿
    """
    获取给定样本记录中LIDAR_TOP传感器在全局坐标系下的4x4位姿矩阵。
    """
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP']) # 获取激光雷达数据记录

    pose_record = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"]) # 获取自车位姿记录
    cs_record = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"]) # 获取传感器标定记录

    # 计算自车到全局的变换矩阵
    ego2global = transform_matrix(pose_record["translation"], Quaternion(pose_record["rotation"]), inverse=False)
    # 计算传感器到自车的变换矩阵
    sensor2ego = transform_matrix(cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=False)
    # 传感器到全局的位姿 = 自车到全局 @ 传感器到自车
    pose = ego2global @ sensor2ego

    return pose

def obtain_sensor2top(nusc, # 计算任意传感器到LIDAR_TOP传感器的相对位姿变换
                      sensor_token, # 源传感器的token
                      l2e_t,        # LIDAR_TOP到当前自车坐标系的平移
                      l2e_r_mat,    # LIDAR_TOP到当前自车坐标系的旋转矩阵
                      e2g_t,        # 当前自车到全局坐标系的平移
                      e2g_r_mat,    # 当前自车到全局坐标系的旋转矩阵
                      sensor_type='lidar'): # 源传感器的类型 ('lidar' 或 'camera')
    """Obtain the info with RT matric from general sensor to Top LiDAR. # 获取从通用传感器到顶部激光雷达的带RT矩阵的信息。

    Args:
        nusc: NuScenes API实例。
        sensor_token (str): 源传感器的样本数据token。
        l2e_t (np.ndarray): 当前关键帧LIDAR_TOP到其自车坐标系的平移 (1x3)。
        l2e_r_mat (np.ndarray): 当前关键帧LIDAR_TOP到其自车坐标系的旋转矩阵 (3x3)。
        e2g_t (np.ndarray): 当前关键帧自车到全局坐标系的平移 (1x3)。
        e2g_r_mat (np.ndarray): 当前关键帧自车到全局坐标系的旋转矩阵 (3x3)。
        sensor_type (str): 源传感器的类型。默认为 'lidar'。

    Returns:
        sweep (dict): 包含源传感器信息及其到当前关键帧LIDAR_TOP的变换的字典。
                      关键的键是 'sensor2lidar_rotation' 和 'sensor2lidar_translation'。
    """
    sd_rec = nusc.get('sample_data', sensor_token) # 获取源传感器的样本数据记录
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token']) # 源传感器的标定记录
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token']) # 源传感器对应时刻的自车位姿记录
    data_path = str(nusc.get_sample_data_path(sd_rec['token'])) # 获取源传感器数据路径
    if os.getcwd() in data_path:  # 将绝对路径转换为相对路径 (如果需要)
        data_path = data_path.split(f'{os.getcwd()}/')[-1]

    sweep = { # 初始化sweep信息字典
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'], # 源传感器到其自车坐标系的平移
        'sensor2ego_rotation': cs_record['rotation'],       # 源传感器到其自车坐标系的旋转
        'ego2global_translation': pose_record['translation'], # 源传感器时刻自车到全局的平移
        'ego2global_rotation': pose_record['rotation'],       # 源传感器时刻自车到全局的旋转
        'timestamp': sd_rec['timestamp'] # 源传感器的时间戳
    }

    # 源传感器时刻的变换参数
    l2e_r_s = sweep['sensor2ego_rotation'] # sensor -> ego_s
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']   # ego_s -> global
    e2g_t_s = sweep['ego2global_translation']

    # 计算从源传感器(sensor_s)到当前关键帧LIDAR_TOP的变换
    # 变换链: sensor_s -> ego_s -> global -> ego_cur -> lidar_cur
    # T_s_to_lc = T_ec_to_lc @ T_g_to_ec @ T_es_to_g @ T_s_to_es

    # sensor_s 到 ego_s 的旋转矩阵
    s_to_es_r_mat = Quaternion(l2e_r_s).rotation_matrix
    # ego_s 到 global 的旋转矩阵
    es_to_g_r_mat = Quaternion(e2g_r_s).rotation_matrix

    # 综合旋转部分: R = R_lc_ec @ R_ec_g @ R_g_es @ R_es_s
    # R = (inv(l2e_r_mat) @ inv(e2g_r_mat)) @ (es_to_g_r_mat @ s_to_es_r_mat)
    # R = inv(e2g_r_mat @ l2e_r_mat) @ (es_to_g_r_mat @ s_to_es_r_mat)
    # T_s_to_lc = inv(T_lc_g) @ T_s_g
    # T_s_g = T_es_g @ T_s_es
    # T_lc_g = T_ec_g @ T_lc_ec

    # 从源传感器(s)到全局(g)的变换矩阵 T_s_g
    T_s_es = transform_matrix(l2e_t_s, Quaternion(l2e_r_s), inverse=False)
    T_es_g = transform_matrix(e2g_t_s, Quaternion(e2g_r_s), inverse=False)
    T_s_g = T_es_g @ T_s_es

    # 从当前激光雷达(lc)到全局(g)的变换矩阵 T_lc_g
    T_lc_ec = transform_matrix(l2e_t, Quaternion(l2e_r), inverse=False)
    T_ec_g = transform_matrix(e2g_t, Quaternion(e2g_r), inverse=False)
    T_lc_g = T_ec_g @ T_lc_ec

    # 从源传感器(s)到当前激光雷达(lc)的变换矩阵 T_s_to_lc
    T_s_to_lc = np.linalg.inv(T_lc_g) @ T_s_g

    # 提取旋转和平移
    sweep['sensor2lidar_rotation'] = Quaternion(matrix=T_s_to_lc[:3,:3]).elements # w,x,y,z
    sweep['sensor2lidar_translation'] = list(T_s_to_lc[:3,3])
    return sweep

def nuscenes_data_prep(root_path, # NuScenes数据准备的总入口函数
                       can_bus_root_path, # CAN总线数据根路径
                       info_prefix, # 输出信息文件的前缀
                       version,     # 数据集版本 (例如 'v1.0-trainval', 'v1.0-test', 'v1.0-mini')
                       dataset_name, # 数据集名称 (未使用在此函数中)
                       out_dir,      # 输出目录
                       max_sweeps=10): # 最大扫描次数
    """Prepare data related to nuScenes dataset. # 准备与NuScenes数据集相关的数据。

    Related data consists of '.pkl' files recording basic infos, # 相关数据包括记录基本信息、
    2D annotations and groundtruth database. # 2D标注和真值数据库的'.pkl'文件。
                                             # (此函数主要生成基本信息和标注)
    Args:
        root_path (str): Path of dataset root. # 数据集根目录的路径。
        info_prefix (str): The prefix of info filenames. # 信息文件名的前缀。
        version (str): Dataset version. # 数据集版本。
        dataset_name (str): The dataset class name. # 数据集类名。
        out_dir (str): Output directory of the groundtruth database info. # 真值数据库信息的输出目录。
        max_sweeps (int): Number of input consecutive frames. Default: 10 # 输入连续帧的数量。默认为10。
        can_bus_root_path (str): Path to CAN bus data. # CAN总线数据的路径。
    """
    # 调用create_nuscenes_infos函数来实际生成和保存信息文件
    create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)


# 设置命令行参数解析器
parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='nuscenes', help='name of the dataset (e.g., nuscenes)') # 数据集名称参数
parser.add_argument(
    '--root-path', # 数据集根路径参数
    type=str,
    default='./data/nuscenes', # 默认路径
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus', # CAN总线数据路径参数
    type=str,
    default='./data', # 默认路径 (通常NuScenes CAN数据在主数据目录下)
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version', # 数据集版本参数
    type=str,
    default='v1.0-trainval', # 默认版本 (注意：脚本中可能会分别处理trainval和test)
    required=False, # 非必需参数
    help='specify the dataset version')
parser.add_argument(
    '--max-sweeps', # 最大激光雷达扫描数参数
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir', # 输出目录参数
    type=str,
    default='./data/nuscenes', # 默认输出目录 (通常与root-path一致或其子目录)
    help='name of info pkl') # (帮助信息不准确，应为directory to save info pkl)
parser.add_argument('--extra-tag', type=str, default='nuscenes') # 输出文件名的额外标签/前缀
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used (not used in this script)') # 工作线程数 (当前脚本中未使用多线程)
args = parser.parse_args() # 解析参数

if __name__ == '__main__': # 如果作为主脚本运行
    if args.dataset == 'nuscenes': # 如果处理的是NuScenes数据集
        if args.version != 'v1.0-mini': # 如果不是mini版本 (即v1.0-trainval或v1.0-test)
            # 分别处理trainval和test数据子集
            train_version = f'{args.version}-trainval' # 拼接训练验证集版本号
            print(f"Preparing {train_version} data...")
            nuscenes_data_prep(
                root_path=args.root_path,
                can_bus_root_path=args.canbus,
                info_prefix=args.extra_tag,
                version=train_version,
                dataset_name='NuScenesDataset', # 传递数据集名称
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps)

            test_version = f'{args.version}-test' # 拼接测试集版本号
            print(f"Preparing {test_version} data...")
            nuscenes_data_prep(
                root_path=args.root_path,
                can_bus_root_path=args.canbus,
                info_prefix=args.extra_tag,
                version=test_version,
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps)
        elif args.version == 'v1.0-mini': # 如果是mini版本
            train_version = f'{args.version}' # mini版本通常不区分trainval和test子集进行数据准备
            print(f"Preparing {train_version} data...")
            nuscenes_data_prep(
                root_path=args.root_path,
                can_bus_root_path=args.canbus,
                info_prefix=args.extra_tag,
                version=train_version, # 使用 'v1.0-mini'
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir, # 输出到指定目录 (可能会在create_nuscenes_infos中附加'mini')
                max_sweeps=args.max_sweeps)
    else:
        print(f"Dataset {args.dataset} is not supported in this script.")
