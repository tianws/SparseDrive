import random  # 导入 random 模块，用于生成随机数
import math  # 导入 math 模块，用于数学运算
import os  # 导入 os 模块，用于与操作系统交互
from os import path as osp  # 从 os.path 模块导入 path，并重命名为 osp，用于路径操作
import cv2  # 导入 OpenCV 库，用于图像处理
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import copy  # 导入 copy 模块，用于对象的浅拷贝和深拷贝
import prettytable  # 导入 prettytable 模块，用于创建格式化的表格输出

import numpy as np  # 导入 numpy 模块，用于科学计算
import torch  # 导入 torch 模块，PyTorch 深度学习框架
from torch.utils.data import Dataset  # 从 torch.utils.data 模块导入 Dataset 基类
import pyquaternion  # 导入 pyquaternion 模块，用于处理四元数
from shapely.geometry import LineString  # 从 shapely.geometry 模块导入 LineString，用于处理线几何对象
from nuscenes.utils.data_classes import Box as NuScenesBox  # 从 nuscenes.utils.data_classes 模块导入 Box 类，并重命名为 NuScenesBox
from nuscenes.eval.detection.config import config_factory as det_configs  # 从 nuscenes.eval.detection.config 导入检测评估配置工厂函数
from nuscenes.eval.common.config import config_factory as track_configs  # 从 nuscenes.eval.common.config 导入通用评估配置工厂函数 (用于跟踪)

import mmcv  # 导入 mmcv 模块，OpenMMLab 计算机视觉基础库
from mmcv.utils import print_log  # 从 mmcv.utils 模块导入 print_log 函数，用于打印日志
from mmdet.datasets import DATASETS  # 从 mmdet.datasets 模块导入 DATASETS 注册表
from mmdet.datasets.pipelines import Compose  # 从 mmdet.datasets.pipelines 模块导入 Compose 类，用于组合数据处理流程
from .utils import (  # 从当前目录的 utils 模块导入
    draw_lidar_bbox3d_on_img,  # 在图像上绘制激光雷达3D边界框的函数
    draw_lidar_bbox3d_on_bev,  # 在鸟瞰图上绘制激光雷达3D边界框的函数
)


@DATASETS.register_module()  # 将 NuScenes3DDataset 类注册到 DATASETS 注册表中
class NuScenes3DDataset(Dataset):  # 定义 NuScenes3DDataset 类，继承自 PyTorch 的 Dataset 类
    DefaultAttribute = {  # 默认属性映射，用于某些类别在特定情况下的属性
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    ErrNameMapping = {  # 错误名称到评估指标名称的映射
        "trans_err": "mATE",  # 平移误差 -> 平均平移误差
        "scale_err": "mASE",  # 尺度误差 -> 平均尺度误差
        "orient_err": "mAOE",  # 方向误差 -> 平均方向误差
        "vel_err": "mAVE",  # 速度误差 -> 平均速度误差
        "attr_err": "mAAE",  # 属性误差 -> 平均属性误差
    }
    CLASSES = (  # 数据集中定义的目标类别
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )
    MAP_CLASSES = (  # 地图元素的类别 (用于地图分割等任务)
        'ped_crossing',  # 人行横道
        'divider',  # 分隔带
        'boundary',  # 边界
    )
    ID_COLOR_MAP = [  # 用于可视化时，不同ID的颜色映射
        (59, 59, 238),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    def __init__(  # 初始化函数
        self,
        ann_file,  # 标注文件路径
        pipeline=None,  # 数据处理流水线配置
        data_root=None,  # 数据根目录
        classes=None,  # 类别列表 (如果为 None，则使用类定义的 CLASSES)
        map_classes=None,  # 地图类别列表 (如果为 None，则使用类定义的 MAP_CLASSES)
        load_interval=1,  # 加载数据的间隔 (例如，每隔多少帧加载一帧)
        with_velocity=True,  # 是否加载速度信息
        modality=None,  # 使用的模态配置 (如相机、激光雷达等)
        test_mode=False,  # 是否为测试模式
        det3d_eval_version="detection_cvpr_2019",  # 3D 检测评估版本
        track3d_eval_version="tracking_nips_2019",  # 3D 跟踪评估版本
        version="v1.0-trainval",  # NuScenes 数据集版本
        use_valid_flag=False,  # 是否使用有效标志过滤标注
        vis_score_threshold=0.25,  # 可视化时的分数阈值
        data_aug_conf=None,  # 数据增强配置
        sequences_split_num=1,  # 序列分割数量 (用于按序列分组数据)
        with_seq_flag=False,  # 是否启用序列分组标志
        keep_consistent_seq_aug=True,  # 是否在序列内保持一致的数据增强
        work_dir=None,  # 工作目录 (用于保存评估结果等)
        eval_config=None,  # 评估配置
    ):
        self.version = version  # NuScenes 数据集版本
        self.load_interval = load_interval  # 加载间隔
        self.use_valid_flag = use_valid_flag  # 是否使用有效标志
        super().__init__()  # 调用父类 (Dataset) 的初始化函数
        self.data_root = data_root  # 数据根目录
        self.ann_file = ann_file  # 标注文件路径
        self.test_mode = test_mode  # 是否为测试模式
        self.modality = modality  # 模态配置
        self.box_mode_3d = 0  # 3D 边界框模式 (此处固定为0，具体含义需查阅代码上下文)

        if classes is not None:  # 如果在构造函数中传入了 classes
            self.CLASSES = classes  # 则使用传入的 classes
        if map_classes is not None:   # 如果在构造函数中传入了 map_classes
            self.MAP_CLASSES = map_classes  # 则使用传入的 map_classes
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}  # 类别名称到ID的映射
        self.data_infos = self.load_annotations(self.ann_file)  # 加载标注信息

        if pipeline is not None:  # 如果提供了数据处理流水线配置
            self.pipeline = Compose(pipeline)  # 创建 Compose 对象

        self.with_velocity = with_velocity  # 是否加载速度
        self.det3d_eval_version = det3d_eval_version  # 3D 检测评估版本
        self.det3d_eval_configs = det_configs(self.det3d_eval_version)  # 获取3D检测评估配置
        self.det3d_eval_configs.class_names = list(self.det3d_eval_configs.class_range.keys()) # 设置评估配置中的类别名称
        self.track3d_eval_version = track3d_eval_version  # 3D 跟踪评估版本
        self.track3d_eval_configs = track_configs(self.track3d_eval_version)  # 获取3D跟踪评估配置
        self.track3d_eval_configs.class_names = list(self.track3d_eval_configs.class_range.keys()) # 设置评估配置中的类别名称
        if self.modality is None:  # 如果未提供模态配置
            self.modality = dict(  # 使用默认模态配置
                use_camera=False,  # 不使用相机
                use_lidar=True,  # 使用激光雷达
                use_radar=False,  # 不使用雷达
                use_map=False,  # 不使用地图
                use_external=False,  # 不使用外部数据
            )
        self.vis_score_threshold = vis_score_threshold  # 可视化分数阈值

        self.data_aug_conf = data_aug_conf  # 数据增强配置
        self.sequences_split_num = sequences_split_num  # 序列分割数量
        self.keep_consistent_seq_aug = keep_consistent_seq_aug  # 是否在序列内保持一致的数据增强
        if with_seq_flag:  # 如果启用序列分组标志
            self._set_sequence_group_flag()  # 设置序列分组标志
        
        self.work_dir = work_dir  # 工作目录
        self.eval_config = eval_config  # 评估配置

    def __len__(self):  # 定义数据集的长度 (样本数量)
        return len(self.data_infos)  # 返回已加载标注信息的数量

    def _set_sequence_group_flag(self):  # 设置序列分组标志的内部方法
        """
        Set each sequence to be a different group  # 将每个序列设置为一个不同的组
        """
        if self.sequences_split_num == -1:  # 如果序列分割数为-1，表示每个样本一个组
            self.flag = np.arange(len(self.data_infos))  # flag 为 0 到 len-1 的数组
            return
        
        res = []  # 用于存储每个样本所属的序列组ID

        curr_sequence = 0  # 当前序列组ID
        for idx in range(len(self.data_infos)):  # 遍历所有数据信息
            if idx != 0 and len(self.data_infos[idx]["sweeps"]) == 0:  # 如果不是第一帧且 sweeps 列表为空 (通常表示新序列的开始)
                # Not first frame and # of sweeps is 0 -> new sequence  # 不是第一帧且扫描次数为0 -> 新序列
                curr_sequence += 1  # 序列组ID加1
            res.append(curr_sequence)  # 将当前序列组ID添加到结果列表

        self.flag = np.array(res, dtype=np.int64)  # 将结果转换为 numpy 数组作为 flag

        if self.sequences_split_num != 1:  # 如果序列分割数不为1 (即需要进一步细分序列)
            if self.sequences_split_num == "all":  # 如果分割数为 "all"，表示每个样本独立成组
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:  # 否则，根据指定的分割数细分每个原始序列
                bin_counts = np.bincount(self.flag)  # 计算每个原始序列组中的样本数量
                new_flags = []  # 存储新的细分后的组ID
                curr_new_flag = 0  # 当前新的组ID
                for curr_flag in range(len(bin_counts)):  # 遍历每个原始序列组
                    curr_sequence_length = np.array(  # 计算当前原始序列的长度以及分割点
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],  # 当前序列的结束点
                                math.ceil(  # 向上取整，计算每个子序列的长度
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]  # 添加序列结束点，确保最后一个子序列完整
                    )

                    for sub_seq_idx in (  # 遍历每个子序列的长度
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):  # 为子序列中的每个样本分配新的组ID
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1  # 更新到下一个新的组ID

                assert len(new_flags) == len(self.flag)  # 断言新旧 flag 长度一致
                assert (  # 断言新的组数量等于旧的组数量乘以分割数
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)  # 更新 flag 为新的细分后的组ID

    def get_augmentation(self):  # 获取数据增强配置的方法
        if self.data_aug_conf is None:  # 如果未配置数据增强
            return None  # 返回 None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]  # 原始图像高度和宽度
        fH, fW = self.data_aug_conf["final_dim"]  # 增强后最终输出的图像高度和宽度
        if not self.test_mode:  # 如果是训练模式
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])  # 在指定范围内随机选择缩放比例
            resize_dims = (int(W * resize), int(H * resize))  # 计算缩放后的尺寸
            newW, newH = resize_dims  # 新的宽度和高度
            crop_h = (  # 计算裁剪区域的垂直起始位置
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))  # 随机底部裁剪百分比
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))  # 计算裁剪区域的水平起始位置
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)  # 定义裁剪区域 (x1, y1, x2, y2)
            flip = False  # 初始化翻转标志为 False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):  # 如果配置了随机翻转且随机选择为1
                flip = True  # 设置翻转标志为 True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])  # 在指定范围内随机选择2D旋转角度
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])  # 在指定范围内随机选择3D旋转角度
        else:  # 如果是测试模式
            resize = max(fH / H, fW / W)  # 计算固定的缩放比例以适应最终尺寸
            resize_dims = (int(W * resize), int(H * resize))  # 计算缩放后的尺寸
            newW, newH = resize_dims  # 新的宽度和高度
            crop_h = (  # 计算固定的裁剪区域垂直起始位置
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)  # 使用底部裁剪百分比的平均值
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)  # 计算固定的裁剪区域水平起始位置 (居中裁剪)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)  # 定义裁剪区域
            flip = False  # 测试模式不翻转
            rotate = 0  # 测试模式不进行2D旋转
            rotate_3d = 0  # 测试模式不进行3D旋转
        aug_config = {  # 将所有增强参数存入字典
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config  # 返回增强配置字典

    def __getitem__(self, idx):  # 获取单个数据样本的方法
        if isinstance(idx, dict):  # 如果 idx 是字典 (通常用于序列采样时传递增强配置)
            aug_config = idx["aug_config"]  # 从字典中获取增强配置
            idx = idx["idx"]  # 从字典中获取真实的数据索引
        else:  # 如果 idx 是普通的整数索引
            aug_config = self.get_augmentation()  # 调用 get_augmentation 获取增强配置
        data = self.get_data_info(idx)  # 根据索引获取原始数据信息
        data["aug_config"] = aug_config  # 将增强配置添加到数据字典中
        data = self.pipeline(data)  # 将数据传递给数据处理流水线进行处理
        return data  # 返回处理后的数据

    def get_cat_ids(self, idx):  # 根据索引获取该样本中包含的类别ID列表
        info = self.data_infos[idx]  # 获取指定索引的标注信息
        if self.use_valid_flag:  # 如果使用有效标志
            mask = info["valid_flag"]  # 获取有效标志掩码
            gt_names = set(info["gt_names"][mask])  # 获取有效标注的类别名称集合
        else:  # 如果不使用有效标志
            gt_names = set(info["gt_names"])  # 获取所有标注的类别名称集合

        cat_ids = []  # 初始化类别ID列表
        for name in gt_names:  # 遍历类别名称
            if name in self.CLASSES:  # 如果该类别在预定义的类别列表中
                cat_ids.append(self.cat2id[name])  # 将对应的类别ID添加到列表中
        return cat_ids  # 返回类别ID列表

    def load_annotations(self, ann_file):  # 加载标注文件的方法
        data = mmcv.load(ann_file, file_format="pkl")  # 使用 mmcv 加载 pkl 格式的标注文件
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))  # 根据时间戳对标注信息进行排序
        data_infos = data_infos[:: self.load_interval]  # 根据加载间隔进行下采样
        self.metadata = data["metadata"]  # 存储元数据
        self.version = self.metadata["version"]  # 从元数据中获取数据集版本
        print(self.metadata)  # 打印元数据
        return data_infos  # 返回加载和处理后的标注信息列表
    
    def anno2geom(self, annos):  # 将地图标注转换为几何对象的方法
        map_geoms = {}  # 初始化存储几何对象的字典
        for label, anno_list in annos.items():  # 遍历每个类别的标注列表
            map_geoms[label] = []  # 为当前类别创建一个空列表
            for anno in anno_list:  # 遍历当前类别的每个标注
                geom = LineString(anno)  # 将标注点列表转换为 LineString 几何对象
                map_geoms[label].append(geom)  # 将几何对象添加到对应类别的列表中
        return map_geoms  # 返回包含几何对象的字典
    
    def get_data_info(self, index):  # 获取单个样本的详细数据信息的方法
        info = self.data_infos[index]  # 获取指定索引的原始标注信息
        input_dict = dict(  # 初始化输入字典，包含各种传感器数据和元信息
            token=info["token"],  # 样本 token
            map_location=info["map_location"],  # 地图位置
            pts_filename=info["lidar_path"],  # 激光雷达点云文件路径
            sweeps=info["sweeps"],  # 激光雷达扫描序列 (用于多帧融合)
            timestamp=info["timestamp"] / 1e6,  # 时间戳 (转换为秒)
            lidar2ego_translation=info["lidar2ego_translation"],  # 激光雷达到自车坐标系的平移
            lidar2ego_rotation=info["lidar2ego_rotation"],  # 激光雷达到自车坐标系的旋转 (四元数)
            ego2global_translation=info["ego2global_translation"],  # 自车到全局坐标系的平移
            ego2global_rotation=info["ego2global_rotation"],  # 自车到全局坐标系的旋转 (四元数)
            ego_status=info['ego_status'].astype(np.float32),  # 自车状态 (如速度、加速度等)
            map_infos=info["map_annos"],  # 原始地图标注信息
        )
        lidar2ego = np.eye(4)  # 初始化激光雷达到自车的4x4变换矩阵
        lidar2ego[:3, :3] = pyquaternion.Quaternion(  # 设置旋转部分
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])  # 设置平移部分
        ego2global = np.eye(4)  # 初始化自车到全局的4x4变换矩阵
        ego2global[:3, :3] = pyquaternion.Quaternion(  # 设置旋转部分
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])  # 设置平移部分
        input_dict["lidar2global"] = ego2global @ lidar2ego  # 计算激光雷达到全局的变换矩阵

        map_geoms = self.anno2geom(info["map_annos"])  # 将原始地图标注转换为几何对象
        input_dict["map_geoms"] = map_geoms  # 添加到输入字典

        if self.modality["use_camera"]:  # 如果使用相机模态
            image_paths = []  # 图像文件路径列表
            lidar2img_rts = []  # 激光雷达到图像的变换矩阵列表
            lidar2cam_rts = []  # 激光雷达到相机的变换矩阵列表
            cam_intrinsic = []  # 相机内参矩阵列表
            for cam_type, cam_info in info["cams"].items():  # 遍历所有相机的信息
                image_paths.append(cam_info["data_path"])  # 添加图像路径
                # obtain lidar to image transformation matrix  # 获取激光雷达到图像的变换矩阵
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])  # 计算激光雷达到相机的旋转矩阵的逆 (即相机到激光雷达的旋转)
                lidar2cam_t = (  # 计算激光雷达到相机的平移向量
                    cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                )
                lidar2cam_rt = np.eye(4)  # 初始化激光雷达到相机的4x4变换矩阵
                lidar2cam_rt[:3, :3] = lidar2cam_r.T  # 设置旋转部分 (转置回来得到激光雷达到相机的旋转)
                lidar2cam_rt[3, :3] = -lidar2cam_t  # 设置平移部分
                intrinsic = copy.deepcopy(cam_info["cam_intrinsic"])  # 深拷贝相机内参
                cam_intrinsic.append(intrinsic)  # 添加到列表
                viewpad = np.eye(4)  # 创建一个4x4的单位矩阵用于填充内参矩阵
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic  # 将内参矩阵填充到左上角
                lidar2img_rt = viewpad @ lidar2cam_rt.T  # 计算激光雷达到图像的变换矩阵 (内参 @ 外参的转置)
                lidar2img_rts.append(lidar2img_rt)  # 添加到列表
                lidar2cam_rts.append(lidar2cam_rt)  # 添加激光雷达到相机的变换矩阵到列表

            input_dict.update(  # 更新输入字典，添加相机相关信息
                dict(
                    img_filename=image_paths,  # 图像文件名列表
                    lidar2img=lidar2img_rts,  # 激光雷达到图像的变换矩阵列表
                    lidar2cam=lidar2cam_rts,  # 激光雷达到相机的变换矩阵列表
                    cam_intrinsic=cam_intrinsic,  # 相机内参列表
                )
            )

        annos = self.get_ann_info(index)  # 获取当前样本的标注信息 (如3D边界框、类别等)
        input_dict.update(annos)  # 将标注信息添加到输入字典
        return input_dict  # 返回包含所有数据的输入字典

    def get_ann_info(self, index):  # 获取单个样本的标注信息 (如3D边界框、类别等)
        info = self.data_infos[index]  # 获取原始标注信息
        if self.use_valid_flag:  # 如果使用有效标志
            mask = info["valid_flag"]  # 获取有效标志掩码
        else:  # 否则，使用激光雷达点数大于0作为有效性判断依据
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]  # 根据掩码获取有效的3D边界框
        gt_names_3d = info["gt_names"][mask]  # 根据掩码获取有效的类别名称
        gt_labels_3d = []  # 初始化3D标签列表
        for cat in gt_names_3d:  # 遍历类别名称
            if cat in self.CLASSES:  # 如果类别在预定义类别列表中
                gt_labels_3d.append(self.CLASSES.index(cat))  # 添加对应的类别索引
            else:  # 如果类别不在预定义列表中
                gt_labels_3d.append(-1)  # 添加-1作为无效标签
        gt_labels_3d = np.array(gt_labels_3d)  # 转换为 numpy 数组

        if self.with_velocity:  # 如果需要加载速度信息
            gt_velocity = info["gt_velocity"][mask]  # 获取有效的速度信息
            nan_mask = np.isnan(gt_velocity[:, 0])  # 检查速度中是否存在 NaN 值
            gt_velocity[nan_mask] = [0.0, 0.0]  # 将 NaN 值替换为0
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)  # 将速度信息拼接到3D边界框数据后面

        anns_results = dict(  # 初始化标注结果字典
            gt_bboxes_3d=gt_bboxes_3d,  # 3D边界框
            gt_labels_3d=gt_labels_3d,  # 3D标签
            gt_names=gt_names_3d,  # 类别名称
        )
        if "instance_inds" in info:  # 如果标注信息中包含实例索引 (用于跟踪)
            instance_inds = np.array(info["instance_inds"], dtype=np.int)[mask]  # 获取有效的实例索引
            anns_results["instance_inds"] = instance_inds  # 添加到结果字典
            
        if 'gt_agent_fut_trajs' in info:  # 如果包含智能体的未来轨迹标注 (用于运动预测)
            anns_results['gt_agent_fut_trajs'] = info['gt_agent_fut_trajs'][mask]  # 未来轨迹点
            anns_results['gt_agent_fut_masks'] = info['gt_agent_fut_masks'][mask]  # 未来轨迹的有效性掩码

        if 'gt_ego_fut_trajs' in info:  # 如果包含自车的未来轨迹标注 (用于规划)
            anns_results['gt_ego_fut_trajs'] = info['gt_ego_fut_trajs']  # 自车未来轨迹点
            anns_results['gt_ego_fut_masks'] = info['gt_ego_fut_masks']  # 自车未来轨迹有效性掩码
            anns_results['gt_ego_fut_cmd'] = info['gt_ego_fut_cmd']  # 自车未来轨迹的命令 (如左转、右转、直行)
        
            ## get future box for planning eval  # 为规划评估获取未来的边界框
            fut_ts = int(info['gt_ego_fut_masks'].sum())  # 未来轨迹的时间步长
            fut_boxes = []  # 存储未来边界框的列表
            cur_scene_token = info["scene_token"]  # 当前场景的 token
            cur_T_global = get_T_global(info)  # 获取当前帧的全局变换矩阵
            for i in range(1, fut_ts + 1):  # 遍历未来时间步
                if index + i >= len(self.data_infos): break # 防止索引越界
                fut_info = self.data_infos[index + i]  # 获取未来帧的信息
                fut_scene_token = fut_info["scene_token"]  # 未来帧的场景 token
                if cur_scene_token != fut_scene_token:  # 如果场景发生变化，则停止
                    break
                if self.use_valid_flag:  # 根据有效标志获取掩码
                    mask_fut = fut_info["valid_flag"]
                else:
                    mask_fut = fut_info["num_lidar_pts"] > 0

                fut_gt_bboxes_3d = fut_info["gt_boxes"][mask_fut]  # 获取未来帧的有效3D边界框
                if len(fut_gt_bboxes_3d) == 0: # 如果未来没有物体，则添加空列表
                    fut_boxes.append(np.array([]))
                    continue
                
                fut_T_global = get_T_global(fut_info)  # 获取未来帧的全局变换矩阵
                T_fut2cur = np.linalg.inv(cur_T_global) @ fut_T_global  # 计算从未来帧到当前帧的变换矩阵

                center = fut_gt_bboxes_3d[:, :3] @ T_fut2cur[:3, :3].T + T_fut2cur[:3, 3]  # 将未来框的中心点变换到当前帧坐标系
                yaw_mat = pyquaternion.Quaternion(matrix=T_fut2cur[:3,:3]).rotation_matrix # 未来帧到当前帧的旋转

                # 将未来框的yaw角变换到当前帧坐标系
                fut_yaw_quat = [pyquaternion.Quaternion(axis=[0,0,1], angle=y) for y in fut_gt_bboxes_3d[:,6]]
                cur_yaw_quat = [pyquaternion.Quaternion(matrix=yaw_mat) * q for q in fut_yaw_quat]
                yaw = np.array([q.angle if q.axis[2] > 0 else -q.angle for q in cur_yaw_quat])


                fut_gt_bboxes_3d[:, :3] = center  # 更新边界框中心点
                fut_gt_bboxes_3d[:, 6] = yaw  # 更新边界框yaw角
                fut_boxes.append(fut_gt_bboxes_3d)  # 添加到未来边界框列表

            anns_results['fut_boxes'] = fut_boxes  # 将未来边界框列表添加到结果字典
        
        return anns_results  # 返回标注结果字典