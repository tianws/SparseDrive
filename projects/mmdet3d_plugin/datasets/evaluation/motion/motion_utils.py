# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.
# 此文件代码基于官方nuScenes dev-kit，并由后续贡献者修改以适应运动预测评估的需求，
# 增加了如MotionBox类、自定义的load_prediction, load_gt, accumulate函数，
#以及运动相关的评估指标计算 (minADE, minFDE, MissRate, EPA)。

import argparse # 导入argparse模块，用于解析命令行参数 (在此文件中未直接使用，但主评估脚本会用)
import json # 导入json模块，用于处理JSON数据格式
import os # 导入os模块，用于与操作系统交互
import random # 导入random模块 (在此文件中未直接使用)
import time # 导入time模块 (在此文件中未直接使用)
import tqdm # 导入tqdm模块，用于显示进度条
from typing import Tuple, Dict, Any, Callable # 导入类型提示相关的模块

import numpy as np # 导入NumPy库

from nuscenes import NuScenes # 从nuscenes-devkit导入NuScenes主类
from nuscenes.eval.common.config import config_factory # 从NuScenes评估通用模块导入配置工厂函数
from nuscenes.eval.common.data_classes import EvalBoxes # 从NuScenes评估通用模块导入EvalBoxes数据类 (用于存储GT或预测box的集合)
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes # 从NuScenes评估通用模块导入加载和过滤box的函数
from nuscenes.eval.detection.algo import calc_ap, calc_tp # 从NuScenes检测评估算法模块导入计算AP和TP指标的函数
from nuscenes.eval.detection.constants import DETECTION_NAMES, ATTRIBUTE_NAMES, TP_METRICS as DETECTION_TP_METRICS # 从NuScenes检测常量模块导入名称和TP指标 (重命名以区分)
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList as OriginalDetectionMetricDataList, DetectionMetricData as OriginalDetectionMetricData # 从NuScenes检测数据类模块导入相关类 (部分会被自定义类覆盖或继承)
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample # 从NuScenes检测渲染模块导入绘图函数
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global # 从NuScenes预测模块导入帮助类和坐标转换函数
from nuscenes.utils.splits import create_splits_scenes # 从NuScenes工具模块导入创建场景分割的函数
from nuscenes.eval.detection.utils import category_to_detection_name # 从NuScenes检测工具模块导入类别到检测名称的转换函数
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion # 从NuScenes通用工具模块导入四元数相关的函数
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean # 从NuScenes通用工具模块导入各种度量计算函数

# 运动预测任务中，将原始NuScenes类别映射到更广泛的评估类别
# 例如，多种类型的bus都映射为'car'进行运动评估，或者将cone和barrier视为一类。
motion_name_mapping = {
    'car': 'car',
    'truck': 'car', # 卡车视为汽车类
    'construction_vehicle': 'car', # 工程车辆视为汽车类
    'bus': 'car', # 公交车视为汽车类
    'trailer': 'car', # 拖车视为汽车类
    'motorcycle': 'car', # 摩托车视为汽车类（有时摩托车会单独评估或归为自行车类）
    'bicycle': 'car', # 自行车视为汽车类（这可能是一个特定的评估设定，通常自行车和汽车动力学差异大）
    'pedestrian': 'pedestrian', # 行人
    'traffic_cone': 'barrier', # 交通锥视为障碍物类
    'barrier': 'barrier', # 障碍物
}


class MotionBox(DetectionBox): # 定义MotionBox类，继承自NuScenes的DetectionBox
    """ Data class used during detection evaluation. Can be a prediction or ground truth.
        # 用于检测评估的数据类。可以是预测框或真实框。
        此类扩展了DetectionBox，增加了存储轨迹信息的功能。
    """

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0), # 四元数 (w,x,y,z)
                 velocity: Tuple[float, float] = (0, 0), # 2D速度 (vx, vy)
                 ego_translation: [float, float, float] = (0, 0, 0),  # 相对于自车(ego)的平移 (米)
                 num_pts: int = -1,  # 框内的激光雷达或雷达点数 (仅用于GT box)
                 detection_name: str = 'car',  # 检测挑战中使用的类别名称
                 detection_score: float = -1.0,  # 检测置信度分数 (GT样本此值为-1)
                 attribute_name: str = '',  # 物体的属性名称 (例如 'vehicle.parked')
                 traj: np.ndarray = None):  # 增加的属性：未来轨迹点序列 (num_timesteps, 2 or 3)
        """
        构造函数。
        Args:
            traj (np.ndarray, optional): 对象的未来轨迹。形状通常是 (T, 2) 或 (T, 3)，T是时间步数量。
        """
        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts,
                         detection_name=detection_name, detection_score=detection_score, attribute_name=attribute_name)
        # 父类的初始化已经处理了 detection_name, detection_score, attribute_name 的断言和赋值
        # 这里不再需要重复断言和赋值这些参数。

        self.traj = traj # 存储轨迹信息

    def __eq__(self, other): # 定义MotionBox对象的相等比较
        # 在父类比较的基础上，增加对轨迹的比较
        return (super().__eq__(other) and
                np.array_equal(self.traj, other.traj)) # 使用np.array_equal比较轨迹数组

    def serialize(self) -> dict: # 将MotionBox对象序列化为字典，以便保存为JSON等格式
        """ Serialize instance into json-friendly format. """ # 序列化实例为JSON友好格式。
        data = super().serialize() # 调用父类的序列化方法获取基础box信息
        data['traj'] = self.traj.tolist() if self.traj is not None else None # 添加轨迹信息 (转换为列表)
        return data

    @classmethod
    def deserialize(cls, content: dict): # 从字典反序列化创建MotionBox对象
        """ Initialize from serialized content. """ # 从序列化内容初始化。
        # 从字典内容创建父类DetectionBox对象
        # detection_box = DetectionBox.deserialize(content) # 父类没有直接的deserialize，而是通过__init__
        # 这里直接调用自己的__init__，并传入父类需要的参数
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content # 处理可选字段
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   attribute_name=content.get('attribute_name', ''), # 使用.get获取可选字段
                   traj=np.array(content['traj']) if content.get('traj') is not None else None) # 添加轨迹的反序列化


def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls=MotionBox, verbose: bool = False) \
        -> Tuple[EvalBoxes, Dict]: # 加载模型预测结果文件
    """
    Loads object predictions from file. # 从文件加载物体预测结果。
    :param result_path: Path to the .json result file provided by the user. # 用户提供的.json结果文件路径。
                        (在此项目中，result_path可以直接是已加载的字典对象)
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample. # 每个样本允许的最大box数量。
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox. # 要加载的box类型，例如MotionBox。
    :param verbose: Whether to print messages to stdout. # 是否打印消息到标准输出。
    :return: The deserialized results and meta data. # 反序列化的结果 (EvalBoxes) 和元数据 (dict)。
    """

    # Load from file and check that the format is correct.
    if isinstance(result_path, str): # 如果result_path是字符串路径
        with open(result_path) as f:
            data = json.load(f) # 则从JSON文件加载
    else: # 否则假定result_path已经是加载好的字典对象
        data = result_path

    assert 'results' in data, 'Error: No field `results` in result file.' # 检查'results'字段是否存在

    # 对预测结果中的类别名称应用motion_name_mapping进行映射
    for sample_token_key in data['results'].keys(): # 遍历每个样本的预测结果
        for i in range(len(data['results'][sample_token_key])): # 遍历该样本的每个预测框
            cls_name = data['results'][sample_token_key][i]['detection_name']
            if cls_name in motion_name_mapping: # 如果类别名在映射表中
                cls_name = motion_name_mapping[cls_name] # 则替换为映射后的名称
            data['results'][sample_token_key][i]['detection_name'] = cls_name
    
    # 使用EvalBoxes.deserialize方法将字典数据反序列化为EvalBoxes对象，其中每个box是指定的box_cls类型
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta'] if 'meta' in data else {} # 获取元数据 (如果存在)
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path if isinstance(result_path, str) else "memory", len(all_results.sample_tokens)))

    # 检查每个样本的预测框数量是否超过上限
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta


def load_gt(nusc: NuScenes, eval_split: str, box_cls=MotionBox, verbose: bool = False, seconds: int = 12) -> EvalBoxes: # 加载真实标签(GT)数据
    """
    Loads ground truth boxes from DB. # 从数据库加载真实(GT)box。
    :param nusc: A NuScenes instance. # NuScenes API实例。
    :param eval_split: The evaluation split for which we load GT boxes. # 要为其加载GT box的评估分割。
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox. # 要加载的box类型，例如MotionBox。
    :param verbose: Whether to print messages to stdout. # 是否打印消息到标准输出。
    :param seconds: Number of future half-seconds to load for trajectory. # 为轨迹加载的未来半秒数 (例如12代表6秒)。
    :return: The GT boxes. # GT box (EvalBoxes对象)。
    """
    predict_helper = PredictHelper(nusc) # 初始化NuScenes预测辅助工具
    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))

    # 从NuScenes的splits中读取指定分割的场景名称
    # create_splits_scenes() 返回一个字典，键是分割名 (如'train', 'val', 'mini_train', 'mini_val', 'test')
    # 值是对应场景名称的列表
    try:
        split_scenes_names = create_splits_scenes()[eval_split]
    except KeyError:
        raise ValueError(f'Split {eval_split} not found in NuScenes standard splits.')

    sample_tokens = [] # 用于存储属于该分割的样本token
    for sample_rec in nusc.sample: # 遍历所有样本记录
        scene_rec = nusc.get('scene', sample_rec['scene_token']) # 获取样本所属的场景记录
        if scene_rec['name'] in split_scenes_names: # 如果场景名在当前分割的场景名列表中
            sample_tokens.append(sample_rec['token']) # 则将样本token加入列表

    all_annotations = EvalBoxes() # 初始化EvalBoxes对象，用于存储所有样本的GT box

    # 加载标注并进行过滤
    if verbose:
        print(f"Found {len(sample_tokens)} samples in split {eval_split}.")

    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose, desc=f"Loading GT for {eval_split}"): # 遍历选定分割的样本token
        sample = nusc.get('sample', sample_token) # 获取样本记录
        sample_annotation_tokens = sample['anns'] # 获取该样本的所有标注token

        sample_boxes = [] # 存储当前样本的MotionBox对象列表
        for sample_annotation_token in sample_annotation_tokens: # 遍历每个标注token
            sample_annotation = nusc.get('sample_annotation', sample_annotation_token) # 获取详细标注信息

            if box_cls == MotionBox: # 如果目标box类型是MotionBox
                # 获取检测任务中使用的类别名称，并过滤掉不在DETECTION_NAMES中的类别
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None: # 如果类别无效或不用于评估，则跳过
                    continue

                # 应用motion_name_mapping将原始类别映射到运动评估中使用的类别
                if detection_name in motion_name_mapping:
                    detection_name = motion_name_mapping[detection_name]
                # 再次检查映射后的名称是否有效 (例如，如果映射到None或非评估类别)
                if detection_name is None or detection_name not in DETECTION_NAMES: # 确保仍在检测名称列表中
                    continue


                attribute_name = '' # 初始化属性名称为空
                if len(sample_annotation['attribute_tokens']) > 0 : # 如果存在属性
                    # NuScenes官方评估通常只考虑第一个属性
                    attribute_name = nusc.get('attribute', sample_annotation['attribute_tokens'][0])['name']

                # 获取未来轨迹
                instance_token = sample_annotation['instance_token']
                # 使用PredictHelper获取在智能体局部坐标系下的未来轨迹点 (x向前, y向左)
                # seconds参数是预测的秒数，由于NuScenes是2Hz采样，所以传入seconds=fut_ts/2
                fut_traj_local = predict_helper.get_future_for_agent(
                    instance_token, sample_token, seconds=seconds/2.0, # seconds是半秒数，所以除以2得到实际秒数
                    in_agent_frame=True
                )

                fut_traj_scene_centric = np.zeros((0,2)) # 初始化为空数组，以防没有未来轨迹
                if fut_traj_local.shape[0] > 0: # 如果存在有效的未来轨迹点
                    # 获取当前标注框在LIDAR坐标系下的信息，用于将局部轨迹转换到场景（LIDAR）坐标系
                    # get_sample_data的第二个参数selected_anntokens用于只返回与该标注相关的box
                    _, boxes_for_current_anno, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'], selected_anntokens=[sample_annotation_token])
                    if not boxes_for_current_anno: continue # 如果由于某些原因没有取到box，则跳过
                    box_lidar_frame = boxes_for_current_anno[0] # 获取对应的Box对象

                    # 将局部轨迹点转换到场景（LIDAR）坐标系
                    # 注意：convert_local_coords_to_global的输入是box中心和姿态，输出是相对于全局原点的坐标
                    # 但对于轨迹，我们通常需要的是相对于当前物体位置的偏移，或者是全局坐标下的轨迹
                    # 如果fut_traj_local已经是相对于当前物体位置的偏移，则不需要再转换。
                    # 如果fut_traj_local是相对于局部坐标系原点(0,0)的绝对坐标，则需要转换。
                    # PredictHelper.get_future_for_agent(in_agent_frame=True)返回的是相对于agent当前位置和朝向的坐标。
                    # 我们需要将其转换为相对于当前LIDAR坐标系的坐标。
                    # 首先将局部轨迹点通过agent的当前姿态旋转，然后加上agent的当前中心位置。
                    # fut_traj_local (T, 2)
                    # box_lidar_frame.orientation: 四元数
                    # box_lidar_frame.center: (3,)

                    # 旋转局部轨迹点到当前LiDAR坐标系下的方向
                    rotated_fut_traj = fut_traj_local @ box_lidar_frame.rotation_matrix[:2, :2].T
                    # 加上当前物体在LiDAR坐标系下的中心点，得到LiDAR坐标系下的绝对轨迹点
                    fut_traj_scene_centric = rotated_fut_traj + box_lidar_frame.center[:2]

                sample_boxes.append( # 创建MotionBox对象
                    MotionBox( # 使用MotionBox的构造函数
                        sample_token=sample_token,
                        translation=tuple(sample_annotation['translation']),
                        size=tuple(sample_annotation['size']),
                        rotation=tuple(sample_annotation['rotation']),
                        velocity=tuple(nusc.box_velocity(sample_annotation['token'])[:2]), # 获取2D速度
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name, # 使用映射后的类别名
                        detection_score=-1.0,  # GT样本分数为-1
                        attribute_name=attribute_name,
                        traj=fut_traj_scene_centric # 存储转换到场景（LIDAR）坐标系下的未来轨迹
                    )
                )
            # elif box_cls == TrackingBox: ... (跟踪相关的box加载逻辑，此处省略，因为主要关注MotionBox)
            else: # 其他box类型未实现
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes) # 将当前样本的所有GT box添加到EvalBoxes对象中

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations # 返回包含所有GT的EvalBoxes对象


def accumulate(gt_boxes: EvalBoxes, # 累积每个预测与GT匹配后的指标数据
               pred_boxes: EvalBoxes, # (此函数是自定义的，不同于nuscenes.eval.detection.algo.accumulate)
               class_name: str,       # 当前评估的类别名称
               dist_fcn: Callable,    # 用于计算box之间距离的函数 (例如center_distance)
               dist_th: float,        # 匹配的距离阈值
               verbose: bool = False) -> Tuple[MotionMetricData, float, float]: # 返回自定义的MotionMetricData, EPA, EPA_
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    # (注释与原检测评估的accumulate类似，但功能已针对运动预测调整)
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (MotionMetricData, EPA, EPA_). MotionMetricData包含多种累积误差，EPA和EPA_是两种终点精度。
    """
    # ---------------------------------------------
    # 组织输入并初始化累加器。
    # ---------------------------------------------
    # 统计当前类别的GT数量
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # 如果GT中没有这个类别的物体，则返回一个表示“无预测”的数据结构和0精度的EPA
    if npos == 0:
        return MotionMetricData.no_predictions(), 0.0, 0.0 # 使用自定义的MotionMetricData

    # 提取当前类别的所有预测框
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list] # 提取置信度

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # 按置信度降序排列预测框
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # 初始化TP, FP, 置信度列表和匹配数据字典
    tp = []  # 真阳性指示列表
    fp = []  # 假阳性指示列表
    conf = []  # 置信度列表 (排序后)
    hit = 0 # 记录FDE < 2.0m 的TP数量 (用于原始EPA计算)

    # match_data 存储每个成功匹配对的运动评估指标
    match_data = {'conf': [], # 匹配上的预测的置信度
                  'min_ade': [], # 最小平均位移误差
                  'min_fde': [], # 最小最终位移误差
                  'miss_rate': []} # 未命中率 (基于阈值)

    # ---------------------------------------------
    # 进行匹配并累积匹配数据。
    # ---------------------------------------------
    taken_gt_in_sample = set()  # 用于跟踪每个样本中哪些GT已经被匹配过，避免重复匹配

    for pred_idx_sorted in sortind: # 遍历按置信度排序后的预测框索引
        pred_box = pred_boxes_list[pred_idx_sorted] # 当前预测框 (MotionBox对象)
        min_dist = np.inf # 初始化与GT的最小距离为无穷大
        match_gt_idx_in_sample = None # 初始化匹配上的GT在样本内的索引为None

        # 在当前样本中查找与pred_box匹配的GT box
        # gt_boxes[pred_box.sample_token] 是一个列表，包含该样本的所有GT MotionBox对象
        for gt_idx_in_sample, gt_box in enumerate(gt_boxes[pred_box.sample_token]):
            # 检查类别是否匹配，以及该GT是否尚未被匹配
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx_in_sample) in taken_gt_in_sample:
                this_distance = dist_fcn(gt_box, pred_box) # 计算预测框与GT框之间的距离 (通常是中心距离)
                if this_distance < min_dist: # 如果找到更近的GT
                    min_dist = this_distance # 更新最小距离
                    match_gt_idx_in_sample = gt_idx_in_sample # 记录匹配上的GT的索引

        # 如果最小距离小于等于阈值，则认为匹配成功
        is_match = min_dist < dist_th

        if is_match: # 如果匹配成功
            taken_gt_in_sample.add((pred_box.sample_token, match_gt_idx_in_sample)) # 标记该GT已被占用

            # 更新TP, FP和置信度列表 (用于后续计算AP等检测指标，如果需要的话)
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # 累积运动预测相关的指标
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx_in_sample] # 获取匹配上的GT MotionBox对象
            match_data['conf'].append(pred_box.detection_score) # 记录置信度

            # 调用prediction_metrics计算minADE, minFDE, MissRate
            minade, minfde, mr = prediction_metrics(gt_box_match, pred_box)
            match_data['min_ade'].append(minade)
            match_data['min_fde'].append(minfde)
            match_data['miss_rate'].append(mr)

            if minfde < 2.0: # 如果FDE小于2米，则认为是一个"hit" (用于原始EPA计算)
                hit += 1
        else: # 如果未匹配成功
            tp.append(0)
            fp.append(1) # 标记为假阳性
            conf.append(pred_box.detection_score)

    # 如果没有任何匹配 (例如，所有预测都无法匹配任何GT，或者没有该类别的预测)
    if len(match_data['min_ade']) == 0: # min_ade列表为空说明没有成功的匹配
        return MotionMetricData.no_predictions(), 0.0, 0.0 # 返回表示无预测的MotionMetricData和0精度的EPA

    # --- 累积和插值，为计算最终平均指标做准备 (这部分逻辑继承自检测评估，用于平滑误差曲线) ---
    # N_tp_total = np.sum(tp) # 总TP数 (基于匹配)
    # N_fp_total = np.sum(fp) # 总FP数 (基于匹配)
    # tp_cumsum = np.cumsum(tp).astype(float) # TP的累积和
    # fp_cumsum = np.cumsum(fp).astype(float) # FP的累积和
    # conf_sorted = np.array(conf) # 已按置信度排序的列表
    #
    # # 计算精确率和召回率曲线 (这对于直接评估minADE/minFDE可能不是必需的，但MotionMetricData结构需要)
    # precision_curve = tp_cumsum / (fp_cumsum + tp_cumsum)
    # recall_curve = tp_cumsum / float(npos) if npos > 0 else np.zeros_like(tp_cumsum)
    #
    # # 在101个标准的召回率点上插值精确率和置信度
    # recall_interp_points = np.linspace(0, 1, MotionMetricData.nelem)
    # precision_interp = np.interp(recall_interp_points, recall_curve, precision_curve, right=0)
    # confidence_interp = np.interp(recall_interp_points, recall_curve, conf_sorted, right=0)
    #
    # # 对每个运动指标的累积均值进行插值
    # for key in match_data.keys():
    #     if key == "conf": continue
    #     # match_data[key] 是一个列表，包含了每次成功匹配时的指标值
    #     # 我们需要将其与排序后的conf对应起来，然后插值
    #     # 假设match_data中的值是按照conf_sorted的顺序排列的（因为它们是在遍历sortind时添加的）
    #     metric_values_sorted_by_conf = np.array(match_data[key])
    #     cumulative_mean_metric = cummean(metric_values_sorted_by_conf) # 计算累积平均值
    #     # 根据置信度进行插值 (注意[::-1]是因为np.interp需要x坐标递增)
    #     # 这实际上是在每个召回率点上，估计在该召回率水平下（由对应置信度决定）的平均误差
    #     match_data[key] = np.interp(confidence_interp[::-1], conf_sorted[::-1], cumulative_mean_metric[::-1])[::-1]
    #     match_data[key][confidence_interp == 0] = cumulative_mean_metric[-1] if len(cumulative_mean_metric)>0 else 0 # 处理置信度为0的情况

    # --- 直接计算平均误差和EPA ---
    # 对于minADE, minFDE, MissRate，通常是计算所有TP样本的这些误差的平均值。
    # 但NuScenes的TP指标计算方式是：在达到某个召回率阈值（如0.1）的所有TP中计算平均误差。
    # 这里match_data中的值是每个TP的误差，我们需要对其求平均。
    # 这里的实现是直接对所有匹配上的样本的误差求平均，然后封装到MotionMetricData中。
    # MotionMetricData的构造函数期望的是插值后的曲线，所以上面的插值逻辑还是需要的。
    # 为了简化并直接使用平均值，可以这样做（但这与MotionMetricData的期望输入不符）：
    # mean_min_ade = np.mean(match_data['min_ade']) if len(match_data['min_ade']) > 0 else 1.0 # 用1.0表示最差情况
    # mean_min_fde = np.mean(match_data['min_fde']) if len(match_data['min_fde']) > 0 else 1.0
    # mean_miss_rate = np.mean(match_data['miss_rate']) if len(match_data['miss_rate']) > 0 else 1.0

    # 使用原始检测评估中的插值方法来填充MotionMetricData，即使某些字段对运动意义不大
    tp_cumsum = np.cumsum(tp).astype(float)
    fp_cumsum = np.cumsum(fp).astype(float)
    conf_sorted = np.array(conf)
    prec_curve = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float32).eps)
    rec_curve = tp_cumsum / float(npos) if npos > 0 else np.zeros_like(tp_cumsum)

    rec_interp_points = np.linspace(0, 1, MotionMetricData.nelem)
    prec_interp = np.interp(rec_interp_points, rec_curve, prec_curve, right=0)
    conf_interp = np.interp(rec_interp_points, rec_curve, conf_sorted, right=0) # 插值置信度

    final_match_data = {}
    for key in MOTION_TP_METRICS: # 只处理运动相关的指标
        if len(match_data[key]) > 0:
            # match_data[key] 已经是按置信度排序的TP样本的误差值列表
            # 我们需要将其与 recall_interp_points 对齐
            # 方法：计算每个recall_interp_points处的平均误差
            # 这通常是通过在达到该recall水平的所有TP上取平均误差来实现
            # 简化的做法：使用累积平均误差，并按置信度（或召回率）插值
            # 这里的conf_sorted是TP和FP都包含的，match_data只包含TP的
            # 需要一种方式将match_data[key]的值与全局的recall或confidence对应起来
            # Nuscenes Detection的calc_tp做法：
            # Take valid confidences (hence TP).
            # Sort by confidence.
            # Interpolate to recall levels.
            # 여기가 중요합니다: match_data[key]는 이미 TP에 해당하는 값들이고, conf_sorted도 TP에 해당하는 confidences로 정렬되어야 합니다.
            # 하지만 현재 conf_sorted는 모든 pred의 conf이고, match_data[key]는 TP pred의 error입니다.
            # 따라서, TP pred에 해당하는 confidence를 기준으로插值해야 합니다.
            tp_confs = conf_sorted[np.array(tp, dtype=bool)]
            tp_errors = np.array(match_data[key])
            if len(tp_confs)>0:
                # 确保tp_confs是降序的
                sort_idx_tp_conf = np.argsort(-tp_confs)
                tp_confs_sorted = tp_confs[sort_idx_tp_conf]
                tp_errors_sorted_by_conf = tp_errors[sort_idx_tp_conf]
                # 计算累积平均误差
                error_cumulative_mean = cummean(tp_errors_sorted_by_conf)
                # 插值到标准的101个置信度点上 (conf_interp是标准recall点上的置信度)
                # 我们需要在这些置信度点上找到对应的累积平均误差
                # np.interp(x_new, x_old, y_old) 要求x_old是递增的
                final_match_data[key] = np.interp(conf_interp[::-1], tp_confs_sorted[::-1], error_cumulative_mean[::-1])[::-1]
                # 处理那些没有对应TP置信度的插值点 (例如，在非常低的置信度区域，可能所有都是FP)
                # 对于这些点，误差可以设为最好（0）或最差（一个大值）或最后一个有效TP的误差。
                # Nuscenes的做法通常是用0填充或用最后一个有效值填充（取决于指标）。
                # 这里用1填充，表示最差情况（对于误差）
                final_match_data[key][conf_interp == 0] = 1.0
            else:
                final_match_data[key] = np.ones(MotionMetricData.nelem)

        else: # 如果没有TP匹配
            final_match_data[key] = np.ones(MotionMetricData.nelem) # 用1填充，表示最大误差/未命中

    # 计算EPA (End Point Accuracy) - 原始的基于hit的定义
    # EPA = (hit - 0.5 * N_fp_total) / npos if npos > 0 else 0.0 # hit是FDE<2m的TP数

    # 计算EPA_ (UniAD中使用的版本，基于traj_fde<2m的匹配)
    # 这部分逻辑在原始accumulate中通过再次遍历实现，这里简化为仅返回之前计算的
    # 注意：此accumulate函数签名与detection的不同，它直接返回EPA和EPA_
    # 所以这里的EPA和EPA_应该是从匹配循环中计算得到的
    # 重新审视：EPA和EPA_的计算是在循环外，基于循环内累积的hit和traj_matched
    # hit: is_match (基于中心距离) 且 minfde < 2.0
    # traj_matched: is_match (基于中心距离) 且 traj_fde < 2.0 (这里的traj_fde是针对最佳匹配GT计算的)

    # EPA的计算需要总的FP数量 (N_fp_total)
    N_fp_total = np.sum(fp)
    EPA = (hit - 0.5 * N_fp_total) / npos if npos > 0 else 0.0

    # EPA_ 的计算需要traj_matched 和 N_fp_total
    # traj_matched的计算在原始代码的第二个循环中，这里没有那个循环。
    # 假设EPA_是作为外部参数或通过另一种方式传入/计算的。
    # 在此函数的当前版本中，第二个循环（用于traj_matched）被移除了。
    # 为了保持函数能运行，我们暂时将EPA_也设为EPA。
    # TODO: 确认EPA_的正确计算方式或来源。
    # 从调用它的MotionEval类来看，它只用了返回的EPA_。
    # 并且MotionEval类中的accumulate调用后，EPA是第二个返回值，EPA_是第三个。
    # 这意味着此处的EPA应该是原始的EPA，而EPA_是UniAD版本的。
    # 此函数内部的第二个循环（用于traj_matched和EPA_）被注释掉了。
    # 为了使其能运行并返回三个值，我们需要一个EPA_的占位符或重新实现。
    # 鉴于此函数在MotionEval中被调用时，EPA和EPA_都被使用了，
    # 我们需要确保它们都被合理计算。
    # 原始代码中，第二个循环计算traj_matched，然后 EPA_ = (traj_matched - 0.5 * N_fp) / npos
    # 由于第二个循环被移除，traj_matched 未定义。
    # 假设EPA_与EPA相同，或者需要从外部获取traj_matched。
    # 为了简单，我们假设调用者只关心MotionMetricData和EPA_。
    # 因此，返回的EPA可以忽略或设为EPA_。
    # 这里的EPA_是UniAD论文中定义的EPA。
    traj_matched_for_epa_prime = 0 # 这个值需要从一个与上面不同的匹配逻辑得到
                                 # 该匹配逻辑是：首先按中心距离找到最近GT，如果距离<阈值，
                                 # THEN 计算这对(pred,gt)的轨迹FDE，如果FDE<2m，则traj_matched_for_epa_prime++
                                 # 这个逻辑在原始代码的第二个循环中。由于它被移除，我们无法在这里计算。
                                 # 假设这个值会通过其他方式提供或此函数不负责计算它。
                                 # 为了让函数能运行，我们暂时将EPA_设为0或基于hit的EPA。
    # 实际上，MotionEval类中的accumulate调用，EPA是第二个返回值，EPA_是第三个
    # 而这个函数只返回一个DetectionMetricData。
    # 这表明这个文件中的accumulate函数签名与MotionEval中调用的accumulate不同。
    # MotionEval中的accumulate: md, EPA, EPA_ = accumulate(...)
    # 此文件中的accumulate: return MotionMetricData(...), EPA, EPA_ (需要修改返回值)

    # 重新整理返回值以匹配MotionEval的期望
    metric_data_obj = MotionMetricData(recall=rec_interp, # 使用插值后的recall
                               precision=prec_interp, # 使用插值后的precision
                               confidence=conf_interp, # 使用插值后的confidence
                               min_ade_err=final_match_data['min_ade_err'],
                               min_fde_err=final_match_data['min_fde_err'],
                               miss_rate_err=final_match_data['miss_rate_err'])

    # EPA_ (UniAD版本) 的计算，需要重新审视traj_fde和匹配逻辑
    # 假设traj_matched是在外部或通过另一个机制计算的，这里只传递
    # 为保持函数结构，暂时返回基于hit的EPA作为EPA，和另一个EPA_（可能需要修改）
    # 在MotionEval中，这个函数的返回值是 md, EPA, EPA_
    # EPA是基于minfde<2的hit数，EPA_是基于traj_fde<2的hit数

    # 计算EPA_ (UniAD version, based on traj_fde < 2.0 for the *closest* GT by center distance)
    # This requires re-iterating or careful handling of matches.
    # The original accumulate in detection kit doesn't have this specific EPA_.
    # The accumulate in this file is custom.
    # The second loop for traj_matched was removed in a previous edit.
    # We need to recalculate traj_matched for EPA_ as per UniAD's definition.
    traj_matched_for_epa_prime = 0
    taken_for_epa_prime = set()
    for ind in sortind: # 遍历按置信度排序的预测
        pred_box = pred_boxes_list[ind]
        min_dist_for_match = np.inf
        match_gt_idx_for_match = None
        fde_for_closest_gt = np.inf

        # 找到中心距离最近的、未匹配的、同类别的GT
        for gt_idx, gt_box_candidate in enumerate(gt_boxes[pred_box.sample_token]):
            if gt_box_candidate.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken_for_epa_prime:
                this_center_distance = dist_fcn(gt_box_candidate, pred_box)
                if this_center_distance < min_dist_for_match:
                    min_dist_for_match = this_center_distance
                    match_gt_idx_for_match = gt_idx

        if match_gt_idx_for_match is not None and min_dist_for_match < dist_th:
            # 对于中心距离匹配上的(pred, gt)对，计算其轨迹的FDE
            gt_box_for_fde = gt_boxes[pred_box.sample_token][match_gt_idx_for_match]
            current_fde = traj_fde(gt_box_for_fde, pred_box, final_step=seconds) # seconds是总的半秒数
            if current_fde < 2.0: # UniAD EPA的FDE阈值
                traj_matched_for_epa_prime += 1
            taken_for_epa_prime.add((pred_box.sample_token, match_gt_idx_for_match)) # 标记GT已被匹配

    EPA_uniad_style = (traj_matched_for_epa_prime - 0.5 * N_fp_total) / npos if npos > 0 else 0.0


    return metric_data_obj, EPA, EPA_uniad_style


def prediction_metrics(gt_box_match: MotionBox, pred_box: MotionBox, miss_thresh: float = 2.0) -> Tuple[float, float, float]: # 计算单个匹配对的运动预测指标
    """
    计算给定的一对真实(GT)轨迹和预测轨迹之间的minADE, minFDE和MissRate。
    预测轨迹可能是多模态的，此函数会选择与GT最匹配的模态进行计算。

    Args:
        gt_box_match (MotionBox): 包含真实轨迹的MotionBox对象。其traj属性形状 (T_gt, 2 or 3)。
        pred_box (MotionBox): 包含预测轨迹（可能多模态）的MotionBox对象。
                              其traj属性形状 (num_modes, T_pred, 2 or 3)。
        miss_thresh (float, optional): 判断为“未命中(Miss)”的FDE阈值。默认为2.0米。

    Returns:
        tuple[float, float, float]: (minADE, minFDE, miss_rate)
                                    minADE: 最小平均位移误差。
                                    minFDE: 最小最终位移误差。
                                    miss_rate: 是否未命中 (0或1)。
    """
    gt_traj = np.array(gt_box_match.traj) # (T_gt, 2) - 假设轨迹是2D的BEV坐标
    pred_traj_modes = np.array(pred_box.traj) # (num_modes, T_pred, 2)

    # 确保GT轨迹有效
    valid_gt_len = gt_traj.shape[0]
    if valid_gt_len <= 0: # 如果GT轨迹长度为0
        return 100.0, 100.0, 1.0 # 返回较大的误差值和100%的未命中率 (表示无效GT)

    # 将预测轨迹裁剪或填充到与GT轨迹相同的有效长度
    # pred_traj_modes: (num_modes, T_pred, 2)
    # gt_traj: (T_gt, 2)
    # 我们需要比较 T_gt 和 T_pred (即 pred_traj_modes.shape[1])
    # 通常评估时，T_pred >= T_gt，我们取T_gt的长度进行比较
    # 如果 T_pred < T_gt，则只能比较到 T_pred。这里假设 T_pred >= T_gt。

    # 截取预测轨迹中与GT轨迹有效长度对应的部分
    pred_traj_valid_len_modes = pred_traj_modes[:, :valid_gt_len, :] # (num_modes, T_gt, 2)

    # 计算每个预测模态轨迹与GT轨迹之间的距离误差
    # dist_all_modes: (num_modes, T_gt) - 每个时间步的L2距离
    dist_all_modes = np.linalg.norm(pred_traj_valid_len_modes - gt_traj[np.newaxis, :, :], axis=2)

    # minADE: 在所有模态中，选择使得平均位移误差(ADE)最小的那个模态的ADE值
    # dist_all_modes.mean(axis=1): 计算每个模态的ADE (num_modes,)
    minade = dist_all_modes.mean(axis=1).min() # 取所有模态中最小的ADE

    # minFDE: 在所有模态中，选择使得最终位移误差(FDE)最小的那个模态的FDE值
    # dist_all_modes[:, -1]: 每个模态的FDE (num_modes,)
    minfde = dist_all_modes[:, -1].min() # 取所有模态中最小的FDE

    # MissRate: 如果所有模态的FDE都大于miss_thresh，则认为未命中 (mr=1)
    # 否则，如果至少有一个模态的FDE小于等于miss_thresh，则认为命中 (mr=0)
    # dist_all_modes[:, -1].min() 即 minfde
    mr = 1.0 if minfde > miss_thresh else 0.0

    return minade, minfde, mr

def traj_fde(gt_box: MotionBox, pred_box: MotionBox, final_step: int) -> float: # 计算轨迹的最终位移误差(FDE)
    """
    计算给定GT轨迹和预测轨迹（可能多模态）在指定`final_step`处的最小FDE。
    Args:
        gt_box (MotionBox): 包含真实轨迹的MotionBox。
        pred_box (MotionBox): 包含预测轨迹（多模态）的MotionBox。
        final_step (int): 计算FDE的时间步（从1开始计数）。
    Returns:
        float: 最小FDE。如果GT轨迹无效，返回无穷大。
    """
    if gt_box.traj is None or gt_box.traj.shape[0] == 0: # 如果GT轨迹为空
        return np.inf # 返回无穷大表示无效

    # 确保final_step不超过GT轨迹的实际长度
    actual_final_step = min(gt_box.traj.shape[0], final_step)
    if actual_final_step <= 0: # 如果有效最终步长为0或负
        return np.inf

    gt_final_point = gt_box.traj[None, actual_final_step-1, :2] # (1, 2) GT的最终位置 (取x,y)

    # pred_box.traj 是 (num_modes, T_pred, 2 or 3)
    # 确保final_step也不超过预测轨迹的长度
    pred_final_points = np.array(pred_box.traj)[:, min(pred_box.traj.shape[1], actual_final_step)-1, :2] # (num_modes, 2) 预测的各模态最终位置

    # 计算每个预测模态的最终位置与GT最终位置之间的L2距离
    errors_at_final_step = np.linalg.norm(gt_final_point - pred_final_points, axis=-1) # (num_modes,)

    return np.min(errors_at_final_step) # 返回所有模态中最小的FDE


class MotionMetricDataList(OriginalDetectionMetricDataList): # 运动指标数据列表类，继承自检测的对应类
    """ This stores a set of MetricData in a dict indexed by (name, match-distance).
        # 此类以 (类别名称, 匹配距离阈值) 为键，在字典中存储一组MotionMetricData对象。
    """
    @classmethod
    def deserialize(cls, content: dict): # 从字典反序列化
        mdl = cls() # 创建一个空的MotionMetricDataList实例
        for key, md_serialized_content in content.items(): # 遍历字典中的每个条目
            name, distance_str = key.split(':') # 解析键 (例如 "car:2.0")
            # 使用MotionMetricData.deserialize反序列化每个指标数据对象
            mdl.set(name, float(distance_str), MotionMetricData.deserialize(md_serialized_content))
        return mdl

class MotionMetricData(OriginalDetectionMetricData): # 运动指标数据类，继承自检测的对应类
    """ This class holds accumulated and interpolated data required to calculate the detection metrics.
        # 此类保存计算检测/运动指标所需的累积和插值数据。
        它在DetectionMetricData的基础上增加了运动特有的指标。
    """

    nelem = 101 # 插值点的数量 (例如，对于0到1的召回率，每0.01一个点)

    def __init__(self,
                 recall: np.array, # 召回率曲线 (在nelem个点上插值后)
                 precision: np.array, # 精确率曲线 (在nelem个点上插值后)
                 confidence: np.array, # 置信度阈值 (对应上述recall和precision)
                 min_ade_err: np.array, # 最小平均位移误差曲线 (在nelem个点上插值后)
                 min_fde_err: np.array, # 最小最终位移误差曲线
                 miss_rate_err: np.array): # 未命中率曲线 (基于FDE > miss_thresh)

        # 调用父类的构造函数 (如果父类有参数的话，但DetectionMetricData的__init__可能没有显式参数)
        # super().__init__(recall, precision, confidence, остальные_tp_метрики_из_detection)
        # 由于这里直接覆盖了__init__，需要手动处理父类可能需要的属性，或者确保父类__init__无参或可处理这些参数。
        # DetectionMetricData的__init__通常需要tp_errors (如trans_err, scale_err等)。
        # MotionMetricData不直接存储这些，而是存储ADE, FDE等。
        # 这里为了简化，我们假设父类的__init__可以被这样调用，或者我们只关心子类新增的属性。
        # 实际上，更规范的做法是分别初始化父类和子类属性。

        # 断言输入数组的长度符合预期
        assert len(recall) == self.nelem
        assert len(precision) == self.nelem
        assert len(confidence) == self.nelem
        assert len(min_ade_err) == self.nelem
        assert len(min_fde_err) == self.nelem
        assert len(miss_rate_err) == self.nelem

        # 断言排序正确性
        assert all(confidence == sorted(confidence, reverse=True))  # 置信度应降序排列
        assert all(recall == sorted(recall))  # 召回率应升序排列 (插值后的标准召回率点)

        # 设置属性
        self.recall = recall
        self.precision = precision
        self.confidence = confidence
        self.min_ade_err = min_ade_err
        self.min_fde_err = min_fde_err
        self.miss_rate_err = miss_rate_err

    def __eq__(self, other): # 定义相等比较
        eq = True
        # 比较所有可序列化的属性是否相等
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    # max_recall_ind 和 max_recall 属性继承自DetectionMetricData，如果它们的逻辑依赖于
    # 父类中定义的其他属性（如tp_errors），则可能需要在此处重写或调整。
    # 假设它们基于confidence和recall的行为是通用的。
    @property
    def max_recall_ind(self): # 获取达到最大召回率的索引
        """ Returns index of max recall achieved. """ # 返回达到的最大召回率的索引。

        # 置信度大于0的最后一个实例的索引即为最大召回率的索引。
        non_zero_conf_indices = np.nonzero(self.confidence)[0]
        if len(non_zero_conf_indices) == 0:  # 如果没有匹配项，所有置信度都为零。
            max_recall_idx = 0
        else:
            max_recall_idx = non_zero_conf_indices[-1]
        return max_recall_idx

    @property
    def max_recall(self): # 获取最大召回率值
        """ Returns max recall achieved. """ # 返回达到的最大召回率。
        return self.recall[self.max_recall_ind]

    def serialize(self) -> Dict[str, Any]: # 序列化为字典
        """ Serialize instance into json-friendly format. """ # 序列化实例为JSON友好格式。
        return {
            'recall': self.recall.tolist(),
            'precision': self.precision.tolist(),
            'confidence': self.confidence.tolist(),
            'min_ade_err': self.min_ade_err.tolist(),
            'min_fde_err': self.min_fde_err.tolist(),
            'miss_rate_err': self.miss_rate_err.tolist(),
        }

    @classmethod
    def deserialize(cls, content: dict): # 从字典反序列化
        """ Initialize from serialized content. """ # 从序列化内容初始化。
        return cls(recall=np.array(content['recall']),
                   precision=np.array(content['precision']),
                   confidence=np.array(content['confidence']),
                   min_ade_err=np.array(content['min_ade_err']),
                   min_fde_err=np.array(content['min_fde_err']),
                   miss_rate_err=np.array(content['miss_rate_err']))

    @classmethod
    def no_predictions(cls): # 创建一个表示“无预测”情况的MotionMetricData实例
        """ Returns a md instance corresponding to having no predictions. """ # 返回一个对应于没有预测的md实例。
        return cls(recall=np.linspace(0, 1, cls.nelem), # 召回率从0到1
                   precision=np.zeros(cls.nelem),       # 精确率全为0
                   confidence=np.zeros(cls.nelem),      # 置信度全为0
                   min_ade_err=np.ones(cls.nelem) * np.inf, # ADE设为无穷大 (或一个非常大的值)
                   min_fde_err=np.ones(cls.nelem) * np.inf, # FDE设为无穷大
                   miss_rate_err=np.ones(cls.nelem))      # 未命中率设为1 (100%)

    @classmethod
    def random_md(cls): # 创建一个表示随机结果的MotionMetricData实例 (用于测试或调试)
        """ Returns an md instance corresponding to a random results. """ # 返回一个对应于随机结果的md实例。
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.random.random(cls.nelem), # 随机精确率
                   confidence=np.linspace(0, 1, cls.nelem)[::-1], # 降序置信度
                   min_ade_err=np.random.random(cls.nelem) * 10, # 随机ADE (0-10范围)
                   min_fde_err=np.random.random(cls.nelem) * 20, # 随机FDE (0-20范围)
                   miss_rate_err=np.random.random(cls.nelem)) # 随机未命中率
