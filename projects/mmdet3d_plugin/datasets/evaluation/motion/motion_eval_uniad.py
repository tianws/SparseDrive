# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.
# 此文件代码基于官方nuScenes dev-kit，并由后续贡献者修改以适应运动预测评估的需求，
# 特别是针对UniAD这类模型的输出格式和评估指标。

import argparse # 导入argparse模块，用于解析命令行参数
import json # 导入json模块，用于处理JSON数据格式 (例如加载配置文件或结果文件)
import os # 导入os模块，用于与操作系统交互，如路径操作和目录创建
import random # 导入random模块，用于生成随机数 (例如，随机选择可视化样本)
import time # 导入time模块，用于计时
import tqdm # 导入tqdm模块，用于显示进度条 (当前代码中未直接使用，但mmcv.track_iter_progress内部可能使用)
from typing import Tuple, Dict, Any # 导入类型提示相关的模块

import numpy as np # 导入NumPy库，用于高效的数值计算

from nuscenes import NuScenes # 从nuscenes-devkit导入NuScenes主类
from nuscenes.eval.common.config import config_factory # 从NuScenes评估通用模块导入配置工厂函数
from nuscenes.eval.common.data_classes import EvalBoxes # 从NuScenes评估通用模块导入EvalBoxes数据类
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes # 从NuScenes评估通用模块导入加载和过滤box的函数
# from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp # 从NuScenes检测评估算法模块导入累积、计算AP和TP的函数 (注意：此处的accumulate被下面的自定义版本覆盖)
from nuscenes.eval.detection.constants import DETECTION_NAMES, ATTRIBUTE_NAMES, TP_METRICS as DETECTION_TP_METRICS # 从NuScenes检测常量模块导入名称和TP指标 (重命名以区分)
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList, DetectionMetricData # 从NuScenes检测数据类模块导入相关类
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample # 从NuScenes检测渲染模块导入绘图函数
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global # 从NuScenes预测模块导入帮助类和坐标转换函数
from nuscenes.utils.splits import create_splits_scenes # 从NuScenes工具模块导入创建场景分割的函数
from nuscenes.eval.detection.utils import category_to_detection_name # 从NuScenes检测工具模块导入类别到检测名称的转换函数
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion # 从NuScenes通用工具模块导入四元数相关的函数
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean # 从NuScenes通用工具模块导入各种度量计算函数

from .motion_utils import MotionBox, load_prediction, load_gt, accumulate # 从同级目录的motion_utils模块导入运动预测相关的自定义类和函数 (这里的accumulate会覆盖上面的nuscenes.eval.detection.algo.accumulate)

# 定义运动预测任务中关心的True Positive (TP) 指标
MOTION_TP_METRICS = ['min_ade_err', 'min_fde_err', 'miss_rate_err']


class MotionEval: # 定义运动预测评估类
    """
    This is the official nuScenes detection evaluation code. # 这是官方nuScenes检测评估代码的修改版本，用于运动预测。
    Results are written to the provided output_dir. # 结果会写入到指定的输出目录。

    nuScenes uses the following detection metrics: # (这部分是原始检测评估的注释，对于运动预测不完全适用)
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method: # 此类中主要函数的概述：
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes. # 初始化：加载GT标注和JSON格式的预测结果，并过滤box。
    - run: Performs evaluation and dumps the metric data to disk. # (在此类中，主要评估逻辑在evaluate方法中)
    - render: Renders various plots and dumps to disk. # 渲染：渲染各种图表并保存到磁盘。

    We assume that: # 我们假设：
    - Every sample_token is given in the results, although there may be not predictions for that sample. # 结果中给出了每个sample_token，即使该样本可能没有任何预测。

    Please see https://www.nuscenes.org/object-detection for more details. # (指向原始检测评估的链接)
    """
    def __init__(self,
                 nusc: NuScenes, # NuScenes API 实例
                 config: DetectionConfig, # 检测评估配置对象 (部分参数可能被运动评估重用或忽略)
                 result_path: str, # 预测结果文件路径 (JSON格式)
                 eval_set: str, # 要评估的数据集分割 (例如 'train', 'val', 'test')
                 output_dir: str = None, # 保存图表和结果的输出目录
                 verbose: bool = True, # 是否打印详细信息到标准输出
                 seconds: int = 12): # 预测的未来时间长度（单位：半秒，因为NuScenes通常是2Hz，所以12代表6秒）
        """
        Initialize a MotionEval object. # 初始化一个MotionEval对象。
        :param nusc: A NuScenes object. # NuScenes对象。
        :param config: A DetectionConfig object. # 检测配置对象 (部分用于运动评估)。
        :param result_path: Path of the nuScenes JSON result file. # nuScenes JSON结果文件的路径。
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test. # 要评估的数据集分割。
        :param output_dir: Folder to save plots and results to. # 保存图表和结果的文件夹。
        :param verbose: Whether to print to stdout. # 是否打印到标准输出。
        :param seconds: Number of future half-seconds to evaluate. # 要评估的未来半秒数 (例如，12表示6秒)。
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config # 存储配置对象

        # Check result file exists. (原始代码中的断言，如果文件不存在会报错)
        # assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # 创建输出目录和绘图子目录
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # 加载数据
        if verbose:
            print('Initializing nuScenes motion evaluation') # 初始化nuScenes运动评估
        # 加载预测结果，使用自定义的load_prediction和MotionBox类
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, MotionBox,
                                                     verbose=verbose)
        # 加载真实GT轨迹，使用自定义的load_gt和MotionBox类
        self.gt_boxes = load_gt(self.nusc, self.eval_set, MotionBox, verbose=verbose, seconds=seconds)

        # 确保预测和GT中的样本token一致
        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # 为预测和GT box添加中心距离信息 (到自车位置的距离)
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # 根据类别定义的距离范围过滤box (例如，只评估一定范围内的车辆)
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens # 获取有效的样本token列表

    def evaluate(self) -> Tuple[Dict[str, Any], DetectionMetricDataList]: # 执行实际的评估计算
        """
        Performs the actual evaluation. # 执行实际的评估。
        :return: A tuple of high-level and the raw metric data. # 返回一个包含高级指标和原始指标数据的元组。
                                                                 # 当前实现主要返回高级指标。
        """
        start_time = time.time() # 记录开始时间

        # 为运动预测重定义评估类别和距离阈值 (这些通常在config中定义，这里是硬编码或默认)
        self.cfg.class_names = ['car', 'pedestrian'] # 只评估'car'和'pedestrian'的运动预测
        self.cfg.dist_ths = [2.0] # 使用2.0米作为匹配的距离阈值

        # -----------------------------------
        # 步骤1: 为所有类别和距离阈值累积指标数据。
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList() # 用于存储原始匹配和误差信息的数据结构 (主要用于检测AP，这里用法可能简化)
        metrics = {} # 初始化存储最终指标的字典

        for class_name in self.cfg.class_names: # 遍历每个评估类别
            for dist_th in self.cfg.dist_ths: # 遍历每个距离阈值 (这里只有一个2.0m)
                # 调用自定义的accumulate函数 (来自motion_utils)
                # md: DetectionMetricData 对象，存储匹配结果
                # EPA: End Point Accuracy (可能是指某个特定时间点的精度，或平均终点精度)
                # EPA_: 可能是另一种形式的EPA或详细的EPA数据
                md, EPA, EPA_ = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md) # 存储原始匹配数据 (虽然后续可能没怎么用)
                metrics[f'{class_name}_EPA'] = EPA_ # 存储EPA指标

        # -----------------------------------
        # 步骤2: 从累积的数据中计算最终指标。
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        for class_name in self.cfg.class_names: # 遍历每个评估类别
            # 计算TP指标 (minADE, minFDE, MissRate)
            for metric_name in MOTION_TP_METRICS: # 遍历定义的运动TP指标
                # metric_data_list中存储了每个(类别,距离阈值)下的匹配信息和误差
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)] # 使用配置中定义的TP指标距离阈值
                # calc_tp函数 (来自nuscenes.eval.detection.algo) 被用于计算这些基于误差的指标
                # 注意：原始calc_tp是为检测的TP指标设计的，这里可能被重载或其内部逻辑能适应误差值的计算
                tp_metric_value = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics[f'{class_name}_{metric_name}']  = tp_metric_value # 存储计算得到的指标

        return metrics, metric_data_list # 返回计算的指标和原始数据列表

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None: # 渲染图表
        """
        Renders various PR and TP curves. # 渲染各种PR（精确率-召回率）和TP（真阳性）曲线。
        :param metrics: DetectionMetrics instance. # DetectionMetrics实例。
        :param md_list: DetectionMetricDataList instance. # DetectionMetricDataList实例。
        """
        # 此渲染函数基本继承自NuScenes的检测评估，可能不完全适用于运动预测指标的特性。
        # 例如，运动预测通常不直接用P-R曲线评估。
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name): # 定义保存图表的路径辅助函数
            return os.path.join(self.plot_dir, name + '.pdf')

        # 绘制总结图 (通常包含mAP和各种TP误差的雷达图)
        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names: # 遍历每个类别
            # 绘制该类别的P-R曲线
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))
            # 绘制该类别的TP指标曲线 (例如，不同误差类型随召回率的变化)
            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        # 绘制不同距离阈值下的P-R曲线
        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self, # 主执行函数，整合评估流程
             plot_examples: int = 0, # 要绘制并保存的可视化样本数量
             render_curves: bool = True) -> Dict[str, Any]: # 是否渲染并保存评估曲线
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        # 主函数，加载评估代码，可视化样本，运行评估并渲染统计图表。
        :param plot_examples: How many example visualizations to write to disk. # 要写入磁盘的可视化样本数量。
        :param render_curves: Whether to render PR and TP curves to disk. # 是否将PR和TP曲线渲染到磁盘。
        :return: A dict that stores the high-level metrics and meta data. # 一个存储高级指标和元数据的字典。
        """
        if plot_examples > 0: # 如果需要绘制示例
            # 选择一个随机但固定的子集进行绘制，以保证可复现性
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # 可视化样本
            example_dir = os.path.join(self.output_dir, 'examples') # 定义示例保存目录
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens: # 遍历选定的样本token
                # 调用NuScenes的visualize_sample函数进行可视化
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(), # 测试集不渲染GT
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()), # 使用配置中最大的类别评估范围
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # 运行评估
        metrics, metric_data_list = self.evaluate()

        # # 将指标保存到JSON文件 (原始检测评估代码中通常有此步骤)
        # metrics_summary = metrics.serialize()
        # metrics_summary['meta'] = self.meta.copy()
        # with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
        #     json.dump(metrics_summary, f, indent=2)
        # 当前实现直接返回 metrics 字典，保存逻辑可能在调用此main函数之后处理。

        # if render_curves: # 如果需要渲染曲线
        #     self.render(metrics, metric_data_list) # 调用render方法 (注意：metrics类型可能不完全匹配DetectionMetrics)

        return metrics # 返回计算得到的指标字典

class NuScenesEval(MotionEval): # 定义一个别名类，保持与旧代码或检测评估的兼容性
    """
    Dummy class for backward-compatibility. Same as MotionEval. # 用于向后兼容的虚拟类。与MotionEval相同。
    """

if __name__ == "__main__": # 如果作为主脚本运行

    # 默认设置和命令行参数解析
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path) # 展开用户路径 (例如 ~)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '': # 如果未提供配置文件路径
        cfg_ = config_factory('detection_cvpr_2019') # 使用默认的CVPR 2019检测配置
    else: # 否则从JSON文件加载配置
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_) # 初始化NuScenes API
    # 注意：这里实例化的是 DetectionEval，而不是上面定义的 MotionEval 或 NuScenesEval (MotionEval的别名)
    # 这可能是一个笔误，或者这个脚本的这个部分是用于运行标准检测评估的。
    # 如果要运行此文件中的MotionEval，应替换为:
    # nusc_eval = MotionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
    # output_dir=output_dir_, verbose=verbose_)
    # 或者如果NuScenesEval确实是MotionEval的别名，那么：
    # nusc_eval = NuScenesEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
    # output_dir=output_dir_, verbose=verbose_)
    # 但DetectionEval是nuscenes.eval.detection.evaluate.DetectionEval，与此文件的MotionEval不同。
    # 假设这里是想要运行此文件定义的评估：
    print("注意: 命令行执行部分可能需要调整以使用本文件定义的 MotionEval/NuScenesEval 类。")
    print("当前配置将尝试运行标准的NuScenes DetectionEval (如果DetectionEval被正确导入)。")

    # 以下是运行标准DetectionEval的示例代码，如果上面导入了DetectionEval的话
    from nuscenes.eval.detection.evaluate import DetectionEval # 需要确保这个导入存在
    nusc_eval_instance = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_)
    nusc_eval_instance.main(plot_examples=plot_examples_, render_curves=render_curves_)

    # 如果要运行此文件中定义的MotionEval:
    # motion_cfg = config_factory('detection_cvpr_2019') # MotionEval也使用DetectionConfig的结构
    # # 可能需要根据motion_eval的需求调整motion_cfg的参数，例如class_names, dist_ths等
    # motion_eval_instance = MotionEval(nusc_, config=motion_cfg, result_path=result_path_, eval_set=eval_set_,
    #                                   output_dir=output_dir_, verbose=verbose_, seconds=12) # seconds对应fut_ts
    # metrics_results = motion_eval_instance.main(plot_examples=plot_examples_, render_curves=False) # render_curves对motion意义不大
    # print("Motion Evaluation Metrics:", metrics_results)
    # # 保存指标到文件
    # metrics_summary_path = os.path.join(output_dir_, 'motion_metrics_summary.json')
    # with open(metrics_summary_path, 'w') as f:
    #     json.dump(metrics_results, f, indent=2)
    # print(f"Motion metrics saved to {metrics_summary_path}")
