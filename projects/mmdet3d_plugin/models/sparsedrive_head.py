from typing import List, Optional, Tuple, Union  # 导入类型提示相关的模块
import warnings  # 导入warnings模块，用于发出警告

import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块

from mmcv.runner import BaseModule  # 从MMCV导入BaseModule基类
from mmdet.models import HEADS  # 从MMDetection导入HEADS注册表
from mmdet.models import build_head  # 从MMDetection导入构建检测头的函数


@HEADS.register_module()  # 将SparseDriveHead注册到MMDetection的HEADS注册表中
class SparseDriveHead(BaseModule):  # 定义SparseDriveHead类，继承自BaseModule
    """
    SparseDrive的总检测头/任务头。
    这个头部模块根据 `task_config` 的配置，管理和调用一个或多个特定任务的子头部，
    例如3D目标检测头 (`det_head`)、地图分割/检测头 (`map_head`)，
    以及运动预测和规划头 (`motion_plan_head`)。
    它负责将从骨干网络和颈部网络提取的特征分发给相应的子头部，
    并汇总它们的输出、损失和后处理结果。
    """
    def __init__(
        self,
        task_config: dict,  # 任务配置字典，指定启用哪些子任务 (如检测、地图、运动规划)
        det_head = dict,  # 3D目标检测头的配置字典
        map_head = dict,  # 地图相关任务头的配置字典
        motion_plan_head = dict,  # 运动预测和规划头的配置字典
        init_cfg=None,  # 初始化配置 (用于BaseModule)
        **kwargs,  # 其他关键字参数
    ):
        super(SparseDriveHead, self).__init__(init_cfg)  # 调用父类BaseModule的初始化函数
        self.task_config = task_config  # 存储任务配置

        # 根据任务配置，条件性地构建各个子头部
        if self.task_config.get('with_det', False):  # 如果启用了检测任务
            self.det_head = build_head(det_head)  # 构建检测头
        if self.task_config.get('with_map', False):  # 如果启用了地图任务
            self.map_head = build_head(map_head)  # 构建地图头
        if self.task_config.get('with_motion_plan', False):  # 如果启用了运动规划任务
            self.motion_plan_head = build_head(motion_plan_head)  # 构建运动规划头

    def init_weights(self): # 初始化权重方法
        """
        初始化各个子头部的权重。
        """
        if self.task_config.get('with_det', False):
            self.det_head.init_weights()
        if self.task_config.get('with_map', False):
            self.map_head.init_weights()
        if self.task_config.get('with_motion_plan', False):
            self.motion_plan_head.init_weights()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],  # 输入的特征图谱，可以是单个张量或张量列表
        metas: dict,  # 包含元信息的字典
    ):
        """
        前向传播函数。将特征图谱和元信息传递给启用的各个子头部。

        Args:
            feature_maps (torch.Tensor or list[torch.Tensor]): 从颈部网络输出的特征图谱。
            metas (dict): 包含样本元信息的字典。

        Returns:
            tuple: 包含各个子头部输出的元组 (det_output, map_output, motion_output, planning_output)。
                   如果某个子头部未启用，则对应的输出为None。
        """
        det_output = None  # 初始化检测输出为None
        if self.task_config.get('with_det', False):  # 如果检测任务启用
            det_output = self.det_head(feature_maps, metas)  # 获取检测头输出

        map_output = None  # 初始化地图输出为None
        if self.task_config.get('with_map', False):  # 如果地图任务启用
            map_output = self.map_head(feature_maps, metas)  # 获取地图头输出
        
        motion_output, planning_output = None, None  # 初始化运动和规划输出为None
        if self.task_config.get('with_motion_plan', False):  # 如果运动规划任务启用
            # 运动规划头可能需要其他头的输出或内部组件作为输入
            motion_output, planning_output = self.motion_plan_head(
                det_output,  # 检测头的输出
                map_output,  # 地图头的输出
                feature_maps,  # 原始特征图谱
                metas,  # 元信息
                self.det_head.anchor_encoder,  # 检测头的锚点编码器 (如果运动头需要)
                self.det_head.instance_bank.mask,  # 检测头的实例库掩码 (如果运动头需要)
                self.det_head.instance_bank.anchor_handler,  # 检测头的锚点处理器 (如果运动头需要)
            )

        return det_output, map_output, motion_output, planning_output  # 返回所有头的输出

    def loss(self, model_outs, data): # 计算损失函数
        """
        计算所有启用子头部的损失。

        Args:
            model_outs (tuple): 包含各个子头部前向传播输出的元组。
            data (dict): 包含真实标签和元信息的数据字典。

        Returns:
            dict: 包含所有子头部损失项的字典。
        """
        det_output, map_output, motion_output, planning_output = model_outs  # 解包模型输出
        losses = dict()  # 初始化损失字典

        if self.task_config.get('with_det', False) and det_output is not None:  # 如果检测任务启用且有输出
            loss_det = self.det_head.loss(det_output, data)  # 计算检测损失
            losses.update(loss_det)  # 更新到总损失字典
        
        if self.task_config.get('with_map', False) and map_output is not None:  # 如果地图任务启用且有输出
            loss_map = self.map_head.loss(map_output, data)  # 计算地图损失
            losses.update(loss_map)  # 更新到总损失字典

        if self.task_config.get('with_motion_plan', False) and motion_output is not None:  # 如果运动规划任务启用且有输出
            # 运动损失计算可能需要一些来自检测头的信息 (例如匹配的索引)
            motion_loss_cache = dict(
                indices=self.det_head.sampler.indices if hasattr(self.det_head, 'sampler') else None,
            )
            loss_motion = self.motion_plan_head.loss(
                motion_output, 
                planning_output, 
                data, 
                motion_loss_cache # 传递缓存的额外信息
            )
            losses.update(loss_motion)  # 更新到总损失字典

        return losses  # 返回总损失字典

    def post_process(self, model_outs, data): # 后处理模型输出以生成最终结果
        """
        对所有启用子头部的输出进行后处理，以生成最终的预测结果。

        Args:
            model_outs (tuple): 包含各个子头部前向传播输出的元组。
            data (dict): 包含元信息的数据字典（主要用于测试时）。

        Returns:
            list[dict]: 结果列表，每个字典包含一个样本的所有任务的预测结果。
        """
        det_output, map_output, motion_output, planning_output = model_outs # 解包模型输出

        batch_size = 0 # 初始化批量大小
        # 获取批量大小，优先从检测结果确定，然后是地图结果
        if self.task_config.get('with_det', False) and det_output is not None:
            # 假设det_head.post_process返回一个列表，其长度为batch_size
            det_result_temp_for_bs = self.det_head.post_process(det_output)
            batch_size = len(det_result_temp_for_bs)
        elif self.task_config.get('with_map', False) and map_output is not None:
            map_result_temp_for_bs = self.map_head.post_process(map_output)
            batch_size = len(map_result_temp_for_bs)
        elif self.task_config.get('with_motion_plan', False) and motion_output is not None:
            # 假设motion_plan_head.post_process返回两个列表，其长度为batch_size
             _, planning_result_temp_for_bs = self.motion_plan_head.post_process(det_output, motion_output, planning_output, data)
             batch_size = len(planning_result_temp_for_bs)
        
        if batch_size == 0 and 'img_metas' in data: # 如果无法从输出推断bs，尝试从元数据获取
             batch_size = len(data['img_metas'])
        if batch_size == 0: # 如果仍然为0，可能是一个问题或空输入
            warnings.warn("Batch size could not be determined in post_process.")
            return []


        results = [dict() for _ in range(batch_size)] # 初始化结果列表，每个元素是一个空字典

        if self.task_config.get('with_det', False) and det_output is not None:
            det_result = self.det_head.post_process(det_output) # 检测后处理
            for i in range(batch_size):
                results[i].update(det_result[i])
        
        if self.task_config.get('with_map', False) and map_output is not None:
            map_result= self.map_head.post_process(map_output) # 地图后处理
            for i in range(batch_size):
                results[i].update(map_result[i])

        if self.task_config.get('with_motion_plan', False) and motion_output is not None:
            # 运动规划后处理
            motion_result, planning_result = self.motion_plan_head.post_process(
                det_output, # 可能需要原始检测输出
                motion_output, 
                planning_output,
                data, # 可能需要元数据
            )
            for i in range(batch_size):
                results[i].update(motion_result[i])
                results[i].update(planning_result[i])

        return results # 返回格式化后的结果列表
