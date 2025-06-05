from inspect import signature  # 导入inspect模块，用于获取函数签名等信息

import torch  # 导入PyTorch库

from mmcv.runner import force_fp32, auto_fp16  # 从MMCV导入强制fp32和自动fp16的装饰器
from mmcv.utils import build_from_cfg  # 从MMCV导入从配置字典构建对象的函数
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS  # 从MMCV导入插件层注册表
from mmdet.models import (  # 从MMDetection导入相关模块
    DETECTORS,  # 检测器注册表
    BaseDetector,  # 检测器基类
    build_backbone,  # 构建骨干网络的函数
    build_head,  # 构建检测头的函数
    build_neck,  # 构建颈部网络的函数
)
from .grid_mask import GridMask  # 从当前目录的grid_mask模块导入GridMask数据增强类

try:
    from ..ops import feature_maps_format  # 尝试从相对路径导入自定义的feature_maps_format操作
    DAF_VALID = True  # 如果成功，则标记DAF（Deformable Aggregation Function）有效
except:
    DAF_VALID = False # 如果导入失败，则标记DAF无效

__all__ = ["SparseDrive"]  # 定义当使用 from .sparsedrive import * 时，"SparseDrive" 会被导入


@DETECTORS.register_module()  # 将SparseDrive类注册到MMDetection的DETECTORS注册表中
class SparseDrive(BaseDetector):  # 定义SparseDrive类，继承自MMDetection的BaseDetector
    """
    SparseDrive模型，一个用于处理（可能是多视图）图像输入的3D检测器。
    它集成了图像骨干网络、可选的颈部网络、检测头，并支持GridMask数据增强
    以及一个可选的深度预测分支。
    """
    def __init__(
        self,
        img_backbone,  # 图像骨干网络配置
        head,  # 检测头配置
        img_neck=None,  # 图像颈部网络配置 (可选)
        init_cfg=None,  # 初始化配置 (用于BaseDetector)
        train_cfg=None,  # 训练配置 (未使用在此代码片段中)
        test_cfg=None,  # 测试配置 (未使用在此代码片段中)
        pretrained=None,  # 预训练权重路径 (如果提供，会设置给骨干网络)
        use_grid_mask=True,  # 是否使用GridMask数据增强
        use_deformable_func=False,  # 是否使用自定义的可变形聚合函数 (DAF)
        depth_branch=None,  # 深度预测分支配置 (可选)
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg)  # 调用父类BaseDetector的初始化函数
        if pretrained is not None:  # 如果指定了预训练权重
            img_backbone.pretrained = pretrained  # 将预训练权重设置给图像骨干网络配置

        self.img_backbone = build_backbone(img_backbone)  # 构建图像骨干网络
        if img_neck is not None:  # 如果配置了图像颈部网络
            self.img_neck = build_neck(img_neck)  # 构建图像颈部网络
        else:
            self.img_neck = None

        self.head = build_head(head)  # 构建检测头
        self.use_grid_mask = use_grid_mask  # 是否使用GridMask

        if use_deformable_func:  # 如果使用自定义DAF
            assert DAF_VALID, "deformable_aggregation needs to be set up." # 确保DAF已成功导入
        self.use_deformable_func = use_deformable_func

        if depth_branch is not None:  # 如果配置了深度预测分支
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)  # 从配置构建深度分支
        else:
            self.depth_branch = None

        if use_grid_mask:  # 如果使用GridMask
            self.grid_mask = GridMask(  # 初始化GridMask模块
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            ) 

    @auto_fp16(apply_to=("img",), out_fp32=True)  # 自动处理输入'img'的fp16转换，输出保持fp32
    def extract_feat(self, img, return_depth=False, metas=None): # 提取特征函数
        """
        从输入图像中提取特征。

        Args:
            img (torch.Tensor): 输入图像张量。形状可以是 (bs, C, H, W) 或 (bs, num_cams, C, H, W)。
            return_depth (bool): 是否返回深度预测结果。默认为False。
            metas (dict, optional): 包含元信息的字典，可能包含相机参数等。

        Returns:
            tuple or torch.Tensor: 如果return_depth为True，返回 (feature_maps, depths)。
                                   否则，只返回 feature_maps。
        """
        bs = img.shape[0]  # 批量大小
        if img.dim() == 5:  # 如果输入图像维度为5，表示是多视图图像 (bs, num_cams, C, H, W)
            num_cams = img.shape[1]  # 获取相机数量
            img = img.flatten(end_dim=1)  # 将bs和num_cams维度合并，变为 (bs*num_cams, C, H, W) 以适应骨干网络输入
        else:
            num_cams = 1  # 单视图图像

        if self.use_grid_mask and self.training: # 如果使用GridMask且在训练模式
            img = self.grid_mask(img)  # 应用GridMask

        # 检查骨干网络的forward方法是否接受'metas'参数
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams=num_cams, metas=metas) # 传递num_cams和metas
        else: # 兼容旧的骨干网络接口
            feature_maps = self.img_backbone(img) # 提取骨干特征

        if self.img_neck is not None:  # 如果存在颈部网络
            feature_maps = list(self.img_neck(feature_maps))  # 通过颈部网络处理特征

        # 将提取的特征图谱调整回 (bs, num_cams, C', H', W') 的形状
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )

        depths = None # 初始化深度为None
        if return_depth and self.depth_branch is not None: # 如果需要返回深度且深度分支存在
            # 使用深度分支预测深度，可能需要焦距信息 (从metas中获取)
            depths = self.depth_branch(feature_maps, metas.get("focal"))

        if self.use_deformable_func: # 如果使用自定义DAF
            feature_maps = feature_maps_format(feature_maps) # 对特征图谱进行特定格式化

        if return_depth: # 如果需要返回深度
            return feature_maps, depths # 返回特征图谱和深度
        return feature_maps # 否则只返回特征图谱

    @force_fp32(apply_to=("img",))  # 强制forward方法中的'img'输入为fp32
    def forward(self, img, **data): # 模型的前向传播入口
        """
        模型的前向传播函数。根据是否处于训练模式，调用不同的处理流程。

        Args:
            img (torch.Tensor): 输入图像。
            **data: 其他数据，如标签、元信息等，通常以字典形式传递。

        Returns:
            dict or list: 训练时返回损失字典，测试时返回检测结果列表。
        """
        if self.training:  # 如果是训练模式
            return self.forward_train(img, **data)  # 调用训练前向传播
        else:  # 如果是测试/评估模式
            return self.forward_test(img, **data)  # 调用测试前向传播

    def forward_train(self, img, **data): # 训练模式的前向传播
        """
        Args:
            img (torch.Tensor): 输入图像。
            **data: 包含gt_labels, gt_bboxes, gt_depths, metas等训练所需数据。

        Returns:
            dict: 包含各项损失的字典。
        """
        # 提取特征和深度（如果深度分支存在）
        feature_maps, depths = self.extract_feat(img, return_depth=True, metas=data)
        # 将特征图谱和数据传递给检测头进行预测
        model_outs = self.head(feature_maps, data)
        # 计算检测头的损失
        output = self.head.loss(model_outs, data)

        if depths is not None and "gt_depth" in data: # 如果预测了深度且存在真实深度标签
            # 计算并添加密集深度损失
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output # 返回损失字典

    def forward_test(self, img, **data): # 测试模式的前向传播
        """
        Args:
            img (torch.Tensor or list[torch.Tensor]): 输入图像。如果是列表，则假定为测试时数据增强的不同版本。
            **data: 包含metas等测试所需数据。

        Returns:
            list[dict]: 检测结果列表，每个字典包含一个样本的检测结果。
        """
        if isinstance(img, list): # 如果img是列表 (通常用于测试时数据增强，如多尺度测试)
            return self.aug_test(img, **data) # 调用aug_test处理
        else: # 单个图像输入
            return self.simple_test(img, **data) # 调用simple_test处理

    def simple_test(self, img, **data): # 简单测试 (单个尺度，无复杂数据增强)
        """
        Args:
            img (torch.Tensor): 输入图像。
            **data: 包含metas等测试所需数据。

        Returns:
            list[dict]: 检测结果列表。
        """
        feature_maps = self.extract_feat(img, metas=data) # 提取特征 (测试时不一定需要深度)
        model_outs = self.head(feature_maps, data) # 通过检测头获取模型输出
        results = self.head.post_process(model_outs, data) # 对模型输出进行后处理得到最终结果
        # 将结果格式化为列表，每个元素是一个包含'img_bbox'键的字典
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data): # 测试时数据增强 (当前实现为伪增强，只取第一个)
        """
        Args:
            img (list[torch.Tensor]): 不同增强版本的图像列表。
            **data: 包含metas等测试所需数据。对于TTA，metas也应该是列表。

        Returns:
            list[dict]: 检测结果列表。
        """
        # fake test time augmentation # 伪测试时数据增强
        # 当前实现仅简单地取列表中的第一个元素进行测试，并未真正融合多次增强的结果
        for key in data.keys(): # 遍历data字典中的所有项
            if isinstance(data[key], list): # 如果某项是列表
                data[key] = data[key][0] # 则只取其第一个元素
        return self.simple_test(img[0], **data) # 调用simple_test处理第一个图像和对应数据
