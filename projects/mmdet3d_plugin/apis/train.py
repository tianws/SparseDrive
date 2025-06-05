# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from .mmdet_train import custom_train_detector  # 从同一目录下的 mmdet_train 模块导入 custom_train_detector 函数
# from mmseg.apis import train_segmentor  # 从 mmseg.apis 模块导入 train_segmentor 函数 (已注释掉)
from mmdet.apis import train_detector  # 从 mmdet.apis 模块导入 train_detector 函数


def custom_train_model(  # 定义自定义训练模型函数
    model,  # 模型对象
    dataset,  # 数据集对象
    cfg,  # 配置字典
    distributed=False,  # 是否进行分布式训练，默认为 False
    validate=False,  # 是否进行验证，默认为 False
    timestamp=None,  # 时间戳，默认为 None
    meta=None,  # 元数据，默认为 None
):
    """A function wrapper for launching model training according to cfg.  # 根据配置启动模型训练的函数包装器。

    Because we need different eval_hook in runner. Should be deprecated in the  # 因为在 runner 中需要不同的 eval_hook。将来应该弃用。
    future.
    """
    if cfg.model.type in ["EncoderDecoder3D"]:  # 检查模型类型是否为 "EncoderDecoder3D"
        assert False  # 如果是，则断言失败 (可能表示尚未支持或需要特定处理)
    else:  # 如果模型类型不是 "EncoderDecoder3D"
        custom_train_detector(  # 调用自定义的检测器训练函数
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta,
        )


def train_model(  # 定义标准训练模型函数
    model,  # 模型对象
    dataset,  # 数据集对象
    cfg,  # 配置字典
    distributed=False,  # 是否进行分布式训练，默认为 False
    validate=False,  # 是否进行验证，默认为 False
    timestamp=None,  # 时间戳，默认为 None
    meta=None,  # 元数据，默认为 None
):
    """A function wrapper for launching model training according to cfg.  # 根据配置启动模型训练的函数包装器。

    Because we need different eval_hook in runner. Should be deprecated in the  # 因为在 runner 中需要不同的 eval_hook。将来应该弃用。
    future.
    """
    train_detector(  # 调用 mmdet 中标准的检测器训练函数
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        meta=meta,
    )
