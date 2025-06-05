# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import random  # 导入 random 模块，用于生成随机数
import warnings  # 导入 warnings 模块，用于处理警告信息

import numpy as np  # 导入 numpy 模块，用于进行科学计算
import torch  # 导入 torch 模块，PyTorch 深度学习框架
import torch.distributed as dist  # 导入 torch.distributed 模块，用于分布式训练
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel  # 从 mmcv.parallel 模块导入 MMDataParallel 和 MMDistributedDataParallel，用于数据并行和分布式数据并行
from mmcv.runner import (  # 从 mmcv.runner 模块导入所需类和函数
    HOOKS,  # HOOKS 注册表，用于管理钩子函数
    DistSamplerSeedHook,  # DistSamplerSeedHook 类，用于在分布式训练中设置采样器的随机种子
    EpochBasedRunner,  # EpochBasedRunner 类，基于 epoch 的执行器
    Fp16OptimizerHook,  # Fp16OptimizerHook 类，用于混合精度训练的优化器钩子
    OptimizerHook,  # OptimizerHook 类，优化器钩子
    build_optimizer,  # build_optimizer 函数，用于构建优化器
    build_runner,  # build_runner 函数，用于构建执行器
    get_dist_info,  # get_dist_info 函数，用于获取分布式训练信息
)
from mmcv.utils import build_from_cfg  # 从 mmcv.utils 模块导入 build_from_cfg 函数，用于从配置字典构建对象

from mmdet.core import EvalHook  # 从 mmdet.core 模块导入 EvalHook 类，评估钩子

from mmdet.datasets import build_dataset, replace_ImageToTensor  # 从 mmdet.datasets 模块导入 build_dataset 和 replace_ImageToTensor 函数，用于构建数据集和替换数据处理流程中的 ImageToTensor
from mmdet.utils import get_root_logger  # 从 mmdet.utils 模块导入 get_root_logger 函数，用于获取根日志记录器
import time  # 导入 time 模块，用于处理时间相关操作
import os.path as osp  # 导入 os.path 模块，并重命名为 osp，用于处理文件路径
from projects.mmdet3d_plugin.datasets.builder import build_dataloader  # 从项目中导入 build_dataloader 函数，用于构建数据加载器
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import (  # 从项目中导入 CustomDistEvalHook 类
    CustomDistEvalHook,
)
from projects.mmdet3d_plugin.datasets import custom_build_dataset  # 从项目中导入 custom_build_dataset 函数，用于自定义构建数据集


def custom_train_detector(  # 定义自定义训练检测器函数
    model,  # 模型对象
    dataset,  # 数据集对象或列表
    cfg,  # 配置字典
    distributed=False,  # 是否进行分布式训练，默认为 False
    validate=False,  # 是否进行验证，默认为 False
    timestamp=None,  # 时间戳，默认为 None
    meta=None,  # 元数据，默认为 None
):
    logger = get_root_logger(cfg.log_level)  # 获取根日志记录器

    # prepare data loaders  # 准备数据加载器

    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]  #确保数据集是列表或元组类型
    # assert len(dataset)==1s  # 断言数据集长度为1（已注释掉）
    if "imgs_per_gpu" in cfg.data:  # 检查配置中是否包含 "imgs_per_gpu"
        logger.warning(  # 记录警告信息
            '"imgs_per_gpu" is deprecated in MMDet V2.0. '  # "imgs_per_gpu" 在 MMDet V2.0 中已弃用
            'Please use "samples_per_gpu" instead'  # 请改用 "samples_per_gpu"
        )
        if "samples_per_gpu" in cfg.data:  # 检查配置中是否包含 "samples_per_gpu"
            logger.warning(  # 记录警告信息
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '  # 获取到 "imgs_per_gpu" 和 "samples_per_gpu"
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f"={cfg.data.imgs_per_gpu} is used in this experiments"  # 在本次实验中使用 "imgs_per_gpu"
            )
        else:
            logger.warning(  # 记录警告信息
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='  # 自动设置 "samples_per_gpu" 等于 "imgs_per_gpu"
                f"{cfg.data.imgs_per_gpu} in this experiments"
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu  # 将 "samples_per_gpu" 设置为 "imgs_per_gpu" 的值

    if "runner" in cfg:  # 检查配置中是否包含 "runner"
        runner_type = cfg.runner["type"]  # 获取执行器类型
    else:
        runner_type = "EpochBasedRunner"  # 默认为 "EpochBasedRunner"
    data_loaders = [  # 构建数据加载器列表
        build_dataloader(  # 调用 build_dataloader 函数构建数据加载器
            ds,  # 数据集
            cfg.data.samples_per_gpu,  # 每张 GPU 的样本数
            cfg.data.workers_per_gpu,  # 每张 GPU 的工作线程数
            # cfg.gpus will be ignored if distributed  # 如果是分布式训练，cfg.gpus 将被忽略
            len(cfg.gpu_ids),  # GPU ID 数量
            dist=distributed,  # 是否分布式训练
            seed=cfg.seed,  # 随机种子
            nonshuffler_sampler=dict(  # 非打乱采样器配置
                type="DistributedSampler"
            ),  # dict(type='DistributedSampler'),  # 分布式采样器
            runner_type=runner_type,  # 执行器类型
        )
        for ds in dataset  # 遍历数据集列表
    ]

    # put model on gpus  # 将模型放到 GPU 上
    if distributed:  # 如果是分布式训练
        find_unused_parameters = cfg.get("find_unused_parameters", False)  # 获取是否查找未使用的参数，默认为 False
        # Sets the `find_unused_parameters` parameter in  # 设置 torch.nn.parallel.DistributedDataParallel 中的 `find_unused_parameters` 参数
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(  # 使用 MMDistributedDataParallel 包装模型
            model.cuda(),  # 将模型移至 GPU
            device_ids=[torch.cuda.current_device()],  # 设备 ID 列表
            broadcast_buffers=False,  # 是否广播缓冲区
            find_unused_parameters=find_unused_parameters,  # 是否查找未使用的参数
        )

    else:  # 如果不是分布式训练
        model = MMDataParallel(  # 使用 MMDataParallel 包装模型
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids  # 将模型移至指定 GPU
        )

    # build runner  # 构建执行器
    optimizer = build_optimizer(model, cfg.optimizer)  # 构建优化器

    if "runner" not in cfg:  # 检查配置中是否不包含 "runner"
        cfg.runner = {  # 设置默认的执行器配置
            "type": "EpochBasedRunner",  # 执行器类型
            "max_epochs": cfg.total_epochs,  # 最大 epoch 数
        }
        warnings.warn(  # 发出用户警告
            "config is now expected to have a `runner` section, "  # 配置文件现在应该包含 `runner` 部分
            "please set `runner` in your config.",  # 请在配置文件中设置 `runner`
            UserWarning,
        )
    else:
        if "total_epochs" in cfg:  # 检查配置中是否包含 "total_epochs"
            assert cfg.total_epochs == cfg.runner.max_epochs  # 断言总 epoch 数与执行器配置中的最大 epoch 数相等

    runner = build_runner(  # 构建执行器
        cfg.runner,  # 执行器配置
        default_args=dict(  # 默认参数
            model=model,  # 模型
            optimizer=optimizer,  # 优化器
            work_dir=cfg.work_dir,  # 工作目录
            logger=logger,  # 日志记录器
            meta=meta,  # 元数据
        ),
    )

    # an ugly workaround to make .log and .log.json filenames the same  # 一个丑陋的解决方法，使 .log 和 .log.json 文件名相同
    runner.timestamp = timestamp  # 设置执行器的时间戳

    # fp16 setting  # fp16 设置
    fp16_cfg = cfg.get("fp16", None)  # 获取 fp16 配置，默认为 None
    if fp16_cfg is not None:  # 如果存在 fp16 配置
        optimizer_config = Fp16OptimizerHook(  # 创建 Fp16OptimizerHook
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed  # 传递优化器配置、fp16 配置和分布式标志
        )
    elif distributed and "type" not in cfg.optimizer_config:  # 如果是分布式训练且优化器配置中没有 "type"
        optimizer_config = OptimizerHook(**cfg.optimizer_config)  # 创建 OptimizerHook
    else:
        optimizer_config = cfg.optimizer_config  # 使用配置文件中的优化器配置

    # register hooks  # 注册钩子
    runner.register_training_hooks(  # 注册训练钩子
        cfg.lr_config,  # 学习率配置
        optimizer_config,  # 优化器配置
        cfg.checkpoint_config,  # 检查点配置
        cfg.log_config,  # 日志配置
        cfg.get("momentum_config", None),  # 动量配置，默认为 None
    )

    # register profiler hook  # 注册性能分析钩子（已注释掉）
    # trace_config = dict(type='tb_trace', dir_name='work_dir')
    # profiler_config = dict(on_trace_ready=trace_config)
    # runner.register_profiler_hook(profiler_config)

    if distributed:  # 如果是分布式训练
        if isinstance(runner, EpochBasedRunner):  # 如果执行器是 EpochBasedRunner 类型
            runner.register_hook(DistSamplerSeedHook())  # 注册 DistSamplerSeedHook

    # register eval hooks  # 注册评估钩子
    if validate:  # 如果进行验证
        # Support batch_size > 1 in validation  # 支持验证时 batch_size > 1
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)  # 获取验证集的每 GPU 样本数，默认为 1
        if val_samples_per_gpu > 1:  # 如果验证集的每 GPU 样本数大于 1
            assert False  # 断言失败（此处可能需要调整逻辑）
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'  # 将 'ImageToTensor' 替换为 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(  # 替换验证集数据处理流程中的 ImageToTensor
                cfg.data.val.pipeline
            )
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))  # 构建验证数据集

        val_dataloader = build_dataloader(  # 构建验证数据加载器
            val_dataset,  # 验证数据集
            samples_per_gpu=val_samples_per_gpu,  # 每 GPU 样本数
            workers_per_gpu=cfg.data.workers_per_gpu,  # 每 GPU 工作线程数
            dist=distributed,  # 是否分布式训练
            shuffle=False,  # 不打乱数据
            nonshuffler_sampler=dict(type="DistributedSampler"),  # 非打乱采样器配置
        )
        eval_cfg = cfg.get("evaluation", {})  # 获取评估配置，默认为空字典
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"  # 根据执行器类型设置按 epoch 评估还是按迭代评估
        eval_cfg["jsonfile_prefix"] = osp.join(  # 设置评估结果 JSON 文件前缀
            "val",  # 验证目录
            cfg.work_dir,  # 工作目录
            time.ctime().replace(" ", "_").replace(":", "_"),  # 当前时间作为文件名一部分
        )
        eval_hook = CustomDistEvalHook if distributed else EvalHook  # 根据是否分布式训练选择评估钩子类型
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))  # 注册评估钩子

    # user-defined hooks  # 用户自定义钩子
    if cfg.get("custom_hooks", None):  # 如果配置中存在自定义钩子
        custom_hooks = cfg.custom_hooks  # 获取自定义钩子列表
        assert isinstance(  # 断言自定义钩子是列表类型
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:  # 遍历自定义钩子配置
            assert isinstance(hook_cfg, dict), (  # 断言每个钩子配置是字典类型
                "Each item in custom_hooks expects dict type, but got "
                f"{type(hook_cfg)}"
            )
            hook_cfg = hook_cfg.copy()  # 复制钩子配置
            priority = hook_cfg.pop("priority", "NORMAL")  # 获取钩子优先级，默认为 "NORMAL"
            hook = build_from_cfg(hook_cfg, HOOKS)  # 从配置构建钩子对象
            runner.register_hook(hook, priority=priority)  # 注册钩子并设置优先级

    if cfg.resume_from:  # 如果配置了从检查点恢复训练
        runner.resume(cfg.resume_from)  # 从指定检查点恢复训练
    elif cfg.load_from:  # 如果配置了加载预训练模型
        runner.load_checkpoint(cfg.load_from)  # 加载预训练模型权重
    runner.run(data_loaders, cfg.workflow)  # 运行训练流程
