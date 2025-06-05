import copy  # 导入 copy 模块，用于对象拷贝
import platform  # 导入 platform 模块，用于获取系统信息
import random  # 导入 random 模块，用于生成随机数
from functools import partial  # 从 functools 模块导入 partial，用于偏函数应用

import numpy as np  # 导入 numpy 模块，用于科学计算
from mmcv.parallel import collate  # 从 mmcv.parallel 模块导入 collate 函数，用于数据整合
from mmcv.runner import get_dist_info  # 从 mmcv.runner 模块导入 get_dist_info 函数，获取分布式训练信息
from mmcv.utils import Registry, build_from_cfg  # 从 mmcv.utils 模块导入 Registry 和 build_from_cfg，用于注册和从配置构建对象
from torch.utils.data import DataLoader  # 从 torch.utils.data 模块导入 DataLoader，PyTorch 数据加载器

from mmdet.datasets.samplers import GroupSampler  # 从 mmdet.datasets.samplers 模块导入 GroupSampler
from projects.mmdet3d_plugin.datasets.samplers import (  # 从项目特定的采样器模块导入
    GroupInBatchSampler,  # 分组批内采样器
    DistributedGroupSampler,  # 分布式分组采样器
    DistributedSampler,  # 分布式采样器
    build_sampler  # 构建采样器的函数
)


def build_dataloader(  # 定义构建数据加载器的函数
    dataset,  # 数据集对象
    samples_per_gpu,  # 每张 GPU 的样本数
    workers_per_gpu,  # 每张 GPU 的工作线程数
    num_gpus=1,  # GPU 数量，默认为1
    dist=True,  # 是否进行分布式训练，默认为 True
    shuffle=True,  # 是否在每个 epoch 打乱数据，默认为 True
    seed=None,  # 随机种子，默认为 None
    shuffler_sampler=None,  # 打乱数据的采样器配置，默认为 None
    nonshuffler_sampler=None,  # 不打乱数据的采样器配置，默认为 None
    runner_type="EpochBasedRunner",  # 执行器类型，默认为 "EpochBasedRunner"
    **kwargs  # 其他 DataLoader 初始化参数
):
    """Build PyTorch DataLoader.  # 构建 PyTorch DataLoader。
    In distributed training, each GPU/process has a dataloader.  # 在分布式训练中，每个 GPU/进程都有一个数据加载器。
    In non-distributed training, there is only one dataloader for all GPUs.  # 在非分布式训练中，所有 GPU 只有一个数据加载器。
    Args:  # 参数说明
        dataset (Dataset): A PyTorch dataset.  # PyTorch 数据集。
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,  # 每张 GPU 上的训练样本数，即
            batch size of each GPU.  # 每张 GPU 的批量大小。
        workers_per_gpu (int): How many subprocesses to use for data loading  # 每张 GPU 用于数据加载的子进程数。
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.  # GPU 数量。仅在非分布式训练中使用。
        dist (bool): Distributed training/test or not. Default: True.  # 是否进行分布式训练/测试。默认为 True。
        shuffle (bool): Whether to shuffle the data at every epoch.  # 是否在每个 epoch 打乱数据。
            Default: True.  # 默认为 True。
        kwargs: any keyword argument to be used to initialize DataLoader  # 用于初始化 DataLoader 的任何关键字参数。
    Returns:  # 返回值说明
        DataLoader: A PyTorch dataloader.  # PyTorch 数据加载器。
    """
    rank, world_size = get_dist_info()  # 获取分布式训练的 rank 和 world_size
    batch_sampler = None  # 初始化批采样器为 None
    if runner_type == 'IterBasedRunner':  # 如果执行器类型是 'IterBasedRunner'
        print("Use GroupInBatchSampler !!!")  # 打印提示信息
        batch_sampler = GroupInBatchSampler(  # 使用 GroupInBatchSampler
            dataset,
            samples_per_gpu,
            world_size,
            rank,
            seed=seed,
        )
        batch_size = 1  # 迭代式运行器通常将 batch_size 设置为1，由 batch_sampler 控制实际批次
        sampler = None  # 主采样器设为 None
        num_workers = workers_per_gpu  # 工作线程数
    elif dist:  # 如果是分布式训练
        # DistributedGroupSampler will definitely shuffle the data to satisfy  # DistributedGroupSampler 肯定会打乱数据以满足
        # that images on each GPU are in the same group  # 每张 GPU 上的图像在同一组内
        if shuffle:  # 如果需要打乱数据
            print("Use DistributedGroupSampler !!!")  # 打印提示信息
            sampler = build_sampler(  # 构建采样器
                shuffler_sampler  # 如果提供了打乱数据的采样器配置则使用
                if shuffler_sampler is not None
                else dict(type="DistributedGroupSampler"),  # 否则默认使用 DistributedGroupSampler
                dict(  # 采样器参数
                    dataset=dataset,
                    samples_per_gpu=samples_per_gpu,
                    num_replicas=world_size,  # 副本数等于 world_size
                    rank=rank,  # 当前进程的 rank
                    seed=seed,  # 随机种子
                ),
            )
        else:  # 如果不需要打乱数据
            sampler = build_sampler(  # 构建采样器
                nonshuffler_sampler  # 如果提供了不打乱数据的采样器配置则使用
                if nonshuffler_sampler is not None
                else dict(type="DistributedSampler"),  # 否则默认使用 DistributedSampler
                dict(  # 采样器参数
                    dataset=dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,  # 传递 shuffle 标志 (此时为 False)
                    seed=seed,
                ),
            )

        batch_size = samples_per_gpu  # 批量大小等于每张 GPU 的样本数
        num_workers = workers_per_gpu  # 工作线程数
    else:  # 如果不是分布式训练 (单机模式)
        # assert False, 'not support in bevformer'  # 断言，表示在 bevformer 中可能不支持此模式 (已注释掉)
        print("WARNING!!!!, Only can be used for obtain inference speed!!!!")  # 打印警告信息，此模式可能仅用于获取推理速度
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None  # 如果需要打乱则使用 GroupSampler，否则为 None
        batch_size = num_gpus * samples_per_gpu  # 批量大小为 GPU 数量乘以每张 GPU 的样本数
        num_workers = num_gpus * workers_per_gpu  # 工作线程数为 GPU 数量乘以每张 GPU 的工作线程数

    init_fn = (  # 定义工作线程初始化函数
        partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)  # 使用偏函数固定参数
        if seed is not None  # 如果提供了随机种子
        else None  # 否则为 None
    )

    data_loader = DataLoader(  # 创建 DataLoader 对象
        dataset,
        batch_size=batch_size,  # 批量大小
        sampler=sampler,  # 主采样器
        batch_sampler=batch_sampler,  # 批采样器 (如果 IterBasedRunner)
        num_workers=num_workers,  # 工作线程数
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),  # 数据整合函数，使用偏函数固定 samples_per_gpu
        pin_memory=False,  # 是否将数据加载到 CUDA 固定内存，默认为 False
        worker_init_fn=init_fn,  # 工作线程初始化函数
        **kwargs  # 其他参数
    )

    return data_loader  # 返回创建的数据加载器


def worker_init_fn(worker_id, num_workers, rank, seed):  # 定义工作线程初始化函数
    # The seed of each worker equals to  # 每个工作线程的种子等于
    # num_worker * rank + worker_id + user_seed  # num_worker * rank + worker_id + 用户种子
    worker_seed = num_workers * rank + worker_id + seed  # 计算工作线程的随机种子
    np.random.seed(worker_seed)  # 设置 numpy 的随机种子
    random.seed(worker_seed)  # 设置 random 的随机种子


# Copyright (c) OpenMMLab. All rights reserved.  # 版权声明
import platform  # 再次导入 platform (通常情况下，重复导入是不必要的)
from mmcv.utils import Registry, build_from_cfg  # 再次导入 Registry, build_from_cfg

from mmdet.datasets import DATASETS  # 从 mmdet.datasets 导入 DATASETS 注册表
from mmdet.datasets.builder import _concat_dataset  # 从 mmdet.datasets.builder 导入 _concat_dataset 内部函数

if platform.system() != "Windows":  # 如果当前系统不是 Windows
    # https://github.com/pytorch/pytorch/issues/973  # 引用 PyTorch 关于文件描述符限制的 issue
    import resource  # 导入 resource 模块，用于系统资源管理

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)  # 获取当前进程的文件描述符限制
    base_soft_limit = rlimit[0]  # 软限制
    hard_limit = rlimit[1]  # 硬限制
    soft_limit = min(max(4096, base_soft_limit), hard_limit)  # 计算新的软限制，确保至少为 4096，但不超过硬限制
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))  # 设置新的文件描述符限制

OBJECTSAMPLERS = Registry("Object sampler")  # 创建名为 "Object sampler" 的注册表，用于对象采样器


def custom_build_dataset(cfg, default_args=None):  # 定义自定义构建数据集的函数
    try:
        from mmdet3d.datasets.dataset_wrappers import CBGSDataset  # 尝试从 mmdet3d 导入 CBGSDataset
    except:
        CBGSDataset = None  # 如果导入失败，则 CBGSDataset 为 None
    from mmdet.datasets.dataset_wrappers import (  # 从 mmdet.datasets.dataset_wrappers 导入
        ClassBalancedDataset,  # 类别平衡数据集包装器
        ConcatDataset,  # 数据集连接包装器
        RepeatDataset,  # 数据集重复包装器
    )

    if isinstance(cfg, (list, tuple)):  # 如果配置是列表或元组
        dataset = ConcatDataset(  # 创建 ConcatDataset
            [custom_build_dataset(c, default_args) for c in cfg]  # 递归构建并连接列表中的每个数据集
        )
    elif cfg["type"] == "ConcatDataset":  # 如果配置类型是 "ConcatDataset"
        dataset = ConcatDataset(  # 创建 ConcatDataset
            [custom_build_dataset(c, default_args) for c in cfg["datasets"]],  # 递归构建并连接 "datasets" 列表中的每个数据集
            cfg.get("separate_eval", True),  # 获取 "separate_eval" 参数，默认为 True
        )
    elif cfg["type"] == "RepeatDataset":  # 如果配置类型是 "RepeatDataset"
        dataset = RepeatDataset(  # 创建 RepeatDataset
            custom_build_dataset(cfg["dataset"], default_args), cfg["times"]  # 递归构建内部数据集并重复指定次数
        )
    elif cfg["type"] == "ClassBalancedDataset":  # 如果配置类型是 "ClassBalancedDataset"
        dataset = ClassBalancedDataset(  # 创建 ClassBalancedDataset
            custom_build_dataset(cfg["dataset"], default_args),  # 递归构建内部数据集
            cfg["oversample_thr"],  # 过采样阈值
        )
    elif cfg["type"] == "CBGSDataset" and CBGSDataset is not None:  # 如果配置类型是 "CBGSDataset" 且 CBGSDataset 已成功导入
        dataset = CBGSDataset(  # 创建 CBGSDataset
            custom_build_dataset(cfg["dataset"], default_args)  # 递归构建内部数据集
        )
    elif isinstance(cfg.get("ann_file"), (list, tuple)):  # 如果 "ann_file" 是列表或元组 (表示多个标注文件)
        dataset = _concat_dataset(cfg, default_args)  # 使用 mmdet 内部的 _concat_dataset 函数处理
    else:  # 其他情况，假定为单个数据集配置
        dataset = build_from_cfg(cfg, DATASETS, default_args)  # 使用 DATASETS 注册表从配置构建数据集

    return dataset  # 返回构建的数据集
