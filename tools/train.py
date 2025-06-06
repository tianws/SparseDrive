# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division # 确保 / 运算符执行真除法 (Python 2兼容性，但在Python 3中是默认行为)
import sys # 导入sys模块，用于访问Python解释器相关的变量和函数
import os # 导入os模块，用于与操作系统交互

print(sys.executable, os.path.abspath(__file__)) # 打印Python解释器路径和当前文件绝对路径
# import init_paths # for conda pkgs submitting method # 注释掉的导入，可能用于特定环境的路径初始化
import argparse # 导入argparse模块，用于解析命令行参数
import copy # 导入copy模块，用于对象的复制
import mmcv # 导入MMCV库
import time # 导入time模块，用于时间相关操作
import torch # 导入PyTorch库
import warnings # 导入warnings模块，用于发出警告
from mmcv import Config, DictAction # 从MMCV导入Config类和DictAction (用于解析字典型命令行参数)
from mmcv.runner import get_dist_info, init_dist # 从MMCV的runner模块导入获取分布式信息和初始化分布式环境的函数
from os import path as osp # 从os.path导入path并重命名为osp，方便路径操作

from mmdet import __version__ as mmdet_version # 导入MMDetection的版本号
from mmdet.apis import train_detector # 从MMDetection的API导入核心的train_detector函数
from mmdet.datasets import build_dataset # 从MMDetection导入构建数据集的函数
from mmdet.models import build_detector # 从MMDetection导入构建检测器模型的函数
from mmdet.utils import collect_env, get_root_logger # 从MMDetection的工具模块导入收集环境信息和获取根日志记录器的函数
from mmdet.apis import set_random_seed # 从MMDetection的API导入设置随机种子的函数
from torch import distributed as dist # 导入PyTorch的分布式模块
from datetime import timedelta # 导入timedelta，用于设置分布式训练的超时时间

import cv2 # 导入OpenCV库

cv2.setNumThreads(8) # 设置OpenCV使用的线程数，避免与PyTorch数据加载线程过多竞争


def parse_args(): # 定义解析命令行参数的函数
    parser = argparse.ArgumentParser(description="Train a detector") # 创建参数解析器
    parser.add_argument("config", help="train config file path") # 必要参数：训练配置文件路径
    parser.add_argument("--work-dir", help="the dir to save logs and models") # 可选参数：保存日志和模型的工作目录
    parser.add_argument(
        "--resume-from", help="the checkpoint file to resume from" # 可选参数：从指定的checkpoint文件恢复训练
    )
    parser.add_argument(
        "--no-validate", # 可选参数：训练过程中不进行验证
        action="store_true", # 指定为开关动作
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group() # 创建一个互斥参数组，用于指定GPU
    group_gpus.add_argument(
        "--gpus", # 可选参数：使用的GPU数量 (仅适用于非分布式训练)
        type=int,
        help="number of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids", # 可选参数：使用的GPU ID列表 (仅适用于非分布式训练)
        type=int,
        nargs="+", # 允许一个或多个参数
        help="ids of gpus to use "
        "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed") # 可选参数：随机种子
    parser.add_argument(
        "--deterministic", # 可选参数：是否为CUDNN后端设置确定性选项，以保证实验可复现性
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options", # 可选参数（已废弃，推荐使用--cfg-options）
        nargs="+",
        action=DictAction, # 将key=value形式的参数解析为字典
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options", # 可选参数：覆盖配置文件中的某些设置
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.", # 详细的帮助信息说明如何覆盖配置项
    )
    parser.add_argument(
        "--dist-url", # 可选参数：分布式初始化的URL
        type=str,
        default="auto", # 默认为auto，通常由启动器自动处理
        help="dist url for init process, such as tcp://localhost:8000",
    )
    parser.add_argument("--gpus-per-machine", type=int, default=8) # 可选参数：每台机器的GPU数量（用于MPI_NCCL启动器）
    parser.add_argument(
        "--launcher", # 可选参数：分布式任务启动器类型
        choices=["none", "pytorch", "slurm", "mpi", "mpi_nccl"], # 可选项
        default="none", # 默认为none（非分布式）
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0) # 可选参数：本地rank（由启动器自动设置）
    parser.add_argument(
        "--autoscale-lr", # 可选参数：是否根据GPU数量自动缩放学习率
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args() # 解析命令行参数
    if "LOCAL_RANK" not in os.environ: # 如果环境变量中没有LOCAL_RANK（通常由pytorch启动器设置）
        os.environ["LOCAL_RANK"] = str(args.local_rank) # 则使用命令行参数中的local_rank（主要用于单节点测试）

    # 处理已废弃的--options参数
    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both specified, "
            "--options is deprecated in favor of --cfg-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options

    return args # 返回解析后的参数对象


def main(): # 主函数
    args = parse_args() # 解析命令行参数

    cfg = Config.fromfile(args.config) # 从指定的配置文件加载配置
    if args.cfg_options is not None: # 如果通过命令行传递了配置覆盖选项
        cfg.merge_from_dict(args.cfg_options) # 则合并这些选项到配置对象中

    # 导入自定义模块 (如果配置中指定了)
    # 这允许用户在不修改mmdetection代码库的情况下添加自定义的backbone, head, dataset等
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    # 导入插件模块 (如果配置中指定了，这是mmdet3d_plugin这类项目常用的方式)
    # 这会动态加载插件目录下的模块，并更新MMCV/MMDet的注册表，使得配置文件可以直接使用插件中定义的模块
    if hasattr(cfg, "plugin"): # 检查配置中是否有'plugin'属性
        if cfg.plugin: # 如果plugin为True
            import importlib
            if hasattr(cfg, "plugin_dir"): # 如果指定了插件目录
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir) # 获取插件目录的父目录路径
                _module_dir = _module_dir.split("/") # 按'/'分割路径
                _module_path = _module_dir[0] # 模块路径的起始部分 (通常是 'projects')
                for m in _module_dir[1:]: # 逐级拼接模块路径
                    _module_path = _module_path + "." + m
                print(f"动态导入插件模块: {_module_path}")
                plg_lib = importlib.import_module(_module_path) # 动态导入插件库
            else: # 如果未指定插件目录，则尝试从配置文件所在目录的相对路径导入
                  # (这种方式在MMDet原始代码中较常见，但对于外部插件可能不太适用)
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(f"动态导入插件模块 (基于config路径): {_module_path}")
                plg_lib = importlib.import_module(_module_path)
            # 特别地，从插件中导入自定义的训练函数 (如果存在)
            from projects.mmdet3d_plugin.apis.train import custom_train_model

    # 设置 cudnn_benchmark，如果输入尺寸固定，可以加速卷积运算
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # 设置工作目录的优先级: 命令行参数 > 配置文件中的设置 > 根据配置文件名自动生成
    if args.work_dir is not None: # 如果通过命令行指定了工作目录
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None: # 如果配置文件中也没有指定
        # 使用配置文件名（不含扩展名）作为默认工作目录的一部分
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.resume_from is not None: # 如果指定了恢复训练的checkpoint路径
        cfg.resume_from = args.resume_from

    # 设置使用的GPU ID
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus) # 默认为1个GPU或指定的GPU数量

    if args.autoscale_lr: # 如果启用学习率自动缩放
        # 根据GPU数量和默认的8卡配置，线性缩放学习率 (参考 "Accurate, Large Minibatch SGD"论文)
        cfg.optimizer["lr"] = cfg.optimizer["lr"] * len(cfg.gpu_ids) / 8

    # 初始化分布式环境
    if args.launcher == "none": # 非分布式训练
        distributed = False
    elif args.launcher == "mpi_nccl": # 使用MPI和NCCL进行分布式训练 (一种特殊的启动方式)
        distributed = True
        import mpi4py.MPI as MPI # 导入mpi4py
        comm = MPI.COMM_WORLD # 获取MPI通信器
        mpi_local_rank = comm.Get_rank() # 获取当前进程在MPI中的rank
        mpi_world_size = comm.Get_size() # 获取MPI的总进程数
        print(f"MPI local_rank={mpi_local_rank}, world_size={mpi_world_size}")

        # 设置CUDA_VISIBLE_DEVICES，确保每个MPI进程使用正确的GPU
        # num_gpus = torch.cuda.device_count() # 这行被注释掉了，可能因为args.gpus_per_machine更可靠
        device_ids_on_machines = list(range(args.gpus_per_machine))
        str_ids = list(map(str, device_ids_on_machines))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_ids)
        torch.cuda.set_device(mpi_local_rank % args.gpus_per_machine) # 每个进程绑定到特定的GPU

        # 初始化PyTorch的分布式进程组
        dist.init_process_group(
            backend="nccl", # 使用NCCL后端
            init_method=args.dist_url, # 初始化URL (例如 tcp://localhost:port)
            world_size=mpi_world_size, # 总进程数
            rank=mpi_local_rank, # 当前进程rank
            timeout=timedelta(seconds=3600), # 超时设置 (1小时)
        )
        cfg.gpu_ids = range(mpi_world_size) # 在MPI模式下，gpu_ids通常是所有参与训练的进程的rank
        print("cfg.gpu_ids (MPI):", cfg.gpu_ids)
    else: # 其他分布式启动器 (如pytorch, slurm)
        distributed = True
        init_dist(args.launcher, timeout=timedelta(seconds=3600), **cfg.dist_params) # 使用MMCV的init_dist初始化
        _, world_size = get_dist_info() # 获取分布式信息
        cfg.gpu_ids = range(world_size) # 设置gpu_ids为所有world_size个进程

    # 创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # 将最终的配置保存到工作目录中，以便复现和记录
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # 初始化日志记录器
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) # 获取当前时间戳
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log") # 定义日志文件名
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level) # 获取根日志记录器

    # 初始化元数据字典，用于记录实验相关信息
    meta = dict()
    # 记录环境信息
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("------------------------------------------------------------")
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text # 将可读的配置文本存入元数据

    # 记录一些基本信息到日志
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # 设置随机种子
    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, deterministic: {args.deterministic}"
        )
        set_random_seed(args.seed, deterministic=args.deterministic) # 设置随机种子
    cfg.seed = args.seed # 将种子也保存到配置中
    meta["seed"] = args.seed # 保存到元数据
    meta["exp_name"] = osp.basename(args.config) # 实验名称通常是配置文件名

    # 构建检测器模型
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.init_weights() # 初始化模型权重
    logger.info(f"Model:\n{model}") # 打印模型结构

    # 将工作目录路径添加到训练和验证集配置中 (某些组件可能需要)
    if hasattr(cfg.data.train, 'work_dir'): cfg.data.train.work_dir = cfg.work_dir
    if hasattr(cfg.data.val, 'work_dir'): cfg.data.val.work_dir = cfg.work_dir

    # 构建训练数据集
    datasets = [build_dataset(cfg.data.train)]

    # 如果工作流程中包含验证 (通常 len(cfg.workflow) == 2 表示包含 ('val', 1))
    if len(cfg.workflow) == 2:
        val_dataset_cfg = copy.deepcopy(cfg.data.val) # 深拷贝验证集配置
        # 确保验证集使用与训练集相同的pipeline (除了数据增强部分，这通常在pipeline内部处理)
        # 这对于确保数据格式一致性很重要
        if "dataset" in cfg.data.train: # 处理Dataset wrapper的情况 (如ConcatDataset)
            val_dataset_cfg.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset_cfg.pipeline = cfg.data.train.pipeline
        # 将验证集的test_mode设为False，因为在训练过程中的验证，模型仍然需要GT进行评估
        # 而不是像纯测试模式那样不加载GT
        val_dataset_cfg.test_mode = False
        datasets.append(build_dataset(val_dataset_cfg)) # 构建并添加验证集到数据集列表

    if cfg.checkpoint_config is not None: # 如果配置了checkpoint保存
        # 在checkpoint的meta信息中保存mmdet版本、配置文件内容和类别名称
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES, # 从训练数据集中获取类别名称
        )

    # 为模型添加CLASSES属性，方便后续使用 (例如可视化或某些后处理)
    model.CLASSES = datasets[0].CLASSES

    # 根据是否使用插件，调用不同的训练函数
    if hasattr(cfg, "plugin") and cfg.plugin: # 如果配置了插件并且启用了
        custom_train_model( # 调用插件中定义的custom_train_model
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate), # 是否进行验证
            timestamp=timestamp, # 时间戳，用于日志或文件名
            meta=meta, # 元数据
        )
    else: # 否则使用MMDetection标准的train_detector
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta,
        )


if __name__ == "__main__":
    # 设置多进程启动方法为 'fork' (在POSIX系统上)。
    # 'fork' 方法比 'spawn' 更快，因为它共享父进程的内存，但可能在某些CUDA场景下不稳定。
    # 如果遇到CUDA相关的多进程错误，可以尝试改为 'spawn'。
    if os.name == 'posix': # 仅在POSIX系统上设置，Windows不支持fork
        torch.multiprocessing.set_start_method("fork", force=True)
    main() # 执行主函数
