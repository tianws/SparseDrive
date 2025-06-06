# Copyright (c) OpenMMLab. All rights reserved.
import argparse # 导入argparse模块，用于解析命令行参数
import mmcv # 导入MMCV库
import os # 导入os模块，用于与操作系统交互
from os import path as osp # 从os.path导入path，并重命名为osp，方便路径操作

import torch # 导入PyTorch库
import warnings # 导入warnings模块，用于发出警告
from mmcv import Config, DictAction # 从MMCV导入Config类（用于处理配置文件）和DictAction（用于解析字典类型的命令行参数）
from mmcv.cnn import fuse_conv_bn # 从MMCV导入融合Conv和BN层的函数
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel # 从MMCV导入数据并行和分布式数据并行的包装器
from mmcv.runner import ( # 从MMCV的runner模块导入相关函数
    get_dist_info, # 获取分布式训练的信息（rank, world_size）
    init_dist, # 初始化分布式环境
    load_checkpoint, # 加载模型checkpoint
    wrap_fp16_model, # 包装模型以支持FP16混合精度
)

from mmdet.apis import single_gpu_test, multi_gpu_test, set_random_seed # 从MMDetection的API导入单GPU测试、多GPU测试和设置随机种子的函数
from mmdet.datasets import replace_ImageToTensor, build_dataset # 从MMDetection的数据集模块导入工具函数和构建数据集的函数
from mmdet.datasets import build_dataloader as build_dataloader_origin # 从MMDetection导入原始的数据加载器构建函数
from mmdet.models import build_detector # 从MMDetection导入构建检测器模型的函数

from projects.mmdet3d_plugin.datasets.builder import build_dataloader # 从项目插件中导入自定义的数据加载器构建函数
from projects.mmdet3d_plugin.apis.test import custom_multi_gpu_test # 从项目插件中导入自定义的多GPU测试函数


def parse_args(): # 定义解析命令行参数的函数
    parser = argparse.ArgumentParser(
        description="MMDet test (and eval) a model" # 创建参数解析器，描述脚本用途
    )
    parser.add_argument("config", help="test config file path") # 必要参数：配置文件路径
    parser.add_argument("checkpoint", help="checkpoint file") # 必要参数：模型权重文件路径
    parser.add_argument("--out", help="output result file in pickle format") # 可选参数：输出结果文件的路径（pickle格式）
    parser.add_argument(
        "--fuse-conv-bn", # 可选参数：是否融合Conv和BN层
        action="store_true", # 指定为开关动作，出现则为True
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed", # 帮助信息：融合Conv和BN可以略微提高推理速度
    )
    parser.add_argument(
        "--format-only", # 可选参数：是否仅格式化输出结果而不进行评估
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server", # 帮助信息：当你希望将结果格式化为特定格式并提交到测试服务器时很有用
    )
    parser.add_argument(
        "--eval", # 可选参数：指定评估指标
        type=str,
        nargs="+", # 允许接收一个或多个值
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC', # 帮助信息：评估指标取决于数据集
    )
    parser.add_argument("--show", action="store_true", help="show results") # 可选参数：是否显示结果（可视化）
    parser.add_argument(
        "--show-dir", help="directory where results will be saved" # 可选参数：保存可视化结果的目录
    )
    parser.add_argument(
        "--gpu-collect", # 可选参数：是否使用GPU收集分布式测试的结果
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir", # 可选参数：用于在多卡收集结果时存储临时文件的目录
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed") # 可选参数：设置随机种子
    parser.add_argument(
        "--deterministic", # 可选参数：是否为CUDNN后端设置确定性选项
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options", # 可选参数：覆盖配置文件中的某些设置
        nargs="+",
        action=DictAction, # 将key=value形式的参数解析为字典
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.", # 帮助信息：详细说明了如何覆盖配置
    )
    parser.add_argument(
        "--options", # 可选参数（已废弃，请使用--eval-options）
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options", # 可选参数：为dataset.evaluate()函数传递自定义选项
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher", # 可选参数：分布式任务启动器类型
        choices=["none", "pytorch", "slurm", "mpi"], # 可选项
        default="none", # 默认为none（非分布式）
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0) # 可选参数：本地rank（用于分布式训练）
    parser.add_argument("--result_file", type=str, default=None, help="Path to pkl file with precomputed results") # 可选参数：预计算结果文件路径
    parser.add_argument("--show_only", action="store_true", help="Only run show method from pipeline") # 可选参数：仅运行数据集的show方法进行可视化

    args = parser.parse_args() # 解析命令行参数
    if "LOCAL_RANK" not in os.environ: # 如果环境变量中没有LOCAL_RANK
        os.environ["LOCAL_RANK"] = str(args.local_rank) # 则使用命令行参数中的local_rank设置

    if args.options and args.eval_options: # --options 和 --eval-options 不能同时指定
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options: # 如果使用了已废弃的 --options
        warnings.warn("--options is deprecated in favor of --eval-options") # 发出警告
        args.eval_options = args.options # 将其值赋给 --eval-options
    return args # 返回解析后的参数对象


def main(): # 主函数
    args = parse_args() # 解析命令行参数

    # 断言至少指定了一个操作（保存结果、评估、仅格式化、显示结果）
    assert (
        args.out or args.eval or args.format_only or args.show or args.show_dir
    ), (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    # --eval 和 --format_only 不能同时指定
    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    # 如果指定了输出路径，确保其以 .pkl 或 .pickle 结尾
    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    # 从文件加载配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None: # 如果有命令行配置覆盖选项
        cfg.merge_from_dict(args.cfg_options) # 合并到配置对象中

    # 导入自定义模块 (如果配置中指定了)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    # 导入插件模块 (如果配置中指定了)
    # 这允许动态加载项目特定的代码，并更新MMCV/MMDet的注册表
    if hasattr(cfg, "plugin"): # 检查配置中是否有'plugin'属性
        if cfg.plugin: # 如果plugin为True
            import importlib
            if hasattr(cfg, "plugin_dir"): # 如果指定了插件目录
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir) # 获取插件目录的父目录
                _module_dir = _module_dir.split("/") # 按'/'分割路径
                _module_path = _module_dir[0] # 模块路径的起始部分
                for m in _module_dir[1:]: # 逐级拼接模块路径
                    _module_path = _module_path + "." + m
                print(f"动态导入插件模块: {_module_path}")
                plg_lib = importlib.import_module(_module_path) # 动态导入插件库
            else: # 如果未指定插件目录，则尝试从配置文件所在目录导入
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(f"动态导入插件模块 (基于config路径): {_module_path}")
                plg_lib = importlib.import_module(_module_path)

    # 设置 cudnn_benchmark，可以加速固定输入尺寸下的卷积运算
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None # 测试时不使用预训练的骨干网络权重 (因为我们加载的是整个模型的checkpoint)

    # 处理测试数据集配置，确保是测试模式，并处理多样本每GPU的情况
    samples_per_gpu = 1 # 默认每个GPU一个样本
    if isinstance(cfg.data.test, dict): # 如果测试数据配置是字典
        cfg.data.test.test_mode = True # 设置为测试模式
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1) # 获取并移除samples_per_gpu配置，默认为1
        if samples_per_gpu > 1: # 如果每个GPU样本数大于1
            # 需要将数据处理流水线中的 'ImageToTensor' 替换为 'DefaultFormatBundle'
            # 因为 'ImageToTensor' 不处理批量数据
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline
            )
    elif isinstance(cfg.data.test, list): # 如果测试数据配置是列表 (例如用于ConcatDataset)
        for ds_cfg in cfg.data.test: # 遍历每个子数据集配置
            ds_cfg.test_mode = True
        samples_per_gpu = max( # 取所有子数据集中最大的samples_per_gpu
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # 初始化分布式环境
    if args.launcher == "none": # 如果启动器是'none'，则为非分布式
        distributed = False
    else: # 否则为分布式
        distributed = True
        init_dist(args.launcher, **cfg.dist_params) # 初始化分布式环境

    # 设置随机种子以保证结果可复现
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # 设置工作目录
    if cfg.get('work_dir', None) is None: # 如果配置文件中没有指定工作目录
        # 使用配置文件名作为默认工作目录的一部分
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0]) 
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir)) # 创建工作目录 (如果不存在)
    cfg.data.test.work_dir = cfg.work_dir # 将工作目录也传递给测试数据集配置 (某些评估可能需要)
    print('工作目录: ',cfg.work_dir)

    # 构建数据加载器
    dataset = build_dataset(cfg.data.test) # 构建数据集对象
    print("是否为分布式测试:", distributed)
    if distributed: # 如果是分布式测试
        # 使用项目插件中的build_dataloader (可能支持自定义的分布式采样器)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False, # 测试时不打乱数据
            nonshuffler_sampler=dict(type="DistributedSampler"), # 使用标准的分布式采样器
        )
    else: # 如果是非分布式测试
        # 使用MMDetection原始的build_dataloader
        data_loader = build_dataloader_origin(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )

    # 构建模型并加载checkpoint
    cfg.model.train_cfg = None # 测试时不需要训练配置
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg")) # 构建检测器模型
    fp16_cfg = cfg.get("fp16", None) # 获取FP16配置
    if fp16_cfg is not None: # 如果启用FP16
        wrap_fp16_model(model) # 包装模型以支持FP16

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu") # 加载模型权重，先映射到CPU

    if args.fuse_conv_bn: # 如果指定了融合Conv和BN
        model = fuse_conv_bn(model) # 执行融合操作

    # 兼容旧版本checkpoint中可能没有保存类别信息的情况
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    # 处理分割任务的调色板信息 (如果存在)
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        model.PALETTE = dataset.PALETTE

    if args.result_file is not None and osp.exists(args.result_file): # 如果提供了预计算的结果文件并且文件存在
        print(f"从 {args.result_file} 加载预计算的结果...")
        outputs = mmcv.load(args.result_file) # 直接加载结果
    elif not distributed: # 如果不是分布式测试
        model = MMDataParallel(model, device_ids=[0]) # 使用MMDataParallel包装模型 (单机多卡或单卡)
        # 调用MMDetection的单GPU测试函数
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else: # 如果是分布式测试
        model = MMDistributedDataParallel( # 使用MMDistributedDataParallel包装模型
            model.cuda(), # 将模型移到当前GPU
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        # 调用项目插件中的自定义多GPU测试函数
        outputs = custom_multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
        )

    rank, _ = get_dist_info() # 获取当前进程的rank
    if rank == 0: #只有rank 0的进程执行后续的保存、评估、可视化操作
        if args.out: # 如果指定了输出文件路径
            print(f"\n将结果写入到 {args.out}")
            mmcv.dump(outputs, args.out) # 保存结果到pickle文件

        kwargs = {} if args.eval_options is None else args.eval_options # 获取评估选项

        if args.show_only: # 如果只是展示结果（通常用于可视化）
            print("仅显示结果，不进行评估或格式化。")
            eval_kwargs = cfg.get("evaluation", {}).copy() # 拷贝评估配置
            # 移除评估钩子中不相关的参数
            for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(kwargs) # 更新为命令行传入的评估选项
            dataset.show(outputs, show=True, **eval_kwargs) # 调用数据集的show方法

        elif args.format_only: # 如果只是格式化结果
            dataset.format_results(outputs, **kwargs) # 调用数据集的format_results方法

        elif args.eval: # 如果指定了评估指标，则进行评估
            eval_kwargs = cfg.get("evaluation", {}).copy() # 拷贝评估配置
            # 移除评估钩子中不相关的参数
            for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs)) # 添加命令行指定的评估指标
            print("评估参数:", eval_kwargs)
            results_dict = dataset.evaluate(outputs, **eval_kwargs) # 调用数据集的evaluate方法
            print("评估结果:", results_dict)


if __name__ == "__main__":
    # 设置多进程启动方法为 'fork'，可以使得 workers_per_gpu > 0 在某些情况下正常工作
    # 但要注意，'fork' 方法在某些CUDA场景下可能不稳定，'spawn' 更安全但可能稍慢。
    if os.name == 'posix': # 'fork' 通常只在POSIX系统上可用
        torch.multiprocessing.set_start_method("fork", force=True)
    main() # 执行主函数
