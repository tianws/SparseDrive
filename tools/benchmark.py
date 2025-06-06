# Copyright (c) OpenMMLab. All rights reserved.
import argparse # 导入argparse模块，用于解析命令行参数
import time # 导入time模块，用于时间相关的操作 (例如计时)
import torch # 导入PyTorch库
from mmcv import Config # 从MMCV导入Config类，用于加载和管理配置文件
from mmcv.parallel import MMDataParallel # 从MMCV导入MMDataParallel，用于单机多卡数据并行
from mmcv.runner import load_checkpoint, wrap_fp16_model # 从MMCV导入加载checkpoint和包装fp16模型的函数
import sys # 导入sys模块，用于访问与Python解释器相关的变量和函数
sys.path.append('.') # 将当前目录添加到Python路径，以便导入项目内部模块
from projects.mmdet3d_plugin.datasets.builder import build_dataloader # 从项目中导入构建数据加载器的函数
from projects.mmdet3d_plugin.datasets import custom_build_dataset # 从项目中导入自定义构建数据集的函数
from mmdet.models import build_detector # 从MMDetection导入构建检测器的函数
from mmcv.cnn.utils.flops_counter import add_flops_counting_methods # 从MMCV导入添加FLOPs计数方法到模型的函数
from mmcv.parallel import scatter # 从MMCV导入scatter函数，用于将数据分发到不同设备


def parse_args(): # 定义解析命令行参数的函数
    parser = argparse.ArgumentParser(description='MMDet benchmark a model') # 创建参数解析器
    parser.add_argument('config', help='test config file path') # 添加'config'参数，指定测试配置文件的路径
    parser.add_argument('--checkpoint', default=None, help='checkpoint file') # 添加'--checkpoint'参数，指定模型权重文件路径 (可选)
    parser.add_argument('--samples', default=1000, type=int, help='samples to benchmark') # 添加'--samples'参数，指定用于基准测试的样本数量
    parser.add_argument(
        '--log-interval', default=50, type=int, help='interval of logging') # 添加'--log-interval'参数，指定打印日志的间隔
    parser.add_argument(
        '--fuse-conv-bn', # 添加'--fuse-conv-bn'参数，一个开关选项
        action='store_true', # 当出现此参数时，其值为True
        help='Whether to fuse conv and bn, this will slightly increase' # 是否融合卷积层和BN层，这会略微提高推理速度
        'the inference speed')
    args = parser.parse_args() # 解析命令行参数
    return args # 返回解析后的参数


def get_max_memory(model): # 定义获取模型最大GPU显存占用的函数
    """获取模型在当前设备上已分配的最大GPU显存（MB）。"""
    device = getattr(model, 'output_device', None) # 获取模型的输出设备 (通常是单个GPU ID)
    mem = torch.cuda.max_memory_allocated(device=device) # 获取指定设备上已分配的最大显存 (bytes)
    mem_mb = torch.tensor([mem / (1024 * 1024)], # 将bytes转换为MB
        dtype=torch.int, # 使用整数类型
        device=device) # 张量也放在同一设备上 (虽然在这里影响不大)
    return mem_mb.item() # 返回MB值


def main(): # 主函数
    args = parse_args() # 解析命令行参数
    print("Calculating FLOPs and Parameters...")
    get_flops_params(args) # 调用函数计算并打印FLOPs和参数量
    print("\nCalculating Inference Speed and GPU Memory Usage...")
    get_mem_fps(args) # 调用函数计算并打印推理速度和显存占用


def get_mem_fps(args): # 定义计算推理速度和最大显存占用的函数
    cfg = Config.fromfile(args.config) # 从配置文件加载配置
    # set cudnn_benchmark # 设置cuDNN benchmark模式
    if cfg.get('cudnn_benchmark', False): # 如果配置中启用了cudnn_benchmark
        torch.backends.cudnn.benchmark = True # 则开启它，可以让cuDNN自动寻找最快的卷积算法
    cfg.model.pretrained = None # 不加载预训练权重 (因为这里是推理速度测试，通常用已训练好的模型)
    cfg.data.test.test_mode = True # 确保数据集配置处于测试模式

    # build the dataloader # 构建数据加载器
    # TODO: support multiple images per gpu (only minor changes are needed) # 待办：支持每个GPU处理多张图片
    print("Test dataset config:\n", cfg.data.test) # 打印测试数据集配置
    dataset = custom_build_dataset(cfg.data.test) # 构建测试数据集
    data_loader = build_dataloader( # 构建数据加载器
        dataset,
        samples_per_gpu=1, # 推理速度测试时，通常每个GPU只处理1个样本
        workers_per_gpu=cfg.data.workers_per_gpu, # 每个GPU的数据加载进程数
        dist=False, # 非分布式测试
        shuffle=False) # 不打乱数据

    # build the model and load checkpoint # 构建模型并加载checkpoint
    cfg.model.train_cfg = None # 测试时不需要训练配置
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # 构建检测器模型
    fp16_cfg = cfg.get('fp16', None) # 获取FP16配置
    if fp16_cfg is not None: # 如果启用了FP16
        wrap_fp16_model(model) # 包装模型以支持FP16推理
    if args.checkpoint is not None: # 如果指定了checkpoint文件
        load_checkpoint(model, args.checkpoint, map_location='cpu') # 加载模型权重，先映射到CPU避免GPU显存不足
    # if args.fuse_conv_bn: # 如果启用了conv和bn的融合 (通常在推理前进行，可以加速)
    #     model = fuse_module(model) # MMDetection中通常用fuse_conv_bn(model)

    model = MMDataParallel(model, device_ids=[0]) # 使用MMDataParallel包装模型，用于单机单卡或多卡 (这里device_ids=[0]表示单卡)
    model.eval() # 设置模型为评估模式

    # the first several iterations may be very slow so skip them # 前几次迭代可能较慢（例如CUDA初始化），因此跳过它们作为预热
    num_warmup = 5 # 预热迭代次数
    pure_inf_time = 0 # 纯推理时间累加器

    # benchmark with several samples and take the average # 使用多个样本进行基准测试并取平均值
    max_memory = 0 # 最大显存占用记录器
    for i, data in enumerate(data_loader): # 遍历数据加载器
        # torch.cuda.synchronize() # 确保之前的CUDA操作完成 (可选，但更精确计时)
        with torch.no_grad(): # 在不计算梯度的上下文中进行推理
            start_time = time.perf_counter() # 记录开始时间 (高精度计时器)
            model(return_loss=False, rescale=True, **data) # 执行模型前向传播，不计算损失，rescale表示将结果缩放到原始图像尺寸

            torch.cuda.synchronize() # 确保模型推理的CUDA操作完成
            elapsed = time.perf_counter() - start_time # 计算推理耗时
            max_memory = max(max_memory, get_max_memory(model)) # 更新最大显存占用

        if i >= num_warmup: # 如果当前迭代在预热之后
            pure_inf_time += elapsed # 累加纯推理时间
            if (i + 1) % args.log_interval == 0: # 如果达到日志打印间隔
                fps = (i + 1 - num_warmup) / pure_inf_time # 计算当前平均FPS
                print(f'Done image [{i + 1:<3}/ {args.samples}], ' # 打印进度和FPS
                      f'fps: {fps:.1f} img / s, '
                      f"gpu mem: {max_memory} M") # 打印当前最大显存占用

        if (i + 1) == args.samples: # 如果已处理指定数量的样本
            # pure_inf_time += elapsed # 注意：这行代码会导致最后一个样本的时间被加了两次，应该在if i >= num_warmup内部更新
            fps = (i + 1 - num_warmup) / pure_inf_time # 计算最终的平均FPS
            print(f'Overall fps: {fps:.1f} img / s, gpu mem: {max_memory} M') # 打印最终结果
            break # 结束基准测试


def get_flops_params(args): # 定义计算FLOPs和参数量的函数
    gpu_id = 0 # 指定使用的GPU ID
    cfg = Config.fromfile(args.config) # 加载配置文件
    # 使用验证集的一个样本来计算FLOPs，因为通常FLOPs与输入尺寸有关
    dataset = custom_build_dataset(cfg.data.val)
    dataloader = build_dataloader( # 构建数据加载器
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0, # FLOPs计算不需要多进程加载数据
        dist=False,
        shuffle=False,
    )
    data_iter = iter(dataloader) # 创建数据迭代器
    data = next(data_iter) # 获取一个样本数据
    data = scatter(data, [gpu_id])[0] # 将数据分发到指定GPU

    cfg.model.train_cfg = None # 不需要训练配置
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # 构建模型
    fp16_cfg = cfg.get('fp16', None) # FP16配置
    if fp16_cfg is not None:
        wrap_fp16_model(model) # 包装FP16模型
    if args.checkpoint is not None: # 如果指定了checkpoint
        load_checkpoint(model, args.checkpoint, map_location='cpu') # 加载权重
    model = model.cuda(gpu_id) # 将模型移到GPU
    model.eval() # 设置为评估模式

    # --- 特殊处理自定义操作的FLOPs ---
    # 下面的代码估算了 deformable_aggregation 操作的FLOPs，因为标准FLOPs计数器可能无法准确计算自定义CUDA操作
    bilinear_flops = 11 # 假设一次双线性插值的FLOPs (通常涉及几次乘法和加法)
    # 计算检测头的deformable_aggregation的FLOPs
    num_key_pts_det = ( # 检测头的每个query的采样点数
        cfg.model["head"]['det_head']["deformable_model"]["kps_generator"]["num_learnable_pts"]
        + len(cfg.model["head"]['det_head']["deformable_model"]["kps_generator"]["fix_scale"])
    )
    deformable_agg_flops_det = ( # 检测头的总deformable_aggregation FLOPs估算
        # num_decoder_layers * embed_dims * num_levels * num_queries * num_cams * num_key_points_per_query * bilinear_flops_per_point
        cfg.model["head"]["det_head"].get("num_decoder", cfg.num_decoder) # 兼容不同配置位置
        * cfg.embed_dims
        * cfg.num_levels
        * cfg.model["head"]['det_head']["instance_bank"]["num_anchor"] # num_queries
        * cfg.model["head"]['det_head']["deformable_model"]["num_cams"]
        * num_key_pts_det
        * bilinear_flops
    )
    # 计算地图头的deformable_aggregation的FLOPs
    num_key_pts_map = ( # 地图头的每个query的采样点数
        cfg.model["head"]['map_head']["deformable_model"]["kps_generator"]["num_learnable_pts"]
        + len(cfg.model["head"]['map_head']["deformable_model"]["kps_generator"]["fix_height"])
    ) * cfg.model["head"]['map_head']["deformable_model"]["kps_generator"]["num_sample"]
    deformable_agg_flops_map = ( # 地图头的总deformable_aggregation FLOPs估算
        cfg.model["head"]["map_head"].get("num_decoder", cfg.num_decoder)
        * cfg.embed_dims
        * cfg.num_levels
        * cfg.model["head"]['map_head']["instance_bank"]["num_anchor"]
        * cfg.model["head"]['map_head']["deformable_model"]["num_cams"]
        * num_key_pts_map
        * bilinear_flops
    )
    deformable_agg_flops = deformable_agg_flops_det + deformable_agg_flops_map # 总的自定义操作FLOPs

    # --- 使用MMCV的FLOPs计数器计算标准操作的FLOPs ---
    total_flops = 0
    total_params = 0
    for module_name in ["total", "img_backbone", "img_neck", "head"]: # 遍历模型的主要模块
        if module_name != "total": # 如果不是计算整个模型
            flops_model = add_flops_counting_methods(getattr(model, module_name)) # 获取子模块并添加FLOPs计数方法
        else: # 计算整个模型
            flops_model = add_flops_counting_methods(model)
        flops_model.eval() # 设置为评估模式
        flops_model.start_flops_count() # 开始FLOPs计数

        # 执行一次前向传播以触发FLOPs计数
        if module_name == "img_backbone":
            flops_model(data["img"].flatten(0, 1)) # 对骨干网络输入展平后的图像
        elif module_name == "img_neck":
            # 对颈部网络输入骨干网络的输出
            with torch.no_grad(): # 确保骨干网络部分不参与梯度计算或FLOPs重复计数
                 backbone_output = model.img_backbone(data["img"].flatten(0,1))
            flops_model(backbone_output)
        elif module_name == "head":
            # 对头部输入骨干和颈部网络的输出
            with torch.no_grad():
                extracted_feats = model.extract_feat(data["img"], metas=data)
            flops_model(extracted_feats, data) # 假设头部forward接受 (features, metas_dict)
        else: # 整个模型
            with torch.no_grad(): # 确保在FLOPs计数时不进行梯度计算
                 flops_model(**data) # 整个模型的前向传播

        flops_count, params_count = flops_model.compute_average_flops_cost() # 计算平均FLOPs和参数量
        # flops_count *= flops_model.__batch_counter__ # MMCV的计数器通常返回的是单样本的，这里乘以batch_counter可能是为了得到总的（但batch_counter可能是1）
                                                    # 通常我们关心的是单样本的FLOPs，所以这行可能不需要或需要根据具体计数器行为调整
        flops_model.stop_flops_count() # 停止FLOPs计数
        
        if module_name == "head" or module_name == "total": # 如果是头部或整个模型，加上自定义操作的FLOPs
            flops_count += deformable_agg_flops
        if module_name == "total": # 保存总的FLOPs和参数量，用于计算百分比
            total_flops = flops_count
            total_params = params_count

        print( # 打印当前模块的FLOPs和参数量及其占比
            f"{module_name:<13} complexity: "
            f"FLOPs={flops_count/ 10.**9:>8.4f} G / {flops_count/total_flops*100 if total_flops > 0 else 0:>6.2f}%, "
            f"Params={params_count/10**6:>8.4f} M / {params_count/total_params*100 if total_params > 0 else 0:>6.2f}%."
        )

if __name__ == '__main__':
    main() # 执行主函数
