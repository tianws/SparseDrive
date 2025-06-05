# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp  # 导入 os.path 模块，并重命名为 osp，用于处理文件路径
import pickle  # 导入 pickle 模块，用于序列化和反序列化 Python 对象
import shutil  # 导入 shutil 模块，用于高级文件操作（如复制、删除）
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import time  # 导入 time 模块，用于处理时间相关操作

import mmcv  # 导入 mmcv 模块，OpenMMLab 计算机视觉基础库
import torch  # 导入 torch 模块，PyTorch 深度学习框架
import torch.distributed as dist  # 导入 torch.distributed 模块，用于分布式训练
from mmcv.image import tensor2imgs  # 从 mmcv.image 模块导入 tensor2imgs 函数，用于将张量转换为图像
from mmcv.runner import get_dist_info  # 从 mmcv.runner 模块导入 get_dist_info 函数，用于获取分布式训练信息

from mmdet.core import encode_mask_results  # 从 mmdet.core 模块导入 encode_mask_results 函数，用于编码掩码结果


import mmcv  # 再次导入 mmcv (通常情况下，重复导入是不必要的，但在这里保持原样)
import numpy as np  # 导入 numpy 模块，用于进行科学计算
import pycocotools.mask as mask_util  # 导入 pycocotools.mask 模块，并重命名为 mask_util，用于处理 COCO 数据集的掩码


def custom_encode_mask_results(mask_results):  # 定义自定义编码掩码结果函数
    """Encode bitmap mask to RLE code. Semantic Masks only  # 将位图掩码编码为 RLE 代码。仅适用于语义分割掩码
    Args:  # 参数说明
        mask_results (list | tuple[list]): bitmap mask results.  # 位图掩码结果，可以是列表或元组列表
            In mask scoring rcnn, mask_results is a tuple of (segm_results,  # 在 mask scoring rcnn 中，mask_results 是 (segm_results, segm_cls_score) 的元组
            segm_cls_score).
    Returns:  # 返回值说明
        list | tuple: RLE encoded mask.  # RLE 编码的掩码
    """
    cls_segms = mask_results  # 获取类别分割结果
    num_classes = len(cls_segms)  # 获取类别数量
    encoded_mask_results = []  # 初始化编码后的掩码结果列表
    for i in range(len(cls_segms)):  # 遍历每个类别的分割结果
        encoded_mask_results.append(  # 添加编码后的掩码
            mask_util.encode(  # 使用 mask_util.encode 进行 RLE 编码
                np.array(  # 将分割结果转换为 numpy 数组
                    cls_segms[i][:, :, np.newaxis], order="F", dtype="uint8"  # 调整数组形状、顺序和数据类型
                )
            )[0]
        )  # encoded with RLE  # 使用 RLE 编码
    return [encoded_mask_results]  # 返回包含编码后掩码的列表


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):  # 定义自定义多 GPU 测试函数
    """Test model with multiple gpus.  # 使用多个 GPU 测试模型
    This method tests model with multiple gpus and collects the results  # 此方法使用多个 GPU 测试模型并收集结果
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'  # 支持两种模式：GPU 模式和 CPU 模式。通过设置 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results  # 它会将结果编码为 GPU 张量，并使用 GPU 通信进行结果收集
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'  # 在 CPU 模式下，它会将不同 GPU 上的结果保存到 'tmpdir'
    and collects them by the rank 0 worker.  # 并由 rank 0 的 worker 收集它们
    Args:  # 参数说明
        model (nn.Module): Model to be tested.  # 要测试的模型 (nn.Module)
        data_loader (nn.Dataloader): Pytorch data loader.  # PyTorch 数据加载器
        tmpdir (str): Path of directory to save the temporary results from  # 在 CPU 模式下保存不同 GPU 临时结果的目录路径
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.  # 选择使用 GPU 还是 CPU 收集结果的选项
    Returns:  # 返回值说明
        list: The prediction results.  # 预测结果列表
    """
    model.eval()  # 将模型设置为评估模式
    bbox_results = []  # 初始化边界框结果列表
    mask_results = []  # 初始化掩码结果列表
    dataset = data_loader.dataset  # 获取数据加载器中的数据集对象
    rank, world_size = get_dist_info()  # 获取分布式训练的 rank 和 world_size
    if rank == 0:  # 如果是 rank 0 进程
        prog_bar = mmcv.ProgressBar(len(dataset))  # 创建进度条
    time.sleep(2)  # This line can prevent deadlock problem in some cases.  # 这行代码在某些情况下可以防止死锁问题
    have_mask = False  # 初始化是否存在掩码的标志
    for i, data in enumerate(data_loader):  # 遍历数据加载器中的数据
        with torch.no_grad():  # 在不计算梯度的情况下执行
            result = model(return_loss=False, rescale=True, **data)  # 模型前向传播，获取预测结果
            # encode mask results  # 编码掩码结果
            if isinstance(result, dict):  # 如果结果是字典类型
                if "bbox_results" in result.keys():  # 如果结果中包含 "bbox_results"
                    bbox_result = result["bbox_results"]  # 获取边界框结果
                    batch_size = len(result["bbox_results"])  # 获取批量大小
                    bbox_results.extend(bbox_result)  # 将当前批次的边界框结果添加到列表中
                if (
                    "mask_results" in result.keys()  # 如果结果中包含 "mask_results"
                    and result["mask_results"] is not None  # 且掩码结果不为 None
                ):
                    mask_result = custom_encode_mask_results(  # 调用自定义编码掩码结果函数
                        result["mask_results"]
                    )
                    mask_results.extend(mask_result)  # 将当前批次的编码后掩码结果添加到列表中
                    have_mask = True  # 设置存在掩码的标志为 True
            else:  # 如果结果不是字典类型（通常是列表）
                batch_size = len(result)  # 获取批量大小
                bbox_results.extend(result)  # 将当前批次的结果（假定为边界框结果）添加到列表中

        if rank == 0:  # 如果是 rank 0 进程
            for _ in range(batch_size * world_size):  # 根据批量大小和 world_size 更新进度条
                prog_bar.update()

    # collect results from all ranks  # 从所有 rank 收集结果
    if gpu_collect:  # 如果使用 GPU 收集结果
        bbox_results = collect_results_gpu(bbox_results, len(dataset))  # 调用 GPU 收集边界框结果函数
        if have_mask:  # 如果存在掩码
            mask_results = collect_results_gpu(mask_results, len(dataset))  # 调用 GPU 收集掩码结果函数
        else:
            mask_results = None  # 否则掩码结果为 None
    else:  # 如果使用 CPU 收集结果
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)  # 调用 CPU 收集边界框结果函数
        tmpdir_mask = tmpdir + "_mask" if tmpdir is not None else None  # 为掩码结果创建临时目录路径
        if have_mask:  # 如果存在掩码
            mask_results = collect_results_cpu(  # 调用 CPU 收集掩码结果函数
                mask_results, len(dataset), tmpdir_mask
            )
        else:
            mask_results = None  # 否则掩码结果为 None

    if mask_results is None:  # 如果掩码结果为 None
        return bbox_results  # 只返回边界框结果
    return {"bbox_results": bbox_results, "mask_results": mask_results}  # 返回包含边界框和掩码结果的字典


def collect_results_cpu(result_part, size, tmpdir=None):  # 定义 CPU 收集结果函数
    rank, world_size = get_dist_info()  # 获取分布式训练的 rank 和 world_size
    # create a tmp dir if it is not specified  # 如果未指定临时目录，则创建一个
    if tmpdir is None:  # 如果临时目录为 None
        MAX_LEN = 512  # 定义最大长度
        # 32 is whitespace  # 32 是空格的 ASCII 码
        dir_tensor = torch.full(  # 创建一个用空格填充的张量，用于存储临时目录路径
            (MAX_LEN,), 32, dtype=torch.uint8, device="cuda"
        )
        if rank == 0:  # 如果是 rank 0 进程
            mmcv.mkdir_or_exist(".dist_test")  # 创建 .dist_test 目录（如果不存在）
            tmpdir = tempfile.mkdtemp(dir=".dist_test")  # 在 .dist_test 目录下创建临时目录
            tmpdir = torch.tensor(  # 将临时目录路径转换为字节张量
                bytearray(tmpdir.encode()), dtype=torch.uint8, device="cuda"
            )
            dir_tensor[: len(tmpdir)] = tmpdir  # 将临时目录路径字节张量复制到 dir_tensor
        dist.broadcast(dir_tensor, 0)  # 将 dir_tensor 从 rank 0 广播到所有其他进程
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()  # 将接收到的 dir_tensor 转换回字符串路径
    else:
        mmcv.mkdir_or_exist(tmpdir)  # 如果指定了临时目录，则创建它（如果不存在）
    # dump the part result to the dir  # 将部分结果转储到目录中
    mmcv.dump(result_part, osp.join(tmpdir, f"part_{rank}.pkl"))  # 将当前 rank 的部分结果保存为 pkl 文件
    dist.barrier()  # 同步所有进程，等待所有进程都保存完各自的部分结果
    # collect all parts  # 收集所有部分的结果
    if rank != 0:  # 如果不是 rank 0 进程
        return None  # 非 rank 0 进程不执行收集操作，直接返回 None
    else:  # 如果是 rank 0 进程
        # load results of all parts from tmp dir  # 从临时目录加载所有部分的结果
        part_list = []  # 初始化部分结果列表
        for i in range(world_size):  # 遍历所有 rank
            part_file = osp.join(tmpdir, f"part_{i}.pkl")  # 构建每个 rank 的 pkl 文件路径
            part_list.append(mmcv.load(part_file))  # 加载 pkl 文件并添加到列表中
        # sort the results  # 对结果进行排序（实际是将所有部分的结果合并）
        ordered_results = []  # 初始化排序后的结果列表
        """
        bacause we change the sample of the evaluation stage to make sure that
        each gpu will handle continuous sample,
        """
        # for res in zip(*part_list):  # 原先的按列合并方式（已注释掉）
        for res in part_list:  # 遍历每个 rank 的结果列表
            ordered_results.extend(list(res))  # 将每个 rank 的结果扩展到 ordered_results 列表中
        # the dataloader may pad some samples  # 数据加载器可能会填充一些样本
        ordered_results = ordered_results[:size]  # 截取到实际数据集大小的结果
        # remove tmp dir  # 删除临时目录
        shutil.rmtree(tmpdir)  # 递归删除临时目录及其内容
        return ordered_results  # 返回收集并排序后的完整结果


def collect_results_gpu(result_part, size):  # 定义 GPU 收集结果函数
    collect_results_cpu(result_part, size)  # 当前实现直接调用 CPU 收集结果函数 (可能是一个占位符或简化实现)
