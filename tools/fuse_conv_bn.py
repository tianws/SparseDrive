# Copyright (c) OpenMMLab. All rights reserved.
import argparse # 导入argparse模块，用于解析命令行参数

import torch # 导入PyTorch库
from mmcv.runner import save_checkpoint # 从MMCV导入保存checkpoint的函数
from torch import nn as nn # 导入PyTorch的神经网络模块，并重命名为nn

from mmdet3d.apis import init_model # 从MMDetection3D的API导入初始化模型的函数


def fuse_conv_bn(conv, bn): # 定义融合卷积层和批归一化层的函数
    """During inference, the functionary of batch norm layers is turned off but # 在推理过程中，批归一化层的功能被关闭，
    only the mean and var alone channels are used, which exposes the chance to # 只使用通道维度的均值和方差，这使得
    fuse it with the preceding conv layers to save computations and simplify # 我们可以将其与前面的卷积层融合，以节省计算量并简化
    network structures. # 网络结构。

    Args:
        conv (nn.Conv2d): 要融合的卷积层。
        bn (nn.BatchNorm2d or nn.SyncBatchNorm): 要融合的批归一化层。

    Returns:
        nn.Conv2d: 融合了批归一化参数的卷积层。
    """
    # 获取卷积层的权重和偏置
    conv_w = conv.weight # 卷积权重 (out_channels, in_channels, kernel_h, kernel_w)
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like( # 卷积偏置，如果不存在则初始化为0
        bn.running_mean) # 使用bn.running_mean的形状创建全0张量

    # 计算BN层的等效缩放因子和偏置
    # factor = gamma / sqrt(running_var + eps)
    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps) # (out_channels)
    # 新的卷积权重: W_new = W_conv * factor (需要reshape factor以匹配权重维度)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1])) # factor reshape为 (out_channels, 1, 1, 1) 以进行广播
    # 新的卷积偏置: B_new = (B_conv - running_mean) * factor + beta
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv # 返回修改后的卷积层


def fuse_module(m: nn.Module): # 定义递归融合模型中所有Conv-BN对的函数
    """
    递归地遍历模型的所有子模块，并融合所有相邻的Conv2d和BatchNorm2d/SyncBatchNorm层。

    Args:
        m (nn.Module): 要进行融合操作的模型或模块。

    Returns:
        nn.Module: 融合了Conv-BN的模型或模块。
    """
    last_conv = None # 用于存储遇到的最后一个卷积层
    last_conv_name = None # 用于存储最后一个卷积层的名称

    for name, child in m.named_children(): # 遍历模块m的所有直接子模块
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)): # 如果子模块是BN层
            if last_conv is None:  # 仅融合BN之前的Conv层 (即 Conv -> BN 结构)
                continue # 如果前一个不是Conv层，则跳过
            # 执行融合操作
            fused_conv = fuse_conv_bn(last_conv, child)
            # 用融合后的卷积层替换模型中原来的卷积层
            m._modules[last_conv_name] = fused_conv
            # 为减少结构更改，将BN层设置为空的Identity层，而不是直接删除它
            m._modules[name] = nn.Identity()
            last_conv = None # 重置last_conv，因为BN层已经被融合了
        elif isinstance(child, nn.Conv2d): # 如果子模块是卷积层
            last_conv = child # 记录下来，以备后续可能的BN层融合
            last_conv_name = name
        else: # 如果子模块是其他类型的层 (例如Sequential, Bottleneck等)
            fuse_module(child) # 递归调用fuse_module处理该子模块
    return m # 返回处理后的模块


def parse_args(): # 定义解析命令行参数的函数
    parser = argparse.ArgumentParser(
        description='fuse Conv and BN layers in a model') # 创建参数解析器
    parser.add_argument('config', help='config file path') # 模型配置文件路径
    parser.add_argument('checkpoint', help='checkpoint file path') # 模型权重文件路径
    parser.add_argument('out', help='output path of the converted model') # 转换后模型的输出路径
    args = parser.parse_args() # 解析参数
    return args


def main(): # 主函数
    args = parse_args() # 解析命令行参数
    # 从配置文件和checkpoint文件初始化模型
    # init_model 通常会自动处理设备分配 (例如CPU或GPU)
    model = init_model(args.config, args.checkpoint, device='cpu') # 明确指定在CPU上初始化，避免GPU显存问题

    # 融合模型中的Conv和BN层
    print("Fusing Conv and BN layers...")
    fused_model = fuse_module(model)
    print("Fusion complete.")

    # 保存融合后的模型权重
    print(f"Saving fused model to {args.out}...")
    save_checkpoint(fused_model, args.out)
    print("Fused model saved.")


if __name__ == '__main__':
    main() # 执行主函数
