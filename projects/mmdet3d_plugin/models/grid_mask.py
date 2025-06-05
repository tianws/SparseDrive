import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import numpy as np  # 导入NumPy库，用于数值运算
from PIL import Image  # 导入Pillow库中的Image模块，用于图像处理


class Grid(object):  # 定义Grid类，实现GridMask数据增强的核心逻辑，可作为callable对象使用
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        """
        GridMask的构造函数（Grid类版本）。

        Args:
            use_h (bool): 是否在高度方向应用GridMask。
            use_w (bool): 是否在宽度方向应用GridMask。
            rotate (int): 旋转角度范围上限。实际旋转角度会在 [0, rotate-1] 度之间随机选择。
                          如果为1，则不进行旋转。
            offset (bool): 是否在遮挡区域添加随机偏移值（而不是直接置0）。默认为False。
            ratio (float): 网格中遮挡条带的宽度与网格单元大小(d)的比例。默认为0.5。
            mode (int): 0表示遮挡区域置0 (或offset值)，1表示保留遮挡区域，非遮挡区域置0。默认为0。
            prob (float): 应用GridMask的初始概率。默认为1.0。
        """
        self.use_h = use_h  # 是否在高度方向应用GridMask
        self.use_w = use_w  # 是否在宽度方向应用GridMask
        self.rotate = rotate  # 旋转角度的上限（随机选择0到rotate-1度）
        self.offset = offset  # 是否使用随机偏移值填充遮挡区域
        self.ratio = ratio  # 遮挡条带宽度与单元格大小的比例
        self.mode = mode  # 模式: 0为标准GridMask (遮挡)，1为反转GridMask (保留网格线)
        self.st_prob = prob  # 初始概率，用于set_prob方法
        self.prob = prob  # 当前应用GridMask的概率

    def set_prob(self, epoch, max_epoch):  # 根据训练周期动态调整GridMask的应用概率
        """
        动态设置GridMask的应用概率，通常随着训练的进行而增加。

        Args:
            epoch (int): 当前训练周期。
            max_epoch (int): 总训练周期数。
        """
        self.prob = self.st_prob * epoch / max_epoch  # 线性增加概率

    def __call__(self, img, label):  # 使Grid对象可调用，通常用于 torchvision.transforms
        """
        对输入的图像应用GridMask。

        Args:
            img (torch.Tensor): 输入图像张量，形状通常为 (C, H, W)。
            label: 输入标签 (此实现中未使用，但保留以兼容常见的transform接口)。

        Returns:
            tuple[torch.Tensor, any]: 应用GridMask后的图像张量和原始标签。
        """
        if np.random.rand() > self.prob:  # 以设定的概率决定是否应用GridMask
            return img, label  # 如果不应用，直接返回原图和标签

        h = img.size(1)  # 获取图像高度
        w = img.size(2)  # 获取图像宽度

        # d: 网格单元的大小，在 [d1, d2-1] 范围内随机选择
        self.d1 = 2  # 网格单元最小尺寸
        self.d2 = min(h, w)  # 网格单元最大尺寸不超过图像的最小边
        # hh, ww: 扩展后的画布尺寸，用于旋转，避免裁剪掉有效区域
        hh = int(1.5 * h)
        ww = int(1.5 * w)

        d = np.random.randint(self.d1, self.d2)  # 随机选择网格单元大小

        # l: 遮挡条带的宽度
        if self.ratio == 1: # 如果比例为1，则条带宽度在[1, d-1]之间随机
            self.l = np.random.randint(1, d)
        else: # 否则根据ratio计算条带宽度
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1) # 确保宽度至少为1且小于d

        mask = np.ones((hh, ww), np.float32)  # 初始化一个全1的扩展画布mask

        # st_h, st_w: 网格的起始随机偏移
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        # 在高度方向应用GridMask
        if self.use_h:
            for i in range(hh // d):  # 遍历所有可能的水平条带
                s = d * i + st_h  # 计算条带起始行
                t = min(s + self.l, hh)  # 计算条带结束行，不超过画布高度
                mask[s:t, :] *= 0  # 将条带区域置0
        # 在宽度方向应用GridMask
        if self.use_w:
            for i in range(ww // d):  # 遍历所有可能的垂直条带
                s = d * i + st_w  # 计算条带起始列
                t = min(s + self.l, ww)  # 计算条带结束列，不超过画布宽度
                mask[:, s:t] *= 0  # 将条带区域置0

        # 随机旋转mask
        r = np.random.randint(self.rotate)  # 随机选择旋转角度
        mask = Image.fromarray(np.uint8(mask))  # 转换为Pillow Image对象进行旋转
        mask = mask.rotate(r)  # 旋转
        mask = np.asarray(mask)  # 转回NumPy数组

        # 从扩展画布中裁剪出与原图同样大小的mask区域
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask = torch.from_numpy(mask).float()  # 将NumPy mask转换为PyTorch张量
        if self.mode == 1:  # 如果mode为1，反转mask (保留网格线，遮挡其他区域)
            mask = 1 - mask

        mask = mask.expand_as(img)  # 将单通道mask扩展到与图像通道数一致
        if self.offset:  # 如果使用offset模式
            # 生成随机偏移值，范围[-1, 1]
            offset_values = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            # 只在mask为0的区域（即原先的遮挡区域）应用offset
            offset_tensor = (1 - mask) * offset_values
            img = img * mask + offset_tensor  # 应用mask和offset
        else:  # 标准模式，直接将遮挡区域置0
            img = img * mask

        return img, label  # 返回处理后的图像和原始标签


class GridMask(nn.Module):  # 定义GridMask的nn.Module封装，方便在PyTorch模型中使用
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        """
        GridMask的构造函数（nn.Module版本）。参数含义同Grid类。
        """
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob  # 初始概率
        self.prob = prob  # 当前应用概率

    def set_prob(self, epoch, max_epoch): # 根据训练周期动态调整GridMask的应用概率
        """
        动态设置GridMask的应用概率。
        """
        self.prob = self.st_prob * epoch / max_epoch  # 线性增加概率, e.g., 0.5 * (epoch/max_epoch)

    def forward(self, x): # 前向传播函数，应用GridMask到输入张量x
        """
        对输入的张量x应用GridMask。

        Args:
            x (torch.Tensor): 输入张量，形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 应用GridMask后的张量。
        """
        if np.random.rand() > self.prob or not self.training: # 以设定概率且仅在训练时应用
            return x # 否则直接返回原张量

        n, c, h, w = x.size() # 获取输入的批次大小、通道数、高度、宽度
        x_input_type = x.dtype # 记录输入的数据类型
        x_device = x.device # 记录输入的设备

        # 将x的形状从 (N, C, H, W) 调整为 (N*C, H, W) 以便逐个通道处理mask
        # 但mask本身是2D的，会广播到每个通道
        # GridMask论文中mask是针对每个图像的，而不是每个通道的。
        # 此处实现将mask应用到每个通道，但mask本身是根据H,W生成的2D mask。

        # 以下逻辑与Grid类中的__call__方法类似
        hh = int(1.5 * h) # 扩展画布高度
        ww = int(1.5 * w) # 扩展画布宽度
        d = np.random.randint(2, min(h,w)) # 网格单元大小，确保不大于图像最小边

        # 遮挡条带宽度，确保在[1, d-1]之间
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        mask = np.ones((hh, ww), np.float32) # 初始化扩展画布mask
        st_h = np.random.randint(d) # 高度方向起始偏移
        st_w = np.random.randint(d) # 宽度方向起始偏移

        # 应用水平条带
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        # 应用垂直条带
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        # 旋转mask
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)

        # 裁剪mask到原始图像尺寸
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask = torch.from_numpy(mask.copy()).float().to(x_device) # 转为Tensor并移到输入设备
        if self.mode == 1: # 反转mask
            mask = 1 - mask

        # 将2D mask扩展到 (1, 1, H, W) 或 (1, C, H, W) 以便与 (N, C, H, W) 的x进行广播相乘
        # PyTorch的广播机制会自动处理 mask.expand_as(x) 失败的情况，如果mask是 (H,W)
        # 正确的做法是 mask.unsqueeze(0).unsqueeze(0) -> (1,1,H,W) 或者 mask.reshape(1,1,h,w)
        mask = mask.reshape(1, 1, h, w).expand(n, c, h, w) # 扩展mask以匹配x的形状

        if self.offset: # 如果使用offset模式
            # 生成随机偏移值，范围[-0.5, 0.5] * 2 = [-1, 1]
            # 注意：这里offset是为整个batch或整个图像生成的，而不是逐像素随机（如果h,w是原图尺寸）
            # 如果x是(N*C,H,W)，那么offset也应该是(N*C,H,W)或者可广播的
            # 但这里offset是(H,W)，然后通过(1-mask)选择性应用，这可能意味着offset值在通道和批次上是共享的
            offset_values = (
                torch.from_numpy(2 * (np.random.rand(h, w) - 0.5))
                .float()
                .to(x_device)
            )
            # (1-mask) 得到需要填充offset的区域 (原遮挡区域)
            x = x * mask + offset_values.unsqueeze(0).unsqueeze(0) * (1 - mask)
        else: # 标准模式
            x = x * mask

        return x.view(n, c, h, w).type(x_input_type) # 恢复x的原始形状和数据类型
