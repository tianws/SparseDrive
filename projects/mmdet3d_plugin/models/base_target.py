from abc import ABC, abstractmethod  # 导入ABC（抽象基类）和abstractmethod（抽象方法装饰器）

__all__ = ["BaseTargetWithDenoising"]  # 定义当使用 from .base_target import * 时，哪些名称会被导入


class BaseTargetWithDenoising(ABC):  # 定义一个名为 BaseTargetWithDenoising 的抽象基类
    """
    带有去噪功能的目标分配器基类。
    该类定义了目标分配的基本接口，特别关注于结合去噪（Denoising, DN）机制，
    这在一些基于Transformer的检测器（如DN-DETR）中用于加速收敛和提高性能。
    """
    def __init__(self, num_dn_groups=0, num_temp_dn_groups=0):
        """
        初始化函数。

        Args:
            num_dn_groups (int, optional): 当前帧去噪组的数量。默认为0。
                                         表示为当前帧的真实目标生成多少组加噪样本。
            num_temp_dn_groups (int, optional): 时序去噪组的数量。默认为0。
                                              表示从过去的帧中缓存并用于当前帧的加噪样本组数。
        """
        super(BaseTargetWithDenoising, self).__init__()  # 调用父类(ABC)的构造函数
        self.num_dn_groups = num_dn_groups  # 存储当前帧去噪组的数量
        self.num_temp_dn_groups = num_temp_dn_groups  # 存储时序去噪组的数量
        self.dn_metas = None  # 初始化去噪相关的元数据缓存为None

    @abstractmethod  # 标记 sample 方法为一个抽象方法，子类必须实现它
    def sample(self, cls_pred, box_pred, cls_target, box_target):
        """
        Perform Hungarian matching between predictions and ground truth,  # 在预测和真实目标之间执行匈牙利匹配，
        returning the matched ground truth corresponding to the predictions  # 返回与预测匹配的真实目标
        along with the corresponding regression weights.  # 以及相应的回归权重。
        """
        pass # 抽象方法，具体实现由子类提供

    def get_dn_anchors(self, cls_target, box_target, *args, **kwargs):
        """
        Generate noisy instances for the current frame, with a total of  # 为当前帧生成加噪实例（作为去噪任务的正样本），总共有
        'self.num_dn_groups' groups.  # 'self.num_dn_groups' 组。

        Args:
            cls_target (Tensor): 真实目标的类别标签。
            box_target (Tensor): 真实目标的边界框。
            *args: 其他位置参数。
            **kwargs: 其他关键字参数。

        Returns:
            Tensor or None: 生成的加噪锚点。如果num_dn_groups为0或未实现，则可能返回None。
        """
        return None  # 默认实现返回None，子类可以重写此方法以生成加噪锚点

    def update_dn(self, instance_feature, anchor, *args, **kwargs):
        """
        Insert the previously saved 'self.dn_metas' into the noisy instances  # 将先前保存的 'self.dn_metas'（时序去噪元数据）插入到
        of the current frame.  # 当前帧的加噪实例中。
                               # 这允许模型利用历史信息进行去噪学习。
        Args:
            instance_feature (Tensor): 当前帧的实例特征。
            anchor (Tensor): 当前帧的锚点或查询。
            *args: 其他位置参数。
            **kwargs: 其他关键字参数。
        """
        pass # 默认实现为空，子类可以重写此方法以更新时序去噪信息

    def cache_dn(
        self,
        dn_instance_feature,  # 去噪实例的特征
        dn_anchor,  # 去噪锚点/查询
        dn_cls_target,  # 去噪实例对应的类别目标
        valid_mask,  # 有效掩码，指示哪些去噪实例是有效的
        dn_id_target,  # 去噪实例对应的ID目标 (用于跟踪等)
    ):
        """
        Randomly save information for 'self.num_temp_dn_groups' groups of  # 为 'self.num_temp_dn_groups' 组时序加噪实例
        temporal noisy instances to 'self.dn_metas'.  # 随机保存信息到 'self.dn_metas'。
                                                     # 这些缓存的信息可以在后续帧的 update_dn 方法中使用。
        """
        if self.num_temp_dn_groups <= 0:  # 如果不使用时序去噪组或数量无效
            return  # 则直接返回
        # 默认实现仅缓存了部分dn_anchor信息，具体缓存哪些内容由子类决定
        self.dn_metas = dict(dn_anchor=dn_anchor[:, : self.num_temp_dn_groups])
