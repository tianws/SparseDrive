import copy  # 导入copy模块，用于对象的复制

import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch库

from projects.mmdet3d_plugin.core.box3d import *  # 从项目中导入3D边界框相关的常量定义


def box3d_to_corners(box3d):  # 将3D边界框参数转换为8个角点的坐标
    """
    Args:
        box3d (np.ndarray or torch.Tensor): 3D边界框参数，形状为 (N, 7+)，其中7个参数通常为 (x, y, z, w, l, h, yaw)。
                                          x,y,z是中心点坐标, w,l,h是维度, yaw是偏航角。
                                          支持包含额外参数如速度等，但仅使用前7个。
    Returns:
        np.ndarray: 边界框的8个角点坐标，形状为 (N, 8, 3)。
    """
    if isinstance(box3d, torch.Tensor):  # 如果输入是PyTorch张量
        box3d = box3d.detach().cpu().numpy()  # 分离计算图，转移到CPU并转换为numpy数组

    # corners_norm 定义了一个单位立方体的8个角点坐标，顺序经过调整以匹配常用的绘制顺序
    # 原始顺序: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    # 调整后顺序 (通常用于绘制): (0,0,0), (0,0,1), (0,1,1), (0,1,0), (1,0,0), (1,0,1), (1,1,1), (1,1,0)
    # 这里使用的顺序是: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1) -> 索引 [0, 1, 3, 2, 4, 5, 7, 6]
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1).astype(np.float32) # 生成 [8,3] 的单位角点，值为0或1
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]  # 调整角点顺序

    # 将角点中心从(0,0,0)移到(0.5,0.5,0.5)，这样后续乘以维度时，角点是相对于box中心
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])

    # 根据box的维度 (W, L, H) 缩放单位角点
    # box3d[:, None, [W, L, H]] 提取每个box的宽度、长度、高度，并增加维度以进行广播
    # W, L, H 是从 projects.mmdet3d_plugin.core.box3d 导入的常量，分别对应索引 3, 4, 5
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])

    # 围绕 z 轴旋转角点
    # YAW 是从 projects.mmdet3d_plugin.core.box3d 导入的常量，对应yaw角的索引 (通常是6)
    rot_cos = np.cos(box3d[:, YAW])  # 计算yaw角的余弦值
    rot_sin = np.sin(box3d[:, YAW])  # 计算yaw角的正弦值

    # 构建旋转矩阵
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))  # 为每个box创建一个3x3的单位矩阵
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos

    # 应用旋转: (rot_mat @ corners.T).T，这里使用爱因斯坦求和约定更高效
    # corners[..., None] 增加一个维度用于矩阵乘法: (N, 8, 3) -> (N, 8, 3, 1)
    # rot_mat[:, None] 增加一个维度以匹配角点: (N, 3, 3) -> (N, 1, 3, 3)
    # (N, 1, 3, 3) @ (N, 8, 3, 1) -> (N, 8, 3, 1)
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)  # 移除最后一个维度

    # 将旋转后的角点平移到box的中心位置 (X, Y, Z)
    # X, Y, Z 是从 projects.mmdet3d_plugin.core.box3d 导入的常量，分别对应索引 0, 1, 2
    corners += box3d[:, None, :3]  # :3 取 x, y, z 坐标
    return corners  # 返回计算得到的8个角点坐标


def plot_rect3d_on_img(  # 在2D图像上绘制3D矩形框的边界线
    img, num_rects, rect_corners, color=(0, 255, 0), thickness=1
):
    """Plot the boundary lines of 3D rectangular on 2D images.  # 在2D图像上绘制3D矩形框的边界线。

    Args:
        img (numpy.array): 图像的numpy数组。
        num_rects (int): 3D矩形框的数量。
        rect_corners (numpy.array): 3D矩形框角点的坐标，形状应为 [num_rect, 8, 2]。
                                     这些角点是3D角点投影到2D图像平面后的坐标。
        color (tuple[int], optional): 用于绘制边界框的颜色。默认为 (0, 255, 0) (绿色)。
                                      如果color列表的第一个元素是int，则所有框使用相同颜色。
                                      否则，每个框使用color[i]指定的颜色。
        thickness (int, optional): 边界框线条的粗细。默认为 1。
    """
    line_indices = (  # 定义连接哪些角点以形成边界框线条的索引对
        (0, 1),  # 例如，连接第0个角点和第1个角点
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    h, w = img.shape[:2]  # 获取图像的高度和宽度
    for i in range(num_rects):  # 遍历每个矩形框
        corners = np.clip(rect_corners[i], -1e4, 1e5).astype(np.int32)  # 获取当前框的角点，并将其坐标裁剪到合理范围并转换为整数
        for start, end in line_indices:  # 遍历预定义的线条索引对
            # 检查线条的起点和终点是否都在图像的可见区域之外，如果是，则跳过绘制该线条
            if (
                (corners[start, 1] >= h or corners[start, 1] < 0)  # 起点y坐标越界
                or (corners[start, 0] >= w or corners[start, 0] < 0)  # 起点x坐标越界
            ) and (
                (corners[end, 1] >= h or corners[end, 1] < 0)  # 终点y坐标越界
                or (corners[end, 0] >= w or corners[end, 0] < 0)  # 终点x坐标越界
            ):
                continue

            current_color = color  # 默认使用传入的color
            if not isinstance(color[0], int): # 如果color不是单个颜色元组 (即，是一个颜色列表)
                current_color = color[i] # 则为当前框选择对应的颜色

            cv2.line(  # 使用OpenCV的line函数绘制线条
                img,  # 目标图像
                (corners[start, 0], corners[start, 1]),  # 线条起点坐标
                (corners[end, 0], corners[end, 1]),  # 线条终点坐标
                current_color,  # 线条颜色
                thickness,  # 线条粗细
                cv2.LINE_AA,  # 线条类型 (抗锯齿)
            )

    return img.astype(np.uint8)  # 返回绘制了边界框的图像，并确保数据类型为uint8


def draw_lidar_bbox3d_on_img(  # 将3D激光雷达边界框投影到2D平面并绘制在输入图像上
    bboxes3d, raw_img, lidar2img_rt, img_metas=None, color=(0, 255, 0), thickness=1
):
    """Project the 3D bbox on 2D plane and draw on input image. # 将3D边界框投影到2D平面并在输入图像上绘制。

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes` or np.ndarray or torch.Tensor): # 要可视化的激光雷达坐标系中的3D边界框。
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): 图像的numpy数组。
        lidar2img_rt (numpy.array, shape=[4, 4]): 根据相机内参得到的投影矩阵。
        img_metas (dict): 此处未使用。
        color (tuple[int], optional): 用于绘制边界框的颜色。默认为 (0, 255, 0)。
        thickness (int, optional): 边界框的粗细。默认为 1。
    """
    img = raw_img.copy()  # 复制原始图像，避免直接修改
    corners_3d = box3d_to_corners(bboxes3d)  # 将3D边界框参数转换为8个角点的3D坐标
    num_bbox = corners_3d.shape[0]  # 获取边界框的数量
    pts_4d = np.concatenate(  # 将3D角点坐标转换为齐次坐标 (增加一维，值为1)
        [corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1
    )
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)  # 深拷贝投影矩阵并确保形状为4x4
    if isinstance(lidar2img_rt, torch.Tensor):  # 如果投影矩阵是PyTorch张量
        lidar2img_rt = lidar2img_rt.cpu().numpy()  # 转换为NumPy数组
    pts_2d = pts_4d @ lidar2img_rt.T  # 将4D角点通过投影矩阵转换到图像坐标系 (得到的是齐次坐标)

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)  # 裁剪深度值(z坐标)，防止除以过小或过大的数
    pts_2d[:, 0] /= pts_2d[:, 2]  # 归一化x坐标 (x = x/z)
    pts_2d[:, 1] /= pts_2d[:, 2]  # 归一化y坐标 (y = y/z)
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)  # 提取2D坐标 (x,y) 并重塑为 [num_bbox, 8, 2]

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)  # 调用plot_rect3d_on_img在图像上绘制2D矩形


def draw_points_on_img(points, img, lidar2img_rt, color=(0, 255, 0), circle=4): # 在图像上绘制点 (例如激光雷达点)
    """
    Args:
        points (torch.Tensor or np.ndarray): 点云数据，形状为 (N, num_points, 3) 或 (num_points, 3)。
        img (numpy.array): 目标图像。
        lidar2img_rt (numpy.array or torch.Tensor): 激光雷达到图像的投影矩阵 (4x4)。
        color (tuple[int] or list[tuple[int]]): 点的颜色。如果是单个元组，所有点使用相同颜色。
                                                如果是列表，则每个点集使用列表中的对应颜色。
        circle (int): 绘制圆点的半径。
    Returns:
        numpy.array: 绘制了点的图像。
    """
    img = img.copy()  # 复制图像，避免修改原图
    N = points.shape[0] if len(points.shape) == 3 else 1 # 获取点集的数量 (如果是单个点集，N=1)
    if len(points.shape) == 2: # 如果输入是单个点集，增加一个维度以统一处理
        points = points.unsqueeze(0) if isinstance(points, torch.Tensor) else points[None, ...]

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()  # 如果是Tensor，转为Numpy

    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)  # 深拷贝投影矩阵
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()  # 如果投影矩阵是Tensor，转为Numpy

    # 将点云从激光雷达坐标系转换到图像坐标系
    # (N, num_points, 3) @ (3, 3) -> (N, num_points, 3)
    # lidar2img_rt[:3, :3] 是旋转部分， lidar2img_rt[:3, 3] 是平移部分
    pts_2d = (
        np.sum(points[:, :, None, :] * lidar2img_rt[None, None, :3, :3], axis=-1) # 旋转
        + lidar2img_rt[None, None, :3, 3] # 平移
    )
    # 另一种写法: pts_2d = points @ lidar2img_rt[:3, :3].T + lidar2img_rt[:3, 3]

    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=1e5)  # 裁剪深度值
    pts_2d = pts_2d[..., :2] / pts_2d[..., 2:3]  # 归一化x,y坐标
    pts_2d = np.clip(pts_2d, -1e4, 1e4).astype(np.int32)  # 裁剪坐标到图像范围并转为整数

    for i in range(N):  # 遍历每个点集 (如果N=1，则只遍历一次)
        for point in pts_2d[i]:  # 遍历当前点集中的每个点
            if isinstance(color[0], int):  # 判断颜色是单个还是列表
                color_tmp = color
            else:
                color_tmp = color[i]  # 为当前点集选择颜色
            cv2.circle(img, tuple(point.tolist()), circle, color_tmp, thickness=-1)  # 绘制圆点
    return img.astype(np.uint8)  # 返回带有点的图像


def draw_lidar_bbox3d_on_bev(  # 在鸟瞰图(BEV)上绘制3D激光雷达边界框
    bboxes_3d, bev_size, bev_range=115, color=(255, 0, 0), thickness=3):
    """
    Args:
        bboxes_3d (np.ndarray or torch.Tensor): 3D边界框参数，形状为 (N, 7+)。
        bev_size (int or tuple[int]): BEV图像的尺寸。如果是单个整数，则高宽相同。
        bev_range (float or int): BEV图像表示的物理范围 (米)。默认为115米。
                                   假设BEV图像中心对应物理世界的(0,0)点。
        color (tuple[int] or list[tuple[int]]): 边界框颜色。
        thickness (int): 线条粗细。
    Returns:
        numpy.array: 绘制了边界框的BEV图像。
    """
    if isinstance(bev_size, (list, tuple)):  # 判断bev_size是列表/元组还是单个值
        bev_h, bev_w = bev_size  # 如果是列表/元组，分别赋值给高和宽
    else:
        bev_h, bev_w = bev_size, bev_size  # 如果是单个值，高和宽相同
    bev = np.zeros([bev_h, bev_w, 3], dtype=np.uint8)  # 创建一个黑色的BEV图像画布

    marking_color = (127, 127, 127)  # 定义标记线 (如距离圈和坐标轴) 的颜色
    bev_resolution = bev_range / bev_h  # 计算BEV图像的分辨率 (米/像素)

    # 绘制距离参考圈
    for cir in range(int(bev_range / 2 / 10)):  # 以10米为间隔绘制同心圆
        cv2.circle(
            bev,  # 目标图像
            (int(bev_w / 2), int(bev_h / 2)),  # 圆心 (图像中心)
            int((cir + 1) * 10 / bev_resolution),  # 半径 (将物理距离转换为像素)
            marking_color,  # 颜色
            thickness=thickness,  # 线宽
        )
    # 绘制中心坐标轴
    cv2.line(
        bev, (0, int(bev_h / 2)), (bev_w, int(bev_h / 2)), marking_color, thickness=1 # x轴
    )
    cv2.line(
        bev, (int(bev_w / 2), 0), (int(bev_w / 2), bev_h), marking_color, thickness=1 # y轴
    )

    if len(bboxes_3d) != 0:  # 如果存在边界框
        # 将3D边界框转换为BEV下的2D角点 (通常取物体的底面或顶面四个角点)
        # 这里选择的角点索引 [0, 3, 4, 7] 可能对应于 (前左下, 前右下, 前左上, 前右上) 或类似组合，具体取决于box3d_to_corners的角点顺序
        # [..., [0, 1]] 表示只取x,y坐标
        bev_corners = box3d_to_corners(bboxes_3d)[:, [0, 3, 7, 4]][
            ..., [0, 1] # 取x,y坐标 (对应BEV平面)
        ]
        # 将物理坐标转换为BEV图像像素坐标
        # x坐标：物理x / 分辨率 + 图像宽度的一半 (因为图像中心是(0,0))
        # y坐标：-物理y / 分辨率 + 图像高度的一半 (负号是因为图像y轴通常向下，而车辆坐标系y轴通常向前或向左)
        xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
        ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2

        for obj_idx, (x, y) in enumerate(zip(xs, ys)):  # 遍历每个物体的角点坐标
            for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):  # 定义连接角点的顺序以形成矩形
                current_color = color
                if not isinstance(color[0], int): # 如果颜色是列表
                    current_color = color[obj_idx] # 为当前物体选择颜色
                cv2.line(  # 绘制矩形的边
                    bev,
                    (int(x[p1]), int(y[p1])),
                    (int(x[p2]), int(y[p2])),
                    current_color,
                    thickness=thickness,
                )
    return bev.astype(np.uint8)  # 返回绘制好的BEV图像


def draw_lidar_bbox3d(bboxes_3d, imgs, lidar2imgs, color=(255, 0, 0)): # 在多视图图像和BEV图上绘制3D激光雷达边界框
    """
    Args:
        bboxes_3d (np.ndarray or torch.Tensor): 要绘制的3D边界框。
        imgs (list[np.ndarray]): 多视图图像列表。
        lidar2imgs (list[np.ndarray]): 对应的激光雷达到各图像的投影矩阵列表。
        color (tuple[int] or list[tuple[int]]): 边界框颜色。
    Returns:
        np.ndarray: 拼接了所有视图和BEV图的可视化结果图像。
    """
    vis_imgs = []  # 初始化存储绘制了边界框的图像列表
    for i, (img, lidar2img) in enumerate(zip(imgs, lidar2imgs)):  # 遍历每个图像和对应的投影矩阵
        vis_imgs.append(  # 调用 draw_lidar_bbox3d_on_img 函数在当前图像上绘制边界框
            draw_lidar_bbox3d_on_img(bboxes_3d, img, lidar2img, color=color)
        )

    num_imgs = len(vis_imgs)  # 获取图像数量
    if num_imgs < 4 or num_imgs % 2 != 0:  # 如果图像数量小于4或为奇数，则水平拼接所有图像
        vis_imgs_concat = np.concatenate(vis_imgs, axis=1)
    else:  # 否则，通常假设为6个视图，分两行拼接 (例如，前左、前、前右在上，后左、后、后右在下)
        vis_imgs_concat = np.concatenate([
            np.concatenate(vis_imgs[:num_imgs//2], axis=1),  # 拼接上半部分图像
            np.concatenate(vis_imgs[num_imgs//2:], axis=1) # 拼接下半部分图像
        ], axis=0) # 垂直拼接上下两部分

    # 绘制BEV图，尺寸参考拼接后的多视图图像高度
    bev = draw_lidar_bbox3d_on_bev(bboxes_3d, vis_imgs_concat.shape[0], color=color)
    vis_imgs_final = np.concatenate([bev, vis_imgs_concat], axis=1)  # 将BEV图与拼接后的多视图图像水平拼接
    return vis_imgs_final # 返回最终的可视化结果
