import os
import glob
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset

from tools.visualization.bev_render import BEVRender
from tools.visualization.cam_render import CamRender

plot_choices = dict(
    draw_pred = True, # True: draw gt and pred; False: only draw gt
    det = True,
    track = True, # True: draw history tracked boxes
    motion = True,
    map = True,
    planning = True,
)
START = 0
END = 81
INTERVAL = 1


class Visualizer: # 定义可视化器类
    """
    用于生成和组合BEV及相机视图可视化结果的主类。
    它会加载数据集、模型预测结果，并调用BEVRender和CamRender来生成图像，
    然后将这些图像组合起来，并能选择性地生成视频。
    """
    def __init__(
        self,
        args, # 命令行参数对象
        plot_choices, # 控制绘制内容的字典
    ):
        """
        可视化器构造函数。

        Args:
            args (argparse.Namespace): 解析后的命令行参数。需要包含:
                                         - out_dir (str): 总的输出目录。
                                         - config (str): 模型和数据配置文件路径。
                                         - result_path (str): 预计算的模型预测结果文件路径 (.pkl)。
            plot_choices (dict): 控制在BEV和相机视图上绘制哪些元素的字典。
                                 例如: {'det': True, 'map': True, ...}
        """
        self.out_dir = args.out_dir # 设置总的输出目录
        self.combine_dir = os.path.join(self.out_dir, 'combine') # 设置组合图像的保存目录
        os.makedirs(self.combine_dir, exist_ok=True) # 创建组合图像目录，如果已存在则不报错

        cfg = Config.fromfile(args.config) # 从配置文件加载配置
        # 注意：这里加载的是cfg.data.val，意味着通常基于验证集进行可视化
        # 如果要可视化训练集结果，需要修改或确保args.result_path对应训练集的结果
        self.dataset = build_dataset(cfg.data.val) # 构建数据集对象 (通常是验证集)
        
        # 加载预先计算好的模型预测结果
        # 如果 args.result_path 为 None (例如只想可视化GT)，则 self.results 将为 None
        self.results = None
        if args.result_path and os.path.exists(args.result_path):
            print(f"从 {args.result_path} 加载预测结果...")
            self.results = mmcv.load(args.result_path)
        elif args.result_path:
            print(f"警告: 指定的结果文件 {args.result_path} 不存在。")
        else:
            print("信息: 未提供结果文件路径，plot_choices中的 'draw_pred' 将无效 (如果为True)。")

        # 初始化BEV和相机视图渲染器
        self.bev_render = BEVRender(plot_choices, self.out_dir) # 创建BEV渲染器实例
        self.cam_render = CamRender(plot_choices, self.out_dir) # 创建相机视图渲染器实例

    def add_vis(self, index):
        data = self.dataset.get_data_info(index)
        result = self.results[index]['img_bbox']

        bev_gt_path, bev_pred_path = self.bev_render.render(data, result, index)
        cam_pred_path = self.cam_render.render(data, result, index)
        self.combine(bev_gt_path, bev_pred_path, cam_pred_path, index)
    
    def combine(self, bev_gt_path, bev_pred_path, cam_pred_path, index):
        bev_gt = cv2.imread(bev_gt_path)
        bev_image = cv2.imread(bev_pred_path)
        cam_image = cv2.imread(cam_pred_path)
        merge_image = cv2.hconcat([cam_image, bev_image, bev_gt])
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + '.jpg')
        cv2.imwrite(save_path, merge_image)

    def image2video(self, fps=12, downsample=4):
        imgs_path = glob.glob(os.path.join(self.combine_dir, '*.jpg'))
        imgs_path = sorted(imgs_path)
        img_array = []
        for img_path in tqdm(imgs_path):
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height //
                             downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out_path = os.path.join(self.out_dir, 'video.mp4')
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--result-path', 
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized')
    parser.add_argument(
        '--out-dir', 
        default='vis',
        help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args

def main(): # 主执行函数
    args = parse_args() # 解析命令行参数

    # 根据是否提供了结果文件路径，来决定是否绘制预测结果
    current_plot_choices = plot_choices.copy() # 复制全局的plot_choices以可能进行修改
    if args.result_path is None or not os.path.exists(args.result_path):
        # 如果没有提供有效的结果文件路径，则不绘制预测，只绘制GT（如果对应GT的绘制选项为True）
        print("警告: 未提供有效的 --result-path 参数或文件不存在。plot_choices中的 'draw_pred' 将被强制设为 False。")
        current_plot_choices['draw_pred'] = False

    visualizer = Visualizer(args, current_plot_choices) # 初始化可视化器实例

    # 确定要可视化的样本数量
    # 如果有预测结果，则可视化范围不能超过结果的数量
    # 否则，可视化范围不能超过数据集的数量
    num_items_to_process = len(visualizer.dataset)
    if visualizer.results is not None:
        num_items_to_process = min(len(visualizer.dataset), len(visualizer.results))

    actual_end_idx = min(END, num_items_to_process) # 确保结束索引不超过可用样本数

    print(f"开始可视化样本范围: START={START}, END={actual_end_idx}, INTERVAL={INTERVAL}")
    # 遍历指定范围的样本进行可视化
    for idx in tqdm(range(START, actual_end_idx, INTERVAL)): # 使用tqdm显示进度
        # add_vis内部会处理result为None的情况（只绘制GT）
        print(f"\n正在处理样本索引: {idx}")
        visualizer.add_vis(idx) # 为当前样本生成并组合可视化图像
    
    # 将生成的图像序列合成为视频
    if actual_end_idx > START : # 只有在实际处理了图像后才尝试生成视频
        print("\n开始将图像序列合成为视频...")
        visualizer.image2video()
        print("可视化完成。")
    else:
        print("没有样本被可视化，跳过视频生成。")

if __name__ == '__main__':
    main() # 执行主函数