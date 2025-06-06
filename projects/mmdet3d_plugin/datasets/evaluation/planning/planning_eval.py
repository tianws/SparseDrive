from tqdm import tqdm # 导入tqdm库，用于显示进度条
import torch # 导入PyTorch库
import torch.nn as nn # 导入PyTorch神经网络模块
import numpy as np # 导入NumPy库
from shapely.geometry import Polygon # 从shapely库导入Polygon，用于几何对象的表示和操作 (例如碰撞检测)

from mmcv.utils import print_log # 从MMCV导入打印日志的函数
from mmdet.datasets import build_dataset, build_dataloader # 从MMDetection导入构建数据集和数据加载器的函数

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners # 从项目中导入将3D框转换为角点的工具函数


def check_collision(ego_box: torch.Tensor, boxes: torch.Tensor) -> bool: # 检查自车（ego）是否与场景中的其他物体发生碰撞
    '''
        ego_box: tensor with shape [7], [x, y, z, w, l, h, yaw] # 自车的3D边界框参数 (7个参数)
        boxes: tensor with shape [N, 7] # 场景中其他物体的3D边界框参数 (N个物体)
    '''
    if boxes.shape[0] == 0: # 如果场景中没有其他物体
        return False # 则不可能发生碰撞

    # 参照UniAD的实现，在自车前进方向上增加一个0.5米的偏移，用于更保守的碰撞检测
    # 这相当于将自车的碰撞检测点略微前移
    ego_box_offset = ego_box.clone() # 复制自车框以避免修改原始数据
    ego_box_offset[0] += 0.5 * torch.cos(ego_box[6]) # x方向偏移
    ego_box_offset[1] += 0.5 * torch.sin(ego_box[6]) # y方向偏移 (假设yaw=0时车头朝向x轴正方向)

    # 将自车框转换为BEV平面上的角点 (通常是4个角点)
    # box3d_to_corners返回(1, 8, 3)，取[0, [0,3,7,4], :2]得到BEV下的4个角点 (x,y)
    # 角点顺序通常是 front-left, front-right, rear-right, rear-left (具体顺序取决于box3d_to_corners实现)
    ego_corners_bev = box3d_to_corners(ego_box_offset.unsqueeze(0))[0, [0, 3, 7, 4], :2]

    # 将场景中其他物体的框也转换为BEV平面上的角点
    other_boxes_corners_bev = box3d_to_corners(boxes)[:, [0, 3, 7, 4], :2] # (N, 4, 2)

    # 使用shapely创建自车的BEV多边形
    ego_poly = Polygon([(point[0].item(), point[1].item()) for point in ego_corners_bev])

    # 遍历场景中的每个其他物体
    for i in range(len(other_boxes_corners_bev)):
        # 创建当前物体的BEV多边形
        box_poly = Polygon([(point[0].item(), point[1].item()) for point in other_boxes_corners_bev[i]])
        # 检查自车多边形是否与当前物体多边形相交
        if ego_poly.intersects(box_poly): # .intersects判断两个几何对象是否相交
            return True # 如果发生碰撞，立即返回True

    return False # 如果遍历完所有物体都未发生碰撞，则返回False

def get_yaw(traj: torch.Tensor) -> torch.Tensor: # 从轨迹点计算偏航角
    """
    根据输入的轨迹点序列计算每个时间步的偏航角。
    偏航角是根据相邻点之间的方向计算得到的。

    Args:
        traj (torch.Tensor): 轨迹点序列，形状为 (num_timesteps, 2)，包含(x,y)坐标。

    Returns:
        torch.Tensor: 每个时间步的偏航角（弧度），形状为 (num_timesteps,)。
    """
    # 检查轨迹是否过短（例如，车辆几乎未移动）
    start_pos = traj[0]
    end_pos = traj[-1]
    dist = torch.linalg.norm(end_pos - start_pos, dim=-1) # 计算起点到终点的距离
    if dist < 0.5: # 如果总位移小于0.5米，认为车辆几乎未动或方向不可靠
        # 返回一个固定的偏航角 (例如，np.pi/2，表示朝向y轴正方向，或根据车辆初始朝向设定)
        # 这里返回的是一个与输入轨迹长度相同，值都为pi/2的张量
        return traj.new_ones(traj.shape[0]) * (np.pi / 2)

    # 在轨迹起点前添加一个(0,0)点（或轨迹的第一个点），方便计算第一个有效点的朝向
    # zeros = traj.new_zeros((1, 2)) # 创建一个(0,0)点
    # traj_cat = torch.cat([zeros, traj], dim=0) # (num_timesteps+1, 2)
    # 更合理的做法是复制第一个点，或者使用其他方式处理边界
    first_point_expanded = traj[0:1] # (1,2)
    traj_cat = torch.cat([first_point_expanded, traj], dim=0) # (num_timesteps+1, 2)


    yaw = traj.new_zeros(traj.shape[0]+1) # 初始化yaw角数组，长度比原轨迹多1

    # 计算中间点的偏航角：使用 (P[i+1] - P[i-1]) 的方向
    # traj_cat[..., 2:, 1] (y_{i+1}), traj_cat[..., :-2, 1] (y_{i-1})
    yaw[1:-1] = torch.atan2(
        traj_cat[2:, 1] - traj_cat[:-2, 1], # dy = y_{i+1} - y_{i-1}
        traj_cat[2:, 0] - traj_cat[:-2, 0]  # dx = x_{i+1} - x_{i-1}
    )
    # 计算最后一个点的偏航角：使用 (P[end] - P[end-1]) 的方向
    yaw[-1] = torch.atan2(
        traj_cat[-1, 1] - traj_cat[-2, 1],
        traj_cat[-1, 0] - traj_cat[-2, 0],
    )
    # 第一个点的偏航角也用 P[1]-P[0] 的方向 (因为traj_cat[0]=traj_cat[1]=original_traj[0])
    # 这会导致第一个yaw为0，除非traj[1]和traj[0]不同。
    # 如果traj是相对位移累加且起点是(0,0)，那么traj[0]是第一个位移点。
    # 这里的逻辑是，yaw[0]对应traj_cat[0] (即原始轨迹的第一个点)，yaw[1]对应traj_cat[1] (原始轨迹的第二个点)
    # 所以yaw[0]的计算应该基于 traj_cat[1] - traj_cat[0]。
    # 但由于traj_cat[0]是复制的traj[0]，如果traj[0]是(0,0)，那么yaw[0]会基于traj[1]-(0,0)计算
    # 为了保持与后面一致，第一个有效yaw (对应原始traj的第一个点)应该用traj_cat[1]和traj_cat[0]
    # 但如果traj_cat[0]是(0,0)，则这个yaw是traj[0]相对于原点的方向。
    # 原始代码中，yaw[0]会是start_yaw (在get_yaw的rescore版本中)，这里没有start_yaw参数。
    # 我们假设第一个点的yaw可以由第一个有效位移决定。
    if traj.shape[0] > 0 : # 确保轨迹至少有一个点
        yaw[0] = torch.atan2(traj_cat[1,1]-traj_cat[0,1], traj_cat[1,0]-traj_cat[0,0]) if traj.shape[0] > 1 else (np.pi/2)
        if traj.shape[0] == 1: yaw[1] = yaw[0] # 如果只有一个点，则最后一个点的yaw也一样

    return yaw[1:] # 返回与原始traj长度一致的yaw角序列

class PlanningMetric(): # 定义规划指标计算类
    """
    用于计算和累积自车路径规划相关评估指标的类。
    主要指标包括：
    - obj_col: 与场景中其他物体发生碰撞的GT轨迹的碰撞率。
    - obj_box_col: 预测轨迹引入的新碰撞率（即GT不碰，但预测碰了）。
    - L2: 预测轨迹与GT轨迹之间的L2距离。
    """
    def __init__(
        self,
        n_future=6, # 评估的未来时间步数量 (与ego_fut_ts对应)
        compute_on_step: bool = False, # (未使用) 是否在每一步计算指标
    ):
        self.W = 1.85 # 自车的宽度 (米)，用于构建自车包围框
        self.H = 4.084 # 自车的长度 (米)

        self.n_future = n_future # 未来时间步数量
        self.reset() # 初始化或重置所有累积指标

    def reset(self): # 重置累积的指标值
        """将所有累积的指标清零，并将总样本数清零。"""
        self.obj_col = torch.zeros(self.n_future) # GT轨迹在每个未来时间步的碰撞次数
        self.obj_box_col = torch.zeros(self.n_future) # 预测轨迹在每个未来时间步的新增碰撞次数
        self.L2 = torch.zeros(self.n_future) # L2距离在每个未来时间步的累积和
        self.total = torch.tensor(0, dtype=torch.long) # 处理的总样本数量

    def evaluate_single_coll(self, traj_ego_bev: torch.Tensor, fut_boxes_agents: List[torch.Tensor]) -> torch.Tensor:
        """
        评估单条自车轨迹在每个未来时间步是否与其他物体发生碰撞。

        Args:
            traj_ego_bev (torch.Tensor): 自车的未来轨迹点序列 (在BEV平面)，形状 (n_future, 2)。
            fut_boxes_agents (List[torch.Tensor]): 一个列表，长度为n_future。
                                                 每个元素是该时间步场景中其他物体的3D边界框，
                                                 形状为 (num_agents_at_t, 7)。

        Returns:
            torch.Tensor: 布尔张量，形状 (n_future,)，表示自车在每个时间步是否发生碰撞。
        """
        n_future = traj_ego_bev.shape[0]
        # 计算自车在轨迹上每个点的偏航角
        yaw_ego = get_yaw(traj_ego_bev) # (n_future,)

        # 构建自车在每个未来时间步的7自由度边界框 (x, y, z, H, W, h_ego, yaw)
        # z和h_ego使用固定值，因为碰撞主要在BEV平面考虑
        ego_boxes_future = traj_ego_bev.new_zeros((n_future, 7))
        ego_boxes_future[:, :2] = traj_ego_bev # 设置x,y坐标
        ego_boxes_future[:, 2] = -1.8 # 设置一个固定的z坐标 (例如地面高度或车辆中心z)
        ego_boxes_future[:, 3:6] = ego_boxes_future.new_tensor([self.H, self.W, 1.56]) # 设置尺寸 (L,W,H_ego) - 注意H,W顺序
        ego_boxes_future[:, 6] = yaw_ego # 设置偏航角

        collision_at_each_step = torch.zeros(n_future, dtype=torch.bool, device=traj_ego_bev.device)

        for t in range(n_future): # 遍历每个未来时间步
            current_ego_box_at_t = ego_boxes_future[t].clone() # 当前时间步的自车框
            # fut_boxes_agents[t] 是一个包含单个张量的列表，所以取[0]
            # other_agents_boxes_at_t: (num_agents_at_t, 7)
            other_agents_boxes_at_t = fut_boxes_agents[t][0].clone() if fut_boxes_agents[t] and fut_boxes_agents[t][0].numel() > 0 else torch.empty(0,7, device=current_ego_box_at_t.device)
            
            # 检查当前时间步的自车框是否与任何其他物体框发生碰撞
            collision_at_each_step[t] = check_collision(current_ego_box_at_t, other_agents_boxes_at_t)

        return collision_at_each_step

    def evaluate_coll(self, pred_ego_trajs_bev: torch.Tensor, gt_ego_trajs_bev: torch.Tensor, fut_boxes_agents_batch: List[List[torch.Tensor]]):
        """
        评估一批预测轨迹和GT轨迹的碰撞情况。

        Args:
            pred_ego_trajs_bev (torch.Tensor): 预测的自车BEV轨迹，形状 (B, n_future, 2)。B是批量大小。
            gt_ego_trajs_bev (torch.Tensor):   真实的自车BEV轨迹，形状 (B, n_future, 2)。
            fut_boxes_agents_batch (List[List[torch.Tensor]]): 批量的其他物体未来边界框。
                                                              外层列表长度B，内层列表长度n_future。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - obj_coll_sum (torch.Tensor): GT轨迹在各时间步的总碰撞次数 (B, n_future)。
                - obj_box_coll_sum (torch.Tensor): 预测轨迹引入的新增碰撞在各时间步的总次数 (B, n_future)。
        """
        B, n_future, _ = pred_ego_trajs_bev.shape
        # 注意：轨迹的x坐标在传入时被取反，这里再取反以匹配通用坐标系（如果需要）
        # 但通常评估是在统一的坐标系下进行，这里的取反操作需要谨慎确认其必要性。
        # 假设这里的坐标系是统一的，不需要额外取反。
        # pred_ego_trajs_bev = pred_ego_trajs_bev * torch.tensor([-1, 1], device=pred_ego_trajs_bev.device)
        # gt_ego_trajs_bev = gt_ego_trajs_bev * torch.tensor([-1, 1], device=gt_ego_trajs_bev.device)

        obj_coll_sum_batch = torch.zeros(n_future, device=pred_ego_trajs_bev.device)
        obj_box_coll_sum_batch = torch.zeros(n_future, device=pred_ego_trajs_bev.device)

        # 当前实现只支持batch_size=1的评估
        assert B == 1, 'evaluate_coll in PlanningMetric currently only supports batch_size=1'

        for i in range(B): # 遍历batch中的每个样本 (实际只有1个)
            # 评估GT轨迹的碰撞情况
            gt_box_coll_steps = self.evaluate_single_coll(gt_ego_trajs_bev[i], fut_boxes_agents_batch[i])
            # 评估预测轨迹的碰撞情况
            pred_box_coll_steps = self.evaluate_single_coll(pred_ego_trajs_bev[i], fut_boxes_agents_batch[i])

            # 计算新增碰撞：预测发生碰撞，且GT未发生碰撞
            newly_collided_steps = torch.logical_and(pred_box_coll_steps, torch.logical_not(gt_box_coll_steps))

            obj_coll_sum_batch += gt_box_coll_steps.long() # 累加GT碰撞次数
            obj_box_coll_sum_batch += newly_collided_steps.long() # 累加新增碰撞次数

        return obj_coll_sum_batch, obj_box_coll_sum_batch

    def compute_L2(self, pred_trajs_bev: torch.Tensor, gt_trajs_bev: torch.Tensor, gt_trajs_mask: torch.Tensor) -> torch.Tensor:
        '''
        计算预测轨迹和真实轨迹之间的L2距离。
        Args:
            pred_trajs_bev (torch.Tensor): 预测轨迹 (B, n_future, 2)。
            gt_trajs_bev (torch.Tensor):   真实轨迹 (B, n_future, 2)。
            gt_trajs_mask (torch.Tensor):  真实轨迹的有效性掩码 (B, n_future, 2)。
        Returns:
            torch.Tensor: 每个时间步的L2距离 (B, n_future)。
        '''
        # ((pred_xy - gt_xy)^2 * mask).sum(dim=-1) -> (B, n_future) L2距离的平方，已mask
        # torch.sqrt(...) -> (B, n_future) L2距离
        return torch.sqrt((((pred_trajs_bev - gt_trajs_bev) ** 2) * gt_trajs_mask).sum(dim=-1))

    def update(self, pred_ego_trajs_bev: torch.Tensor, gt_ego_trajs_bev: torch.Tensor,
               gt_trajs_mask: torch.Tensor, fut_boxes_agents_batch: List[List[torch.Tensor]]):
        """
        用一个批次的评估结果更新累积的指标。
        Args:
            pred_ego_trajs_bev (torch.Tensor): 预测的自车BEV轨迹 (B, n_future, 2)。
            gt_ego_trajs_bev (torch.Tensor):   真实的自车BEV轨迹 (B, n_future, 2)。
            gt_trajs_mask (torch.Tensor):      真实轨迹的有效性掩码 (B, n_future, 2)。
            fut_boxes_agents_batch (List[List[torch.Tensor]]): 批量的其他物体未来边界框。
        """
        assert pred_ego_trajs_bev.shape == gt_ego_trajs_bev.shape # 确保预测和GT轨迹形状一致

        # 坐标系调整：x坐标取反 (这通常是为了匹配特定的坐标系约定，例如nuScenes中x轴向右，y轴向前)
        # 如果输入已经是期望的坐标系，则不需要此操作。
        # 假设这里的输入是需要调整的。
        pred_ego_trajs_adjusted = pred_ego_trajs_bev.clone()
        pred_ego_trajs_adjusted[..., 0] = -pred_ego_trajs_adjusted[..., 0]
        gt_ego_trajs_adjusted = gt_ego_trajs_bev.clone()
        gt_ego_trajs_adjusted[..., 0] = -gt_ego_trajs_adjusted[..., 0]

        # 计算L2距离
        l2_distances_per_step = self.compute_L2(pred_ego_trajs_adjusted, gt_ego_trajs_adjusted, gt_trajs_mask) # (B, n_future)
        # 计算碰撞
        obj_coll_sum_batch, obj_box_coll_sum_batch = self.evaluate_coll(
            pred_ego_trajs_adjusted[...,:2], # evaluate_coll期望输入是(B, n_future, 2)
            gt_ego_trajs_adjusted[...,:2],
            fut_boxes_agents_batch
        ) # (n_future), (n_future) - 因为evaluate_coll内部假设B=1并已处理

        # 累积指标
        self.obj_col += obj_coll_sum_batch.cpu() # GT碰撞累积 (转移到CPU以防万一)
        self.obj_box_col += obj_box_coll_sum_batch.cpu() # 新增碰撞累积
        self.L2 += l2_distances_per_step.sum(dim=0).cpu() # L2距离按时间步累积 (先在batch维度求和)
        self.total += pred_ego_trajs_bev.shape[0] # 累加处理的样本数量

    def compute(self) -> Dict[str, torch.Tensor]: # 计算最终的平均指标
        """
        计算所有累积指标的平均值。
        Returns:
            dict: 包含平均碰撞率和平均L2距离的字典。
                  值是每个未来时间步的平均指标张量。
        """
        if self.total == 0: # 防止除以零
            return {
                'obj_col': torch.zeros(self.n_future),
                'obj_box_col': torch.zeros(self.n_future),
                'L2' : torch.zeros(self.n_future)
            }
        return {
            'obj_col': self.obj_col / self.total, # 平均GT碰撞率
            'obj_box_col': self.obj_box_col / self.total, # 平均新增碰撞率
            'L2' : self.L2 / self.total # 平均L2距离
        }


def planning_eval(results: List[Dict], eval_config: Config, logger: Logger) -> Dict[str, float]: # 主规划评估函数
    """
    执行规划评估的总体流程。

    Args:
        results (List[Dict]): 模型的预测结果列表，每个字典对应一个样本。
                              每个字典应包含 'img_bbox'键，其值又是一个字典，包含 'final_planning'等。
        eval_config (Config): 用于评估的数据集配置。
        logger (Logger): 用于打印日志的记录器。

    Returns:
        Dict[str, float]: 包含各项平均规划指标的字典。
    """
    dataset = build_dataset(eval_config) # 根据评估配置构建数据集 (主要用于加载GT)
    dataloader = build_dataloader( # 构建数据加载器
            dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False)

    planning_metrics_calculator = PlanningMetric(n_future=dataset.ego_fut_ts if hasattr(dataset, 'ego_fut_ts') else 6) # 初始化指标计算器
                                                                                                # 从dataset获取n_future

    print_log("开始规划评估...", logger=logger)
    for i, data_batch in enumerate(tqdm(dataloader)): # 遍历数据加载器中的每个GT样本
        # 从GT数据中提取自车规划目标和相关信息
        # gt_ego_fut_trajs已经是相对偏移，cumsum将其变为相对于当前ego位置的轨迹点
        # sdc_planning_gt: (1, 1, ego_fut_ts, 2) - 增加mode和query维度以匹配某些预测格式
        sdc_planning_gt = data_batch['gt_ego_fut_trajs'].cumsum(dim=-2).unsqueeze(1)
        # sdc_planning_mask_gt: (1, 1, ego_fut_ts, 2)
        sdc_planning_mask_gt = data_batch['gt_ego_fut_masks'].unsqueeze(-1).repeat(1, 1, 1, 2).unsqueeze(1)
        # command_idx = data_batch['gt_ego_fut_cmd'].argmax(dim=-1).item() # 当前GT命令 (未使用在此处)

        # fut_boxes_agents_batch: 长度为B的列表，每个元素是长度为n_future的列表，
        #                        其中每个子元素是(num_agents_at_t, 7)的Tensor
        # data_batch['fut_boxes']已经是List[List[Tensor]]格式
        fut_boxes_agents_batch = data_batch['fut_boxes']

        # 如果GT轨迹不完整，则跳过该样本的评估
        if not sdc_planning_mask_gt.all():
            print_log(f"样本 {i} 的GT轨迹不完整，跳过规划评估。", logger=logger, level='warning')
            continue

        if i >= len(results): # 确保预测结果列表足够长
            print_log(f"警告: 样本 {i} 没有对应的预测结果，停止评估。", logger=logger)
            break

        res_sample = results[i] # 当前样本的预测结果
        # 从预测结果中提取最终的规划轨迹
        # pred_sdc_traj: (1, ego_fut_ts, 2) - 增加一个batch维度
        pred_sdc_traj = res_sample['img_bbox']['final_planning'].unsqueeze(0)

        # 更新累积指标
        # 注意：planning_metrics_calculator.update期望的轨迹长度是n_future (初始化时设置)
        # 需要确保pred_sdc_traj和sdc_planning_gt的时间步长与之一致，这里截取到planning_metrics_calculator.n_future
        planning_metrics_calculator.update(
            pred_sdc_traj[:, :planning_metrics_calculator.n_future, :2],
            sdc_planning_gt[:, :, :planning_metrics_calculator.n_future, :2], # gt也取对应长度
            sdc_planning_mask_gt[:, :, :planning_metrics_calculator.n_future, :2], # mask也取对应长度
            fut_boxes_agents_batch # fut_boxes也应该只传递对应n_future的长度
        )
       
    final_planning_results = planning_metrics_calculator.compute() # 计算最终的平均指标
    planning_metrics_calculator.reset() # 重置以便下次可能的调用

    # 使用PrettyTable格式化并打印结果
    planning_metrics_table = PrettyTable()
    planning_metrics_table.field_names = [ # 表头
    "指标", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s", "平均值(1,2,3s)"]

    averaged_results_summary = {} # 用于存储表格中最后一列的“平均值”

    for metric_key, values_over_time in final_planning_results.items(): # 遍历每个指标 (obj_col, obj_box_col, L2)
        values_list = values_over_time.tolist() # (n_future)

        # 计算在特定时间点 (0.5s, 1.0s, ..., 3.0s) 的指标值
        # 假设每秒2个时间步 (即每个时间步0.5s)
        # 0.5s -> index 0 (0.5s * 2 - 1)
        # 1.0s -> index 1 (1.0s * 2 - 1) -> 应该是 index 1 (0.5s), 3 (1.0s), 5 (1.5s)...
        # 时间步索引应该是: 1 (0.5s), 2 (1.0s), 3 (1.5s), 4 (2.0s), 5 (2.5s), 6 (3.0s)
        # 如果n_future=6，则values_list长度为6，对应0.5s到3.0s

        row_values_str = [metric_key] # 行的第一个元素是指标名称
        # 提取对应0.5s, 1.0s, ..., 3.0s的值
        # 假设时间步是0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s
        # 对应索引 0, 1, 2, 3, 4, 5 (如果n_future=6)
        for t_idx in range(planning_metrics_calculator.n_future):
            val = values_list[t_idx]
            if 'col' in metric_key: # 如果是碰撞率，格式化为百分比
                row_values_str.append('%.2f%%' % (float(val) * 100))
            else: # 否则格式化为浮点数
                row_values_str.append('%.4f' % float(val))

        # 计算1s, 2s, 3s三个时间点的平均值
        # 对应索引 1 (1s), 3 (2s), 5 (3s)
        avg_indices = [idx for idx in [1, 3, 5] if idx < len(values_list)] # 确保索引不越界
        if avg_indices:
            avg_val = np.mean([values_list[idx] for idx in avg_indices])
        else: # 如果没有足够的时间步来计算平均值 (例如n_future < 2)
            avg_val = values_list[-1] if values_list else 0.0 # 取最后一个有效值或0

        averaged_results_summary[metric_key] = avg_val # 存储平均值
        if 'col' in metric_key:
            row_values_str.append('%.2f%%' % (float(avg_val) * 100))
        else:
            row_values_str.append('%.4f' % float(avg_val))
        planning_metrics_table.add_row(row_values_str) # 添加到表格

    print_log('\n'+str(planning_metrics_table), logger=logger) # 打印表格到日志
    return averaged_results_summary # 返回包含平均指标的字典
