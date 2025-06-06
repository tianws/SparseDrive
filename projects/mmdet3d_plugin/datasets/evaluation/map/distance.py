from scipy.spatial import distance
from numpy.typing import NDArray # 从NumPy导入类型提示NDArray
import torch # 导入PyTorch库

def chamfer_distance(line1: NDArray, line2: NDArray) -> float: # 计算两条线之间的倒角距离 (Chamfer Distance)
    ''' Calculate chamfer distance between two lines. Make sure the 
    lines are interpolated. # 计算两条线（点云序列）之间的倒角距离。确保输入的线是经过插值的（即由足够多的点组成）。

    倒角距离是一种衡量两组点集之间相似度的度量。对于线段（视为点的有序序列），
    它计算第一条线中每个点到第二条线中最近点的平均距离，以及第二条线中每个点
    到第一条线中最近点的平均距离，然后将这两个平均距离再次平均。

    Args:
        line1 (NDArray): 第一个线段（或点云）的坐标数组，形状为 (num_points1, dims)。
                         dims通常为2（例如BEV下的x,y）或3（x,y,z）。
        line2 (NDArray): 第二个线段（或点云）的坐标数组，形状为 (num_points2, dims)。
    
    Returns:
        float: 计算得到的对称倒角距离。
    '''
    
    # 使用SciPy的cdist函数计算line1中每个点到line2中所有点的欧氏距离矩阵。
    # dist_matrix[i, j] 是 line1的第i个点与line2的第j个点之间的欧氏距离。
    # 最终 dist_matrix 的形状为 (num_points1, num_points2)。
    dist_matrix = distance.cdist(line1, line2, 'euclidean')

    # 计算从line1到line2的单向倒角距离:
    # 1. dist_matrix.min(axis=1) 或 dist_matrix.min(-1): 对于line1中的每个点，找到其到line2中所有点的最小距离。
    #    结果是一个形状为 (num_points1,) 的数组。
    # 2. .sum(): 对这些最小距离求和。
    # 3. / len(line1): 除以line1的点数，计算平均最小距离。
    dist12 = dist_matrix.min(axis=1).sum() / len(line1)

    # 计算从line2到line1的单向倒角距离:
    # 1. dist_matrix.min(axis=0) 或 dist_matrix.min(-2): 对于line2中的每个点（对应dist_matrix的列），
    #    找到其到line1中所有点的最小距离。结果是一个形状为 (num_points2,) 的数组。
    # 2. .sum(): 对这些最小距离求和。
    # 3. / len(line2): 除以line2的点数，计算平均最小距离。
    dist21 = dist_matrix.min(axis=0).sum() / len(line2)

    # 对称倒角距离是两个单向倒角距离的平均值。
    return (dist12 + dist21) / 2

def frechet_distance(line1: NDArray, line2: NDArray) -> float:
    ''' Calculate frechet distance between two lines. Make sure the 
    lines are interpolated.

    Args:
        line1 (array): coordinates of line1
        line2 (array): coordinates of line2
    
    Returns:
        distance (float): frechet distance
    '''
    
    raise NotImplementedError # 抛出NotImplementedError，表示该函数尚未实现

def chamfer_distance_batch(pred_lines, gt_lines): # 计算两组线之间的批量倒角距离
    ''' Calculate chamfer distance between two group of lines. Make sure the 
    lines are interpolated. # 计算两组（批次）线之间的倒角距离。确保输入的线是经过插值的。

    此函数用于高效地计算一组预测线段与一组真实线段之间的所有配对的倒角距离。

    Args:
        pred_lines (array or tensor): 预测的线段组，形状为 (m, num_pts, coord_dims)。
                                      m 是预测线段的数量，num_pts 是每条线的点数，
                                      coord_dims 是坐标维度 (通常为2或3)。
        gt_lines (array or tensor): 真实的线段组，形状为 (n, num_pts, coord_dims)。
                                    n 是真实线段的数量。
    
    Returns:
        distance (NDArray): 倒角距离矩阵，形状为 (m, n)。
                            distance[i, j] 表示第i条预测线段与第j条真实线段之间的倒角距离。
    '''
    # 获取预测线段的点数和坐标维度
    num_preds, num_pts, coord_dims = pred_lines.shape
    num_gts = gt_lines.shape[0]

    # 确保输入是PyTorch张量，以便使用torch.cdist等PyTorch操作
    if not isinstance(pred_lines, torch.Tensor):
        pred_lines = torch.tensor(pred_lines, dtype=torch.float32) # 转换为float32以保证精度
    if not isinstance(gt_lines, torch.Tensor):
        gt_lines = torch.tensor(gt_lines, dtype=torch.float32)

    # 将线段组中的所有点展平，以便计算所有预测点与所有真实点之间的两两距离
    # pred_lines_flat: (m * num_pts, coord_dims)
    # gt_lines_flat: (n * num_pts, coord_dims)
    pred_lines_flat = pred_lines.reshape(-1, coord_dims)
    gt_lines_flat = gt_lines.reshape(-1, coord_dims)

    # 使用torch.cdist计算所有预测点与所有真实点之间的L2距离 (欧氏距离)
    # dist_mat_flat: (m * num_pts, n * num_pts)
    # dist_mat_flat[i*num_pts + k, j*num_pts + l] 是 pred_lines[i]的第k个点与gt_lines[j]的第l个点之间的距离
    dist_mat_flat = torch.cdist(pred_lines_flat, gt_lines_flat, p=2)

    # 将扁平化的距离矩阵重新组织，以反映原始线段的结构，得到一个高维的距离矩阵
    # 目标形状: (num_preds, num_gts, num_pts_pred, num_pts_gt)
    # dist_mat_final[i,j,k,l] 表示 pred_lines[i]的第k个点 与 gt_lines[j]的第l个点 之间的距离

    # 1. 按预测线段的点进行分组:
    #    torch.split(dist_mat_flat, num_pts, dim=0) 将 (m*num_pts, n*num_pts) 切分为 m 个 (num_pts, n*num_pts) 的块
    #    torch.stack(...) 将这些块堆叠起来，形成 (m, num_pts, n*num_pts)
    dist_mat_reshaped1 = torch.stack(torch.split(dist_mat_flat, num_pts, dim=0))

    # 2. 再按真实线段的点进行分组:
    #    torch.split(dist_mat_reshaped1, num_pts, dim=-1) 将最后一维 (n*num_pts) 切分为 n 个 num_pts 大小的块
    #    这会得到一个列表，列表的每个元素是 (m, num_pts, num_pts)。列表长度为n。
    #    torch.stack(..., dim=2) 将这个列表沿新的维度2堆叠起来。
    #    形状变为 (m, num_pts, n, num_pts)
    #    然后用 permute 调整维度顺序为 (m, n, num_pts, num_pts)
    dist_mat_final = torch.stack(torch.split(dist_mat_reshaped1, num_pts, dim=-1), dim=2).permute(0,2,1,3)
    # dist_mat_final[pred_idx, gt_idx, pred_pt_idx, gt_pt_idx]

    # 计算从预测线段到真实线段的单向倒角距离 (对于每个 pred_line 和 gt_line 对)
    # dist_mat_final.min(dim=-1)[0]: 对每个预测线段的每个点，找到其到对应真实线段所有点的最小距离
    #                                结果形状 (m, n, num_pts_pred)
    # .sum(dim=-1): 对每个(pred_line, gt_line)对，将上述最小距离求和
    # dist1[i,j] = sum_{p_pred in pred_lines[i]} min_{p_gt in gt_lines[j]} ||p_pred - p_gt||
    dist1 = dist_mat_final.min(dim=-1)[0].sum(dim=-1) # 形状 (m, n)

    # 计算从真实线段到预测线段的单向倒角距离
    # dist_mat_final.min(dim=-2)[0]: 对每个真实线段的每个点，找到其到对应预测线段所有点的最小距离
    #                                (注意dim=-2对应num_pts_pred)
    #                                结果形状 (m, n, num_pts_gt)
    # .sum(dim=-1): 对每个(pred_line, gt_line)对，将上述最小距离求和
    # dist2[i,j] = sum_{p_gt in gt_lines[j]} min_{p_pred in pred_lines[i]} ||p_gt - p_pred||
    dist2 = dist_mat_final.min(dim=-2)[0].sum(dim=-1) # 形状 (m, n)

    # 计算对称倒角距离，并进行归一化 (除以点数和2)
    # (dist1/num_pts + dist2/num_pts) / 2
    chamfer_dist_matrix = (dist1 + dist2) / (2 * num_pts) # 形状 (m, n)
    
    return chamfer_dist_matrix.cpu().numpy() # 返回NumPy数组