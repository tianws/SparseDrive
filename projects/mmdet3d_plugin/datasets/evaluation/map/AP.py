import numpy as np
from .distance import chamfer_distance, frechet_distance, chamfer_distance_batch
from typing import List, Tuple, Union
from numpy.typing import NDArray # 从NumPy导入类型提示NDArray

def average_precision(recalls, precisions, mode='area'): # 计算平均精度(AP)
    """Calculate average precision. # 计算平均精度。
    此函数根据给定的召回率和精确率点计算平均精度。
    支持两种计算模式：'area'（计算P-R曲线下面积）和 '11points'（11点插值法）。

    Args:
        recalls (ndarray): 召回率数组，形状为 (num_dets, )。应按置信度降序排列对应的检测结果。
        precisions (ndarray): 精确率数组，形状为 (num_dets, )。与recalls对应。
        mode (str): 计算AP的模式。可选值为:
                    'area': 计算精确率-召回率曲线下的面积。这是PASCAL VOC竞赛常用的方法。
                    '11points': 使用11点插值法计算AP。在召回率为[0, 0.1, ..., 1.0]这11个点上取最大精确率求平均。

    Returns:
        float: 计算得到的平均精度值。
    """

    # 将输入的recalls和precisions转换为二维数组，即使只有一个评估尺度/类别。
    # 这样做是为了与更通用的、可能处理多尺度/多类别AP计算的代码兼容，尽管当前实现主要处理单尺度。
    recalls = recalls[np.newaxis, :] # 形状变为 (1, num_dets)
    precisions = precisions[np.newaxis, :] # 形状变为 (1, num_dets)

    assert recalls.shape == precisions.shape and recalls.ndim == 2 # 断言recalls和precisions形状相同且为二维
    num_scales = recalls.shape[0] # 评估的尺度数量，这里通常为1
    ap = 0. # 初始化平均精度为0

    if mode == 'area': # 使用面积法计算AP (PASCAL VOC方法)
        # 在召回率和精确率数组的开头和结尾分别添加辅助点，以正确计算曲线下面积。
        # mrec: [0, r1, r2, ..., rn, 1]
        # mpre: [0, p1, p2, ..., pn, 0] (有些实现中，最后一个点也是0，有些是最后一个precision)
        # 这里mpre末尾添加0，符合VOC07/VOC12的AP计算方式。
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype) # (1,1) 的0数组
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)  # (1,1) 的1数组
        mrec = np.hstack((zeros, recalls, ones)) # 水平拼接，mrec形状 (1, num_dets + 2)
        mpre = np.hstack((zeros, precisions, zeros)) # 水平拼接，mpre形状 (1, num_dets + 2)

        # 使精确率曲线单调递减。
        # 对于每个召回率点，其精确率值应不小于所有更高召回率点上的精确率值。
        # 这是通过从右向左遍历，将当前点的精确率更新为它自身和右侧点精确率的最大值来实现的。
        for i in range(mpre.shape[1] - 1, 0, -1): # 从倒数第二个元素开始，到第一个元素
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        
        # 找到召回率(mrec)发生变化的点的索引。
        # 当mrec[0, k+1] != mrec[0, k]时，表示召回率从mrec[0, k]变为mrec[0, k+1]。
        # ind 存储的是这些变化点在 mrec[0, :-1] 中的索引。
        ind = np.where(mrec[0, 1:] != mrec[0, :-1])[0]

        # 计算P-R曲线下面积。
        # 对每个召回率增加的区间 (mrec[0, ind + 1] - mrec[0, ind])，
        # 乘以该区间右端点（也是该区间内）的最大精确率 mpre[0, ind + 1]。
        # 然后将所有这些小矩形的面积加起来。
        ap = np.sum(
            (mrec[0, ind + 1] - mrec[0, ind]) * mpre[0, ind + 1])
    
    elif mode == '11points': # 使用11点插值法计算AP
        # 在11个固定的召回率阈值 [0, 0.1, ..., 1.0] 上进行采样。
        for thr in np.arange(0, 1 + 1e-3, 0.1): # 1e-3是为了确保浮点数比较时能包含1.0
            # 对于当前的召回率阈值thr，找到所有实际召回率大于等于thr的预测点对应的精确率。
            # 注意：原始代码中 `recalls[i, :]` 的 `i` 未定义，应为 `recalls[0, :]` 因为num_scales通常为1。
            precs_at_recall_thr = precisions[0, recalls[0, :] >= thr]
            # 在这些精确率中取最大值。如果没有精确率满足条件（即没有召回率达到thr），则精确率为0。
            prec = precs_at_recall_thr.max() if precs_at_recall_thr.size > 0 else 0
            ap += prec # 累加这11个点的精确率值
        ap /= 11 # 除以11得到平均精确率
    else: # 如果模式无法识别
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported') # 抛出错误
    
    return ap # 返回计算得到的AP值

def instance_match(pred_lines: NDArray, 
                   scores: NDArray, 
                   gt_lines: NDArray, 
                   thresholds: Union[Tuple, List], 
                   metric: str='chamfer') -> List:
    """Compute whether detected lines are true positive or false positive.

    Args:
        pred_lines (array): Detected lines of a sample, of shape (M, INTERP_NUM, 2 or 3).
        scores (array): Confidence score of each line, of shape (M, ).
        gt_lines (array): GT lines of a sample, of shape (N, INTERP_NUM, 2 or 3).
        thresholds (list of tuple): List of thresholds.
        metric (str): Distance function for lines matching. Default: 'chamfer'.

    Returns:
        list_of_tp_fp (list): tp-fp matching result at all thresholds
    """

    if metric == 'chamfer':
        distance_fn = chamfer_distance

    elif metric == 'frechet':
        distance_fn = frechet_distance
    
    else:
        raise ValueError(f'unknown distance function {metric}')

    num_preds = pred_lines.shape[0]
    num_gts = gt_lines.shape[0]

    # tp and fp
    tp_fp_list = []
    tp = np.zeros((num_preds), dtype=np.float32)
    fp = np.zeros((num_preds), dtype=np.float32)

    # if there is no gt lines in this sample, then all pred lines are false positives
    if num_gts == 0:
        fp[...] = 1
        for thr in thresholds:
            tp_fp_list.append((tp.copy(), fp.copy()))
        return tp_fp_list
    
    if num_preds == 0:
        for thr in thresholds:
            tp_fp_list.append((tp.copy(), fp.copy()))
        return tp_fp_list

    assert pred_lines.shape[1] == gt_lines.shape[1], \
        "sample points num should be the same"

    # distance matrix: M x N
    matrix = np.zeros((num_preds, num_gts))

    # for i in range(num_preds):
    #     for j in range(num_gts):
    #         matrix[i, j] = distance_fn(pred_lines[i], gt_lines[j])
    
    matrix = chamfer_distance_batch(pred_lines, gt_lines)
    # for each det, the min distance with all gts
    matrix_min = matrix.min(axis=1)

    # for each det, which gt is the closest to it
    matrix_argmin = matrix.argmin(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-scores)

    # match under different thresholds
    for thr in thresholds:
        tp = np.zeros((num_preds), dtype=np.float32)
        fp = np.zeros((num_preds), dtype=np.float32)

        gt_covered = np.zeros(num_gts, dtype=bool)
        for i in sort_inds:
            if matrix_min[i] <= thr:
                matched_gt = matrix_argmin[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        tp_fp_list.append((tp, fp))

    return tp_fp_list