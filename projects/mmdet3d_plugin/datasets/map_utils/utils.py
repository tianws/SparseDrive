from shapely.geometry import LineString, box, Polygon, LinearRing # 从shapely库导入几何对象：线、矩形、多边形、线性环
from shapely.geometry.base import BaseGeometry # 从shapely库导入基础几何对象类
from shapely import ops # 从shapely库导入几何操作函数，如linemerge, unary_union
import numpy as np # 导入NumPy库
from scipy.spatial import distance # 从SciPy导入距离计算模块 (当前脚本未使用)
from typing import List, Optional, Tuple # 导入类型提示相关的模块
from numpy.typing import NDArray # 从NumPy导入类型提示NDArray

def split_collections(geom: BaseGeometry) -> List[Optional[BaseGeometry]]: # 分割几何集合对象
    ''' Split Multi-geoms to list and check is valid or is empty.
        # 将MultiLineString或MultiPolygon等几何集合对象拆分为单个几何对象的列表，
        # 并检查每个几何对象的有效性（is_valid）和是否为空（is_empty）。
        
    Args:
        geom (BaseGeometry): geoms to be split or validate. # 要被拆分或验证的几何对象。
    
    Returns:
        geometries (List): list of geometries. # 返回一个几何对象列表，只包含有效的、非空的单个几何对象。
    '''
    # 断言输入的几何对象类型是预期的几种之一
    assert geom.geom_type in ['MultiLineString', 'LineString', 'MultiPolygon', 
        'Polygon', 'GeometryCollection'], f"got geom type {geom.geom_type}"

    if 'Multi' in geom.geom_type or geom.geom_type == 'GeometryCollection': # 如果是Multi*或GeometryCollection类型
        outs = [] # 初始化输出列表
        if hasattr(geom, 'geoms'): # shapely >= 2.0 使用 .geoms, 旧版本可能使用 .geoms 或直接迭代
            geoms_to_iterate = geom.geoms
        else: # 兼容旧版本shapely或特殊情况
            geoms_to_iterate = [geom] if not hasattr(geom, 'geoms') else []


        for g in geoms_to_iterate: # 遍历集合中的每个子几何对象
            if g.is_valid and not g.is_empty: # 如果子对象有效且不为空
                outs.append(g) # 添加到输出列表
        return outs
    else: # 如果是单个几何对象 (LineString, Polygon)
        if geom.is_valid and not geom.is_empty: # 检查其有效性
            return [geom,] # 作为列表返回
        else:
            return [] # 无效或空则返回空列表

def get_drivable_area_contour(drivable_areas: List[Polygon], # 获取可行驶区域的轮廓线
                              roi_size: Tuple[float, float]      # 感兴趣区域(ROI)的尺寸 (宽度, 高度)
                              ) -> List[LineString]:
    ''' Extract drivable area contours to get list of boundaries.
        # 提取可行驶区域的轮廓，以获得边界线列表。

    Args:
        drivable_areas (list[Polygon]): list of drivable areas. # 可行驶区域的多边形列表。
        roi_size (tuple[float, float]): bev range size (width, height) # BEV的范围尺寸 (宽度, 高度)。
    
    Returns:
        boundaries (List[LineString]): list of boundaries. # 提取出的边界线（LineString对象）列表。
    '''
    max_x = roi_size[0] / 2 # ROI区域X轴半范围 (对应宽度的一半)
    max_y = roi_size[1] / 2 # ROI区域Y轴半范围 (对应高度的一半)

    # 创建一个略小于ROI的局部patch，用于与轮廓线求交，以避免ROI边缘产生意外的边界线段。
    # box(minx, miny, maxx, maxy)
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    
    exteriors = [] # 存储可行驶区域多边形的外轮廓
    interiors = [] # 存储可行驶区域多边形的内轮廓 (例如，路中间的安全岛)
    
    for poly in drivable_areas: # 遍历每个可行驶区域多边形
        exteriors.append(poly.exterior) # 获取外轮廓
        for inter in poly.interiors: # 获取所有内轮廓 (洞)
            interiors.append(inter)
    
    results = [] # 存储最终提取的边界线段
    for ext in exteriors: # 处理外轮廓
        # 注意: 我们确保所有的外轮廓是顺时针方向的。
        # 这样做的目的是使得每条边界线的右手边是可行驶区域，左手边是人行道或非可行驶区域。
        # shapely中，exterior默认是逆时针的，interior是顺时针的。
        if ext.is_ccw: # is_ccw 判断是否为逆时针
            ext = LinearRing(list(ext.coords)[::-1]) # 如果是逆时针，则反转点顺序使其变为顺时针

        # 将外轮廓与局部patch求交，得到在ROI内的部分
        lines = ext.intersection(local_patch)
        if lines.geom_type == 'MultiLineString': # 如果交集是多个线段
            lines = ops.linemerge(lines) # 尝试将它们合并成一个或多个连续的LineString
        # 断言结果是LineString或MultiLineString (linemerge可能无法合并所有情况)
        assert lines.geom_type in ['MultiLineString', 'LineString', 'GeometryCollection']
        
        results.extend(split_collections(lines)) # 分割并添加有效的线段到结果列表

    for inter in interiors: # 处理内轮廓
        # 注意: 我们确保所有的内轮廓是逆时针方向的 (与外轮廓相反的约定)。
        if not inter.is_ccw: # 如果不是逆时针 (即默认的顺时针)
            inter = LinearRing(list(inter.coords)[::-1]) # 反转点顺序使其变为逆时针

        lines = inter.intersection(local_patch) # 与局部patch求交
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString', 'GeometryCollection']
        
        results.extend(split_collections(lines))

    return results # 返回所有提取并处理过的边界线

def get_ped_crossing_contour(polygon: Polygon, # 获取人行横道多_polygon的轮廓线
                             local_patch: box   # 用于裁剪的局部patch区域 (shapely.geometry.box对象)
                             ) -> Optional[LineString]:
    ''' Extract ped crossing contours to get a closed polyline.
    # 提取人行横道多边形的轮廓，以得到一个闭合的折线。
    Different from `get_drivable_area_contour`, this function ensures a closed polyline.
    # 与 `get_drivable_area_contour` 不同，此函数旨在确保返回一个（如果可能）闭合的LineString。

    Args:
        polygon (Polygon): ped crossing polygon to be extracted. # 要提取轮廓的人行横道多边形。
        local_patch (box): local patch (shapely.geometry.box object) for intersection. # 用于求交的局部patch。
    
    Returns:
        line (LineString, optional): a closed line or None if intersection is empty or invalid.
                                     # 返回一个闭合的LineString，如果交集为空或无效则返回None。
    '''

    ext = polygon.exterior # 获取多边形的外轮廓
    # 同样，可以根据需要调整方向，但对于人行横道轮廓，方向通常不如可行驶区域边界重要
    # if not ext.is_ccw:
    #     ext = LinearRing(list(ext.coords)[::-1])
        
    lines = ext.intersection(local_patch) # 与局部patch求交

    if lines.is_empty: # 如果交集为空
        return None

    # intersection可能返回多种几何类型，需要处理
    if lines.geom_type == 'Point': # 如果交集是点，则无效
        return None
    elif lines.geom_type == 'MultiLineString': # 如果交集是多个线段
        # 尝试移除交集结果中的点对象（如果存在，尽管intersection(LineRing, Polygon)通常不直接产生点）
        valid_lines = [l for l in lines.geoms if l.geom_type == 'LineString']
        if not valid_lines:
            return None
        # 尝试合并这些线段。如果它们能首尾相连形成一条或少数几条连续线，linemerge会处理。
        # 对于闭合轮廓，理想情况下应该合并成一条LineString。
        merged_lines = ops.linemerge(valid_lines)

        if merged_lines.geom_type == 'MultiLineString':
            # 如果合并后仍然是MultiLineString，说明原始轮廓被patch切割成了多段不连续的线。
            # 这种情况对于期望得到单个闭合轮廓来说比较复杂。
            # 一种处理方法是尝试将所有线段的点连接起来，但这可能不总是产生期望的形状。
            # 原始代码中的处理方式：
            ls_coords = []
            for l_segment in merged_lines.geoms:
                ls_coords.append(np.array(l_segment.coords))
            
            if not ls_coords: return None
            concatenated_coords = np.concatenate(ls_coords, axis=0)
            # 尝试从所有点创建一个LineString，但这不保证闭合或简单性。
            # 如果这些线段原本属于一个闭合环，且被patch切割，这里的处理可能不够鲁棒。
            # 对于ped_crossing，通常期望得到一个近似矩形的闭合轮廓。
            # 如果被切割得很碎，可能需要更复杂的几何重建逻辑。
            # 此处简单地将所有点连接起来。
            lines = LineString(concatenated_coords)
        else: # 如果合并后是单个LineString
            lines = merged_lines

    if not lines.is_empty and lines.geom_type == 'LineString': # 确保最终结果是非空且为LineString
        return lines
    
    return None # 其他情况（例如，交集是Point，或处理后仍为空）返回None
