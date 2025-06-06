from shapely.geometry import LineString, box, Polygon # 从shapely库导入几何对象：线、矩形、多边形
from shapely import ops, strtree # 从shapely库导入操作函数和STRtree (一种空间索引)

import numpy as np # 导入NumPy库
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer # 从NuScenes开发工具包导入地图API和地图浏览器
from nuscenes.eval.common.utils import quaternion_yaw # 从NuScenes工具包导入从四元数计算偏航角的函数
from pyquaternion import Quaternion # 导入pyquaternion库，用于处理四元数
from .utils import split_collections, get_drivable_area_contour, \
        get_ped_crossing_contour # 从同级目录的utils模块导入辅助函数
from numpy.typing import NDArray # 从NumPy导入类型提示NDArray
from typing import Dict, List, Tuple, Union # 导入类型提示相关的模块

class NuscMapExtractor(object): # 定义NuScenes地图提取器类
    """NuScenes map ground-truth extractor. # NuScenes地图真实标签(ground-truth)提取器。

    Args:
        data_root (str): path to nuScenes dataset # nuScenes数据集的根目录路径。
        roi_size (tuple or list): bev range # BEV（鸟瞰图）的范围 (宽度, 高度)，单位米，定义了提取地图元素的感兴趣区域。
    """
    def __init__(self, data_root: str, roi_size: Union[List, Tuple]) -> None:
        """
        构造函数。
        初始化NuScenesMapExplorer对象，用于访问不同地点的地图数据。

        Args:
            data_root (str): nuScenes数据集的根目录。
            roi_size (Union[List, Tuple]): 感兴趣区域的尺寸 (宽度, 高度)，单位米。
                                           提取的地图元素将位于以此尺寸定义的、以自车为中心的矩形区域内。
        """
        self.roi_size = roi_size # 存储ROI尺寸
        # NuScenes数据集包含的地图区域名称列表
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        
        self.nusc_maps = {} # 字典，用于存储每个区域的NuScenesMap对象
        self.map_explorer = {} # 字典，用于存储每个区域的NuScenesMapExplorer对象
        for loc in self.MAPS: # 遍历每个地图区域
            # 初始化NuScenesMap对象，用于加载指定区域的地图数据
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=data_root, map_name=loc)
            # 初始化NuScenesMapExplorer对象，提供更便捷的地图数据访问接口
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])
        
        # 定义一个本地坐标系下的矩形patch，表示以自车为中心的ROI区域
        # roi_size[0] 是宽度 (x方向范围)，roi_size[1] 是长度 (y方向范围，通常是车辆前进方向)
        # box(-width/2, -length/2, width/2, length/2)
        self.local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2, 
                roi_size[0] / 2, roi_size[1] / 2)
    
    def _union_ped(self, ped_geoms: List[Polygon]) -> List[Polygon]: # 合并邻近的人行横道多边形
        ''' merge close ped crossings. # 合并邻近的人行横道。
        NuScenes中的人行横道有时被分割成多个小的多边形，此函数尝试将它们合并。
        
        Args:
            ped_geoms (list): Polygon对象的列表，代表人行横道。
        
        Returns:
            union_ped_geoms (List[Polygon]): 合并后的人行横道Polygon对象列表。
        '''

        def get_rec_direction(geom: Polygon) -> Tuple[NDArray, float]: # 获取多边形的最小外接矩形的主方向和长度
            """获取多边形最小旋转外接矩形的最长边的方向向量和长度。"""
            rect = geom.minimum_rotated_rectangle # 获取最小旋转外接矩形
            rect_v_p = np.array(rect.exterior.coords)[:3] # 取矩形的前3个顶点
            rect_v = rect_v_p[1:] - rect_v_p[:-1] # 计算两条边向量 (v01, v12)
            v_len = np.linalg.norm(rect_v, axis=-1) # 计算边长
            longest_v_i = v_len.argmax() # 找到最长边的索引
            return rect_v[longest_v_i], v_len[longest_v_i] # 返回最长边的方向向量和长度

        # 使用STRtree（一种空间索引）来加速查询邻近的几何对象
        tree = strtree.STRtree(ped_geoms)
        # 创建一个从几何对象ID到其在原始列表中的索引的映射
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

        final_pgeom = [] # 存储最终合并后的多边形列表
        remain_idx = list(range(len(ped_geoms))) # 记录尚未处理的多边形的索引

        for i, pgeom in enumerate(ped_geoms): # 遍历每个人行横道多边形
            if i not in remain_idx: # 如果已经被合并处理过，则跳过
                continue

            remain_idx.pop(remain_idx.index(i)) # 从待处理列表中移除当前多边形
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom) # 获取当前多边形的主方向和长度
            current_merged_geom = pgeom # 初始化当前合并的多边形为自身

            # 查询与当前多边形pgeom可能相交或邻近的其他多边形 (o)
            # tree.query(pgeom) 返回与pgeom的包围盒相交的其他几何对象
            for o in tree.query(pgeom):
                if o is pgeom: continue # 跳过自身
                o_idx = index_by_id.get(id(o)) # 获取邻近多边形o的原始索引
                if o_idx is None or o_idx not in remain_idx: # 如果o已被处理或无效，则跳过
                    continue

                o_v, o_v_norm = get_rec_direction(o) # 获取邻近多边形o的主方向和长度
                # 计算两个多边形主方向向量之间的夹角余弦值
                cos_sim = pgeom_v.dot(o_v) / (pgeom_v_norm * o_v_norm)
                # 如果两个多边形的主方向大致相同或相反 (夹角小于约8度)，则认为它们可以合并
                if 1 - np.abs(cos_sim) < 0.01:  # cos(theta) 接近 1 或 -1
                    current_merged_geom = current_merged_geom.union(o) # 使用shapely的union操作合并
                    remain_idx.pop(remain_idx.index(o_idx)) # 从待处理列表中移除被合并的o
            final_pgeom.append(current_merged_geom) # 将当前合并完成的多边形添加到结果列表

        results = [] # 最终结果列表
        for p_collection in final_pgeom: # 合并操作可能产生MultiPolygon，需要拆分
            results.extend(split_collections(p_collection)) # split_collections处理单个Polygon或MultiPolygon
        return results
        
    def get_map_geom(self, # 提取指定位置和姿态下的局部地图几何元素
                     location: str, # 地图位置名称 (例如 'boston-seaport')
                     translation: Union[List, NDArray], # 自车(或传感器)到全局坐标系的平移 (x,y,z)
                     rotation: Union[List, NDArray]     # 自车(或传感器)到全局坐标系的旋转 (四元数 w,x,y,z)
                     ) -> Dict[str, List[Union[LineString, Polygon]]]:
        ''' Extract geometries given `location` and self pose, self may be lidar or ego.
            # 根据给定的位置和自身位姿（自身可以是激光雷达或自车）提取几何形状。
        
        Args:
            location (str): city name # 城市名称。
            translation (Union[List, NDArray]): self2global translation, shape (3,) # 自身到全局的平移。
            rotation (Union[List, NDArray]): self2global quaternion, shape (4, ) # 自身到全局的四元数。
            
        Returns:
            geometries (Dict): extracted geometries by category. # 按类别提取的几何形状字典。
                               键为类别名 ('divider', 'ped_crossing', 'boundary', 'drivable_area')，
                               值为对应的LineString或Polygon对象列表。
        '''

        # 定义一个局部坐标系下的patch box，用于从地图API中查询数据
        # patch_box是 (中心x, 中心y, 长度_y, 长度_x) 的格式，在全局坐标系下定义
        patch_box = (translation[0], translation[1], self.roi_size[1], self.roi_size[0])

        # 将四元数转换为偏航角 (yaw)，并转换为角度制
        current_rotation_quat = Quaternion(rotation)
        yaw_degrees = quaternion_yaw(current_rotation_quat) / np.pi * 180

        # --- 提取车道分割线和道路分割线 ---
        # _get_layer_line 方法从NuScenesMapExplorer中提取指定图层内的线状元素
        # 它会返回在patch_box和yaw定义的、以当前车辆为中心的旋转坐标系内的线段
        lane_dividers = self.map_explorer[location]._get_layer_line(
                    patch_box, yaw_degrees, 'lane_divider')
        road_dividers = self.map_explorer[location]._get_layer_line(
                    patch_box, yaw_degrees, 'road_divider')
        
        all_dividers = [] # 合并所有分割线
        for line in lane_dividers + road_dividers: # _get_layer_line可能返回MultiLineString
            all_dividers += split_collections(line) # split_collections将其拆分为独立的LineString

        # --- 提取人行横道 ---
        ped_crossings_polygons = []
        # _get_layer_polygon 方法提取多边形元素
        raw_ped_polygons = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw_degrees, 'ped_crossing')
        for p in raw_ped_polygons: # 同样需要拆分可能的多部件几何对象
            ped_crossings_polygons += split_collections(p)
        
        # 合并邻近且方向相似的人行横道多边形
        # ped_crossings_polygons = self._union_ped(ped_crossings_polygons) # 注意：原始代码中此行被注释掉了，直接使用原始提取结果
        
        ped_crossing_lines = [] # 将人行横道多边形转换为其轮廓线
        for p_poly in ped_crossings_polygons:
            # get_ped_crossing_contour 提取多边形轮廓，并可能根据local_patch进行裁剪或调整
            line_contour = get_ped_crossing_contour(p_poly, self.local_patch)
            if line_contour is not None:
                ped_crossing_lines.append(line_contour)

        # --- 提取道路边界和可行驶区域 ---
        # 将道路段(road_segment)和车道(lane)图层的多边形合并，作为可行驶区域的基础
        road_segment_polygons = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw_degrees, 'road_segment')
        lane_polygons = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw_degrees, 'lane')
        
        # 使用shapely的unary_union合并所有道路段和车道多边形
        # union_roads = ops.unary_union(road_segments) # 原代码中变量名是road_segments
        # union_lanes = ops.unary_union(lanes)       # 原代码中变量名是lanes
        union_roads = ops.unary_union(road_segment_polygons)
        union_lanes = ops.unary_union(lane_polygons)
        # 再次合并，得到最终的可行驶区域
        drivable_areas_combined = ops.unary_union([union_roads, union_lanes])
        
        # 拆分可能产生的MultiPolygon
        drivable_areas_final_list = split_collections(drivable_areas_combined)

        # 道路边界被定义为可行驶区域的轮廓
        # get_drivable_area_contour 提取轮廓线
        boundaries = get_drivable_area_contour(drivable_areas_final_list, self.roi_size)

        return dict( # 返回包含各类地图元素的字典
            divider=all_dividers,                # List[LineString] 分割线
            ped_crossing=ped_crossing_lines,     # List[LineString] 人行横道轮廓线
            boundary=boundaries,                 # List[LineString] 道路边界线
            drivable_area=drivable_areas_final_list, # List[Polygon] 可行驶区域多边形
        )
