"""
工具函数模块

包含SceneCtx使用的各种辅助函数，主要包括：
- 几何计算相关函数
- 墙体处理相关函数
- 多边形计算相关函数
- Mesh加载和处理相关函数
"""

import numpy as np
import trimesh
from shapely.geometry import Polygon
from typing import List, Dict, Tuple, Optional
from scipy.spatial import ConvexHull
import uuid
import os
import json



def read_jsonl_line(file_path, line_number):
    """
    读取JSONL文件的第n行（0-based）
    
    Args:
        file_path: JSONL文件路径
        line_number: 行号（从0开始）
    
    Returns:
        dict: 该行的JSON数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == line_number:
                return json.loads(line.strip())
    
    raise ValueError(f"行号 {line_number} 超出文件范围")



def generate_unique_id() -> str:
    """
    生成唯一ID

    用途: 为墙体、门窗、家具等对象生成唯一标识符
    实现: 生成4位字母数字混合ID
    """
    import random
    import string
    chars = string.ascii_letters + string.digits  # 包含大小写字母和数字
    return ''.join(random.choices(chars, k=4))


def point_to_line_distance(point: Tuple[float, float], line_start: Tuple[float, float],
                          line_end: Tuple[float, float]) -> Tuple[float, Tuple[float, float]]:
    """
    计算点到线段的距离和最近点

    用途: 在add_door/add_window时，将门窗中心吸附到最近的墙上
    实现: 使用向量投影计算点在线段上的投影点，然后计算距离

    参数:
        point: 目标点坐标 (x, y)
        line_start: 线段起点 (x, y)
        line_end: 线段终点 (x, y)

    返回:
        (距离, 线段上最近点坐标)
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    # 线段方向向量
    dx = x2 - x1
    dy = y2 - y1

    # 起点到目标点的向量
    px = x0 - x1
    py = y0 - y1

    # 投影到线段上
    if dx == 0 and dy == 0:
        return np.sqrt(px*px + py*py), (x1, y1)

    # 计算投影参数 t (0-1之间表示在线段内)
    t = max(0, min(1, (px * dx + py * dy) / (dx * dx + dy * dy)))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    distance = np.sqrt((x0 - closest_x)**2 + (y0 - closest_y)**2)
    return distance, (closest_x, closest_y)


def calculate_wall_orientation(wall_start: Tuple[float, float], wall_end: Tuple[float, float],
                               vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    计算墙体朝向（指向房间内部的法向量）

    用途: 在add_walls时为每面墙计算法向量，用于后续墙体mesh生成
    实现:
        1. 计算墙的两个垂直方向（候选法向量）
        2. 从墙中心沿每个方向偏移一小段距离
        3. 检查测试点是否在地板多边形内
        4. 选择落在多边形内的方向作为朝向房间内部的方向

    参数:
        wall_start: 墙起点
        wall_end: 墙终点
        vertices: 房间顶点列表（地板多边形，支持凹多边形）

    返回:
        单位法向量 (nx, ny)，指向房间内部
    """
    dx = wall_end[0] - wall_start[0]
    dy = wall_end[1] - wall_start[1]
    length = np.sqrt(dx*dx + dy*dy)

    if length == 0:
        return (0, 0)

    # 两个可能的法向量（垂直于墙体方向）
    normal1 = (-dy/length, dx/length)
    normal2 = (dy/length, -dx/length)

    # 墙中心点
    wall_center = ((wall_start[0] + wall_end[0])/2, (wall_start[1] + wall_end[1])/2)

    # 测试距离：从墙中心沿法向量方向偏移
    test_offset = 0.01  # 偏移0.01米（1厘米）进行测试

    # 计算两个测试点
    test_point1 = (
        wall_center[0] + normal1[0] * test_offset,
        wall_center[1] + normal1[1] * test_offset
    )
    test_point2 = (
        wall_center[0] + normal2[0] * test_offset,
        wall_center[1] + normal2[1] * test_offset
    )

    # 检查哪个测试点在多边形内
    in_polygon1 = point_in_polygon(test_point1, vertices)
    in_polygon2 = point_in_polygon(test_point2, vertices)

    # 根据测试结果选择法向量
    if in_polygon1 and not in_polygon2:
        # 只有法向量1指向内部
        return normal1
    elif in_polygon2 and not in_polygon1:
        # 只有法向量2指向内部
        return normal2
    else:
        # 两个都在内部或都在外部（边界情况）
        # 这种情况下，选择第一个法向量
        # 可以根据需要改为其他策略
        return normal1


def try_find_closed_loop(walls: List[Dict]) -> Optional[List[Tuple[float, float]]]:
    """
    尝试判断输入的墙体是否组成一个单一的、闭合的简单环路。
    
    算法：
    1. 统计每个顶点的度数。
    2. 如果所有顶点的度数都为 2，说明这是一个或多个闭合环。
    3. 从一个点出发，沿着墙体遍历，看是否能访问所有墙体并回到起点。
    
    返回：
    如果是单一闭合环路，返回按序排列的顶点列表；否则返回 None。
    """
    if not walls:
        return None
        
    # 构建连接图
    graph = {}
    for wall_idx, wall in enumerate(walls):
        s = tuple(wall["s"])
        e = tuple(wall["e"])
        
        if s not in graph: graph[s] = []
        if e not in graph: graph[e] = []
        
        graph[s].append((wall_idx, e))
        graph[e].append((wall_idx, s))
        
    # 1. 每个顶点的度数必须正好为 2 (说明是简单闭合环，没有分叉，没有悬空)
    for vertex, edges in graph.items():
        if len(edges) != 2:
            return None
            
    # 2. 尝试遍历整个环
    start_vertex = next(iter(graph.keys()))
    current_vertex = start_vertex
    visited_walls = set()
    ordered_vertices = []
    
    while True:
        ordered_vertices.append(current_vertex)
        edges = graph[current_vertex]
        
        # 找到下一条没走过的墙
        next_edge = None
        for wall_idx, neighbor in edges:
            if wall_idx not in visited_walls:
                next_edge = (wall_idx, neighbor)
                break
        
        if not next_edge:
            break
            
        wall_idx, neighbor = next_edge
        visited_walls.add(wall_idx)
        current_vertex = neighbor
        
        if current_vertex == start_vertex:
            break
            
    # 3. 必须覆盖所有的墙体，确保是单一连通的环
    if len(visited_walls) == len(walls):
        return ordered_vertices
        
    return None


def calculate_minimum_area_polygon(walls: List[Dict]) -> List[Tuple[float, float]]:
    """
    计算包围所有墙体的最小面积多边形（支持凹多边形）

    用途: 在add_walls时计算房间的地板顶点，用于生成地板mesh
    实现:
        1. 首先检查输入的 walls 是否已经组成了一个闭合的简单环路
        2. 如果是闭合环路，直接返回按顺序排列的顶点
        3. 如果不是闭合环路，则采用基于凸包的算法：
           a. 找到能包围所有墙体的最小凸多边形
           b. 检查凸多边形的每条边，如果该边不与任何墙体共线
           c. 尝试用墙体路径替代该边，形成凹多边形

    参数:
        walls: 墙体列表，每个墙包含 s(起点) 和 e(终点)

    返回:
        多边形顶点列表，按逆时针顺序排列（可能是凹多边形）
    """
    # 尝试直接寻找闭合环路
    closed_loop = try_find_closed_loop(walls)
    if closed_loop:
        print(f"✅ 检测到输入墙体已组成闭合环路，直接使用该顺序 ({len(closed_loop)} 个顶点)")
        return closed_loop

    # 如果没有闭合环路，采用原有逻辑
    # 收集所有唯一端点
    points = []
    for wall in walls:
        points.extend([tuple(wall["s"]), tuple(wall["e"])])

    unique_points = []
    seen = set()
    for point in points:
        if point not in seen:
            unique_points.append(point)
            seen.add(point)

    if len(unique_points) < 3:
        return unique_points

    # 使用新算法：凸包 -> 凹多边形优化
    return calculate_concave_polygon_from_walls(unique_points, walls)


def calculate_concave_polygon_from_walls(points: List[Tuple[float, float]], walls: List[Dict]) -> List[Tuple[float, float]]:
    """
    基于墙体计算凹多边形

    算法步骤：
    1. 计算凸包（最小凸多边形）
    2. 检查凸包的每条边是否与墙体共线
    3. 对于不共线的边，尝试用墙体路径替代，形成凹多边形

    参数:
        points: 所有墙体端点的唯一列表
        walls: 墙体列表

    返回:
        多边形顶点列表（可能是凹多边形）
    """
    # 步骤1: 计算凸包
    points_array = np.array(points)
    hull = ConvexHull(points_array)
    convex_polygon = [points[i] for i in hull.vertices]
    
    if len(convex_polygon) < 3:
        return convex_polygon
    
    # 步骤2: 构建墙体连接图（用于后续路径搜索）
    wall_graph = build_wall_graph(walls)
    
    # 步骤3: 标记哪些墙体在凸包边上（共线）
    walls_on_convex = set()  # 存储在凸包边上的墙体索引
    
    for i in range(len(convex_polygon)):
        v1 = convex_polygon[i]
        v2 = convex_polygon[(i + 1) % len(convex_polygon)]
        
        # 检查哪些墙体与这条边共线
        for wall_idx, wall in enumerate(walls):
            if is_wall_on_edge(wall, v1, v2):
                walls_on_convex.add(wall_idx)
    
    # 步骤4: 尝试用墙体路径替代凸包的边
    final_polygon = []
    
    for i in range(len(convex_polygon)):
        start_vertex = convex_polygon[i]
        end_vertex = convex_polygon[(i + 1) % len(convex_polygon)]
        
        # 检查这条边是否与任何墙体共线
        edge_has_wall = False
        for wall in walls:
            if is_wall_on_edge(wall, start_vertex, end_vertex):
                edge_has_wall = True
                break
        
        # 添加起点
        final_polygon.append(start_vertex)
        
        # 如果这条边没有墙体共线，尝试用墙体路径替代
        if not edge_has_wall:
            # 只使用尚未在凸包上的墙体进行搜索
            available_walls = [w for idx, w in enumerate(walls) if idx not in walls_on_convex]
            wall_path = find_wall_path(start_vertex, end_vertex, available_walls, wall_graph)
            
            if wall_path and len(wall_path) > 2:
                # 找到了墙体路径，将中间顶点加入多边形（不包括起点和终点）
                # 起点已经添加，终点会在下一轮迭代添加
                for j in range(1, len(wall_path) - 1):
                    final_polygon.append(wall_path[j])
    
    # 去除可能的重复顶点
    final_polygon = remove_consecutive_duplicates(final_polygon)
    
    # 确保多边形至少有3个顶点
    if len(final_polygon) < 3:
        return convex_polygon
    
    # 确保多边形是简单多边形（不自交）
    if not is_simple_polygon(final_polygon):
        # 如果生成的多边形自交，回退到凸包
        print("⚠️  生成的凹多边形自交，回退到凸包")
        return convex_polygon
    
    return final_polygon


def remove_consecutive_duplicates(vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    移除连续的重复顶点

    参数:
        vertices: 顶点列表

    返回:
        去除连续重复后的顶点列表
    """
    if len(vertices) <= 1:
        return vertices
    
    result = [vertices[0]]
    for i in range(1, len(vertices)):
        # 使用小容差比较，避免浮点数精度问题
        if not (abs(vertices[i][0] - result[-1][0]) < 1e-6 and 
                abs(vertices[i][1] - result[-1][1]) < 1e-6):
            result.append(vertices[i])
    
    # 检查首尾是否重复
    if len(result) > 1:
        if abs(result[0][0] - result[-1][0]) < 1e-6 and abs(result[0][1] - result[-1][1]) < 1e-6:
            result.pop()
    
    return result


def build_wall_graph(walls: List[Dict]) -> Dict[Tuple[float, float], List[Tuple[int, Tuple[float, float]]]]:
    """
    构建墙体连接图

    返回一个字典，key是顶点，value是[(墙体索引, 相邻顶点)]的列表
    """
    graph = {}
    
    for wall_idx, wall in enumerate(walls):
        s = tuple(wall["s"])
        e = tuple(wall["e"])
        
        if s not in graph:
            graph[s] = []
        if e not in graph:
            graph[e] = []
        
        graph[s].append((wall_idx, e))
        graph[e].append((wall_idx, s))
    
    return graph


def is_wall_on_edge(wall: Dict, edge_start: Tuple[float, float], edge_end: Tuple[float, float]) -> bool:
    """
    检查墙体是否在凸包的边上（共线且在边的范围内）

    参数:
        wall: 墙体数据
        edge_start: 边的起点
        edge_end: 边的终点

    返回:
        True如果墙体在这条边上
    """
    wall_s = tuple(wall["s"])
    wall_e = tuple(wall["e"])
    
    # 检查墙体的两个端点是否都在边上
    if not (point_on_segment(wall_s, edge_start, edge_end) and 
            point_on_segment(wall_e, edge_start, edge_end)):
        return False
    
    # 检查三点是否共线
    return are_collinear(edge_start, edge_end, wall_s) and are_collinear(edge_start, edge_end, wall_e)


def are_collinear(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> bool:
    """
    检查三点是否共线

    使用叉积判断：(p2-p1) × (p3-p1) = 0
    """
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]
    dx2 = p3[0] - p1[0]
    dy2 = p3[1] - p1[1]
    
    cross_product = dx1 * dy2 - dy1 * dx2
    return abs(cross_product) < 1e-6


def find_wall_path(start: Tuple[float, float], end: Tuple[float, float], 
                   available_walls: List[Dict], wall_graph: Dict) -> List[Tuple[float, float]]:
    """
    使用BFS在可用墙体中搜索从start到end的路径

    参数:
        start: 起点
        end: 终点
        available_walls: 可用的墙体列表
        wall_graph: 墙体连接图

    返回:
        顶点路径列表，如果找不到则返回None
    """
    from collections import deque
    
    # 构建可用墙体的端点集合（用于快速查找）
    available_edges = set()
    for wall in available_walls:
        s = tuple(wall["s"])
        e = tuple(wall["e"])
        available_edges.add((s, e))
        available_edges.add((e, s))  # 双向
    
    # BFS搜索
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        # 找到目标
        if current == end:
            return path
        
        # 检查当前顶点的所有邻居
        if current in wall_graph:
            for wall_idx, neighbor in wall_graph[current]:
                edge = (current, neighbor)
                
                # 只使用可用墙体的边，且邻居未访问过
                if edge in available_edges and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
    
    # 找不到路径
    return None


def brute_force_minimum_polygon(points: List[Tuple[float, float]], walls: List[Dict]) -> List[Tuple[float, float]]:
    """
    穷举法寻找最小面积多边形

    用途: calculate_minimum_area_polygon的子函数，用于小规模点集
    实现: 枚举所有可能的点组合和排列，找到能包含所有墙且面积最小的多边形

    复杂度: O(2^n * n!)，仅适用于点数较少的情况
    """
    from itertools import combinations

    min_area = float('inf')
    best_polygon = None

    # 尝试不同数量的顶点（至少3个）
    for num_vertices in range(3, len(points) + 1):
        for vertex_combination in combinations(points, num_vertices):
            # 找到这些顶点的最佳排列顺序
            polygon = find_best_ordering(list(vertex_combination))
            if polygon and polygon_contains_all_walls(polygon, walls):
                area = polygon_area(polygon)
                if area < min_area:
                    min_area = area
                    best_polygon = polygon

    return best_polygon if best_polygon else points


def heuristic_minimum_polygon(points: List[Tuple[float, float]], walls: List[Dict]) -> List[Tuple[float, float]]:
    """
    启发式方法寻找最小多边形

    用途: calculate_minimum_area_polygon的子函数，用于大规模点集
    实现:
        1. 先计算凸包作为初始解
        2. 尝试移除不影响墙体包围的顶点来优化
    """
    points_array = np.array(points)
    hull = ConvexHull(points_array)
    hull_points = [points[i] for i in hull.vertices]

    # 尝试优化多边形
    optimized = optimize_polygon(hull_points, walls, points)
    return optimized if optimized else hull_points


def find_best_ordering(vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    寻找顶点的最佳排列顺序，形成简单多边形（不自交）

    用途: brute_force_minimum_polygon的子函数
    实现:
        1. 计算顶点相对于中心点的角度
        2. 按角度排序
        3. 检查是否自交，如不满足则尝试凸包排序
    """
    if len(vertices) < 3:
        return vertices

    # 计算中心点
    centroid_x = sum(p[0] for p in vertices) / len(vertices)
    centroid_y = sum(p[1] for p in vertices) / len(vertices)

    def angle_from_centroid(point):
        return np.arctan2(point[1] - centroid_y, point[0] - centroid_x)

    sorted_vertices = sorted(vertices, key=angle_from_centroid)

    # 检查是否为简单多边形
    if is_simple_polygon(sorted_vertices):
        return sorted_vertices

    # 如果不是，尝试凸包排序
    try:
        points_array = np.array(vertices)
        hull = ConvexHull(points_array)
        return [vertices[i] for i in hull.vertices]
    except:
        return sorted_vertices


def is_simple_polygon(vertices: List[Tuple[float, float]]) -> bool:
    """
    检查多边形是否为简单多边形（边不自交）

    用途: find_best_ordering的子函数
    实现: 检查所有边对，判断是否存在相交
    """
    n = len(vertices)
    if n < 3:
        return True

    # 检查边是否自交
    for i in range(n):
        for j in range(i + 2, n):
            if j == n - 1 and i == 0:  # 跳过相邻边
                continue
            if segments_intersect(vertices[i], vertices[(i + 1) % n],
                                vertices[j], vertices[(j + 1) % n]):
                return False
    return True


def segments_intersect(p1: Tuple[float, float], q1: Tuple[float, float],
                      p2: Tuple[float, float], q2: Tuple[float, float]) -> bool:
    """
    检查两条线段是否相交

    用途: is_simple_polygon的子函数
    实现: 使用方向判断法（orientation test）
    """
    def orientation(p, q, r):
        """计算三点的方向：0=共线，1=顺时针，2=逆时针"""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        """检查点q是否在线段pr上（假设三点共线）"""
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # 一般情况：两线段跨越彼此
    if o1 != o2 and o3 != o4:
        return True

    # 特殊情况：点在线段上
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def polygon_contains_all_walls(polygon: List[Tuple[float, float]], walls: List[Dict]) -> bool:
    """
    检查多边形是否包含所有墙体

    用途: brute_force_minimum_polygon的子函数，验证候选多边形是否有效
    实现: 检查每面墙的起点和终点是否都在多边形内或边上
    """
    for wall in walls:
        if not (point_in_or_on_polygon(tuple(wall["s"]), polygon) and
                point_in_or_on_polygon(tuple(wall["e"]), polygon)):
            return False
    return True


def point_in_or_on_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    检查点是否在多边形内部或边上

    用途: polygon_contains_all_walls的子函数
    实现: 先检查是否在边上，再使用射线法检查是否在内部
    """
    # 检查是否在边上
    for i in range(len(polygon)):
        seg_start = polygon[i]
        seg_end = polygon[(i + 1) % len(polygon)]
        if point_on_segment(point, seg_start, seg_end):
            return True

    # 检查是否在内部（射线法）
    return point_in_polygon(point, polygon)


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    射线法判断点是否在多边形内

    用途: point_in_or_on_polygon的子函数
    实现: 从点向右发射射线，计算与多边形边的交点数量，奇数则在内部
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def point_on_segment(point: Tuple[float, float], seg_start: Tuple[float, float],
                    seg_end: Tuple[float, float]) -> bool:
    """
    检查点是否在线段上

    用途: point_in_or_on_polygon的子函数
    实现: 检查点到线段两端点的距离之和是否等于线段长度
    """
    dist_to_start = np.sqrt((point[0] - seg_start[0])**2 + (point[1] - seg_start[1])**2)
    dist_to_end = np.sqrt((point[0] - seg_end[0])**2 + (point[1] - seg_end[1])**2)
    seg_length = np.sqrt((seg_end[0] - seg_start[0])**2 + (seg_end[1] - seg_start[1])**2)

    # 使用容差判断（避免浮点数精度问题）
    return abs(dist_to_start + dist_to_end - seg_length) < 1e-6


def polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """
    计算多边形面积

    用途: brute_force_minimum_polygon中比较不同多边形的大小
    实现: 使用鞋带公式（Shoelace formula）

    公式: Area = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
    """
    if len(polygon) < 3:
        return 0

    area = 0
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]

    return abs(area) / 2


def optimize_polygon(polygon: List[Tuple[float, float]], walls: List[Dict],
                    all_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    优化多边形，尝试移除不必要的顶点

    用途: heuristic_minimum_polygon的子函数，减少凸包顶点数量
    实现: 依次尝试移除每个顶点，如果移除后仍能包含所有墙则保留修改
    """
    optimized = polygon.copy()
    changed = True

    while changed and len(optimized) > 3:
        changed = False
        for i in range(len(optimized)):
            # 尝试移除顶点i
            test_polygon = optimized[:i] + optimized[i+1:]
            if polygon_contains_all_walls(test_polygon, walls):
                optimized = test_polygon
                changed = True
                break

    return optimized


def find_closest_wall(center: List[float], walls_dict: Dict) -> str:
    """
    寻找距离指定中心点最近的墙

    用途: add_door和add_window时自动关联到最近的墙
    实现: 遍历所有墙，计算点到每面墙的距离，返回最近的墙ID

    参数:
        center: 门/窗的中心点坐标 [x, y, z]
        walls_dict: 墙体字典 {wall_id: wall_data}

    返回:
        最近墙体的unique_id
    """
    point = (center[0], center[1])
    min_distance = float('inf')
    closest_wall_id = None

    for wall_id, wall in walls_dict.items():
        distance, _ = point_to_line_distance(point, tuple(wall["s"]), tuple(wall["e"]))
        if distance < min_distance:
            min_distance = distance
            closest_wall_id = wall_id

    return closest_wall_id


def find_closest_wall_from_list(center: List[float], walls_list: List[Tuple]) -> str:
    """
    从墙体列表中找到距离给定中心点最近的墙体
    
    Args:
        center: 点的中心坐标 [x, y, z]
        walls_list: 墙体列表 [(wall_id, p, q), ...]
    
    Returns:
        最接近的墙体的ID
    """
    min_distance = float('inf')
    closest_wall_id = None
    
    point = np.array(center[:2])  # 只使用x, y坐标
    
    for wall_id, p, q in walls_list:
        # 计算点到线段的距离
        wall_start = np.array(p[:2])
        wall_end = np.array(q[:2])
        
        # 线段向量
        wall_vec = wall_end - wall_start
        wall_length_sq = np.dot(wall_vec, wall_vec)
        
        if wall_length_sq == 0:
            # 墙体退化为一个点
            distance = np.linalg.norm(point - wall_start)
        else:
            # 计算投影参数 t
            t = max(0, min(1, np.dot(point - wall_start, wall_vec) / wall_length_sq))
            projection = wall_start + t * wall_vec
            distance = np.linalg.norm(point - projection)
        
        if distance < min_distance:
            min_distance = distance
            closest_wall_id = wall_id
    
    return closest_wall_id


def snap_to_wall(center: List[float], wall: Dict) -> List[float]:
    """
    将点吸附到墙上

    用途: add_door和add_window时，如果门/窗中心不在墙上，将其投影到墙上
    实现: 使用point_to_line_distance计算最近点，保持z坐标不变

    参数:
        center: 原始中心点 [x, y, z]
        wall: 墙体数据（包含s和e）

    返回:
        吸附后的中心点 [x', y', z]
    """
    point = (center[0], center[1])
    _, closest_point = point_to_line_distance(point, tuple(wall["s"]), tuple(wall["e"]))
    return [closest_point[0], closest_point[1], center[2]]


def create_floor_mesh(vertices: List[Tuple[float, float]], bounds: List[float], 
                      texture_scale: float = 2.0) -> trimesh.Trimesh:
    """
    创建地板mesh（使用方案一：extrude 挤出带厚度的实体）

    参数:
        vertices: 地板顶点列表 [(x1, y1), (x2, y2), ...]
        bounds: 边界 [x_min, y_min, x_max, y_max]
        texture_scale: 纹理缩放系数，默认2.0

    返回:
        trimesh.Trimesh对象，具有物理厚度
    """
    texture_scale = texture_scale if texture_scale  else 2.0
    if len(vertices) < 3:
        return trimesh.Trimesh()

    # 1. 提取 2D 坐标并确保顺序正确 (CCW)
    pts2_raw = [(float(v[0]), float(v[1])) for v in vertices]
    pts_arr = np.array(pts2_raw)
    xs, ys = pts_arr[:, 0], pts_arr[:, 1]
    # 计算 2*面积 的符号判断顺逆时针
    area2 = np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
    if area2 < 0:
        pts2_raw = pts2_raw[::-1]

    # 2. 构建 Polygon 对象（支持凹多边形）
    poly = Polygon(pts2_raw)
    
    # 3. 使用 extrude 向下挤出 0.1m
    # 这会自动完成三角化，并生成顶面、底面和侧面
    # 注意：trimesh 中正确的函数名是 extrude_polygon
    floor_mesh = trimesh.creation.extrude_polygon(poly, height=-0.1)

    # 4. 计算并应用投影 UV (Planar Mapping)
    all_verts = floor_mesh.vertices
    
    # 计算 X 和 Y 方向的实际长度
    x_length = bounds[2] - bounds[0]
    y_length = bounds[3] - bounds[1]
    max_length = max(x_length, y_length)
    
    # 使用最大长度作为归一化基准，保持纹理不被拉伸
    # 这样即使房间是长方形，纹理也会保持正方形
    u = (all_verts[:, 0] - bounds[0]) / max_length * texture_scale
    v = (all_verts[:, 1] - bounds[1]) / max_length * texture_scale
    
    # 赋值给 mesh 的纹理视觉属性
    floor_mesh.visual = trimesh.visual.TextureVisuals(uv=np.column_stack((u, v)))
    
    return floor_mesh


def create_ceiling_mesh(vertices: List[Tuple[float, float]], bounds: List[float], z_height: float) -> trimesh.Trimesh:
    """
    创建天花板mesh（法线朝下，使用耳切三角化处理凹多边形）
    
    参数:
        vertices: 天花板顶点列表 [(x1, y1), (x2, y2), ...]（与地板相同的顶点）
        bounds: 边界 [x_min, y_min, x_max, y_max]
        z_height: 天花板高度（z坐标）
    
    返回:
        trimesh.Trimesh对象，带UV坐标，法线朝下
    """
    if len(vertices) < 3:
        return trimesh.Trimesh()
    
    texture_scale = 2.0
    # 顶点在 XY 平面，Z=z_height
    ceiling_verts = np.array([[v[0], v[1], z_height] for v in vertices], dtype=float)
    
    # 天花板需要顺时针顺序（从下方看），使法线朝下（-Z）
    # 与地板相反：如果面积为正（逆时针），需要反转为顺时针
    if ceiling_verts.shape[0] >= 3:
        xs = ceiling_verts[:, 0]
        ys = ceiling_verts[:, 1]
        # 计算 2*面积 的符号
        area2 = np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
        if area2 > 0:
            # 反转顶点顺序以确保为 CW（从 -Z 方向看为顺时针，使法线朝下）
            ceiling_verts = ceiling_verts[::-1]
    
    # 计算 UV（使用当前顶点顺序）
    ceiling_uvs = [[(v[0] - bounds[0]) / (bounds[2] - bounds[0]) * texture_scale,
                    (v[1] - bounds[1]) / (bounds[3] - bounds[1]) * texture_scale] for v in ceiling_verts]
    
    # 使用耳切（ear-clipping）三角化以处理凹多边形
    def _is_convex(a, b, c):
        # 判断三点 (a,b,c) 在多边形中是否为凸角
        # 对于CW顺序（天花板），凹凸判断需要取反
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < -1e-9
    
    def _point_in_triangle(pt, a, b, c):
        # barycentric technique
        v0 = np.array(c) - np.array(a)
        v1 = np.array(b) - np.array(a)
        v2 = np.array(pt) - np.array(a)
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            return False
        u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
        v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
        return (u >= -1e-9) and (v >= -1e-9) and (u + v <= 1 + 1e-9)
    
    pts2 = [(float(x), float(y)) for x, y in ceiling_verts[:, :2]]
    idx_list = list(range(len(pts2)))
    ceiling_faces = []
    
    # Guard for degenerate small polygons
    if len(idx_list) == 3:
        ceiling_faces = [[0, 1, 2]]
    else:
        safety = 0
        while len(idx_list) > 3 and safety < 10000:
            made_cut = False
            n = len(idx_list)
            for i in range(n):
                i_prev = idx_list[(i - 1) % n]
                i_curr = idx_list[i]
                i_next = idx_list[(i + 1) % n]
                a = pts2[i_prev]
                b = pts2[i_curr]
                c = pts2[i_next]
                
                if not _is_convex(a, b, c):
                    continue
                
                # 检查是否有其他顶点在三角形内
                any_inside = False
                for j in idx_list:
                    if j in (i_prev, i_curr, i_next):
                        continue
                    if _point_in_triangle(pts2[j], a, b, c):
                        any_inside = True
                        break
                
                if any_inside:
                    continue
                
                # 这是一个耳，切掉中心点
                ceiling_faces.append([i_prev, i_curr, i_next])
                idx_list.remove(i_curr)
                made_cut = True
                break
            
            if not made_cut:
                # 可能遇到数值问题或自交，退回到扇形备选以避免死循环
                ceiling_faces = [[0, i, i + 1] for i in range(1, len(pts2) - 1)]
                break
            safety += 1
        
        if len(idx_list) == 3:
            ceiling_faces.append([idx_list[0], idx_list[1], idx_list[2]])
    
    # 构建 trimesh，关闭自动处理以保留我们指定的顺序
    ceiling_mesh = trimesh.Trimesh(vertices=ceiling_verts, faces=np.array(ceiling_faces, dtype=int), process=False)
    
    # 设置UV
    ceiling_mesh.visual = trimesh.visual.TextureVisuals(uv=np.array(ceiling_uvs))
    
    return ceiling_mesh


def create_opening_box(center: List[float], width: float, height: float,
                      wall_orientation: List[float], wall_thickness: float) -> trimesh.Trimesh:
    """
    创建门或窗的 Box 用于布尔减法挖洞
    
    参数:
        center: 门/窗中心点 [x, y, z]（原始面片模式的中心）
        width: 门/窗宽度
        height: 门/窗高度
        wall_orientation: 墙的朝向（法向量），指向房间内部
        wall_thickness: 墙的厚度
    
    返回:
        正确定位和旋转的 box mesh
    """
    # 1. 创建一个轴对齐的 box（默认朝向）
    # extents 顺序: [x_size, y_size, z_size]
    # 这里让 y 方向作为厚度方向
    box = trimesh.creation.box(extents=[width, wall_thickness + 0.02, height])
    
    # 2. 计算旋转角度：让 box 的法线（默认 +Y）对齐到墙的 orientation
    # 墙的 orientation 是 2D 向量 [nx, ny]，转换为 3D [nx, ny, 0]
    target_normal = np.array([wall_orientation[0], wall_orientation[1], 0.0])
    target_normal /= np.linalg.norm(target_normal) + 1e-9
    
    # box 默认的法线是 +Y 方向
    default_normal = np.array([0.0, 1.0, 0.0])
    
    # 计算旋转轴（叉乘）和旋转角度（点积）
    rotation_axis = np.cross(default_normal, target_normal)
    rotation_angle = np.arccos(np.clip(np.dot(default_normal, target_normal), -1.0, 1.0))
    
    # 如果需要旋转（即两个向量不平行）
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            rotation_angle, rotation_axis, point=[0, 0, 0]
        )
        box.apply_transform(rotation_matrix)
    
    # 3. 调整中心位置：向墙外（-orientation）移动 thickness/2
    adjusted_center = np.array(center) - np.array([
        wall_orientation[0] * wall_thickness / 2,
        wall_orientation[1] * wall_thickness / 2,
        0.0
    ])
    
    # 4. 平移到最终位置
    box.apply_translation(adjusted_center)
    
    return box


def create_single_wall_mesh(start: List[float], end: List[float], height: float, 
                            orientation: List[float], chip: bool = False,
                            openings: List[Dict] = None, wall_thickness: float = 0.1,
                            texture_scale: float = 2.0) -> trimesh.Trimesh:
    """
    为单面墙创建mesh。默认创建带厚度并向外部挤出的实体墙。
    
    参数:
        start: 墙的起点 [x, y]
        end: 墙的终点 [x, y]
        height: 墙的高度
        orientation: 墙的朝向（法向量），指向房间内部
        chip: 如果为 True，则返回传统的单面薄片面片
        openings: 门窗列表，每项包含 {"center": [x,y,z], "width": w, "height": h}
        wall_thickness: 墙的厚度（米），默认0.1
        texture_scale: 纹理缩放系数，默认2.0
    """
    if height == 0:
        height = 0.01
    texture_scale = texture_scale if texture_scale  else 2.0
    # 基础墙面向量计算
    wall_vec = np.array(end[:2]) - np.array(start[:2])
    wall_length = np.linalg.norm(wall_vec)
    if wall_length == 0: wall_length = 1.0
    unit_wall_vec = wall_vec / wall_length

    if chip:
        # --- 旧逻辑：创建单面薄片 ---
        wall_verts = np.array([
            [start[0], start[1], 0.0],
            [end[0], end[1], 0.0],
            [end[0], end[1], height],
            [start[0], start[1], height]
        ], dtype=float)

        u_scale = wall_length / (height if height != 0 else 1.0)
        wall_uvs = np.array([[0.0, 0.0], [u_scale, 0.0], [u_scale, 1.0], [0.0, 1.0]])

        v0, v1, v2 = wall_verts[0], wall_verts[1], wall_verts[2]
        default_normal = np.cross(v1 - v0, v2 - v0)
        default_normal /= (np.linalg.norm(default_normal) + 1e-9)
        dot = np.dot(default_normal, np.array([orientation[0], orientation[1], 0.0]))

        wall_faces = np.array([[0, 1, 2], [0, 2, 3]] if dot > 0 else [[0, 2, 1], [0, 3, 2]], dtype=int)
        wall_mesh = trimesh.Trimesh(vertices=wall_verts, faces=wall_faces, process=False)
        wall_mesh.visual = trimesh.visual.TextureVisuals(uv=wall_uvs)
        return wall_mesh
    else:
        # --- 新逻辑：创建带厚度的实体墙 (Extrude) ---
        thickness = wall_thickness
        # 计算向墙外偏移的向量 (和 orientation 相反)
        off_x, off_y = -orientation[0] * thickness, -orientation[1] * thickness
        
        # 矩形底面的四个点 (p1, p2 是内墙面底部, p3, p4 是外墙面底部)
        p1 = (float(start[0]), float(start[1]))
        p2 = (float(end[0]), float(end[1]))
        p3 = (p2[0] + off_x, p2[1] + off_y)
        p4 = (p1[0] + off_x, p1[1] + off_y)
        
        # 使用 Polygon 定义足迹并向上拉伸高度
        poly = Polygon([p1, p2, p3, p4])
        # 注意：trimesh 中正确的函数名是 extrude_polygon
        wall_mesh = trimesh.creation.extrude_polygon(poly, height=height)
        
        # --- 布尔运算：挖洞（门窗） ---
        if openings:
            for opening in openings:
                try:
                    opening_box = create_opening_box(
                        center=opening["center"],
                        width=opening["width"],
                        height=opening["height"],
                        wall_orientation=orientation,
                        wall_thickness=thickness
                    )
                    wall_mesh = wall_mesh.difference(opening_box)
                except Exception as e:
                    print(f"⚠️  布尔运算失败 (opening at {opening.get('center')}): {e}")
        
        # --- 按照用户指定的投影逻辑计算 UV ---
        all_verts = wall_mesh.vertices
        
        # 使用墙长和高度中的最大值作为归一化基准，防止纹理变形
        max_dimension = max(wall_length, height)
        
        # u: 点相对于起点(p1)在墙面方向(unit_wall_vec)上的投影长度占比
        rel_vecs = all_verts[:, :2] - np.array(p1)
        u = (np.dot(rel_vecs, unit_wall_vec) / max_dimension) * texture_scale
        
        # v: 点在垂直方向(Z轴)的高度占比
        v = (all_verts[:, 2] / max_dimension) * texture_scale
        
        # 应用投影映射
        wall_mesh.visual = trimesh.visual.TextureVisuals(uv=np.column_stack((u, v)))
        
        return wall_mesh


def create_face_wall_mesh(vertices: List[Tuple[float, float]], height: float) -> trimesh.Trimesh:
    """
    创建无厚度的墙体mesh（双面可见，带UV纹理坐标）

    参数:
        vertices: 墙环顶点序列 [(x1,y1), (x2,y2), ...]，按逆时针顺序排列
        height: 墙体高度

    返回:
        trimesh.Trimesh对象，从内外两侧都可见，带UV坐标
    """
    n = len(vertices)
    if n < 3:
        return trimesh.Trimesh()

    # 构建顶点：底面环 + 顶面环
    mesh_vertices = []

    # 底面环 (z=0)
    for v in vertices:
        mesh_vertices.append([v[0], v[1], 0.0])

    # 顶面环 (z=height)
    for v in vertices:
        mesh_vertices.append([v[0], v[1], height])

    # 计算UV坐标 - 沿着墙的周长展开
    # 计算每段墙的累计长度用于U坐标
    cumulative_length = [0.0]
    for i in range(n):
        i_next = (i + 1) % n
        v_curr = np.array(vertices[i])
        v_next = np.array(vertices[i_next])
        segment_length = np.linalg.norm(v_next - v_curr)
        cumulative_length.append(cumulative_length[-1] + segment_length)

    total_length = cumulative_length[-1]
    if total_length == 0:
        total_length = 1.0  # 避免除零

    # 为每个顶点分配UV坐标
    # U: 沿墙周长的归一化位置 (0-1范围，但可以超过1实现纹理重复)
    # V: 高度的归一化位置 (0=底部, 1=顶部)
    texture_repeat = total_length / height  # 让纹理在水平和垂直方向保持相似比例

    uvs = []
    for i in range(n):
        u = cumulative_length[i] / height  # 使用height作为单位长度
        uvs.append([u, 0.0])  # 底部顶点
    for i in range(n):
        u = cumulative_length[i] / height
        uvs.append([u, 1.0])  # 顶部顶点

    # 构建面片：双面墙体
    # 顶点索引布局:
    # 0~n-1: 底面环
    # n~2n-1: 顶面环
    faces = []

    # 遍历顶点环，为每个墙段创建两个三角形构成矩形
    for i in range(n):
        i_next = (i + 1) % n

        # 4个顶点索引
        b_curr = i          # 当前底部顶点
        b_next = i_next     # 下一个底部顶点
        t_curr = n + i      # 当前顶部顶点
        t_next = n + i_next # 下一个顶部顶点

        # 第一面：法向量指向多边形内部（假设vertices是逆时针）
        # 从内部看，三角形顶点应该是逆时针顺序
        faces.append([b_curr, t_curr, t_next])
        faces.append([b_curr, t_next, b_next])

        # 第二面：法向量指向多边形外部（反向三角形）
        # 从外部看，三角形顶点应该是逆时针顺序
        faces.append([b_curr, t_next, t_curr])
        faces.append([b_curr, b_next, t_next])

    # 创建mesh，启用process=True让trimesh正确计算法线
    mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=faces, process=True)

    # 将UV坐标存储到mesh中
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.array(uvs))

    return mesh


def create_wall_ring_mesh(vertices: List[Tuple[float, float]], walls: Dict, height: float, thickness: float) -> trimesh.Trimesh:
    """
    基于墙环顶点创建整体墙体mesh

    参数:
        vertices: 墙环顶点序列 [(x1,y1), (x2,y2), ...]
        walls: 墙体字典,包含orientation信息
        height: 墙体高度
        thickness: 墙体厚度

    返回:
        trimesh.Trimesh对象
    """
    n = len(vertices)
    if n < 3:
        return trimesh.Trimesh()

    # 计算每个顶点对应的外环偏移
    # 对于墙环顶点v[i],相邻两条边是: v[i-1]→v[i] 和 v[i]→v[i+1]
    # 每条边对应一面墙,找到这两面墙的外法向量,沿角平分线方向计算正确的偏移量
    #
    # 几何原理:
    # 设两个外法向量为 n1, n2 (单位向量), 夹角为 θ
    # 向量和 n1+n2 指向角平分线方向, 模长为 ||n1+n2|| = 2*cos(θ/2)
    # 为了使墙体垂直向外偏移 thickness, 沿角平分线的偏移应为:
    # offset = (n1+n2) * 2*thickness / ||n1+n2||²
    vertex_offsets = []

    for i in range(n):
        v_curr = np.array(vertices[i])
        v_prev = np.array(vertices[(i - 1) % n])
        v_next = np.array(vertices[(i + 1) % n])

        # 找到两条边对应的墙
        edge_prev = (tuple(v_prev), tuple(v_curr))  # 前一条边
        edge_curr = (tuple(v_curr), tuple(v_next))  # 当前边

        offset_sum = np.array([0.0, 0.0])

        for wall in walls.values():
            s = np.array(wall["s"])
            e = np.array(wall["e"])
            orientation = np.array(wall.get("orientation", [0, 0]))

            # 判断墙是否匹配边
            if (np.allclose(s, v_prev, atol=1e-6) and np.allclose(e, v_curr, atol=1e-6)) or \
               (np.allclose(s, v_curr, atol=1e-6) and np.allclose(e, v_prev, atol=1e-6)):
                # 这是前一条边对应的墙
                outer_normal = -orientation
                offset_sum += outer_normal

            if (np.allclose(s, v_curr, atol=1e-6) and np.allclose(e, v_next, atol=1e-6)) or \
               (np.allclose(s, v_next, atol=1e-6) and np.allclose(e, v_curr, atol=1e-6)):
                # 这是当前边对应的墙
                outer_normal = -orientation
                offset_sum += outer_normal

        # 根据角度几何关系计算正确的偏移量
        # offset_sum 是两个单位外法向量的和, 指向角平分线方向
        # 正确的偏移 = offset_sum * (2 * thickness / ||offset_sum||²)
        norm = np.linalg.norm(offset_sum)
        if norm > 1e-6:
            # 使用几何正确的公式: 保证每面墙垂直向外偏移 thickness
            offset = offset_sum * (2.0 * thickness / (norm * norm))
        else:
            # 退化情况: 两个法向量相反(180度角), 使用默认偏移
            offset = np.array([0.0, 0.0])

        vertex_offsets.append(offset)

    # 构建内外环顶点
    mesh_vertices = []
    # 底面内环
    for v in vertices:
        mesh_vertices.append([v[0], v[1], 0.0])
    # 底面外环
    for i, v in enumerate(vertices):
        offset = vertex_offsets[i]
        mesh_vertices.append([v[0] + offset[0], v[1] + offset[1], 0.0])
    # 顶面内环
    for v in vertices:
        mesh_vertices.append([v[0], v[1], height])
    # 顶面外环
    for i, v in enumerate(vertices):
        offset = vertex_offsets[i]
        mesh_vertices.append([v[0] + offset[0], v[1] + offset[1], height])

    # 构建面片 - 墙体由四部分组成
    # 顶点索引布局:
    # 0~n-1: 底面内环
    # n~2n-1: 底面外环
    # 2n~3n-1: 顶面内环
    # 3n~4n-1: 顶面外环
    faces = []

    # 第一部分: 内环面 (面向房间内部)
    # 遍历顶点环，连接相邻顶点形成矩形面
    for i in range(n):
        i_next = (i + 1) % n

        # 内环面的4个顶点: 底i, 底i+1, 顶i+1, 顶i
        b_curr = i
        b_next = i_next
        t_curr = 2 * n + i
        t_next = 2 * n + i_next

        # 法向朝房间内: 从房间内看,逆时针
        faces.append([b_curr, b_next, t_next])
        faces.append([b_curr, t_next, t_curr])

    # 第二部分: 外环面 (背向房间,面向外部)
    # 遍历顶点环
    for i in range(n):
        i_next = (i + 1) % n

        # 外环面的4个顶点
        b_curr_outer = n + i
        b_next_outer = n + i_next
        t_curr_outer = 3 * n + i
        t_next_outer = 3 * n + i_next

        # 法向朝房间外: 从外面看,逆时针 (顶点顺序与内环相反)
        faces.append([b_next_outer, b_curr_outer, t_curr_outer])
        faces.append([b_next_outer, t_curr_outer, t_next_outer])

    # 第三部分: 顶部环带 (法向朝上+z)
    # 遍历顶点环
    for i in range(n):
        i_next = (i + 1) % n

        # 顶部环带的4个顶点: 内i, 内i+1, 外i+1, 外i
        t_inner_curr = 2 * n + i
        t_inner_next = 2 * n + i_next
        t_outer_curr = 3 * n + i
        t_outer_next = 3 * n + i_next

        # 法向朝上: 从上方看,逆时针
        faces.append([t_inner_curr, t_inner_next, t_outer_next])
        faces.append([t_inner_curr, t_outer_next, t_outer_curr])

    # 第四部分: 底部环带 (法向朝下-z)
    # 遍历顶点环
    for i in range(n):
        i_next = (i + 1) % n

        # 底部环带的4个顶点: 内i, 内i+1, 外i+1, 外i
        b_inner_curr = i
        b_inner_next = i_next
        b_outer_curr = n + i
        b_outer_next = n + i_next

        # 法向朝下: 从下方看,逆时针 (从外向内)
        faces.append([b_outer_next, b_outer_curr, b_inner_curr])
        faces.append([b_outer_next, b_inner_curr, b_inner_next])

    return trimesh.Trimesh(vertices=mesh_vertices, faces=faces)


def create_wall_mesh(wall: Dict, thickness: float) -> trimesh.Trimesh:
    """
    创建墙体的3D mesh

    用途: construct_floor中为每面墙创建3D几何体
    实现:
        1. 根据墙的起点、终点、高度和厚度计算8个顶点
        2. 墙的厚度完全朝向房间外侧延伸(使用orientation指向房间内部)
        3. 定义12个三角面（每个矩形面用2个三角形）
        4. 创建trimesh对象

    参数:
        wall: 墙体数据，包含 s(起点), e(终点), height(高度), orientation(朝向房间内部的单位向量)
        thickness: 墙体厚度

    返回:
        trimesh.Trimesh对象
    """
    s = wall["s"]
    e = wall["e"]
    height = wall["height"]
    orientation = wall.get("orientation", [0, 0])  # 朝向房间内部的法向量

    # 墙的外法向量 = -orientation (朝向房间外部)
    # 墙的厚度完全向外延伸
    outer_normal = np.array([-orientation[0], -orientation[1], 0]) * thickness

    # 8个顶点布局:
    # 底面: 0(s内) -- 1(e内)     顶面: 4(s内) -- 5(e内)
    #        |          |                |          |
    #       3(s外) -- 2(e外)            7(s外) -- 6(e外)
    vertices = [
        # 底面
        [s[0], s[1], 0],  # 0: 起点内侧
        [e[0], e[1], 0],  # 1: 终点内侧
        [e[0] + outer_normal[0], e[1] + outer_normal[1], 0],  # 2: 终点外侧
        [s[0] + outer_normal[0], s[1] + outer_normal[1], 0],  # 3: 起点外侧
        # 顶面
        [s[0], s[1], height],  # 4: 起点内侧
        [e[0], e[1], height],  # 5: 终点内侧
        [e[0] + outer_normal[0], e[1] + outer_normal[1], height],  # 6: 终点外侧
        [s[0] + outer_normal[0], s[1] + outer_normal[1], height],  # 7: 起点外侧
    ]

    # 12个三角面（6个矩形面 * 2个三角形）
    # 所有法线朝向墙体立方体外部(远离立方体内部空间)
    faces = [
        # 底面 (法向朝下-z): 顺时针0→3→2→1 (从下方看)
        [0, 3, 2], [0, 2, 1],
        # 顶面 (法向朝上+z): 顺时针4→5→6→7 (从上方看)
        [4, 5, 6], [4, 6, 7],
        # 内侧面 (法向朝房间内+orientation): 逆时针1→0→4→5 (从房间内看墙的内表面)
        [1, 0, 4], [1, 4, 5],
        # 外侧面 (法向朝房间外-orientation): 顺时针2→3→7→6 (从房间外看墙的外表面)
        [2, 3, 7], [2, 7, 6],
        # 起点端面 (s端,法向朝-墙方向): 0→3→7→4
        [0, 3, 7], [0, 7, 4],
        # 终点端面 (e端,法向朝+墙方向): 1→5→6→2
        [1, 5, 6], [1, 6, 2],
    ]

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def cut_opening_from_wall(wall_mesh: trimesh.Trimesh, wall: Dict, opening: Dict,
                         thickness: float) -> trimesh.Trimesh:
    """
    从墙体中挖去门/窗的开口

    用途: construct_floor中处理门窗，在墙上创建开口
    实现: 创建门/窗的box几何体，使用trimesh的boolean difference操作

    注意: 这是一个占位实现，实际的boolean操作可能需要更复杂的处理

    参数:
        wall_mesh: 原始墙体mesh
        wall: 墙体数据
        opening: 门/窗数据，包含 center, width, height
        thickness: 墙体厚度

    返回:
        挖去开口后的墙体mesh
    """
    # 简化实现：直接返回原墙体
    # 实际应用中可以使用trimesh.boolean.difference进行布尔运算
    return wall_mesh


def load_mesh(mesh_id: int, config: Dict):
    """
    加载家具mesh

    用途: construct_scene中加载GLTF/GLB模型
    实现:
        1. 优先尝试加载GLB格式
        2. 如果失败，尝试加载GLTF格式
        3. 使用trimesh.load直接加载

    参数:
        mesh_id: 模型ID
        config: 配置字典，包含model_path

    返回:
        trimesh.Scene或trimesh.Trimesh对象，失败返回None
    """
    glb_path = os.path.join(config["model_path"], f"{mesh_id}.glb")
    gltf_path = os.path.join(config["model_path"], f"{mesh_id}.gltf")

    # 按优先级尝试加载
    for path in [glb_path, gltf_path]:
        if os.path.exists(path):
            try:
                return trimesh.load(path)
            except Exception:
                continue

    return None


def create_door_or_window_mesh(center: List[float], width: float, height: float, 
                               wall_data: Dict, texture_path: str) -> trimesh.Trimesh:
    """
    创建门或窗的mesh，贴在墙体上
    
    参数:
        center: 门/窗中心位置 [x, y, z]
        width: 宽度
        height: 高度
        wall_data: 墙体数据，包含s, e, orientation
        texture_path: 贴图路径
        
    返回:
        trimesh.Trimesh对象，带贴图和正确法线
    """
    # 墙体的起点和终点
    s = np.array(wall_data["s"], dtype=float)
    e = np.array(wall_data["e"], dtype=float)
    orientation = np.array(wall_data["orientation"], dtype=float)
    
    # 墙的方向向量（沿着墙）
    wall_dir = e - s
    wall_length = np.linalg.norm(wall_dir)
    if wall_length < 1e-9:
        return trimesh.Trimesh()
    wall_dir_norm = wall_dir / wall_length
    
    # 门/窗的中心在墙上的投影
    center_2d = np.array([center[0], center[1]], dtype=float)
    
    # 计算门/窗在墙上的局部坐标（沿墙方向）
    to_center = center_2d - s
    along_wall = np.dot(to_center, wall_dir_norm)
    
    # 门/窗的4个角点（3D空间）
    # 沿墙方向: ±width/2
    # 垂直方向: center[2] ±height/2
    
    half_width = width / 2
    half_height = height / 2
    z_bottom = center[2] - half_height
    z_top = center[2] + half_height
    
    # 在墙上的位置
    pos_left = along_wall - half_width
    pos_right = along_wall + half_width
    
    # 计算4个顶点的3D坐标
    vertices = np.array([
        # 底部左
        [s[0] + pos_left * wall_dir_norm[0], s[1] + pos_left * wall_dir_norm[1], z_bottom],
        # 底部右  
        [s[0] + pos_right * wall_dir_norm[0], s[1] + pos_right * wall_dir_norm[1], z_bottom],
        # 顶部右
        [s[0] + pos_right * wall_dir_norm[0], s[1] + pos_right * wall_dir_norm[1], z_top],
        # 顶部左
        [s[0] + pos_left * wall_dir_norm[0], s[1] + pos_left * wall_dir_norm[1], z_top],
    ], dtype=float)
    
    # 向墙内侧偏移一点点，避免z-fighting
    offset = 0.001
    offset_vec = np.array([orientation[0], orientation[1], 0], dtype=float) * offset
    vertices += offset_vec
    
    # UV坐标（整个贴图映射到矩形）
    uvs = np.array([
        [0.0, 0.0],  # 左下
        [1.0, 0.0],  # 右下
        [1.0, 1.0],  # 右上
        [0.0, 1.0],  # 左上
    ])
    
    # 计算法线方向（与墙体一致）
    # 使用和墙体相同的逻辑判断三角形顶点顺序
    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    default_normal = np.cross(edge1, edge2)
    default_normal = default_normal / (np.linalg.norm(default_normal) + 1e-9)
    
    # 目标法线（和墙体一致，指向房间内部）
    target_normal_3d = np.array([orientation[0], orientation[1], 0.0])
    dot = np.dot(default_normal, target_normal_3d)
    
    # 根据法线方向选择顶点顺序
    if dot > 0:
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    else:
        faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=int)
    
    # 创建mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # 加载并设置贴图
    try:
        from PIL import Image
        if os.path.exists(texture_path):
            texture_image = Image.open(texture_path)
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=texture_image)
        else:
            print(f"⚠️  贴图文件不存在: {texture_path}")
            # 使用默认颜色
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    except Exception as e:
        print(f"⚠️  加载贴图失败 {texture_path}: {e}")
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    
    return mesh


def create_windows_and_doors(walls_data: Dict, door_texture_path: str, 
                             window_texture_path: str) -> List[Dict]:
    """
    为所有墙体上的门窗创建mesh
    
    参数:
        walls_data: 墙体数据字典 {wall_id: {s, e, height, orientation, doors: {}, windows: {}}}
        door_texture_path: 门的贴图路径
        window_texture_path: 窗的贴图路径
        
    返回:
        门窗mesh列表 [{type: 'door'/'window', id: xxx, mesh: trimesh}]
    """
    result = []
    
    for wall_id, wall in walls_data.items():
        # 处理门
        for door_id, door_data in wall.get("doors", {}).items():
            door_mesh = create_door_or_window_mesh(
                center=door_data["center"],
                width=door_data["width"],
                height=door_data["height"],
                wall_data=wall,
                texture_path=door_texture_path
            )
            result.append({
                "type": "door",
                "id": door_id,
                "mesh": door_mesh,
                "wall_id": wall_id
            })
        
        # 处理窗
        for window_id, window_data in wall.get("windows", {}).items():
            window_mesh = create_door_or_window_mesh(
                center=window_data["center"],
                width=window_data["width"],
                height=window_data["height"],
                wall_data=wall,
                texture_path=window_texture_path
            )
            result.append({
                "type": "window",
                "id": window_id,
                "mesh": window_mesh,
                "wall_id": wall_id
            })
    
    return result


def segment_intersects_wall(segment_start: np.ndarray, segment_end: np.ndarray, 
                            wall_data: dict) -> bool:
    """
    检测线段是否与墙体相交（3D线段与墙体矩形面相交）
    
    Args:
        segment_start: 线段起点 [x, y, z]
        segment_end: 线段终点 [x, y, z]
        wall_data: 墙体数据，包含s, e, height
        
    Returns:
        bool: 是否相交
    """
    # 墙体的4个顶点（矩形）
    s = np.array(wall_data["s"] + [0], dtype=float)  # 底部起点
    e = np.array(wall_data["e"] + [0], dtype=float)  # 底部终点
    height = wall_data["height"]
    
    # 墙体的4个角点
    p1 = s  # 左下
    p2 = e  # 右下
    p3 = np.array([e[0], e[1], height], dtype=float)  # 右上
    p4 = np.array([s[0], s[1], height], dtype=float)  # 左上
    
    # 墙体法向量 (垂直于墙面)
    wall_dir = e - s
    wall_normal = np.array([-wall_dir[1], wall_dir[0], 0])
    wall_normal = wall_normal / np.linalg.norm(wall_normal)
    
    # 线段方向
    line_dir = segment_end - segment_start
    line_length = np.linalg.norm(line_dir)
    
    if line_length < 1e-9:
        return False
        
    line_dir_norm = line_dir / line_length
    
    # 计算线段与墙面所在平面的交点（使用平面方程）
    denom = np.dot(line_dir_norm, wall_normal)
    
    if abs(denom) < 1e-9:
        # 线段与墙面平行
        return False
    
    # 平面方程: dot(P - p1, wall_normal) = 0
    t = np.dot(p1 - segment_start, wall_normal) / denom
    
    # 检查交点是否在线段范围内
    if t < 0 or t > line_length:
        return False
    
    # 计算交点
    intersection = segment_start + t * line_dir_norm
    
    # 检查交点是否在墙体矩形内
    # 将3D问题投影到2D（墙面的局部坐标系）
    # 墙面的两个轴：沿着墙 (wall_dir) 和 垂直向上 (0,0,1)
    to_intersection = intersection - p1
    
    # 沿墙方向的投影
    wall_dir_norm = wall_dir / np.linalg.norm(wall_dir)
    proj_along_wall = np.dot(to_intersection, wall_dir_norm)
    wall_length = np.linalg.norm(wall_dir)
    
    # 沿高度方向的投影
    proj_along_height = intersection[2] - p1[2]
    
    # 检查是否在矩形范围内（留一点容差）
    epsilon = 0.01
    if (proj_along_wall >= -epsilon and proj_along_wall <= wall_length + epsilon and
        proj_along_height >= -epsilon and proj_along_height <= height + epsilon):
        return True
    
    return False


def find_walls_to_make_transparent(camera_pos_2d: list, look_at_2d: list, 
                                   floor_vertices: List[Tuple[float, float]], 
                                   walls_dict: dict) -> list:
    """
    基于相机位置和视角，确定需要设为透明的墙体
    
    算法：
    1. 检查相机(x,y)是否在凹多边形内
    2. 如果在内部：不透明任何墙体（返回空列表）
    3. 如果在外部：
       - 从相机到每个顶点计算射线方向
       - 找到夹角最大的两条射线（视锥边界）
       - 确定这两个顶点之间的"前方弧段"
       - 前方弧段上的墙体设为透明
    
    Args:
        camera_pos_2d: 相机位置 [x, y]
        look_at_2d: 目标位置 [x, y]
        floor_vertices: 地板多边形顶点列表（凹多边形）
        walls_dict: 墙体字典 {wall_id: wall_data}
        
    Returns:
        list: 需要设为透明的墙体ID列表
    """
    camera_pos = np.array(camera_pos_2d, dtype=float)
    look_at = np.array(look_at_2d, dtype=float)
    
    # 步骤1: 检查相机是否在多边形内部
    camera_tuple = tuple(camera_pos)
    is_inside = point_in_polygon(camera_tuple, floor_vertices)
    
    if is_inside:
        # 相机在房间内，不需要透明任何墙体
        print("📍 相机位于房间内部，不设置透明墙体")
        return []
    
    print("📍 相机位于房间外部，计算需要透明的墙体...")
    
    # 步骤2: 计算相机朝向（参考方向）
    camera_direction = look_at - camera_pos
    if np.linalg.norm(camera_direction) < 1e-6:
        # 相机和目标点重合，无法确定方向
        print("⚠️  相机和目标点重合，无法确定朝向")
        return []
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    
    # 步骤3: 计算从相机到每个顶点的向量和夹角
    vertex_angles = []
    for i, vertex in enumerate(floor_vertices):
        vertex_pos = np.array(vertex, dtype=float)
        to_vertex = vertex_pos - camera_pos
        
        if np.linalg.norm(to_vertex) < 1e-6:
            # 顶点和相机重合
            angle = 0.0
        else:
            to_vertex_norm = to_vertex / np.linalg.norm(to_vertex)
            
            # 计算带符号的夹角（-180° 到 +180°）
            # 使用 atan2 计算相对于相机朝向的角度
            cos_angle = np.dot(camera_direction, to_vertex_norm)
            # 计算垂直方向的分量（用于确定左右）
            cross = camera_direction[0] * to_vertex_norm[1] - camera_direction[1] * to_vertex_norm[0]
            angle = np.arctan2(cross, cos_angle)  # 返回 [-π, π]
        
        vertex_angles.append((i, vertex, angle))
    
    # 步骤4: 找到最左（最负角度）和最右（最正角度）的顶点
    vertex_angles.sort(key=lambda x: x[2])
    leftmost_idx, leftmost_vertex, leftmost_angle = vertex_angles[0]
    rightmost_idx, rightmost_vertex, rightmost_angle = vertex_angles[-1]
    
    print(f"   最左顶点: 索引{leftmost_idx}, 角度{np.degrees(leftmost_angle):.1f}°")
    print(f"   最右顶点: 索引{rightmost_idx}, 角度{np.degrees(rightmost_angle):.1f}°")
    
    # 步骤5: 确定前方弧段（从最左到最右，选择距离相机更近的路径）
    n_vertices = len(floor_vertices)
    
    # 计算两种路径
    if leftmost_idx <= rightmost_idx:
        path1 = list(range(leftmost_idx, rightmost_idx + 1))
        path2 = list(range(rightmost_idx, n_vertices)) + list(range(0, leftmost_idx + 1))
    else:
        path1 = list(range(leftmost_idx, n_vertices)) + list(range(0, rightmost_idx + 1))
        path2 = list(range(rightmost_idx, leftmost_idx + 1))
    
    # 计算每条路径上顶点到相机的平均距离
    def avg_distance_to_camera(path):
        if not path:
            return float('inf')
        distances = []
        for idx in path:
            vertex = np.array(floor_vertices[idx], dtype=float)
            dist = np.linalg.norm(vertex - camera_pos)
            distances.append(dist)
        return np.mean(distances)
    
    dist1 = avg_distance_to_camera(path1)
    dist2 = avg_distance_to_camera(path2)
    
    # 选择距离相机更近的路径作为前方弧段
    front_vertex_indices = path1 if dist1 < dist2 else path2
    front_vertices_set = set(front_vertex_indices)
    
    print(f"   路径1平均距离: {dist1:.2f}m, 路径2平均距离: {dist2:.2f}m")
    print(f"   选择{'路径1' if dist1 < dist2 else '路径2'}作为前方弧段（距离更近）")
    
    print(f"   前方弧段包含 {len(front_vertex_indices)} 个顶点: {front_vertex_indices}")
    print(f"   前方弧段顶点坐标: {[floor_vertices[i] for i in front_vertex_indices]}")
    
    # 步骤6: 找到前方弧段上的墙体
    # 前方弧段上的墙 = 连接前方弧段中相邻顶点的墙
    transparent_wall_ids = []
    
    for wall_id, wall_data in walls_dict.items():
        wall_s = tuple(wall_data["s"])
        wall_e = tuple(wall_data["e"])
        
        # 检查墙的两个端点是否都在顶点列表中
        try:
            s_idx = floor_vertices.index(wall_s)
            e_idx = floor_vertices.index(wall_e)
        except ValueError:
            # 墙的端点不在顶点列表中（不应该发生）
            continue
        
        # 检查墙的两个端点是否都在前方弧段中
        if s_idx not in front_vertices_set or e_idx not in front_vertices_set:
            continue
        
        # 检查这两个端点在前方弧段中是否相邻
        # 在front_vertex_indices中找到它们的位置
        try:
            pos_s = front_vertex_indices.index(s_idx)
            pos_e = front_vertex_indices.index(e_idx)
        except ValueError:
            continue
        
        # 检查是否相邻（考虑环形）
        n_front = len(front_vertex_indices)
        is_adjacent = (abs(pos_s - pos_e) == 1 or 
                      abs(pos_s - pos_e) == n_front - 1)
        
        if is_adjacent:
            transparent_wall_ids.append(wall_id)
            print(f"      透明墙: {wall_id}, 端点索引({s_idx},{e_idx}), 在前方弧段位置({pos_s},{pos_e})")
    
    print(f"   共 {len(transparent_wall_ids)} 面墙体将设为透明")
    
    return transparent_wall_ids


def find_intersecting_walls(camera_position: list, look_at_target: list, mesh_nodes: dict) -> list:
    """
    找到与相机到目标线段相交的所有墙体ID（旧方法，已弃用）
    
    Args:
        camera_position: 相机位置 [x, y, z]
        look_at_target: 目标位置 [x, y, z]
        mesh_nodes: 场景的mesh节点字典
        
    Returns:
        list: 相交的墙体ID列表
    """
    segment_start = np.array(camera_position, dtype=float)
    segment_end = np.array(look_at_target, dtype=float)
    
    intersecting_wall_ids = []
    
    for wall_id, wall_info in mesh_nodes["walls"].items():
        wall_data = wall_info["wall_data"]
        if segment_intersects_wall(segment_start, segment_end, wall_data):
            intersecting_wall_ids.append(wall_id)
    
    return intersecting_wall_ids


def set_mesh_alpha(mesh_nodes: dict, mesh_type: str, mesh_id: str, alpha: float = 0.3):
    """
    设置指定mesh的透明度
    
    Args:
        mesh_nodes: 场景的mesh节点字典
        mesh_type: mesh类型 ("walls", "boxes", "doors", "windows")
        mesh_id: mesh的唯一ID
        alpha: 透明度 (0.0-1.0, 0为完全透明, 1为完全不透明)
    """
    import pyrender
    
    if mesh_type not in mesh_nodes:
        print(f"⚠️  未知的mesh类型: {mesh_type}")
        return
    
    if mesh_id not in mesh_nodes[mesh_type]:
        print(f"⚠️  未找到ID为 {mesh_id} 的{mesh_type}")
        return
    
    mesh_info = mesh_nodes[mesh_type][mesh_id]
    
    # 处理单节点或多节点
    nodes_to_update = []
    if "node" in mesh_info:
        nodes_to_update = [mesh_info["node"]]
    elif "nodes" in mesh_info:
        nodes_to_update = mesh_info["nodes"]
    
    for node in nodes_to_update:
        if node and node.mesh:
            # 更新所有primitives的材质
            for primitive in node.mesh.primitives:
                # 保存原始材质（如果还没保存）
                if not hasattr(primitive, '_original_material'):
                    primitive._original_material = primitive.material
                
                # 创建新的透明材质（使用pyrender的材质类）
                if primitive.material:
                    # 复制现有材质属性
                    mat = primitive.material
                    baseColorFactor = list(getattr(mat, 'baseColorFactor', [1.0, 1.0, 1.0, 1.0]))
                    baseColorFactor[3] = alpha  # 设置alpha通道
                    
                    # 创建pyrender的MetallicRoughnessMaterial
                    new_mat = pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=baseColorFactor,
                        metallicFactor=getattr(mat, 'metallicFactor', 0.0),
                        roughnessFactor=getattr(mat, 'roughnessFactor', 1.0),
                        alphaMode='BLEND',  # 关键：启用透明度混合
                        doubleSided=True
                    )
                    
                    # 保留纹理（如果有）
                    if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                        new_mat.baseColorTexture = mat.baseColorTexture
                    
                    primitive.material = new_mat
                else:
                    # 没有材质，创建一个透明材质
                    primitive.material = pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=[0.8, 0.8, 0.8, alpha],
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                        alphaMode='BLEND',
                        doubleSided=True
                    )
    
    print(f"✅ 已设置 {mesh_type}[{mesh_id}] 的透明度为 {alpha}")


def reset_mesh_alpha(mesh_nodes: dict, mesh_type: str, mesh_id: str):
    """
    恢复指定mesh的原始材质
    
    Args:
        mesh_nodes: 场景的mesh节点字典
        mesh_type: mesh类型
        mesh_id: mesh的唯一ID
    """
    if mesh_type not in mesh_nodes:
        return
    
    if mesh_id not in mesh_nodes[mesh_type]:
        return
    
    mesh_info = mesh_nodes[mesh_type][mesh_id]
    
    nodes_to_update = []
    if "node" in mesh_info:
        nodes_to_update = [mesh_info["node"]]
    elif "nodes" in mesh_info:
        nodes_to_update = mesh_info["nodes"]
    
    for node in nodes_to_update:
        if node and node.mesh:
            for primitive in node.mesh.primitives:
                if hasattr(primitive, '_original_material'):
                    primitive.material = primitive._original_material
                    delattr(primitive, '_original_material')
    
    print(f"✅ 已恢复 {mesh_type}[{mesh_id}] 的原始材质")


def reset_all_alpha(mesh_nodes: dict):
    """
    恢复所有mesh的原始材质
    
    Args:
        mesh_nodes: 场景的mesh节点字典
    """
    for mesh_type in mesh_nodes:
        if isinstance(mesh_nodes[mesh_type], dict):
            for mesh_id in list(mesh_nodes[mesh_type].keys()):
                reset_mesh_alpha(mesh_nodes, mesh_type, mesh_id)


def calculate_optimal_fov(camera_position: np.ndarray, look_at_target: np.ndarray,
                         floor_vertices: List[Tuple[float, float]], z_max: float,
                         bounds: List[float], indoor_fov: float = 120.0, 
                         outdoor_fov_scale: float = 1.05) -> float:
    """
    智能计算最佳FOV
    
    Args:
        camera_position: 相机位置 [x, y, z]
        look_at_target: 目标位置 [x, y, z]
        floor_vertices: 地板顶点列表（凹多边形）
        z_max: 房间高度
        bounds: 边界 [x_min, y_min, x_max, y_max]
        indoor_fov: 室内FOV（度）
        outdoor_fov_scale: 室外FOV缩放系数
        
    Returns:
        fov_y: 垂直视场角（弧度）
        
    算法：
        情况1: 相机在包围盒（XY+Z）内部 → 使用固定FOV 70°
        情况2: 相机的 z 坐标超出 [0, z_max] → 以相机到包围盒中心连线为方向，确保视锥包裹整个 3D 包围盒
        情况3: 相机在垂直范围内但 XY 在包围盒外 → 使用之前的视锥平面与立方体交点计算
            - 垂直平面（forward + up）与立方体相交 → 计算最大夹角
            - 水平平面（forward + right）与立方体相交 → 计算最大夹角
            - FOV = 2 × max(两个夹角) × scale_factor
    """
    # 包围盒中心（用于 z 越界时确定方向）
    center_xy = np.array([(bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0], dtype=float)
    bbox_center = np.array([center_xy[0], center_xy[1], z_max / 2.0], dtype=float)

    # 检查相机是否在包围盒内（xy + z）
    in_bbox_xy = (bounds[0] <= camera_position[0] <= bounds[2] and 
                  bounds[1] <= camera_position[1] <= bounds[3])
    in_bbox_z = (0 <= camera_position[2] <= z_max)

    if in_bbox_xy and in_bbox_z:
        # 情况1: 相机在包围盒内部，使用传入的 indoor_fov
        fixed_fov = indoor_fov
        print(f"📐 相机在包围盒内部，使用固定FOV: {fixed_fov}°")
        return np.radians(fixed_fov)
    
    camera_z_outside = camera_position[2] < 0 or camera_position[2] > z_max
    if camera_z_outside:
        print("📐 相机 z 超出 [0, z_max]，以包围盒中心为方向计算 FOV...")
        target_point = bbox_center
    else:
        print("📐 相机在垂直范围内但 XY 超出包围盒，基于视锥平面计算 FOV...")
        target_point = look_at_target

    # 计算相机坐标系
    forward = target_point - camera_position
    forward = forward / np.linalg.norm(forward)
    
    # 假设up向量为z轴向上
    world_up = np.array([0, 0, 1], dtype=float)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # forward和world_up平行，使用y轴
        right = np.cross(forward, np.array([0, 1, 0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    # 构建立方体边（vertices多边形向上拉伸）
    edges_3d = []
    n = len(floor_vertices)
    
    # 底边（z=0）
    for i in range(n):
        v1 = np.array([floor_vertices[i][0], floor_vertices[i][1], 0.0])
        v2 = np.array([floor_vertices[(i+1)%n][0], floor_vertices[(i+1)%n][1], 0.0])
        edges_3d.append((v1, v2))
    
    # 顶边（z=z_max）
    for i in range(n):
        v1 = np.array([floor_vertices[i][0], floor_vertices[i][1], z_max])
        v2 = np.array([floor_vertices[(i+1)%n][0], floor_vertices[(i+1)%n][1], z_max])
        edges_3d.append((v1, v2))
    
    # 竖边
    for i in range(n):
        v1 = np.array([floor_vertices[i][0], floor_vertices[i][1], 0.0])
        v2 = np.array([floor_vertices[i][0], floor_vertices[i][1], z_max])
        edges_3d.append((v1, v2))
    
    # 计算两个平面与立方体的交点
    # 平面1：垂直平面（forward + up）
    plane1_normal = np.cross(forward, up)
    plane1_normal = plane1_normal / np.linalg.norm(plane1_normal)
    
    # 平面2：水平平面（forward + right）
    plane2_normal = np.cross(forward, right)
    plane2_normal = plane2_normal / np.linalg.norm(plane2_normal)
    
    def plane_edge_intersection(plane_normal, plane_point, edge_start, edge_end):
        """计算平面与线段的交点"""
        # 平面方程: dot(P - plane_point, plane_normal) = 0
        # 线段参数方程: P = edge_start + t * (edge_end - edge_start), t in [0,1]
        
        edge_dir = edge_end - edge_start
        denom = np.dot(edge_dir, plane_normal)
        
        if abs(denom) < 1e-9:
            # 边与平面平行
            return None
        
        t = np.dot(plane_point - edge_start, plane_normal) / denom
        
        if 0 <= t <= 1:
            return edge_start + t * edge_dir
        return None
    
    # 收集两个平面的交点
    intersections_plane1 = []
    intersections_plane2 = []
    
    for edge_start, edge_end in edges_3d:
        pt1 = plane_edge_intersection(plane1_normal, camera_position, edge_start, edge_end)
        if pt1 is not None:
            intersections_plane1.append(pt1)
        
        pt2 = plane_edge_intersection(plane2_normal, camera_position, edge_start, edge_end)
        if pt2 is not None:
            intersections_plane2.append(pt2)
    
    # 计算每个平面交点的最大夹角
    def calc_max_angle(intersections):
        if len(intersections) == 0:
            return 0.0
        
        max_angle = 0.0
        for pt in intersections:
            to_pt = pt - camera_position
            dist = np.linalg.norm(to_pt)
            if dist < 1e-6:
                continue
            to_pt_norm = to_pt / dist
            cos_angle = np.dot(forward, to_pt_norm)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            max_angle = max(max_angle, angle)
        
        return max_angle
    
    angle1 = calc_max_angle(intersections_plane1)
    angle2 = calc_max_angle(intersections_plane2)
    
    print(f"   垂直平面交点数: {len(intersections_plane1)}, 最大夹角: {np.degrees(angle1):.1f}°")
    print(f"   水平平面交点数: {len(intersections_plane2)}, 最大夹角: {np.degrees(angle2):.1f}°")
    
    # FOV = 2 * max_angle * scale_factor
    max_angle = max(angle1, angle2)
    fov_y = 2 * max_angle * outdoor_fov_scale
    
    # 限制范围
    fov_y = np.clip(fov_y, np.radians(10), np.radians(120))
    
    print(f"   FOV = 2 × {np.degrees(max_angle):.1f}° × {outdoor_fov_scale} = {np.degrees(fov_y):.1f}°")
    
    return fov_y
def create_wall_edge_lines(start: List[float], end: List[float], height: float,
                           orientation: List[float], edge_color: List[float] = None,
                           offset: float = 0.01):
    """
    为墙体创建边缘线条，使墙与地面、墙与墙之间的分界线更明显
    边缘线会向墙的内侧（房间内部）偏移，避免相邻墙的边缘线重叠
    
    Args:
        start: 墙的起点 [x, y]
        end: 墙的终点 [x, y]
        height: 墙的高度
        orientation: 墙的朝向（法向量） [nx, ny]，指向房间内部
        edge_color: 边缘线条颜色 [R, G, B, A]，默认为深灰色
        offset: 边缘线向内偏移距离，默认0.01米
        
    Returns:
        pyrender.Mesh 线条对象
    """
    import pyrender
    # 计算向内偏移的量（沿着orientation方向）
    offset_vec = np.array([orientation[0] * offset, orientation[1] * offset, 0.0])
    
    # 定义墙的4个顶点，并向内偏移
    vertices = np.array([
        [start[0], start[1], 0.0],        # 0: 底部起点
        [end[0], end[1], 0.0],            # 1: 底部终点
        [end[0], end[1], height],         # 2: 顶部终点
        [start[0], start[1], height]      # 3: 顶部起点
    ], dtype=np.float32)
    
    # 将所有顶点向房间内部偏移
    vertices += offset_vec
    
    # 定义边缘线的连接关系（线段索引对）
    # 4条边缘线：底边、顶边、左边、右边
    edges = np.array([
        [0, 1],  # 底边（墙与地面）
        [2, 3],  # 顶边（墙与天花板）
        [0, 3],  # 左边（墙与墙）
        [1, 2],  # 右边（墙与墙）
    ], dtype=np.uint32)
    
    # 设置边缘线的颜色
    if edge_color is None:
        edge_color = [0.1, 0.1, 0.1, 1.0]  # 默认深灰色
    edge_color = np.array(edge_color, dtype=np.float32)
    
    # 创建线条的材质
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=edge_color,
        metallicFactor=0.0,
        roughnessFactor=1.0
    )
    
    # 创建线条 primitive
    primitive = pyrender.Primitive(
        positions=vertices,
        indices=edges,
        mode=pyrender.constants.GLTF.LINES,
        material=material
    )
    
    # 创建并返回 pyrender.Mesh 对象
    return pyrender.Mesh(primitives=[primitive])
