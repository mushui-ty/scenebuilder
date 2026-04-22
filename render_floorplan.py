import json
import math
import os
import time
import numpy as np
from typing import Any, Dict, List, Optional

try:
    from .fast_scene_bpy import BpySceneCtx
except ImportError:
    from fast_scene_bpy import BpySceneCtx

def _normalize_height(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 3.5

def _build_walls_from_data(floor_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从新的数据结构构建原始墙体列表
    数据结构包含:
    - vertices: [[x1, y1], [x2, y2], ...] 外部边界
    - partition: [[xs, ys, xe, ye], ...] 内部隔断 (注意：这里是s和e的坐标拼接)
    """
    vertices = floor_plan.get("vertices") or []
    partitions = floor_plan.get("partition") or []
    height = _normalize_height(floor_plan.get("height"))
    
    walls = []
    
    # 1. 处理外部边界
    for i in range(len(vertices)):
        start = vertices[i]
        end = vertices[(i + 1) % len(vertices)]
        walls.append({
            "p": [float(start[0]), float(start[1]), 0.0],
            "q": [float(end[0]), float(end[1]), 0.0],
            "height": height
        })
        
    # 2. 处理内部隔断 (递归兼容多层嵌套格式)
    def _collect_partitions(data: Any):
        if not isinstance(data, list):
            return
        # 检查是否是 [xs, ys, xe, ye] 格式 (至少4个数字)
        if len(data) >= 4 and all(isinstance(x, (int, float)) for x in data[:4]):
            walls.append({
                "p": [float(data[0]), float(data[1]), 0.0],
                "q": [float(data[2]), float(data[3]), 0.0],
                "height": height
            })
        else:
            # 否则继续递归处理子项
            for item in data:
                _collect_partitions(item)

    _collect_partitions(partitions)
            
    return walls

def render_floor_plan(
    floor_plan: Dict[str, Any],
    plan_dir: str,
    scene_name: str,
    description: str,
    width: int = 1024,
    height: int = 1024
) -> Optional[str]:
    """
    增强版的 floor_plan 渲染函数，支持内部隔断墙。
    """
    vertices_input = floor_plan.get("vertices") or []
    if not vertices_input:
        print("⚠️ 该 floor_plan 不含有效边界顶点，略过渲染")
        return None

    # 获取完整场景名称作为类型
    scene_type = scene_name if scene_name else "room"
    
    # 初始化 BpySceneCtx (基于 Blender)
    ctx = BpySceneCtx(scene_type)
    
    # 构建初始墙体
    raw_walls = _build_walls_from_data(floor_plan)
    if not raw_walls:
        print("⚠️ 未能构建任何墙体，略过渲染")
        return None

    # 调用 add_walls
    # 注意：BpySceneCtx.add_walls 内部现在会自动调用 util.calculate_minimum_area_polygon_and_partitions
    # 识别出边界墙和隔断墙，并应用吸附逻辑。
    ctx.add_walls(raw_walls)

    # 确保输出目录存在
    os.makedirs(plan_dir, exist_ok=True)
    
    # 渲染俯视图
    # 默认设置：显示墙体，不显示门窗/天花板，不使用HDRI
    topdown_path = os.path.join(plan_dir, "topdown.png")
    ctx.topdown_view(
        topdown_path, 
        width=width, 
        height=height,
        show_wall=True, 
        show_window=False, 
        show_door=False, 
        show_ceiling=False, 
        use_HDRI=False
    )
    
    # 导出墙体 SSL 数据
    ctx.export_wall_ssl(plan_dir)
    
    # 保存元数据
    metadata_path = os.path.join(plan_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "scene_name": scene_name,
            "description": description,
            "floor_plan": floor_plan,
            "timestamp": int(time.time())
        }, f, ensure_ascii=False, indent=4)
        
    print(f"✅ [Render] 俯视图与元数据已写入: {plan_dir}")
    return plan_dir

def render_floorplan_view(
    floor_plan: Dict[str, Any],
    plan_dir: str,
    camera_position: List[float],
    fov_y: float,
    width: int = 1024,
    height: int = 1024,
) -> Optional[str]:
    """Render floorplan with fixed camera parameters (no auto camera computation).

    Uses render_view instead of topdown_view so that camera_position and fov_y
    are preserved exactly as given. This allows shift adjustments to be
    reflected directly in the rendered image.

    Args:
        floor_plan: Dict with vertices, partition, height (SSL coordinates)
        plan_dir: Output directory
        camera_position: Camera position in SSL coords [x, y, z]
        fov_y: Vertical field of view in radians
        width, height: Render resolution
    """
    vertices_input = floor_plan.get("vertices") or []
    if not vertices_input:
        print("⚠️ floor_plan has no valid vertices, skipping render")
        return None

    ctx = BpySceneCtx("floorplan_view")
    raw_walls = _build_walls_from_data(floor_plan)
    if not raw_walls:
        print("⚠️ No walls built, skipping render")
        return None

    ctx.add_walls(raw_walls)
    os.makedirs(plan_dir, exist_ok=True)

    topdown_path = os.path.join(plan_dir, "topdown.png")
    ctx.render_view(
        output_path=topdown_path,
        camera_position=camera_position,
        look_at_target=[camera_position[0], camera_position[1], 0.0],
        width=width,
        height=height,
        up_vector=[0.0, 1.0, 0.0],
        auto_fov=False,
        manual_fov=math.degrees(fov_y),
        auto_transparent=False,
        show_wall=True,
        show_window=False,
        show_door=False,
        show_ceiling=False,
        use_HDRI=False,
    )

    return plan_dir


if __name__ == "__main__":
    # 测试数据
    test_data = [
        {   
            "scene_name": "General-Restaurant",
            "description": "餐厅整体呈现 L 型布局。入口位于南侧中心位置，进入后正对着宽敞的开放式用餐区，这里整齐排列着深色实木方桌和皮质餐椅。西侧（左侧）沿墙设有一排连续的卡座沙发，配以大理石纹路的小方桌。餐厅东北侧通过隔断墙划分出两个独立的私密包间，每个包间内中心位置摆放着一张配有自动旋转底盘的大型圆桌，墙角点缀有落地盆栽。服务台位于靠近厨房通道的转角处。整体铺设浅灰色哑光地砖，天花板吊挂着阵列式的暖色工业风吊灯。",
            "floor_plan": {
                "vertices": [[0, 0], [14, 0], [14, 8], [6, 8], [6, 4], [0, 4]], 
                "partition": [[10, 8, 10, 4.5], [10, 4.5, 14, 4.5], [12, 8, 12, 4.5]],
                "height": 3.5,
                "shape" : "L型"
            }
        }
    ]
    
    # 运行测试
    output_base = "/data-nas/data/experiments/mushui/projects/utils/fast-scene/output"
    for item in test_data:
        scene_name = item["scene_name"]
        description = item["description"]
        floor_plan = item["floor_plan"]
        
        # 构造输出路径
        timestamp = int(time.time())
        plan_dir = os.path.join(output_base, scene_name.replace("-", "/"), f"plan_{timestamp}")
        render_floor_plan(floor_plan, plan_dir, scene_name, description)
