#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用SSL格式数据渲染场景 - 支持 BPY 和 Pyrender 双后端
"""

import os
import re
import json
import time
import yaml
import numpy as np
from typing import Dict, Any, List, Optional

# 默认 SSL 示例
ssl_example = '''
Room(id="D54g", room_type="dining room")
Wall(id="0", room_id="D54g", p=[0.0, 1.38, 0.0], q=[3.11, 1.38, 0.0], height=2.9)
Wall(id="1", room_id="D54g", p=[3.11, 1.38, 0.0], q=[3.11, 0.0, 0.0], height=2.9)
Wall(id="2", room_id="D54g", p=[3.11, 0.0, 0.0], q=[7.82, 0.02, 0.0], height=2.9)
Wall(id="3", room_id="D54g", p=[7.82, 0.02, 0.0], q=[7.82, 4.28, 0.0], height=2.9)
Wall(id="4", room_id="D54g", p=[7.82, 4.28, 0.0], q=[0.0, 4.28, 0.0], height=2.9)
Wall(id="5", room_id="D54g", p=[0.0, 4.28, 0.0], q=[0.0, 1.38, 0.0], height=2.9)
Door(id="0", wall_id="4", center=[1.59, 4.28, 1.1], width=1.06, height=2.2)
Door(id="1", wall_id="4", center=[4.02, 4.28, 1.1], width=1.41, height=2.2)
Door(id="2", wall_id="4", center=[6.41, 4.28, 1.1], width=2.6, height=2.2)
Door(id="3", wall_id="3", center=[7.82, 2.47, 1.0], width=3.13, height=2.0)
Window(id="0", wall_id="5", center=[0.0, 2.8, 1.43], width=1.56, height=1.2)
Bbox(id="0", room_id="D54g", label="curtain", center=[0.02, 2.81, 1.51], angle_z=90, scale=[1.6, 0.05, 1.3], mesh_id="3922456")
Bbox(id="3", room_id="D54g", label="kitchen cabinet", center=[0.3, 1.88, 0.66], angle_z=90, scale=[1.0, 0.6, 1.33], mesh_id="12038494")
Bbox(id="1", room_id="D54g", label="kitchen cabinet", center=[0.3, 2.87, 0.43], angle_z=90, scale=[0.99, 0.6, 0.86], mesh_id="7886809")
Bbox(id="2", room_id="D54g", label="kitchen cabinet", center=[0.3, 3.81, 0.43], angle_z=90, scale=[0.93, 0.6, 0.86], mesh_id="7887034")
Bbox(id="4", room_id="D54g", label="kitchen cabinet", center=[2.47, 2.76, 0.51], angle_z=180, scale=[2.2, 0.93, 1.03], mesh_id="2840139")
Bbox(id="5", room_id="D54g", label="kitchen cabinet", center=[5.47, 0.36, 1.56], angle_z=180, scale=[4.7, 0.67, 3.13], mesh_id="40342063")
Bbox(id="6", room_id="D54g", label="ornaments", center=[0.09, 3.95, 1.39], angle_z=90, scale=[0.64, 0.18, 0.76], mesh_id="34142540")
Bbox(id="7", room_id="D54g", label="ornaments", center=[0.23, 3.9, 1.02], angle_z=90, scale=[0.74, 0.46, 0.33], mesh_id="34142599")
Bbox(id="8", room_id="D54g", label="ornaments", center=[0.31, 3.01, 0.94], angle_z=0, scale=[0.43, 0.55, 0.17], mesh_id="13848022")
Bbox(id="9", room_id="D54g", label="ornaments", center=[1.87, 2.61, 0.93], angle_z=0, scale=[0.58, 0.39, 0.19], mesh_id="35368125")
Bbox(id="10", room_id="D54g", label="decorative painting", center=[1.97, 1.4, 1.8], angle_z=180, scale=[1.23, 0.05, 0.54], mesh_id="48510583")
Bbox(id="11", room_id="D54g", label="dining chair", center=[3.98, 3.59, 0.42], angle_z=0, scale=[0.6, 0.58, 0.84], mesh_id="5829816")
Bbox(id="12", room_id="D54g", label="dining chair", center=[4.02, 1.97, 0.42], angle_z=180, scale=[0.6, 0.58, 0.84], mesh_id="5829816")
Bbox(id="13", room_id="D54g", label="dining chair", center=[4.74, 3.59, 0.42], angle_z=0, scale=[0.6, 0.58, 0.84], mesh_id="5829816")
Bbox(id="14", room_id="D54g", label="dining chair", center=[4.76, 1.97, 0.42], angle_z=180, scale=[0.6, 0.58, 0.84], mesh_id="5829816")
Bbox(id="15", room_id="D54g", label="dining table", center=[5.1, 2.77, 0.6], angle_z=0, scale=[3.05, 0.93, 1.21], mesh_id="17116819")
Bbox(id="16", room_id="D54g", label="dining chair", center=[5.47, 3.59, 0.42], angle_z=0, scale=[0.6, 0.58, 0.84], mesh_id="5829816")
Bbox(id="17", room_id="D54g", label="dining chair", center=[5.51, 1.97, 0.42], angle_z=180, scale=[0.6, 0.58, 0.84], mesh_id="5829816")
Bbox(id="18", room_id="D54g", label="dining chair", center=[6.16, 3.59, 0.42], angle_z=0, scale=[0.6, 0.58, 0.84], mesh_id="5829816")
Bbox(id="19", room_id="D54g", label="dining chair", center=[6.23, 1.97, 0.42], angle_z=180, scale=[0.6, 0.58, 0.84], mesh_id="5829816")
Bbox(id="20", room_id="D54g", label="dining chair", center=[6.91, 2.76, 0.42], angle_z=270, scale=[0.6, 0.58, 0.84], mesh_id="5829816")
'''

def parse_ssl_to_json(ssl_text: str) -> Dict[str, Any]:
    """
    解析SSL格式场景描述并返回指定的JSON结构
    """
    data = {
        "wall": [],
        "door": [],
        "window": [],
        "bbox": [],
        "room": {"id": "", "room_type": ""}
    }

    # 提取属性的正则辅助函数
    def get_attr(pattern, text, default=""):
        match = re.search(pattern, text)
        return match.group(1) if match else default

    def get_list_attr(pattern, text, default=None):
        match = re.search(pattern, text)
        if match:
            return [float(x.strip()) for x in match.group(1).split(',')]
        return default or []

    for line in ssl_text.strip().split('\n'):
        line = line.strip()
        if not line: continue

        # Room(id="...", room_type="...")
        if line.startswith('Room('):
            data["room"]["id"] = get_attr(r'id="([^"]+)"', line)
            data["room"]["room_type"] = get_attr(r'room_type="([^"]+)"', line)

        # Wall(id="...", p=[...], q=[...], height=...)
        elif line.startswith('Wall('):
            data["wall"].append({
                "id": get_attr(r'id="([^"]+)"', line),
                "p": get_list_attr(r'p=\[([^\]]+)\]', line),
                "q": get_list_attr(r'q=\[([^\]]+)\]', line),
                "height": float(get_attr(r'height=([\d.]+)', line, "2.8"))
            })

        # Door(id="...", wall_id="...", center=[...], width=..., height=..., mesh_id="...", category="...")
        elif line.startswith('Door('):
            data["door"].append({
                "id": get_attr(r'id="([^"]+)"', line),
                "wall_id": get_attr(r'wall_id="([^"]+)"', line),
                "center": get_list_attr(r'center=\[([^\]]+)\]', line),
                "width": float(get_attr(r'width=([\d.]+)', line, "0")),
                "height": float(get_attr(r'height=([\d.]+)', line, "0")),
                "category": get_attr(r'category="([^"]+)"', line),
                "mesh_id": get_attr(r'mesh_id="([^"]+)"', line)
            })

        # Window(...) - 同 Door 逻辑
        elif line.startswith('Window('):
            data["window"].append({
                "id": get_attr(r'id="([^"]+)"', line),
                "wall_id": get_attr(r'wall_id="([^"]+)"', line),
                "center": get_list_attr(r'center=\[([^\]]+)\]', line),
                "width": float(get_attr(r'width=([\d.]+)', line, "0")),
                "height": float(get_attr(r'height=([\d.]+)', line, "0")),
                "category": get_attr(r'category="([^"]+)"', line),
                "mesh_id": get_attr(r'mesh_id="([^"]+)"', line)
            })

        # Bbox(id="...", mesh_id="...", center=[...], angle_z=..., scale=[...], caption="...", label="...")
        elif line.startswith('Bbox('):
            data["bbox"].append({
                "id": get_attr(r'id="([^"]+)"', line),
                "mesh_id": get_attr(r'mesh_id="([^"]+)"', line),
                "center": get_list_attr(r'center=\[([^\]]+)\]', line),
                "angle_z": float(get_attr(r'angle_z=([\d.-]+)', line, "0")),
                "scale": get_list_attr(r'scale=\[([^\]]+)\]', line),
                "caption": get_attr(r'caption="([^"]+)"', line),
                "label": get_attr(r'label="([^"]+)"', line)
            })

    return data

def render_ssl(ssl_text: str, backend: str = 'bpy', output_root: str = 'output_ssl', geometry_mode: str = 'mixed'):
    """
    主渲染函数
    Args:
        ssl_text: SSL文本
        backend: 'bpy' 或 'pyrender'
        output_root: 输出根目录
        geometry_mode: 针对 bpy 后端的几何体加载模式
    """
    print(f"\n🚀 开始渲染 [后端: {backend}]")
    
    # 1. 解析为 JSON
    scene_json = parse_ssl_to_json(ssl_text)
    # print(scene_json)
    room_id = scene_json["room"]["id"] or "unnamed"
    room_type = scene_json["room"]["room_type"] or "default"
    
    # 2. 选择 Context
    if backend == 'bpy':
        try:
            from .fast_scene_bpy import BpySceneCtx
        except (ImportError, ValueError):
            from fast_scene_bpy import BpySceneCtx
        ctx = BpySceneCtx(room_type)
    else:
        try:
            from .fast_scene import SceneCtx
        except (ImportError, ValueError):
            from fast_scene import SceneCtx
        ctx = SceneCtx(room_type)

    # 3. 填充数据
    ctx.add_walls(scene_json["wall"])
    if scene_json["door"]: ctx.add_doors(scene_json["door"])
    if scene_json["window"]: ctx.add_windows(scene_json["window"])
    
    # 将 JSON 中的 bbox 数据适配到 ctx.add_boxes
    # ctx.add_boxes 接受列表，其中元素包含 label, center, angle_z, scale, mesh_id, caption
    ctx.add_boxes(scene_json["bbox"])


    # 4. 准备输出目录
    timestamp = int(time.time())
    output_dir = os.path.join(os.path.dirname(__file__), output_root, f"{timestamp}_{room_id}")
    os.makedirs(output_dir, exist_ok=True)

    # 5. 执行渲染 (俯视图 + 前视图)
    print("🎨 正在生成视图...")
    ctx.topdown_view(os.path.join(output_dir, "topdown.png"), show_ceiling=False)
    
    # center = ctx.context["meta"]["center"]
    # span = ctx.context["meta"]["span"]
    # z_max = ctx.context["meta"]["z_max"]
    # look_at = [center[0], center[1], z_max / 2]
    # camera_pos = [center[0], center[1] - max(span)/2 - 2, z_max * 2/3]
    
    # ctx.render_view(
    #     output_path=os.path.join(output_dir, "front.png"),
    #     camera_position=camera_pos,
    #     look_at_target=look_at
    # )

    # 7. 保存 JSON 和 SSL
    with open(os.path.join(output_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(scene_json, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "ssl.txt"), "w", encoding="utf-8") as f:
        f.write(ssl_text)

    print(f"✅ 渲染完成！结果保存在: {os.path.abspath(output_dir)}")
    return output_dir

if __name__ == "__main__":
    # 示例运行
    render_ssl(ssl_example, backend='bpy')
