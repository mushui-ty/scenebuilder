#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用SSL格式数据渲染场景 - 支持 BPY 和 Pyrender 双后端
"""

import os
import json
import time
import yaml
import numpy as np
from typing import Dict, Any, List, Optional, Literal

try:
    from .util_data import parse_ssl_to_json, get_mesh, update_ssl_with_mesh_id
except (ImportError, ValueError):
    from util_data import parse_ssl_to_json, get_mesh, update_ssl_with_mesh_id # type: ignore

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

def render_ssl(
    ssl_text: str, 
    backend: str = 'bpy', 
    output_root: str = 'output_ssl', 
    image: Optional[str] = None, 
    retrieve_hole: bool = True,
    asset_mode: Literal["none", "retrieve", "generate"] = "none",
    outpaint_image_dir: Optional[str] = None,
    gen_asset_dir: Optional[str] = "/data-nas/data/dataset/qunhe/Manycore-Future/generate",
    gen_3d_model: Literal["hunyuan-3d-rapid", "hunyuan-3d-pro"] = "hunyuan-3d-pro"
):
    """
    主渲染函数
    Args:
        ssl_text: SSL文本
        backend: 'bpy' 或 'pyrender'
        output_root: 输出根目录
        image: 可选的输入图片路径
        retrieve_hole: 是否检索门窗的 mesh_id
        asset_mode: 资产处理模式 ("none", "retrieve", "generate")
        outpaint_image_dir: 扩图结果保存目录 (默认为 output_root)
        gen_asset_dir: 生成资产保存目录 (默认为 output_root)
    """
    if outpaint_image_dir is None:
        outpaint_image_dir = output_root
 

    print(f"\n🚀 开始渲染 [后端: {backend}, 资产模式: {asset_mode}]")
    
    # 1. 解析为 JSON
    scene_json = parse_ssl_to_json(ssl_text)
    # print(scene_json)
    room_type = scene_json["room"]["room_type"]
    
    # 2. 资产处理
    scene_json = get_mesh(
        scene_json, 
        image_path=image, 
        retrieve_hole=retrieve_hole, 
        asset_mode=asset_mode,
        outpaint_image_dir=outpaint_image_dir,
        gen_asset_dir=gen_asset_dir,
        gen_3d_model=gen_3d_model
    )
    
    # 将更新后的 mesh_id 填回 SSL
    updated_ssl = update_ssl_with_mesh_id(ssl_text, scene_json)
      

    # 3. 选择 Context
    if backend == 'bpy':
        try:
            from .fast_scene_bpy import BpySceneCtx
        except (ImportError, ValueError):
            from fast_scene_bpy import BpySceneCtx # type: ignore
        ctx = BpySceneCtx(room_type, gen_asset_dir)
    else:
        try:
            from .fast_scene import SceneCtx
        except (ImportError, ValueError):
            from fast_scene import SceneCtx
        ctx = SceneCtx(room_type, gen_asset_dir)

    # 4. 填充数据
    ctx.add_walls(scene_json["wall"])
    if scene_json["door"]: ctx.add_doors(scene_json["door"])
    if scene_json["window"]: ctx.add_windows(scene_json["window"])
    
    # 将 JSON 中的 bbox 数据适配到 ctx.add_boxes
    # ctx.add_boxes 接受列表，其中元素包含 label, center, angle_z, scale, mesh_id, caption
    ctx.add_boxes(scene_json["bbox"])


    # 5. 准备输出目录
    timestamp = int(time.time())
    output_dir = os.path.join(output_root, f"{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 6. 执行渲染 (俯视图 + 前视图)
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
        f.write(updated_ssl)

    print(f"✅ 渲染完成！结果保存在: {os.path.abspath(output_dir)}")
    return output_dir, updated_ssl

if __name__ == "__main__":
    # 示例运行
    render_ssl(ssl_example)
