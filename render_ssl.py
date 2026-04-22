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
    from .util_data import parse_ssl_to_json, get_mesh, update_ssl_with_asset_id, generate_texture
except (ImportError, ValueError):
    from util_data import parse_ssl_to_json, get_mesh, update_ssl_with_asset_id, generate_texture # type: ignore

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
Bbox(id="0", room_id="D54g", label="curtain", center=[0.02, 2.81, 1.51], angle_z=90, scale=[1.6, 0.05, 1.3], asset_id="3922456")
Bbox(id="3", room_id="D54g", label="kitchen cabinet", center=[0.3, 1.88, 0.66], angle_z=90, scale=[1.0, 0.6, 1.33], asset_id="12038494")
Bbox(id="1", room_id="D54g", label="kitchen cabinet", center=[0.3, 2.87, 0.43], angle_z=90, scale=[0.99, 0.6, 0.86], asset_id="7886809")
Bbox(id="2", room_id="D54g", label="kitchen cabinet", center=[0.3, 3.81, 0.43], angle_z=90, scale=[0.93, 0.6, 0.86], asset_id="7887034")
Bbox(id="4", room_id="D54g", label="kitchen cabinet", center=[2.47, 2.76, 0.51], angle_z=180, scale=[2.2, 0.93, 1.03], asset_id="2840139")
Bbox(id="5", room_id="D54g", label="kitchen cabinet", center=[5.47, 0.36, 1.56], angle_z=180, scale=[4.7, 0.67, 3.13], asset_id="40342063")
Bbox(id="6", room_id="D54g", label="ornaments", center=[0.09, 3.95, 1.39], angle_z=90, scale=[0.64, 0.18, 0.76], asset_id="34142540")
Bbox(id="7", room_id="D54g", label="ornaments", center=[0.23, 3.9, 1.02], angle_z=90, scale=[0.74, 0.46, 0.33], asset_id="34142599")
Bbox(id="8", room_id="D54g", label="ornaments", center=[0.31, 3.01, 0.94], angle_z=0, scale=[0.43, 0.55, 0.17], asset_id="13848022")
Bbox(id="9", room_id="D54g", label="ornaments", center=[1.87, 2.61, 0.93], angle_z=0, scale=[0.58, 0.39, 0.19], asset_id="35368125")
Bbox(id="10", room_id="D54g", label="decorative painting", center=[1.97, 1.4, 1.8], angle_z=180, scale=[1.23, 0.05, 0.54], asset_id="48510583")
Bbox(id="11", room_id="D54g", label="dining chair", center=[3.98, 3.59, 0.42], angle_z=0, scale=[0.6, 0.58, 0.84], asset_id="5829816")
Bbox(id="12", room_id="D54g", label="dining chair", center=[4.02, 1.97, 0.42], angle_z=180, scale=[0.6, 0.58, 0.84], asset_id="5829816")
Bbox(id="13", room_id="D54g", label="dining chair", center=[4.74, 3.59, 0.42], angle_z=0, scale=[0.6, 0.58, 0.84], asset_id="5829816")
Bbox(id="14", room_id="D54g", label="dining chair", center=[4.76, 1.97, 0.42], angle_z=180, scale=[0.6, 0.58, 0.84], asset_id="5829816")
Bbox(id="15", room_id="D54g", label="dining table", center=[5.1, 2.77, 0.6], angle_z=0, scale=[3.05, 0.93, 1.21], asset_id="17116819")
Bbox(id="16", room_id="D54g", label="dining chair", center=[5.47, 3.59, 0.42], angle_z=0, scale=[0.6, 0.58, 0.84], asset_id="5829816")
Bbox(id="17", room_id="D54g", label="dining chair", center=[5.51, 1.97, 0.42], angle_z=180, scale=[0.6, 0.58, 0.84], asset_id="5829816")
Bbox(id="18", room_id="D54g", label="dining chair", center=[6.16, 3.59, 0.42], angle_z=0, scale=[0.6, 0.58, 0.84], asset_id="5829816")
Bbox(id="19", room_id="D54g", label="dining chair", center=[6.23, 1.97, 0.42], angle_z=180, scale=[0.6, 0.58, 0.84], asset_id="5829816")
Bbox(id="20", room_id="D54g", label="dining chair", center=[6.91, 2.76, 0.42], angle_z=270, scale=[0.6, 0.58, 0.84], asset_id="5829816")
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
    gen_3d_model: Literal["hunyuan-3d-rapid", "hunyuan-3d-pro"] = "hunyuan-3d-pro",
    gen_texture: bool = False,
    texture_dir: Optional[str] = None,
    correct_tilt: bool = True,
    correct_yaw: bool = True,
    views: Optional[list] = None,
    export_glb: bool = True,
):
    """
    主渲染函数
    Args:
        ssl_text: SSL文本
        backend: 'bpy' 或 'pyrender'
        output_root: 输出根目录
        image: 可选的输入图片路径
        retrieve_hole: 是否检索门窗的 asset_id
        asset_mode: 资产处理模式 ("none", "retrieve", "generate")
        outpaint_image_dir: 扩图结果保存目录 (默认为 output_root)
        gen_asset_dir: 生成资产保存目录 (默认为 output_root)
        correct_tilt: 是否纠正模型倾斜
        correct_yaw: 是否纠正模型偏航角和输入图片对齐
        texture_dir: 外部纹理目录路径，包含 floor_texture.png, wall_texture.png, ceiling_texture.png
                     如果提供，优先使用外部纹理；否则当 gen_texture=True 时内部生成
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
        gen_3d_model=gen_3d_model,
        correct_tilt=correct_tilt,
        correct_yaw=correct_yaw
    )
    
    # 将更新后的 asset_id 填回 SSL
    updated_ssl = update_ssl_with_asset_id(ssl_text, scene_json)

    
      

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
    # ctx.add_boxes 接受列表，其中元素包含 label, center, angle_z, scale, asset_id, caption
    ctx.add_boxes(scene_json["bbox"])

    # 5. 纹理处理
    if texture_dir and os.path.isdir(texture_dir):
        # 使用外部预生成的纹理
        floor_tex = os.path.join(texture_dir, "floor_texture.png")
        wall_tex = os.path.join(texture_dir, "wall_texture.png")
        ceiling_tex = os.path.join(texture_dir, "ceiling_texture.png")
        if os.path.exists(wall_tex):
            ctx.set_wall_blender_texture_path(wall_tex)
            print(f"🎨 使用外部墙体纹理: {wall_tex}")
        if os.path.exists(floor_tex):
            ctx.set_floor_blender_texture_path(floor_tex)
            print(f"🎨 使用外部地板纹理: {floor_tex}")
        if os.path.exists(ceiling_tex) and hasattr(ctx, 'set_ceiling_blender_texture_path'):
            ctx.set_ceiling_blender_texture_path(ceiling_tex)
            print(f"🎨 使用外部天花板纹理: {ceiling_tex}")
    elif gen_texture:
        generate_texture(ctx, image)

    


    # 6. 准备输出目录
    timestamp = int(time.time())
    output_dir = os.path.join(output_root, f"{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 7. 执行渲染 (俯视图 + 前视图)
    print("🎨 正在生成视图...")
    
    # --- 风险评估逻辑：提前预防 OOM ---
    # 统计物体总数
    total_objects = len(scene_json.get("bbox", [])) + \
                    len(scene_json.get("door", [])) + \
                    len(scene_json.get("window", []))
    
    # 阈值设定：如果物体超过 20 个，认为高风险
    OBJ_RISK_THRESHOLD = 25
    
    if total_objects > OBJ_RISK_THRESHOLD:
        print(f"⚠️ 场景物体较多 ({total_objects} 个)，检测到 OOM 风险。")
        simplified_path = ctx.config.get("model_simplified_path", "/data-nas/data/dataset/qunhe/Manycore-Future/simplified")
        print(f"🚀 直接切换到简化模型路径进行渲染: {simplified_path}")
        ctx.set_model_path(simplified_path)
        
        # # 同时降低采样数以加快渲染并减小压力
        # if ctx.config.get("blender_samples", 32) > 16:
        #     print("📉 降低渲染采样数至 16 以减轻负载。")
        #     ctx.set_blender_samples(16)
    # -------------------------------

    # Render topdown if requested (or if views is None = all)
    if views is None or "topdown" in views:
        ctx.topdown_view(os.path.join(output_dir, "topdown.png"), show_ceiling=False, rebuild=True, use_HDRI=False)


    # Camera parameters
    center = ctx.context["meta"]["center"]
    span = ctx.context["meta"]["span"]
    z_max = ctx.context["meta"]["z_max"]
    look_at = [center[0], center[1], z_max / 2]

    # View → camera position mapping
    _view_cameras = {
        "front":        [center[0],                center[1] - span[1] / 3, z_max * 5 / 6],
        "behind":       [center[0],                center[1] + span[1] / 3, z_max * 5 / 6],
        "left":         [center[0] - span[0] / 3,  center[1],               z_max * 5 / 6],
        "right":        [center[0] + span[0] / 3,  center[1],               z_max * 5 / 6],
        "leftfront":    [center[0] - span[0] / 3,  center[1] - span[1] / 3, z_max * 5 / 6],
        "rightfront":   [center[0] + span[0] / 3,  center[1] - span[1] / 3, z_max * 5 / 6],
        "leftbehind":   [center[0] - span[0] / 3,  center[1] + span[1] / 3, z_max * 5 / 6],
        "rightbehind":  [center[0] + span[0] / 3,  center[1] + span[1] / 3, z_max * 5 / 6],
    }

    # Determine which views to render (None = all)
    render_views = list(_view_cameras.keys()) if views is None else [v for v in views if v in _view_cameras]

    first_perspective = True
    for view_name in render_views:
        ctx.render_view(
            output_path=os.path.join(output_dir, f"{view_name}.png"),
            camera_position=_view_cameras[view_name],
            look_at_target=look_at,
            rebuild=first_perspective,
            use_HDRI=False
        )
        first_perspective = False

    # 8. 导出 GLB
    if export_glb:
        glb_path = os.path.join(output_dir, "scene.glb")
        try:
            ctx.export_glb(glb_path)
        except Exception as e:
            print(f"⚠️ GLB 导出失败: {e}")

    # 9. 保存 JSON 和 SSL
    with open(os.path.join(output_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(scene_json, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "ssl.txt"), "w", encoding="utf-8") as f:
        f.write(updated_ssl)

    print(f"✅ 渲染完成！结果保存在: {os.path.abspath(output_dir)}")
    return output_dir, updated_ssl

if __name__ == "__main__":
    # 示例运行
    render_ssl(ssl_example)
