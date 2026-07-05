#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 SSL 或 JSON 场景数据渲染 - 支持 BPY 和 Pyrender 双后端。
每视角独立子进程渲染；直接运行本文件可查看 render_ssl 用法示例。
"""

import json
import os
import subprocess
import sys
from typing import Optional, Literal

try:
    from .core.util_data import parse_scene_input, format_standard_ssl
except (ImportError, ValueError):
    from core.util_data import parse_scene_input, format_standard_ssl  # type: ignore


# ---------------------------------------------------------------------------
# 子进程 worker（每视角独立进程）
# ---------------------------------------------------------------------------

def _load_job(job_path: str) -> dict:
    with open(job_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _instantiate_ctx(backend: str, room_type: str, asset_dir: Optional[str]):
    """按 backend 创建 SceneCtx / BpySceneCtx。"""
    if backend == "pyrender":
        try:
            from .core.fast_scene import SceneCtx
        except (ImportError, ValueError):
            from core.fast_scene import SceneCtx  # type: ignore
        return SceneCtx(room_type, asset_dir)

    try:
        from .core.fast_scene_bpy import BpySceneCtx
    except (ImportError, ValueError):
        from core.fast_scene_bpy import BpySceneCtx  # type: ignore
    return BpySceneCtx(room_type, asset_dir)


def _prepare_render_ctx(
    ctx,
    scene_json: dict,
    *,
    texture_dir: Optional[str] = None,
    gen_texture: bool = False,
    image: Optional[str] = None,
    samples: Optional[int] = None,
):
    if samples is not None and hasattr(ctx, "set_blender_samples"):
        ctx.set_blender_samples(samples)

    ctx.add_walls(scene_json["wall"])
    if scene_json.get("door"):
        ctx.add_doors(scene_json["door"])
    if scene_json.get("window"):
        ctx.add_windows(scene_json["window"])
    ctx.add_boxes(scene_json["bbox"])

    if texture_dir and os.path.isdir(texture_dir):
        floor_tex = os.path.join(texture_dir, "floor_texture.png")
        wall_tex = os.path.join(texture_dir, "wall_texture.png")
        ceiling_tex = os.path.join(texture_dir, "ceiling_texture.png")
        if os.path.exists(wall_tex):
            ctx.set_wall_blender_texture_path(wall_tex)
        if os.path.exists(floor_tex):
            ctx.set_floor_blender_texture_path(floor_tex)
        if os.path.exists(ceiling_tex) and hasattr(ctx, "set_ceiling_blender_texture_path"):
            ctx.set_ceiling_blender_texture_path(ceiling_tex)
    elif gen_texture:
        try:
            from .core.util_data import generate_texture
        except (ImportError, ValueError):
            from core.util_data import generate_texture  # type: ignore
        generate_texture(ctx, image)

    ctx.normalize_scene_data()
    return ctx


def _create_render_ctx(job: dict):
    scene_json = job["scene_json"]
    ctx = _instantiate_ctx(
        job.get("backend", "bpy"),
        scene_json["room"]["room_type"],
        job.get("asset_dir"),
    )
    return _prepare_render_ctx(
        ctx,
        scene_json,
        texture_dir=job.get("texture_dir"),
        gen_texture=job.get("gen_texture", False),
        image=job.get("image"),
        samples=job.get("samples"),
    )


def _job_render_kwargs(job: dict) -> dict:
    return dict(
        export_glb=job.get("export_glb", False),
        export_point_cloud=job.get("export_point_cloud", False),
        visible_geometry=job.get("visible_geometry", False),
        render_semantic=job.get("semantic", False),
        render_depth=job.get("depth", False),
        view_transform=job.get("view_transform", False),
    )


def worker_render_view(job_path: str, view_name: str) -> None:
    job = _load_job(job_path)
    ctx = _create_render_ctx(job)
    output_dir = job["output_dir"]
    extra = _job_render_kwargs(job)

    if view_name == "topdown":
        ctx.topdown_view(
            output_dir,
            show_ceiling=False,
            rebuild=True,
            use_HDRI=False,
            **extra,
        )
        return

    look_at = job["look_at"]
    view_cameras = job["view_cameras"]
    if view_name not in view_cameras:
        raise ValueError(f"未知视角: {view_name}")

    ctx.render_view(
        output_path=output_dir,
        camera_position=view_cameras[view_name],
        look_at_target=look_at,
        rebuild=True,
        use_HDRI=False,
        **extra,
    )


def worker_render_post(job_path: str) -> None:
    job = _load_job(job_path)
    ctx = _create_render_ctx(job)
    output_dir = job["output_dir"]
    scene_json = job["scene_json"]

    if job.get("export_glb"):
        glb_path = os.path.join(output_dir, "scene.glb")
        try:
            ctx.export_glb(glb_path, rebuild=True, show_ceiling=True)
        except Exception as exc:
            print(f"⚠️ GLB 导出失败: {exc}")

    if job.get("export_point_cloud"):
        point_cloud_dir = os.path.join(output_dir, "pointcloud")
        try:
            ctx.export_point_cloud(point_cloud_dir, rebuild=True, show_ceiling=True)
        except Exception as exc:
            print(f"⚠️ 点云导出失败: {exc}")

    with open(os.path.join(output_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(scene_json, f, indent=2, ensure_ascii=False)


def _spawn_worker_view(job_path: str, view_name: str) -> None:
    subprocess.run([
        sys.executable, "-c",
        "from fast_scene.render_ssl import worker_render_view; "
        f"worker_render_view({job_path!r}, {view_name!r})",
    ], check=True)


def _spawn_worker_post(job_path: str) -> None:
    subprocess.run([
        sys.executable, "-c",
        "from fast_scene.render_ssl import worker_render_post; "
        f"worker_render_post({job_path!r})",
    ], check=True)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def render_ssl(
    input_text: str,
    backend: str = 'bpy',
    output_root: str = 'output_ssl',
    image: Optional[str] = None,
    retrieve_hole: bool = True,
    asset_mode: Literal["none", "retrieve", "generate"] = "none",
    outpaint_image_dir: Optional[str] = None,
    asset_dir: Optional[str] = "/data-nas/data/dataset/qunhe/Manycore-Future/generate",
    gen_3d_model: Literal["hunyuan-3d-rapid", "hunyuan-3d-pro"] = "hunyuan-3d-pro",
    gen_texture: bool = False,
    texture_dir: Optional[str] = None,
    correct_tilt: bool = True,
    correct_yaw: bool = True,
    views: Optional[list] = None,
    export_glb: bool = False,
    export_point_cloud: bool = False,
    visible_geometry: bool = False,
    semantic: bool = False,
    depth: bool = False,
    view_transform: bool = False,
    samples: Optional[int] = None,
):
    """渲染场景。``input_text`` 可为标准 SSL 文本，或 JSON 字符串（含 wall/door/window/bbox/room）。"""
    if outpaint_image_dir is None:
        outpaint_image_dir = output_root

    try:
        from .core.util_data import get_mesh
    except (ImportError, ValueError):
        from core.util_data import get_mesh  # type: ignore

    print(f"\n🚀 开始渲染 [后端: {backend}, 资产模式: {asset_mode}]")

    scene_json = parse_scene_input(input_text)

    scene_json = get_mesh(
        scene_json,
        image_path=image,
        retrieve_hole=retrieve_hole,
        asset_mode=asset_mode,
        outpaint_image_dir=outpaint_image_dir,
        asset_dir=asset_dir,
        gen_3d_model=gen_3d_model,
        correct_tilt=correct_tilt,
        correct_yaw=correct_yaw,
    )

    output_dir = output_root
    os.makedirs(output_dir, exist_ok=True)

    room_type = scene_json["room"]["room_type"]
    ctx = _instantiate_ctx(backend, room_type, asset_dir)
    _prepare_render_ctx(
        ctx,
        scene_json,
        texture_dir=texture_dir,
        gen_texture=gen_texture,
        image=image,
        samples=samples,
    )

    standard_ssl = format_standard_ssl(ctx.context)
    ssl_path = os.path.join(output_dir, "ssl.txt")
    with open(ssl_path, "w", encoding="utf-8") as f:
        f.write(standard_ssl)

    center = ctx.context["meta"]["center"]
    span = ctx.context["meta"]["span"]
    z_max = ctx.context["meta"]["z_max"]
    look_at = [center[0], center[1], z_max / 2]
    z_cam = z_max * 5 / 6
    # 透视视角相机位置（改这里即可，子进程从 job 读取）
    view_cameras = {
        "front": [center[0], center[1] - span[1] / 3, z_cam],
        "behind": [center[0], center[1] + span[1] / 3, z_cam],
        "left": [center[0] - span[0] / 3, center[1], z_cam],
        "right": [center[0] + span[0] / 3, center[1], z_cam],
        "leftfront": [center[0] - span[0] / 3, center[1] - span[1] / 3, z_cam],
        "rightfront": [center[0] + span[0] / 3, center[1] - span[1] / 3, z_cam],
        "leftbehind": [center[0] - span[0] / 3, center[1] + span[1] / 3, z_cam],
        "rightbehind": [center[0] + span[0] / 3, center[1] + span[1] / 3, z_cam],
        "left_seq": [
            [center[0] - span[0] / 3, center[1], z_cam],
            [center[0], center[1] - span[1] / 3, z_cam],
        ],
    }

    job_path = os.path.join(output_dir, ".render_job.json")
    job = {
        "backend": backend,
        "scene_json": scene_json,
        "output_dir": output_dir,
        "asset_dir": asset_dir,
        "texture_dir": texture_dir,
        "gen_texture": gen_texture,
        "image": image,
        "samples": samples,
        "export_glb": export_glb,
        "export_point_cloud": export_point_cloud,
        "visible_geometry": visible_geometry,
        "semantic": semantic,
        "depth": depth,
        "view_transform": view_transform,
        "look_at": look_at,
        "view_cameras": view_cameras,
    }
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2, ensure_ascii=False)

    views_to_run = []
    if views is None or "topdown" in views:
        views_to_run.append("topdown")
    if views is None:
        views_to_run.extend(view_cameras.keys())
    else:
        views_to_run.extend(v for v in views if v in view_cameras)

    print(f"🎨 正在生成视图 ({len(views_to_run)} 个子进程)...")
    for view_name in views_to_run:
        print(f"🔄 子进程渲染: {view_name}")
        _spawn_worker_view(job_path, view_name)
    print("🔄 子进程导出 GLB/点云/metadata...")
    _spawn_worker_post(job_path)
    print(f"✅ 渲染完成！结果保存在: {os.path.abspath(output_dir)}")
    return output_dir, standard_ssl


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
Bbox(id="15", room_id="D54g", label="dining table", center=[5.1, 2.77, 0.6], angle_z=0, scale=[3.05, 0.93, 1.21], asset_id="17116819")
'''


if __name__ == "__main__":
    render_ssl(ssl_example)
