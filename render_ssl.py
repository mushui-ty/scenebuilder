#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 SSL 或 JSON 场景数据渲染 - 支持 BPY 和 Pyrender 双后端。
每视角独立子进程渲染。

命令行用法:
    python fast_scene/render_ssl.py --ssl path/to/ssl.txt --views topdown left_seq --output out_dir
    python fast_scene/render_ssl.py --help
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from typing import Optional, Literal, Any, Dict, Union

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_PKG_ROOT)
for _p in (_PKG_PARENT, _PKG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_util_data_module():
    util_path = os.path.join(_PKG_ROOT, "core", "util_data.py")
    spec = importlib.util.spec_from_file_location("fast_scene_util_data", util_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 util_data: {util_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    from .core.util_data import parse_scene_input, format_standard_ssl
except (ImportError, ValueError):
    _util_data = _load_util_data_module()
    parse_scene_input = _util_data.parse_scene_input
    format_standard_ssl = _util_data.format_standard_ssl


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

def render_normalized_topdown(
    input_text: str,
    output_dir: str,
    backend: str = "bpy",
    asset_dir: Optional[str] = None,
    texture_dir: Optional[str] = None,
    gen_texture: bool = False,
    image: Optional[str] = None,
    samples: Optional[int] = None,
    width: int = 1000,
    height: int = 1000,
    show_ceiling: bool = False,
    retrieve_hole: bool = True,
    asset_mode: Literal["none", "retrieve", "generate"] = "none",
    outpaint_image_dir: Optional[str] = None,
    gen_3d_model: Literal["hunyuan-3d-rapid", "hunyuan-3d-pro"] = "hunyuan-3d-pro",
    correct_tilt: bool = True,
    correct_yaw: bool = True,
    floor_path: bool = True,
) -> Union[str, Dict[str, Any]]:
    """像素对齐俯视图：平移 SSL 后渲染 1000×1000 俯视图，输出 ssl/topdown/camera_para。

    floor_path=True（默认）时额外渲染深度图+语义图，并基于
    (深度mask - 墙门窗mask) ∪ 地板mask 采样地板闭环路径。
    """
    if outpaint_image_dir is None:
        outpaint_image_dir = output_dir

    try:
        from .core.util_data import get_mesh
    except (ImportError, ValueError):
        from core.util_data import get_mesh  # type: ignore

    print(f"\n🚀 像素对齐俯视图渲染 [后端: {backend}]")
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

    if not hasattr(ctx, "normalized_topdown_view"):
        raise RuntimeError(f"后端 {backend!r} 不支持 normalized_topdown_view")

    ctx.normalized_topdown_view(
        output_dir,
        width=width,
        height=height,
        show_ceiling=show_ceiling,
        use_HDRI=False,
        render_depth=floor_path,
        render_semantic=floor_path,
    )
    print(f"✅ 完成: {output_dir}")

    if not floor_path:
        return output_dir

    try:
        from .core.config_utils import load_config
        from .core.nav_mask_path import run_nav_mask_floor_path
    except (ImportError, ValueError):
        from core.config_utils import load_config  # type: ignore
        from core.nav_mask_path import run_nav_mask_floor_path  # type: ignore

    config = load_config()
    floor_result = run_nav_mask_floor_path(output_dir, config)
    return {
        "output_dir": output_dir,
        "path_points_px": floor_result["path_points_px"],
        "path_points_ssl": floor_result["path_points_ssl"],
        "floor_path": floor_result,
    }


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


ALL_VIEWS = [
    "topdown", "front", "behind", "left", "right",
    "leftfront", "rightfront", "leftbehind", "rightbehind",
    "left_seq",
]


def _prepare_ssl_and_dirs(args) -> tuple:
    """读取 SSL/JSON 输入，解析输出目录与资产/贴图路径。"""
    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.ssl), "render_output")

    asset_dir = args.assets
    if asset_dir is None:
        ssl_dir = os.path.dirname(args.ssl)
        for candidate in ("assets", "Assets", "nano_gen_asset"):
            path = os.path.join(ssl_dir, candidate)
            if os.path.isdir(path):
                asset_dir = path
                break

    texture_dir = args.texture
    if texture_dir is None:
        candidate = os.path.join(os.path.dirname(args.ssl), "texture")
        if os.path.isdir(candidate):
            texture_dir = candidate

    with open(args.ssl, "r", encoding="utf-8") as f:
        input_text = f.read()

    if "asset_id=" not in input_text and "mesh_id=" in input_text:
        timestamp = os.path.basename(os.path.dirname(args.ssl))

        def _add_asset_id(match):
            full = match.group(0)
            mesh_id_match = re.search(r'mesh_id="([^"]*)"', full)
            if mesh_id_match:
                mesh_id_str = mesh_id_match.group(1)
                try:
                    mesh_id_padded = f"{int(mesh_id_str):03d}"
                except ValueError:
                    mesh_id_padded = mesh_id_str
                asset_id = f"{mesh_id_padded}_{timestamp}"
                return full[:-1] + f', asset_id="{asset_id}")'
            return full

        input_text = re.sub(r"(?:Bbox|Door|Window)\([^)]+\)", _add_asset_id, input_text)
        print(f"Inferred asset_id from mesh_id + timestamp ({timestamp})")

    return input_text, output_dir, asset_dir, texture_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 SSL/JSON 场景文件渲染多视角图像并导出 GLB/点云等",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（等价于 SpatialFactory/scripts/render_scene.py 的常用命令）:

  python /data-nas/data/experiments/mushui/projects/utils/fast-scene/fast_scene/render_ssl.py \\
    --ssl /data-nas/data/experiments/mushui/projects/SpatialFactory/benchmark/data/Balcony/310449449_4/ssl.txt \\
    --views topdown left_seq \\
    --output /data-nas/data/experiments/mushui/projects/SpatialFactory/benchmark/data/Balcony/310449449_4/out8 \\
    --glb --assets /data-nas/data/dataset/qunhe/Manycore-Future/simplified \\
    --ply --visible_geometry --semantic --depth
""",
    )
    parser.add_argument("--ssl", required=True, help="SSL 或 JSON 场景文件路径")
    parser.add_argument("--output", default=None, help="输出目录（默认: ssl 同目录下 render_output）")
    parser.add_argument(
        "--views", nargs="+", default=["all"],
        help=f"要渲染的视角，可选: {', '.join(ALL_VIEWS)}, all",
    )
    parser.add_argument("--backend", choices=["bpy", "pyrender"], default="bpy", help="渲染后端")
    parser.add_argument("--texture", default=None, help="贴图目录")
    parser.add_argument("--assets", default=None, help="3D 资产目录（未指定时尝试自动检测）")
    parser.add_argument("--glb", action="store_true", help="导出 GLB")
    parser.add_argument("--ply", action="store_true", help="导出彩色点云 PLY")
    parser.add_argument("--visible_geometry", action="store_true",
                        help="按视角视锥裁剪后导出可见 GLB/PLY（需配合 --glb 或 --ply）")
    parser.add_argument("--semantic", action="store_true", help="导出语义分割图")
    parser.add_argument("--depth", action="store_true", help="导出深度图与法线图")
    parser.add_argument("--view_transform", action="store_true",
                        help="渲染前将场景变换到各视角 SSL 坐标系")
    parser.add_argument("--normalized_topdown", action="store_true",
                        help="像素对齐俯视图（ssl.txt + topdown.png + camera_para.json + 地板路径）")
    parser.add_argument("--no_floor_path", action="store_true",
                        help="配合 --normalized_topdown：跳过地板导航路径采样")
    parser.add_argument("--samples", type=int, default=None, help="Blender 采样数")

    args = parser.parse_args()
    input_text, output_dir, asset_dir, texture_dir = _prepare_ssl_and_dirs(args)

    if args.normalized_topdown:
        render_kwargs = dict(
            input_text=input_text,
            output_dir=output_dir,
            backend=args.backend,
            asset_dir=asset_dir,
            texture_dir=texture_dir,
            retrieve_hole=True,
            asset_mode="none",
            floor_path=not args.no_floor_path,
        )
        if args.samples is not None:
            render_kwargs["samples"] = args.samples
        result = render_normalized_topdown(**render_kwargs)
        if isinstance(result, dict):
            print(f"Normalized topdown done: {result['output_dir']}")
            if "path_points_ssl" in result:
                print(f"Floor path (ssl): {len(result['path_points_ssl'])} points")
        else:
            print(f"Normalized topdown done: {result}")
        return

    views = None if "all" in args.views else args.views
    render_kwargs = dict(
        backend=args.backend,
        output_root=output_dir,
        retrieve_hole=True,
        asset_mode="none",
        asset_dir=asset_dir,
        texture_dir=texture_dir,
        views=views,
        export_glb=args.glb,
        export_point_cloud=args.ply,
        visible_geometry=args.visible_geometry,
        semantic=args.semantic,
        depth=args.depth,
        view_transform=args.view_transform,
    )
    if args.samples is not None:
        render_kwargs["samples"] = args.samples

    output_path, _ = render_ssl(input_text, **render_kwargs)
    print(f"Render done: {output_path}")


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
    main()
