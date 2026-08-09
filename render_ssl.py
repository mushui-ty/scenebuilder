#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 SSL 或 JSON 场景数据渲染 - 支持 BPY 和 Pyrender 双后端。
每视角独立子进程渲染。

命令行用法:
    python scenebuilder/render_ssl.py --ssl path/to/ssl.txt --views topdown left_seq --output out_dir
    python scenebuilder/render_ssl.py --help
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from typing import Optional, Literal, Any, Dict, Union, List, Tuple

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_PKG_ROOT)
for _p in (_PKG_PARENT, _PKG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_util_data_module():
    util_path = os.path.join(_PKG_ROOT, "core", "util_data.py")
    spec = importlib.util.spec_from_file_location("scenebuilder_util_data", util_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 util_data: {util_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    from .core.util_data import (
        parse_scene_input,
        format_standard_ssl,
        prepare_pixel_aligned_topdown_context,
        apply_scene_json_xy_translation,
    )
except (ImportError, ValueError):
    _util_data = _load_util_data_module()
    parse_scene_input = _util_data.parse_scene_input
    format_standard_ssl = _util_data.format_standard_ssl
    prepare_pixel_aligned_topdown_context = _util_data.prepare_pixel_aligned_topdown_context
    apply_scene_json_xy_translation = _util_data.apply_scene_json_xy_translation


def resolve_output_dir(output_root: str, normalized_topdown: bool) -> str:
    """确定唯一输出根目录 Y：规范化时 ``{output}_normalized``，否则 ``{output}``。"""
    return normalized_output_dir(output_root) if normalized_topdown else output_root


def normalized_output_dir(output_root: str) -> str:
    return f"{output_root}_normalized"


ViewsSpec = Optional[Union[List[str], Literal["auto"]]]

TOPDOWN_NORMALIZED_SUBDIR = "topdown_normalized"


def _resolve_views_to_run(views: ViewsSpec, view_cameras: dict) -> List[str]:
    """将 views 参数解析为待渲染视角名列表。None / [] → 不渲染；auto 由上层单独处理。"""
    if not views or views == "auto":
        return []
    names: List[str] = []
    for name in views:
        if name == "topdown":
            if "topdown" not in names:
                names.append("topdown")
        elif name in view_cameras and name not in names:
            names.append(name)
        elif name not in view_cameras:
            print(f"⚠️  未知视角 {name!r}，已跳过")
    return names


def _run_auto_views_from_path(
    y_dir: str,
    floor_result: Optional[Dict[str, Any]],
    context: Dict[str, Any],
    *,
    resume: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """根据地板路径生成 auto 视角 spec，并写入 auto_views.json。"""
    if resume:
        try:
            from .core.render_resume import load_auto_views_manifest
        except (ImportError, ValueError):
            from core.render_resume import load_auto_views_manifest  # type: ignore
        specs, names = load_auto_views_manifest(y_dir)
        if specs:
            print(f"📂 从已有 auto_views.json 加载 {len(names)} 个 auto 视角")
            return specs, names

    try:
        from .core.auto_views import build_auto_views_from_path, write_auto_views_manifest
    except (ImportError, ValueError):
        from core.auto_views import build_auto_views_from_path, write_auto_views_manifest  # type: ignore

    if floor_result is None:
        print("⚠️  无 floor_path 结果，auto 视角不可用")
        return {}, []
    path_points = floor_result.get("path_points_ssl") or []
    if not path_points:
        print("⚠️  path_points_ssl 为空，auto 视角不可用")
        return {}, []

    specs, names = build_auto_views_from_path(path_points, context)
    manifest = write_auto_views_manifest(y_dir, specs)
    n_path = len(path_points)
    n_single = sum(1 for n in names if not n.endswith("_seq"))
    print(
        f"🎯 views=auto：路径 {n_path} 点 → "
        f"{n_single} 个单帧视角 + 1 个三帧序列；manifest → {manifest}"
    )
    return specs, names


def _render_auto_view_spec(
    ctx,
    output_dir: str,
    spec: Dict[str, Any],
    extra: dict,
    view_dir_name: str,
) -> None:
    """按 auto_view spec 调用 render_view；输出目录名为 view_dir_name（如 auto_path_0004_seq）。"""
    common = dict(
        rebuild=True,
        use_HDRI=False,
        width=int(spec.get("width", 1000)),
        height=int(spec.get("height", 1000)),
        manual_fov=float(spec["manual_fov"]),
        auto_fov=False,
        auto_transparent=True,
        view_dir_name=view_dir_name,
        **extra,
    )
    if spec["type"] == "single":
        ctx.render_view(
            output_path=output_dir,
            camera_position=spec["camera_position"],
            look_at_target=spec["look_at_target"],
            up_vector=spec.get("up_vector", [0.0, 0.0, 1.0]),
            **common,
        )
        return
    if spec["type"] == "sequence":
        ctx.render_view(
            output_path=output_dir,
            camera_position=spec["camera_positions"],
            look_at_target=spec["look_at_targets"],
            up_vector=spec.get("up_vectors", [0.0, 0.0, 1.0]),
            **common,
        )
        return
    raise ValueError(f"未知 auto_view type: {spec.get('type')!r}")


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
            from .core.scenebuilder import SceneCtx
        except (ImportError, ValueError):
            from core.scenebuilder import SceneCtx  # type: ignore
        return SceneCtx(room_type, asset_dir)

    try:
        from .core.scenebuilder_bpy import BpySceneCtx
    except (ImportError, ValueError):
        from core.scenebuilder_bpy import BpySceneCtx  # type: ignore
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
    if job.get("normalized_topdown"):
        print("   [worker] 使用 job 内已规范化的 scene_json（pixel-aligned SSL）")
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
        export_voxel=job.get("export_voxel", False),
        visible_geometry=job.get("visible_geometry", False),
        render_semantic=job.get("semantic", False),
        render_depth=job.get("depth", False),
        pano=job.get("pano", False),
        pano_resolution=job.get("pano_resolution", 4096),
    )


def worker_render_view(job_path: str, view_name: str) -> None:
    job = _load_job(job_path)
    ctx = _create_render_ctx(job)
    output_dir = job["output_dir"]
    extra = _job_render_kwargs(job)

    try:
        from .core.render_resume import (
            apply_partial_resume_to_kwargs,
            is_view_complete,
            list_primary_frames,
            missing_view_artifacts,
            record_view_complete,
            view_dir_for,
            view_dir_name_for,
        )
    except (ImportError, ValueError):
        from core.render_resume import (  # type: ignore
            apply_partial_resume_to_kwargs,
            is_view_complete,
            list_primary_frames,
            missing_view_artifacts,
            record_view_complete,
            view_dir_for,
            view_dir_name_for,
        )

    auto_specs = job.get("auto_view_specs") or {}
    auto_spec = auto_specs.get(view_name)
    if job.get("resume", True) and is_view_complete(
        output_dir, view_name, job, auto_spec=auto_spec
    ):
        print(f"⏭️  跳过已完成视角: {view_name}")
        return

    if job.get("resume", True):
        missing = missing_view_artifacts(
            output_dir, view_name, job, auto_spec=auto_spec
        )
        extra = apply_partial_resume_to_kwargs(job, missing)
        if missing:
            print(f"🔧 补跑缺失产物 [{view_name}]: {', '.join(sorted(missing))}")
    else:
        extra = _job_render_kwargs(job)

    if view_name in auto_specs:
        _render_auto_view_spec(ctx, output_dir, auto_specs[view_name], extra, view_dir_name=view_name)
        view_dir = view_dir_for(output_dir, view_name)
        record_view_complete(
            output_dir,
            view_name,
            view_dir=view_dir,
            main_pngs=list_primary_frames(view_dir),
        )
        return

    if view_name == "topdown":
        extra.pop("pano", None)
        extra.pop("pano_resolution", None)
        ctx.topdown_view(
            output_dir,
            show_ceiling=False,
            rebuild=True,
            use_HDRI=False,
            **extra,
        )
        view_dir = view_dir_for(output_dir, "topdown")
        record_view_complete(
            output_dir,
            "topdown",
            view_dir=view_dir,
            main_pngs=list_primary_frames(view_dir),
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
        view_dir_name=view_dir_name_for(view_name),
        **extra,
    )
    view_dir = view_dir_for(output_dir, view_name)
    record_view_complete(
        output_dir,
        view_name,
        view_dir=view_dir,
        main_pngs=list_primary_frames(view_dir),
    )


def worker_render_post(job_path: str) -> None:
    job = _load_job(job_path)
    output_dir = job["output_dir"]
    holo_geometry = bool(job.get("holo_geometry"))

    try:
        from .core.render_resume import is_post_export_complete, record_post_complete
    except (ImportError, ValueError):
        from core.render_resume import is_post_export_complete, record_post_complete  # type: ignore

    if not holo_geometry:
        return

    if job.get("resume", True) and is_post_export_complete(output_dir, job):
        print("⏭️  跳过全场景几何导出（已完成）")
        return

    ctx = _create_render_ctx(job)
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

    if job.get("export_voxel"):
        if hasattr(ctx, "export_voxel"):
            try:
                ctx.export_voxel(output_dir, rebuild=True, show_ceiling=True)
            except Exception as exc:
                print(f"⚠️ 体素导出失败: {exc}")
        else:
            print("⚠️ 当前后端不支持全场景体素导出")

    with open(os.path.join(output_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(scene_json, f, indent=2, ensure_ascii=False)

    record_post_complete(output_dir)


def _spawn_worker_view(job_path: str, view_name: str) -> None:
    subprocess.run([
        sys.executable, "-c",
        "from scenebuilder.render_ssl import worker_render_view; "
        f"worker_render_view({job_path!r}, {view_name!r})",
    ], check=True)


def _spawn_worker_post(job_path: str) -> None:
    subprocess.run([
        sys.executable, "-c",
        "from scenebuilder.render_ssl import worker_render_post; "
        f"worker_render_post({job_path!r})",
    ], check=True)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def _run_normalized_topdown_phase(
    ctx,
    output_dir: str,
    *,
    align: Dict[str, Any],
    floor_path: bool = True,
    width: int = 1000,
    height: int = 1000,
    show_ceiling: bool = False,
) -> Optional[Dict[str, Any]]:
    """在 Y/topdown_normalized/ 渲染 1000² 像素对齐俯视图，并可选采样地板路径。"""
    if not hasattr(ctx, "normalized_topdown_view"):
        raise RuntimeError("当前后端不支持 normalized_topdown_view")

    topdown_norm_dir = os.path.join(output_dir, TOPDOWN_NORMALIZED_SUBDIR)
    os.makedirs(topdown_norm_dir, exist_ok=True)
    print(f"🗺️  像素对齐俯视图 + 地板路径 → {topdown_norm_dir}")

    ctx.normalized_topdown_view(
        topdown_norm_dir,
        width=width,
        height=height,
        show_ceiling=show_ceiling,
        use_HDRI=False,
        render_depth=floor_path,
        render_semantic=floor_path,
        align=align,
        write_ssl=False,
    )

    if not floor_path:
        return None

    try:
        from .core.config_utils import load_config
        from .core.nav_mask_path import run_nav_mask_floor_path
    except (ImportError, ValueError):
        from core.config_utils import load_config  # type: ignore
        from core.nav_mask_path import run_nav_mask_floor_path  # type: ignore

    config = load_config()
    floor_result = run_nav_mask_floor_path(topdown_norm_dir, config)
    return floor_result


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
    export_glb: bool = False,
    export_point_cloud: bool = False,
) -> Union[str, Dict[str, Any]]:
    """像素对齐模式：等价于 ``render_ssl(..., normalized_topdown=True, views=None)``。

    唯一输出目录 Y = ``{output_dir}_normalized``；路径规划产物在 ``Y/topdown_normalized/``。
    """
    y_dir, _, floor_result = render_ssl(
        input_text,
        backend=backend,
        output_root=output_dir,
        image=image,
        retrieve_hole=retrieve_hole,
        asset_mode=asset_mode,
        outpaint_image_dir=outpaint_image_dir,
        asset_dir=asset_dir,
        gen_3d_model=gen_3d_model,
        gen_texture=gen_texture,
        texture_dir=texture_dir,
        correct_tilt=correct_tilt,
        correct_yaw=correct_yaw,
        views=None,
        export_glb=export_glb,
        export_point_cloud=export_point_cloud,
        samples=samples,
        normalized_topdown=True,
        floor_path=floor_path,
        normalized_topdown_width=width,
        normalized_topdown_height=height,
        normalized_topdown_show_ceiling=show_ceiling,
    )
    if not floor_path:
        return y_dir
    if floor_result is None:
        return y_dir
    return {
        "output_dir": y_dir,
        "topdown_normalized_dir": os.path.join(y_dir, TOPDOWN_NORMALIZED_SUBDIR),
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
    views: ViewsSpec = None,
    export_glb: bool = False,
    export_point_cloud: bool = False,
    export_voxel: bool = False,
    visible_geometry: bool = False,
    holo_geometry: bool = False,
    semantic: bool = False,
    depth: bool = False,
    pano: bool = False,
    pano_resolution: int = 4096,
    samples: Optional[int] = None,
    normalized_topdown: bool = False,
    floor_path: bool = True,
    normalized_topdown_width: int = 1000,
    normalized_topdown_height: int = 1000,
    normalized_topdown_show_ceiling: bool = False,
    resume: bool = True,
):
    """渲染场景。``input_text`` 可为标准 SSL 文本，或 JSON 字符串。

    ``visible_geometry=True`` 时会强制 ``semantic=True``（可见比例依赖 semantic mask）。

    ``normalized_topdown=True`` 时：
    - 唯一输出根目录 Y = ``{output_root}_normalized``（否则 Y = ``{output_root}``）
    - 先对 SSL 做 pixel-aligned 规范化，后续所有渲染均基于该坐标系
    - 自动在 ``Y/topdown_normalized/`` 渲染像素对齐俯视图 + 地板路径
    - 再在 Y 下按 ``views`` 渲染 ``topdown/``、序列视角等（``views=None`` 时不渲染任何视角）
    - ``views="auto"``：强制规范化 + 地板路径，随后渲染常规 topdown + 按路径自动生成视角

    各视角在世界（或规范化）SSL 下渲染；可见几何主文件为世界 SSL，并写 ``*_opencv`` 副本。
    """
    if visible_geometry:
        semantic = True
    if views == "auto":
        normalized_topdown = True
        floor_path = True
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

    y_dir = resolve_output_dir(output_root, normalized_topdown)
    os.makedirs(y_dir, exist_ok=True)

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

    pixel_align: Optional[Dict[str, Any]] = None
    if normalized_topdown:
        print(f"📐 Step 1: pixel-aligned SSL 规范化 → 输出根目录 Y = {y_dir}")
        pixel_align = prepare_pixel_aligned_topdown_context(ctx.context)
        apply_scene_json_xy_translation(
            scene_json,
            pixel_align["dx_ssl"],
            pixel_align["dy_ssl"],
        )

    standard_ssl = format_standard_ssl(ctx.context)
    ssl_path = os.path.join(y_dir, "ssl.txt")
    with open(ssl_path, "w", encoding="utf-8") as f:
        f.write(standard_ssl)
    with open(os.path.join(y_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(scene_json, f, indent=2, ensure_ascii=False)

    try:
        from .core.render_resume import (
            filter_views_to_run,
            is_normalized_topdown_complete,
            load_floor_result_from_disk,
        )
    except (ImportError, ValueError):
        from core.render_resume import (  # type: ignore
            filter_views_to_run,
            is_normalized_topdown_complete,
            load_floor_result_from_disk,
        )

    floor_result: Optional[Dict[str, Any]] = None
    if normalized_topdown and pixel_align is not None:
        skip_norm = (
            resume
            and is_normalized_topdown_complete(
                y_dir,
                floor_path=floor_path,
                semantic=semantic,
                depth=depth,
            )
        )
        if skip_norm:
            print(f"⏭️  跳过 pixel-aligned topdown_normalized（已完成）→ {y_dir}/{TOPDOWN_NORMALIZED_SUBDIR}/")
            floor_result = load_floor_result_from_disk(y_dir) if floor_path else None
        else:
            print(f"📐 Step 2: topdown_normalized + 地板路径 → {y_dir}/{TOPDOWN_NORMALIZED_SUBDIR}/")
            floor_result = _run_normalized_topdown_phase(
                ctx,
                y_dir,
                align=pixel_align,
                floor_path=floor_path,
                width=normalized_topdown_width,
                height=normalized_topdown_height,
                show_ceiling=normalized_topdown_show_ceiling,
            )

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

    job_path = os.path.join(y_dir, ".render_job.json")
    auto_view_specs: Dict[str, Dict[str, Any]] = {}
    auto_view_names: List[str] = []
    if views == "auto":
        if backend != "bpy":
            print("⚠️  views=auto 目前仅支持 backend=bpy，已跳过 auto 视角")
        else:
            auto_view_specs, auto_view_names = _run_auto_views_from_path(
                y_dir, floor_result, ctx.context, resume=resume
            )

    job = {
        "backend": backend,
        "scene_json": scene_json,
        "output_dir": y_dir,
        "normalized_topdown": normalized_topdown,
        "asset_dir": asset_dir,
        "texture_dir": texture_dir,
        "gen_texture": gen_texture,
        "image": image,
        "samples": samples,
        "export_glb": export_glb,
        "export_point_cloud": export_point_cloud,
        "export_voxel": export_voxel,
        "visible_geometry": visible_geometry,
        "holo_geometry": holo_geometry,
        "semantic": semantic,
        "depth": depth,
        "pano": pano,
        "pano_resolution": int(pano_resolution),
        "look_at": look_at,
        "view_cameras": view_cameras,
        "auto_view_specs": auto_view_specs,
        "resume": resume,
    }
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2, ensure_ascii=False)

    views_to_run = _resolve_views_to_run(views, view_cameras)
    if views == "auto":
        if backend == "bpy":
            # 常规 topdown（Y/topdown/，1024²）+ 路径驱动 auto 视角
            views_to_run = ["topdown"] + list(auto_view_names)
        else:
            views_to_run = []

    if views_to_run:
        views_to_run, skipped_views = filter_views_to_run(
            y_dir, views_to_run, job, resume=resume
        )
        if skipped_views:
            print(f"⏭️  已跳过 {len(skipped_views)} 个已完成视角: {', '.join(skipped_views)}")
    if views_to_run:
        step = "Step 3: " if normalized_topdown else ""
        print(f"🎨 {step}多视角渲染 → {y_dir} ({len(views_to_run)} 个子进程)")
        for view_name in views_to_run:
            print(f"🔄 子进程渲染: {view_name}")
            _spawn_worker_view(job_path, view_name)
    elif views == "auto":
        print("⏭️  auto 视角未生成（backend 不支持 bpy）")
    elif normalized_topdown:
        print("⏭️  未指定 views，跳过多视角渲染")

    need_holo_post = holo_geometry and (export_glb or export_point_cloud or export_voxel)
    if need_holo_post:
        print(f"🔄 子进程导出全场景几何 → {y_dir}")
        _spawn_worker_post(job_path)

    print(f"✅ 渲染完成！输出目录: {os.path.abspath(y_dir)}")
    return y_dir, standard_ssl, floor_result


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

  python /data-nas/data/experiments/mushui/projects/utils/fast-scene/scenebuilder/render_ssl.py \\
    --ssl /data-nas/data/experiments/mushui/projects/SpatialFactory/benchmark/data/Balcony/310449449_4/ssl.txt \\
    --views topdown left_seq \\
    --output /data-nas/data/experiments/mushui/projects/SpatialFactory/benchmark/data/Balcony/310449449_4/out9 \\
    --glb --assets /data-nas/data/dataset/qunhe/Manycore-Future/simplified \\
    --ply --visible_geometry --semantic --depth --pano
""",
    )
    parser.add_argument("--ssl", required=True, help="SSL 或 JSON 场景文件路径")
    parser.add_argument("--output", default=None, help="输出目录（默认: ssl 同目录下 render_output）")
    parser.add_argument(
        "--views", nargs="*", default=None, metavar="VIEW",
        help=f"视角列表（{', '.join(ALL_VIEWS)}）；不指定则不渲染视角；"
             f"指定 auto 则强制 normalized_topdown + 地板路径并按路径自动生成视角",
    )
    parser.add_argument("--backend", choices=["bpy", "pyrender"], default="bpy", help="渲染后端")
    parser.add_argument("--texture", default=None, help="贴图目录")
    parser.add_argument("--assets", default=None, help="3D 资产目录（未指定时尝试自动检测）")
    parser.add_argument("--glb", action="store_true", help="导出 GLB")
    parser.add_argument("--ply", action="store_true", help="导出彩色点云 PLY")
    parser.add_argument("--visible_geometry", action="store_true",
                        help="各视角视锥内 GLB/PLY/体素（需配合 --glb、--ply 或 --voxel）")
    parser.add_argument("--holo_geometry", action="store_true",
                        help="根目录全场景 GLB/点云/体素（需配合 --glb、--ply 或 --voxel）")
    parser.add_argument("--voxel", action="store_true",
                        help="256³ 彩色占用体素（各视角需 --visible_geometry，根目录需 --holo_geometry）")
    parser.add_argument("--semantic", action="store_true", help="导出语义分割图")
    parser.add_argument("--depth", action="store_true", help="导出深度图与法线图")
    parser.add_argument("--pano", action="store_true",
                        help="为 render_view 普通视角额外导出兄弟目录 {毫秒时间戳}_pano（topdown 不生成）")
    parser.add_argument("--pano_resolution", type=int, default=4096,
                        help="全景图横向分辨率，仅 --pano 时生效；高度为横向分辨率的一半（默认: 4096，即 4096x2048）")
    parser.add_argument("--normalized_topdown", action="store_true",
                        help="启用 pixel-aligned SSL：唯一输出 Y={output}_normalized，"
                             "先 Y/topdown_normalized 路径规划，再渲染各 views")
    parser.add_argument("--no_floor_path", action="store_true",
                        help="配合 --normalized_topdown：跳过 topdown_normalized 下的地板路径采样")
    parser.add_argument("--samples", type=int, default=None, help="Blender 采样数")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用断点续跑，强制重新渲染所有视角与导出",
    )

    args = parser.parse_args()
    input_text, output_dir, asset_dir, texture_dir = _prepare_ssl_and_dirs(args)

    views: ViewsSpec
    normalized_topdown = args.normalized_topdown
    floor_path = not args.no_floor_path

    if args.views is None:
        views = None
    elif len(args.views) == 1 and args.views[0] == "auto":
        views = "auto"
        normalized_topdown = True
        floor_path = True
        print("ℹ️  views=auto → normalized_topdown=True, floor_path=True")
    else:
        if "auto" in args.views:
            parser.error("--views auto 不能与其它视角同时使用")
        views = list(args.views)

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
        export_voxel=args.voxel,
        visible_geometry=args.visible_geometry,
        holo_geometry=args.holo_geometry,
        semantic=args.semantic or args.visible_geometry,
        depth=args.depth,
        pano=args.pano,
        pano_resolution=args.pano_resolution,
        normalized_topdown=normalized_topdown,
        floor_path=floor_path,
        resume=not args.no_resume,
    )
    if args.samples is not None:
        render_kwargs["samples"] = args.samples

    y_dir, _, floor_result = render_ssl(input_text, **render_kwargs)
    print(f"Output: {y_dir}")
    if args.normalized_topdown and floor_result and floor_result.get("path_points_ssl"):
        print(f"Floor path (ssl): {len(floor_result['path_points_ssl'])} points")
        print(f"Topdown normalized: {os.path.join(y_dir, TOPDOWN_NORMALIZED_SUBDIR)}")


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

'''
python /data-nas/data/experiments/mushui/projects/utils/fast-scene/scenebuilder/render_ssl.py --ssl /data-nas/data/experiments/mushui/projects/SpatialFactory/benchmark/data/Balcony/310449449_4/ssl.txt --views auto --output /data-nas/data/experiments/mushui/projects/SpatialFactory/benchmark/data/Balcony/310449449_4/out14  --glb --assets /data-nas/data/dataset/qunhe/Manycore-Future/simplified --ply --visible_geometry --semantic --depth --pano --voxel --holo_geometry
'''