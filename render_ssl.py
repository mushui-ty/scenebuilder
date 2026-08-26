#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render from SSL or JSON scene data — supports BPY and Pyrender backends.
Each view is rendered in an independent subprocess.

CLI usage:
    python scenebuilder/render_ssl.py --ssl path/to/ssl.txt --views topdown left_seq --output out_dir
    python scenebuilder/render_ssl.py --help
"""

import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
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
        raise ImportError(f"Failed to load util_data: {util_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    from .core.util_data import (
        parse_scene_input,
        format_standard_ssl,
        context_for_ssl_export,
        prepare_pixel_aligned_topdown_context,
        apply_scene_json_xy_translation,
    )
except (ImportError, ValueError):
    _util_data = _load_util_data_module()
    parse_scene_input = _util_data.parse_scene_input
    format_standard_ssl = _util_data.format_standard_ssl
    context_for_ssl_export = _util_data.context_for_ssl_export
    prepare_pixel_aligned_topdown_context = _util_data.prepare_pixel_aligned_topdown_context
    apply_scene_json_xy_translation = _util_data.apply_scene_json_xy_translation


def resolve_output_dir(output_root: str, normalized_topdown: bool) -> str:
    """Resolve the unique output root Y: ``{output}_normalized`` when normalized, else ``{output}``."""
    return normalized_output_dir(output_root) if normalized_topdown else output_root


def normalized_output_dir(output_root: str) -> str:
    return f"{output_root}_normalized"


ViewsSpec = Optional[Union[List[str], Literal["auto"]]]

TOPDOWN_NORMALIZED_SUBDIR = "topdown_normalized"


def _resolve_views_to_run(views: ViewsSpec, view_cameras: dict) -> List[str]:
    """Parse views into a list of view names to render. None / [] → skip; auto handled upstream."""
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
            print(f"⚠️  Unknown view {name!r}, skipped")
    return names


def _path_points_ssl_from_floor(
    y_dir: str,
    floor_result: Optional[Dict[str, Any]],
) -> List[List[float]]:
    path_points = (floor_result or {}).get("path_points_ssl") or []
    if path_points:
        return path_points
    try:
        from .core.render_resume import load_floor_result_from_disk
    except (ImportError, ValueError):
        from core.render_resume import load_floor_result_from_disk  # type: ignore
    disk = load_floor_result_from_disk(y_dir)
    return (disk or {}).get("path_points_ssl") or []


def _video_view_names(names: List[str]) -> List[str]:
    return [n for n in names if n.startswith("video_")]


def _video_manifest_ready(names: List[str]) -> bool:
    try:
        from .core.video_views import video_manifest_complete
    except (ImportError, ValueError):
        from core.video_views import video_manifest_complete  # type: ignore
    return video_manifest_complete(names)


def _run_auto_views_from_path(
    y_dir: str,
    floor_result: Optional[Dict[str, Any]],
    context: Dict[str, Any],
    *,
    resume: bool = True,
    width: int = 1000,
    height: int = 1000,
    sparse_auto: bool = True,
    video: bool = False,
    video_frame_spacing: float = 0.2,
    video_frames: Optional[int] = None,
    pano_num: int = 3,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Build path-driven view specs and write auto_views.json.

    ``sparse_auto=True`` (``--views auto``): sparse auto_path_* + optional video.
    ``sparse_auto=False`` (``--video`` only): video trajectories only.
    """
    try:
        from .core.auto_views import build_auto_views_from_path, write_auto_views_manifest
    except (ImportError, ValueError):
        from core.auto_views import build_auto_views_from_path, write_auto_views_manifest  # type: ignore
    try:
        from .core.render_resume import load_auto_views_manifest
    except (ImportError, ValueError):
        from core.render_resume import load_auto_views_manifest  # type: ignore

    specs: Dict[str, Dict[str, Any]] = {}
    names: List[str] = []
    loaded = False
    if resume:
        specs, names = load_auto_views_manifest(y_dir)
        if specs:
            loaded = True
            print(f"📂 Loaded {len(names)} auto views from existing auto_views.json")

    path_points = _path_points_ssl_from_floor(y_dir, floor_result)
    if path_points:
        try:
            from .core.floor_path_utils import ensure_closed_path_ccw_ssl
        except (ImportError, ValueError):
            from core.floor_path_utils import ensure_closed_path_ccw_ssl  # type: ignore
        path_points = ensure_closed_path_ccw_ssl(path_points)

    has_sparse = any(n.startswith("auto_path_") for n in names)
    if sparse_auto and not has_sparse:
        if not path_points:
            print("⚠️  No floor_path result; auto views unavailable")
            return {}, []
        specs, names = build_auto_views_from_path(
            path_points, context, width=width, height=height, pano_num=pano_num
        )
        manifest = write_auto_views_manifest(y_dir, specs)
        n_path = len(path_points)
        n_single = sum(1 for n in names if not n.endswith("_seq") and not n.startswith("video_"))
        print(
            f"🎯 views=auto: {n_path} path points → "
            f"{n_single} single-frame views + 1 three-frame sequence; manifest → {manifest}"
        )
    elif sparse_auto and has_sparse and _refresh_auto_path_resolutions(specs, names):
        write_auto_views_manifest(y_dir, specs)
    elif not sparse_auto and not video:
        return {}, []

    if sparse_auto and specs:
        try:
            from .core.auto_views import apply_auto_path_pano_flags, list_auto_path_single_view_names
        except (ImportError, ValueError):
            from core.auto_views import apply_auto_path_pano_flags, list_auto_path_single_view_names  # type: ignore
        pano_views = apply_auto_path_pano_flags(specs, pano_num)
        n_single = len(list_auto_path_single_view_names(specs))
        if n_single:
            print(
                f"🌐 auto_path pano sampling: {len(pano_views)}/{n_single} views "
                f"(pano_num={pano_num}) → {', '.join(pano_views)}"
            )
            write_auto_views_manifest(y_dir, specs)

    if video and path_points:
        try:
            from .core.video_views import (
                build_video_views_from_path,
                strip_legacy_video_entries,
                video_specs_current,
            )
        except (ImportError, ValueError):
            from core.video_views import (  # type: ignore
                build_video_views_from_path,
                strip_legacy_video_entries,
                video_specs_current,
            )
        if not video_specs_current(specs, names):
            strip_legacy_video_entries(specs, names)
            _drop_video_view_specs(specs, names)
            video_specs, video_names = build_video_views_from_path(
                path_points,
                context,
                frame_spacing_m=video_frame_spacing,
                n_frames=video_frames,
                width=width,
                height=height,
            )
            for name in video_names:
                specs[name] = video_specs[name]
                if name not in names:
                    names.append(name)
            manifest = write_auto_views_manifest(y_dir, specs)
            n_vf = len((video_specs.get(video_names[0]) or {}).get("camera_positions") or [])
            path_len = 0.0
            try:
                from .core.video_views import closed_path_length_m
            except (ImportError, ValueError):
                from core.video_views import closed_path_length_m  # type: ignore
            try:
                path_len = closed_path_length_m(path_points)
            except ValueError:
                pass
            frame_note = (
                f"fixed {n_vf} frames"
                if video_frames is not None
                else f"spacing≈{video_frame_spacing}m"
            )
            print(
                f"🎬 --video: path≈{path_len:.1f}m → center look-at persp 896×896 "
                f"(+ pano 1024×512 with --pano, {frame_note}); manifest → {manifest}"
            )
        elif loaded:
            print(f"📂 Reusing {len(_video_view_names(names))} video trajectories from auto_views.json")
    elif video and not path_points:
        print("⚠️  --video: path_points_ssl is empty; video views skipped")

    if not sparse_auto:
        video_names = _video_view_names(names)
        specs = {n: specs[n] for n in video_names if n in specs}
        names = video_names

    return specs, names


def _is_camera_sequence(camera_position: Any) -> bool:
    """Return True if camera_position is a multi-frame sequence ([[x,y,z], ...])."""
    if not isinstance(camera_position, list) or not camera_position:
        return False
    first = camera_position[0]
    return isinstance(first, (list, tuple)) and len(first) >= 3


def _normalize_vec3(value: Any, name: str) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"{name} must be JSON in the form [x, y, z] or [[x,y,z], ...]")
    return [float(value[0]), float(value[1]), float(value[2])]


def _parse_camera_cli_arg(raw: Optional[str], arg_name: str) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON: {raw!r}") from exc


def _apply_up_vector_to_spec(
    spec: Dict[str, Any],
    up_vector: Any,
    n_frames: int,
) -> None:
    """Write optional up_vector into spec; omit when unset — render_view picks default up (falls back to [0,1,0] when collinear)."""
    if up_vector is None:
        return
    if _is_camera_sequence(up_vector):
        ups = [_normalize_vec3(p, "up_vector") for p in up_vector]
        if len(ups) != n_frames:
            raise ValueError(
                f"up_vector sequence length ({len(ups)}) does not match camera frame count ({n_frames})"
            )
        spec["up_vectors"] = ups
    else:
        up = _normalize_vec3(up_vector, "up_vector")
        if n_frames == 1:
            spec["up_vector"] = up
        else:
            spec["up_vectors"] = [up] * n_frames


def _build_custom_view_spec(
    camera_position: Any,
    look_at: Any = None,
    *,
    up_vector: Any = None,
    width: int = 1000,
    height: int = 1000,
    manual_fov: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build a single-frame or sequence view spec from camera_position / look_at. Returns (view_name, spec).

    up_vector is optional: ``render_view`` defaults to ``[0,0,1]``; when the view axis is collinear (e.g. top-down), uses ``[0,1,0]``.
    """
    if camera_position is None:
        raise ValueError("camera_position cannot be empty")

    if _is_camera_sequence(camera_position):
        cam_positions = [_normalize_vec3(p, "camera_position") for p in camera_position]
        if look_at is None:
            raise ValueError("look_at is required when camera_position is a sequence")
        if _is_camera_sequence(look_at):
            look_targets = [_normalize_vec3(p, "look_at") for p in look_at]
            if len(look_targets) != len(cam_positions):
                raise ValueError(
                    f"look_at sequence length ({len(look_targets)}) does not match camera_position ({len(cam_positions)})"
                )
        else:
            look_targets = [_normalize_vec3(look_at, "look_at")] * len(cam_positions)
        spec = {
            "type": "sequence",
            "camera_positions": cam_positions,
            "look_at_targets": look_targets,
            "width": int(width),
            "height": int(height),
        }
        if manual_fov is not None:
            spec["manual_fov"] = float(manual_fov)
        _apply_up_vector_to_spec(spec, up_vector, len(cam_positions))
        return "custom_seq", spec

    cam_pos = _normalize_vec3(camera_position, "camera_position")
    look_target = _normalize_vec3(look_at, "look_at") if look_at is not None else None
    spec = {
        "type": "single",
        "camera_position": cam_pos,
        "look_at_target": look_target,
        "width": int(width),
        "height": int(height),
    }
    if manual_fov is not None:
        spec["manual_fov"] = float(manual_fov)
    _apply_up_vector_to_spec(spec, up_vector, 1)
    return "custom", spec


def _apply_auto_path_pano_gate(
    view_name: str,
    spec: Dict[str, Any],
    extra: dict,
    job: dict,
) -> None:
    """Only selected auto_path singles render pano (+ pano geometry); video views unchanged."""
    if not view_name.startswith("auto_path_") or view_name.endswith("_seq"):
        return
    try:
        from .core.render_resume import _auto_path_needs_pano
    except (ImportError, ValueError):
        from core.render_resume import _auto_path_needs_pano  # type: ignore
    if not _auto_path_needs_pano(view_name, job, spec):
        extra["pano"] = False


def _apply_video_render_flags(spec: Dict[str, Any], target: Dict[str, Any]) -> None:
    """Video views: center pano follows --pano; legacy tangent = pano-only."""
    if not spec.get("video"):
        return
    target["isolated_semantic_masks"] = False
    if spec.get("video_trajectory") == "center":
        target.pop("pano_only", None)
        # Keep pano from job (--pano); do not force True
        target["pano"] = bool(target.get("pano", False))
    else:
        target["pano"] = True
        if spec.get("pano_only"):
            target["pano_only"] = True
        else:
            target.pop("pano_only", None)
    if spec.get("pano_resolution") is not None:
        target["pano_resolution"] = int(spec["pano_resolution"])


def _render_view_spec(
    ctx,
    output_dir: str,
    spec: Dict[str, Any],
    extra: dict,
    view_dir_name: str,
    camera_kwargs: Optional[dict] = None,
    *,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    defer_geometry_export: bool = False,
    geometry_export_only: bool = False,
) -> None:
    """Invoke render_view from a view spec (shared by auto / custom)."""
    camera_kwargs = camera_kwargs or {}
    common = dict(
        rebuild=True,
        use_HDRI=False,
        width=int(spec.get("width", camera_kwargs.get("width", 1000))),
        height=int(spec.get("height", camera_kwargs.get("height", 1000))),
        auto_transparent=True,
        view_dir_name=view_dir_name,
        **extra,
    )
    if frame_start is not None:
        common["frame_start"] = int(frame_start)
        common["frame_end"] = int(frame_end) if frame_end is not None else None
        common["defer_geometry_export"] = True
    if defer_geometry_export:
        common["defer_geometry_export"] = True
    if geometry_export_only:
        common["geometry_export_only"] = True
    if spec.get("manual_fov") is not None:
        common["manual_fov"] = float(spec["manual_fov"])
        common["auto_fov"] = False
    elif camera_kwargs.get("manual_fov") is not None:
        common["manual_fov"] = float(camera_kwargs["manual_fov"])
        common["auto_fov"] = False
    else:
        common["auto_fov"] = bool(camera_kwargs.get("auto_fov", True))

    _apply_video_render_flags(spec, common)

    if spec["type"] == "single":
        single_kwargs = dict(
            output_path=output_dir,
            camera_position=spec["camera_position"],
            look_at_target=spec.get("look_at_target"),
            **common,
        )
        if "up_vector" in spec:
            single_kwargs["up_vector"] = spec["up_vector"]
        ctx.render_view(**single_kwargs)
        return
    if spec["type"] == "sequence":
        seq_kwargs = dict(
            output_path=output_dir,
            camera_position=spec["camera_positions"],
            look_at_target=spec["look_at_targets"],
            **common,
        )
        if "up_vectors" in spec:
            seq_kwargs["up_vector"] = spec["up_vectors"]
        ctx.render_view(**seq_kwargs)
        return
    raise ValueError(f"Unknown view spec type: {spec.get('type')!r}")


def _render_auto_view_spec(
    ctx,
    output_dir: str,
    spec: Dict[str, Any],
    extra: dict,
    view_dir_name: str,
    camera_kwargs: Optional[dict] = None,
    *,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    defer_geometry_export: bool = False,
    geometry_export_only: bool = False,
) -> None:
    """Invoke render_view from an auto_view spec; output dir name is view_dir_name (e.g. auto_path_0004_seq)."""
    _render_view_spec(
        ctx,
        output_dir,
        spec,
        extra,
        view_dir_name,
        camera_kwargs=camera_kwargs,
        frame_start=frame_start,
        frame_end=frame_end,
        defer_geometry_export=defer_geometry_export,
        geometry_export_only=geometry_export_only,
    )


# ---------------------------------------------------------------------------
# Subprocess workers (one process per view)
# ---------------------------------------------------------------------------

def _describe_view_start(view_name: str, spec: Optional[Dict[str, Any]] = None, job: Optional[dict] = None) -> str:
    """Human-readable label for view-start timing logs."""
    spec = spec or {}
    w, h = spec.get("width"), spec.get("height")
    size = f"{w}×{h}" if w and h else "?×?"

    if view_name.startswith("auto_path_"):
        if view_name.endswith("_seq"):
            n = len(spec.get("camera_positions") or [])
            return f"auto_path sequence ({n or '?'} frames, {size})"
        pano_note = ""
        if (job or {}).get("pano") and spec.get("render_pano"):
            pano_res = (job or {}).get("pano_resolution", 4096)
            pano_note = f", +pano {pano_res}×{int(pano_res) // 2}"
        return f"auto_path ({size}{pano_note})"

    if view_name.startswith("video_center_"):
        n = spec.get("video_frames", "?")
        pano_res = spec.get("pano_resolution") or (job or {}).get("pano_resolution", 1024)
        if spec.get("pano_only"):
            return f"video pano-only ({n} frames, {pano_res}×{int(pano_res) // 2})"
        label = f"video center ({n} frames, persp {size})"
        if (job or {}).get("pano"):
            label += f", +pano {pano_res}×{int(pano_res) // 2}"
        return label

    if view_name.startswith("video_tangent_"):
        n = spec.get("video_frames", "?")
        return f"video tangent pano ({n} frames)"

    if view_name == "topdown":
        return "topdown"
    return view_name


def _load_job(job_path: str) -> dict:
    with open(job_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _instantiate_ctx(
    backend: str,
    room_type: str,
    asset_dir: Optional[str],
    hole_asset_dir: Optional[str] = None,
):
    """Create SceneCtx / BpySceneCtx for the given backend."""
    if backend == "pyrender":
        try:
            from .core.scenebuilder import SceneCtx
        except (ImportError, ValueError):
            from core.scenebuilder import SceneCtx  # type: ignore
        return SceneCtx(room_type, asset_dir, hole_asset_dir)

    try:
        from .core.scenebuilder_bpy import BpySceneCtx
    except (ImportError, ValueError):
        from core.scenebuilder_bpy import BpySceneCtx  # type: ignore
    return BpySceneCtx(room_type, asset_dir, hole_asset_dir)


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
        print("   [worker] Using normalized scene_json from job (pixel-aligned SSL)")
    ctx = _instantiate_ctx(
        job.get("backend", "bpy"),
        scene_json["room"]["room_type"],
        job.get("asset_dir"),
        job.get("hole_asset_dir"),
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


def _view_camera_kwargs(job: dict) -> dict:
    """Width/height/FOV kwargs shared by topdown, preset, and custom views."""
    kwargs = {
        "width": int(job.get("width", 1000)),
        "height": int(job.get("height", 1000)),
        "auto_fov": bool(job.get("auto_fov", True)),
    }
    manual_fov = job.get("manual_fov")
    if manual_fov is not None:
        kwargs["manual_fov"] = float(manual_fov)
        kwargs["auto_fov"] = False
    return kwargs


def _apply_job_view_size(job: dict, spec: Dict[str, Any]) -> None:
    """Override auto/custom view spec resolution only when CLI explicitly sets --width/--height."""
    if not job.get("explicit_view_size"):
        return
    spec["width"] = int(job["width"])
    spec["height"] = int(job["height"])
    manual_fov = job.get("manual_fov")
    if manual_fov is not None:
        spec["manual_fov"] = float(manual_fov)


def _refresh_auto_path_resolutions(
    specs: Dict[str, Dict[str, Any]],
    names: List[str],
) -> bool:
    """Sync auto_path_* specs to current AUTO_VIEW_WIDTH/HEIGHT constants."""
    try:
        from .core.auto_views import AUTO_VIEW_WIDTH, AUTO_VIEW_HEIGHT
    except (ImportError, ValueError):
        from core.auto_views import AUTO_VIEW_WIDTH, AUTO_VIEW_HEIGHT  # type: ignore
    patched = False
    for name in names:
        if not name.startswith("auto_path_"):
            continue
        spec = specs.get(name)
        if not spec:
            continue
        if int(spec.get("width", 0)) != AUTO_VIEW_WIDTH or int(spec.get("height", 0)) != AUTO_VIEW_HEIGHT:
            spec["width"] = AUTO_VIEW_WIDTH
            spec["height"] = AUTO_VIEW_HEIGHT
            patched = True
    return patched


def _drop_video_view_specs(
    specs: Dict[str, Dict[str, Any]],
    names: List[str],
) -> None:
    for name in list(names):
        if name.startswith("video_"):
            specs.pop(name, None)
            names.remove(name)


def _video_frame_chunk_size(job: dict) -> int:
    try:
        from .core.video_views import VIDEO_FRAME_CHUNK_SIZE
    except (ImportError, ValueError):
        from core.video_views import VIDEO_FRAME_CHUNK_SIZE  # type: ignore
    return max(1, int(job.get("video_frame_chunk_size", VIDEO_FRAME_CHUNK_SIZE)))


def _video_needs_deferred_geometry(job: dict) -> bool:
    if not job.get("visible_geometry"):
        return False
    return bool(job.get("export_glb") or job.get("export_point_cloud") or job.get("export_voxel"))


def _video_view_frame_count(spec: Optional[Dict[str, Any]]) -> int:
    if not spec:
        return 0
    if spec.get("type") == "sequence":
        return len(spec.get("camera_positions") or [])
    return 1


def _should_chunk_video_view(view_name: str, spec: Optional[Dict[str, Any]], job: dict) -> bool:
    if not view_name.startswith("video_center_"):
        return False
    if not spec or spec.get("type") != "sequence":
        return False
    return _video_view_frame_count(spec) > _video_frame_chunk_size(job)


def _should_record_view_complete(
    job: dict,
    view_name: str,
    spec: Optional[Dict[str, Any]],
    *,
    frame_start: Optional[int],
    frame_end: Optional[int],
    geometry_export_only: bool,
) -> bool:
    if geometry_export_only:
        return True
    if frame_start is not None:
        if _video_needs_deferred_geometry(job):
            return False
        n_frames = _video_view_frame_count(spec)
        end = int(frame_end) if frame_end is not None else n_frames
        return end >= n_frames
    return True


def worker_render_view(
    job_path: str,
    view_name: str,
    *,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    geometry_export_only: bool = False,
) -> None:
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
    custom_specs = job.get("custom_view_specs") or {}
    view_spec = auto_specs.get(view_name) or custom_specs.get(view_name)
    is_chunk = frame_start is not None
    if not is_chunk and not geometry_export_only:
        if job.get("resume", True) and is_view_complete(
            output_dir, view_name, job, auto_spec=view_spec
        ):
            print(f"⏭️  Skipping completed view: {view_name}")
            return

    if job.get("resume", True):
        missing = missing_view_artifacts(
            output_dir, view_name, job, auto_spec=view_spec
        )
        extra = apply_partial_resume_to_kwargs(job, missing)
        if missing:
            print(f"🔧 Re-running missing artifacts [{view_name}]: {', '.join(sorted(missing))}")
    else:
        extra = _job_render_kwargs(job)

    camera_kwargs = _view_camera_kwargs(job)

    try:
        from .core import util
    except (ImportError, ValueError):
        from core import util  # type: ignore

    view_start = time.perf_counter()
    view_label = _describe_view_start(view_name, view_spec, job)
    if is_chunk:
        view_label += f" [frames {frame_start}:{frame_end})"
    elif geometry_export_only:
        view_label += " [geometry export]"
    print(f"⏱️ [{util.log_timestamp()}] Start view: {view_name} — {view_label}")

    chunk_kwargs = dict(
        frame_start=frame_start,
        frame_end=frame_end,
        geometry_export_only=geometry_export_only,
    )
    record_complete = _should_record_view_complete(
        job,
        view_name,
        view_spec,
        frame_start=frame_start,
        frame_end=frame_end,
        geometry_export_only=geometry_export_only,
    )

    try:
        if view_name in auto_specs:
            spec = dict(auto_specs[view_name])
            _apply_job_view_size(job, spec)
            extra = dict(extra)
            _apply_video_render_flags(spec, extra)
            _apply_auto_path_pano_gate(view_name, spec, extra, job)
            _render_auto_view_spec(
                ctx,
                output_dir,
                spec,
                extra,
                view_dir_name=view_name,
                camera_kwargs=camera_kwargs,
                **chunk_kwargs,
            )
            if record_complete:
                view_dir = view_dir_for(output_dir, view_name)
                if spec.get("pano_only"):
                    try:
                        from .core.render_resume import _pano_dir_for
                    except (ImportError, ValueError):
                        from core.render_resume import _pano_dir_for  # type: ignore
                    main_pngs = list_primary_frames(_pano_dir_for(view_dir))
                else:
                    main_pngs = list_primary_frames(view_dir)
                record_view_complete(
                    output_dir,
                    view_name,
                    view_dir=view_dir,
                    main_pngs=main_pngs,
                )
            return

        if view_name in custom_specs:
            spec = dict(custom_specs[view_name])
            _apply_job_view_size(job, spec)
            _render_view_spec(
                ctx,
                output_dir,
                spec,
                extra,
                view_dir_name=view_name,
                camera_kwargs=camera_kwargs,
                **chunk_kwargs,
            )
            if record_complete:
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
                **_view_camera_kwargs(job),
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
            raise ValueError(f"Unknown view: {view_name}")

        ctx.render_view(
            output_path=output_dir,
            camera_position=view_cameras[view_name],
            look_at_target=look_at,
            rebuild=True,
            use_HDRI=False,
            view_dir_name=view_dir_name_for(view_name),
            **_view_camera_kwargs(job),
            **extra,
        )
        view_dir = view_dir_for(output_dir, view_name)
        record_view_complete(
            output_dir,
            view_name,
            view_dir=view_dir,
            main_pngs=list_primary_frames(view_dir),
        )
    finally:
        elapsed = time.perf_counter() - view_start
        print(f"⏱️ [{util.log_timestamp()}] Done view: {view_name} ({elapsed:.1f}s)")


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
        print("⏭️  Skipping full-scene geometry export (already complete)")
        return

    ctx = _create_render_ctx(job)
    scene_json = job["scene_json"]

    try:
        from .core import util_data
    except (ImportError, ValueError):
        from core import util_data  # type: ignore

    if job.get("export_glb"):
        glb_path = util_data.holo_glb_path(output_dir)
        try:
            ctx.export_glb(glb_path, rebuild=True, show_ceiling=True)
        except Exception as exc:
            print(f"⚠️ GLB export failed: {exc}")

    if job.get("export_point_cloud"):
        point_cloud_dir = util_data.geometry_pointcloud_dir(output_dir)
        try:
            ctx.export_point_cloud(point_cloud_dir, rebuild=True, show_ceiling=True)
        except Exception as exc:
            print(f"⚠️ Point cloud export failed: {exc}")

    if job.get("export_voxel"):
        if hasattr(ctx, "export_voxel"):
            try:
                ctx.export_voxel(output_dir, rebuild=True, show_ceiling=True)
            except Exception as exc:
                print(f"⚠️ Voxel export failed: {exc}")
        else:
            print("⚠️ Current backend does not support full-scene voxel export")

    with open(os.path.join(output_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(scene_json, f, indent=2, ensure_ascii=False)

    record_post_complete(output_dir)


_WORKER_SEGV_RETRIES = 3
_WORKER_SEGV_RETRY_DELAY_S = 3.0


def _is_worker_retryable_error(exc: subprocess.CalledProcessError) -> bool:
    return exc.returncode in (-signal.SIGSEGV, -signal.SIGKILL)


def _is_worker_segv_error(exc: subprocess.CalledProcessError) -> bool:
    return exc.returncode == -signal.SIGSEGV


def _worker_env() -> dict:
    """Subprocess env: inherit parent PYTHONPATH and ensure scenebuilder package is importable."""
    env = os.environ.copy()
    prefix_paths = [_PKG_PARENT, _PKG_ROOT]
    existing = env.get("PYTHONPATH", "")
    merged = os.pathsep.join(prefix_paths + ([existing] if existing else []))
    env["PYTHONPATH"] = merged
    return env


def _spawn_worker(cmd: List[str], *, label: str) -> None:
    """Run a render worker subprocess; retry on intermittent Blender SIGSEGV / OOM SIGKILL."""
    last_exc: Optional[subprocess.CalledProcessError] = None
    for attempt in range(1, _WORKER_SEGV_RETRIES + 1):
        try:
            subprocess.run(cmd, check=True, env=_worker_env())
            return
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            if not _is_worker_retryable_error(exc) or attempt >= _WORKER_SEGV_RETRIES:
                raise
            reason = "SIGSEGV" if exc.returncode == -signal.SIGSEGV else "SIGKILL (likely OOM)"
            print(
                f"⚠️  {label} crashed with {reason} (attempt {attempt}/{_WORKER_SEGV_RETRIES}); "
                f"retrying in {_WORKER_SEGV_RETRY_DELAY_S:.0f}s..."
            )
            time.sleep(_WORKER_SEGV_RETRY_DELAY_S)
    if last_exc is not None:
        raise last_exc


def _spawn_worker_view(
    job_path: str,
    view_name: str,
    *,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    geometry_export_only: bool = False,
) -> None:
    kwargs: List[str] = []
    if frame_start is not None:
        kwargs.append(f"frame_start={int(frame_start)}")
    if frame_end is not None:
        kwargs.append(f"frame_end={int(frame_end)}")
    if geometry_export_only:
        kwargs.append("geometry_export_only=True")
    kw = ", " + ", ".join(kwargs) if kwargs else ""
    label = f"View {view_name!r}"
    if frame_start is not None:
        label += f" frames [{frame_start}:{frame_end})"
    if geometry_export_only:
        label += " geometry export"
    _spawn_worker([
        sys.executable, "-c",
        "from scenebuilder.render_ssl import worker_render_view; "
        f"worker_render_view({job_path!r}, {view_name!r}{kw})",
    ], label=label)


def _spawn_video_view_workers(job_path: str, view_name: str, spec: Dict[str, Any], job: dict) -> None:
    n_frames = _video_view_frame_count(spec)
    chunk_size = _video_frame_chunk_size(job)
    if n_frames <= chunk_size:
        _spawn_worker_view(job_path, view_name)
        return
    print(
        f"🎬 Video chunking: {view_name} → {n_frames} frames in "
        f"{(n_frames + chunk_size - 1) // chunk_size} subprocesses "
        f"(chunk_size={chunk_size})"
    )
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        _spawn_worker_view(job_path, view_name, frame_start=start, frame_end=end)
    if _video_needs_deferred_geometry(job):
        _spawn_worker_view(job_path, view_name, geometry_export_only=True)


def _spawn_worker_post(job_path: str) -> None:
    _spawn_worker([
        sys.executable, "-c",
        "from scenebuilder.render_ssl import worker_render_post; "
        f"worker_render_post({job_path!r})",
    ], label="Full-scene geometry export")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _run_normalized_topdown_phase(
    ctx,
    output_dir: str,
    *,
    align: Dict[str, Any],
    floor_path: bool = True,
    show_ceiling: bool = False,
) -> Optional[Dict[str, Any]]:
    """Render fixed 1000² pixel-aligned top-down under Y/topdown_normalized/ and optionally sample the floor path."""
    if not hasattr(ctx, "normalized_topdown_view"):
        raise RuntimeError("Current backend does not support normalized_topdown_view")

    topdown_norm_dir = os.path.join(output_dir, TOPDOWN_NORMALIZED_SUBDIR)
    os.makedirs(topdown_norm_dir, exist_ok=True)
    print(f"🗺️  Pixel-aligned top-down view + floor path → {topdown_norm_dir}")

    ctx.normalized_topdown_view(
        topdown_norm_dir,
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
    hole_asset_dir: Optional[str] = None,
    texture_dir: Optional[str] = None,
    gen_texture: bool = False,
    image: Optional[str] = None,
    samples: Optional[int] = None,
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
    """Pixel-aligned mode: equivalent to ``render_ssl(..., normalized_topdown=True, views=None)``.

    Output root Y = ``{output_dir}_normalized``; path-planning artifacts live under ``Y/topdown_normalized/``.
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
        hole_asset_dir=hole_asset_dir,
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
    hole_asset_dir: Optional[str] = None,
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
    pano_num: int = 3,
    samples: Optional[int] = None,
    normalized_topdown: bool = False,
    floor_path: bool = True,
    normalized_topdown_show_ceiling: bool = False,
    width: int = 1000,
    height: int = 1000,
    auto_fov: bool = True,
    manual_fov: Optional[float] = None,
    resume: bool = True,
    camera_position: Any = None,
    look_at: Any = None,
    up_vector: Any = None,
    video: bool = False,
    video_frame_spacing: float = 0.2,
    video_frames: Optional[int] = None,
    video_frame_chunk_size: Optional[int] = None,
):
    """Render a scene. ``input_text`` may be standard SSL text or a JSON string.

    ``visible_geometry=True`` forces ``semantic=True`` (visible ratio depends on semantic mask).

    When ``normalized_topdown=True``:
    - Output root Y = ``{output_root}_normalized`` (otherwise Y = ``{output_root}``)
    - Apply pixel-aligned SSL normalization first; all subsequent rendering uses that frame
    - Automatically render pixel-aligned top-down + floor path under ``Y/topdown_normalized/``
    - Then render ``topdown/``, sequence views, etc. under Y per ``views`` (``views=None`` skips preset views)
    - ``views="auto"``: force normalization + floor path, then render regular topdown + path-driven auto views
    - ``video=True`` (``--video`` alone or with ``views="auto"``): normalization + floor path; alone skips sparse auto / regular topdown and renders video trajectories only

    When ``camera_position`` / ``look_at`` are provided, append a custom single frame (``custom/``)
    or sequence (``custom_seq/``) after the ``views`` specified above finish.
    ``up_vector`` is optional: ``render_view`` defaults to ``[0,0,1]``; uses ``[0,1,0]`` when collinear.

    ``width`` / ``height`` / ``auto_fov`` / ``manual_fov`` follow ``render_view`` semantics for preset,
    custom, auto, and regular ``topdown/`` views (defaults: 1000×1000, auto FOV).
    Pixel-aligned ``topdown_normalized/`` is **fixed at 1000×1000** (SpatialFactory convention; not configurable).
    Auto views keep randomized FOV unless ``manual_fov`` is set.

    Each view renders in world (or normalized) SSL; visible-geometry primary files use world SSL with ``*_opencv`` copies.
    """
    width = int(width)
    height = int(height)
    explicit_view_size = width != 1000 or height != 1000
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    if manual_fov is not None and float(manual_fov) <= 0:
        raise ValueError("manual_fov must be positive when set")
    if manual_fov is not None:
        auto_fov = False
    if visible_geometry:
        semantic = True
    try:
        from .core.video_views import VIDEO_FRAME_CHUNK_SIZE
    except (ImportError, ValueError):
        from core.video_views import VIDEO_FRAME_CHUNK_SIZE  # type: ignore
    resolved_video_chunk_size = max(
        1,
        int(video_frame_chunk_size if video_frame_chunk_size is not None else VIDEO_FRAME_CHUNK_SIZE),
    )
    if views == "auto":
        normalized_topdown = True
        floor_path = True
    if video:
        normalized_topdown = True
        floor_path = True
    if outpaint_image_dir is None:
        outpaint_image_dir = output_root

    try:
        from .core.util_data import get_mesh
    except (ImportError, ValueError):
        from core.util_data import get_mesh  # type: ignore

    print(f"\n🚀 Starting render [backend: {backend}, asset_mode: {asset_mode}]")

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
    ctx = _instantiate_ctx(backend, room_type, asset_dir, hole_asset_dir)
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
        print(f"📐 Step 1: pixel-aligned SSL normalization → output root Y = {y_dir}")
        pixel_align = prepare_pixel_aligned_topdown_context(ctx.context)
        apply_scene_json_xy_translation(
            scene_json,
            pixel_align["dx_ssl"],
            pixel_align["dy_ssl"],
        )

    ssl_exclude_box_ids = None
    if normalized_topdown:
        try:
            from .core import util as core_util
        except (ImportError, ValueError):
            from core import util as core_util  # type: ignore
        ssl_exclude_box_ids = core_util.identify_topdown_occluding_box_ids(
            ctx.context, getattr(ctx, "config", {})
        ) or None

    standard_ssl = format_standard_ssl(
        context_for_ssl_export(ctx.context, ssl_exclude_box_ids)
    )
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
            print(f"⏭️  Skipping pixel-aligned topdown_normalized (already complete) → {y_dir}/{TOPDOWN_NORMALIZED_SUBDIR}/")
            floor_result = load_floor_result_from_disk(y_dir) if floor_path else None
        else:
            print(f"📐 Step 2: topdown_normalized + floor path → {y_dir}/{TOPDOWN_NORMALIZED_SUBDIR}/")
            floor_result = _run_normalized_topdown_phase(
                ctx,
                y_dir,
                align=pixel_align,
                floor_path=floor_path,
                show_ceiling=normalized_topdown_show_ceiling,
            )

    center = ctx.context["meta"]["center"]
    span = ctx.context["meta"]["span"]
    z_max = ctx.context["meta"]["z_max"]
    look_at = [center[0], center[1], z_max / 2]
    z_cam = z_max * 5 / 6
    # Perspective view camera positions (edit here; subprocesses read from job)
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
    custom_view_specs: Dict[str, Dict[str, Any]] = {}
    custom_view_names: List[str] = []
    if views == "auto" or video:
        if backend != "bpy":
            print("⚠️  views=auto / --video currently only supports backend=bpy; skipped path views")
        else:
            auto_view_specs, auto_view_names = _run_auto_views_from_path(
                y_dir,
                floor_result,
                ctx.context,
                resume=resume,
                width=width,
                height=height,
                sparse_auto=(views == "auto"),
                video=video,
                video_frame_spacing=video_frame_spacing,
                video_frames=video_frames,
                pano_num=pano_num,
            )

    if camera_position is not None:
        custom_name, custom_spec = _build_custom_view_spec(
            camera_position,
            look_at,
            up_vector=up_vector,
            width=width,
            height=height,
            manual_fov=manual_fov,
        )
        custom_view_specs[custom_name] = custom_spec
        custom_view_names.append(custom_name)
        print(
            f"🎯 Custom view (appended): {custom_name} "
            f"({'sequence' if custom_spec['type'] == 'sequence' else 'single frame'}, "
            f"{len(custom_spec.get('camera_positions', [custom_spec.get('camera_position')]))} frames)"
        )

    job = {
        "backend": backend,
        "scene_json": scene_json,
        "output_dir": y_dir,
        "normalized_topdown": normalized_topdown,
        "asset_dir": asset_dir,
        "hole_asset_dir": hole_asset_dir,
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
        "pano_num": max(1, int(pano_num)),
        "width": int(width),
        "height": int(height),
        "explicit_view_size": bool(explicit_view_size),
        "auto_fov": bool(auto_fov),
        "manual_fov": float(manual_fov) if manual_fov is not None else None,
        "look_at": look_at,
        "view_cameras": view_cameras,
        "auto_view_specs": auto_view_specs,
        "custom_view_specs": custom_view_specs,
        "resume": resume,
        "video_frame_chunk_size": resolved_video_chunk_size,
    }
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2, ensure_ascii=False)

    views_to_run = _resolve_views_to_run(views, view_cameras)
    if views == "auto":
        if backend == "bpy":
            # Regular topdown (Y/topdown/) + path-driven auto views (+ video when --video)
            views_to_run = ["topdown"] + list(auto_view_names)
        else:
            views_to_run = []
    elif video:
        if backend == "bpy":
            views_to_run = list(auto_view_names)
        else:
            views_to_run = []

    for custom_name in custom_view_names:
        if custom_name not in views_to_run:
            views_to_run.append(custom_name)

    if views_to_run:
        views_to_run, skipped_views = filter_views_to_run(
            y_dir, views_to_run, job, resume=resume
        )
        if skipped_views:
            print(f"⏭️  Skipped {len(skipped_views)} completed views: {', '.join(skipped_views)}")
        if views_to_run:
            step = "Step 3: " if normalized_topdown else ""
            n_sub = 0
            auto_specs = job.get("auto_view_specs") or {}
            for view_name in views_to_run:
                spec = auto_specs.get(view_name) or (job.get("custom_view_specs") or {}).get(view_name)
                if _should_chunk_video_view(view_name, spec, job):
                    n_frames = _video_view_frame_count(spec)
                    chunk_size = _video_frame_chunk_size(job)
                    n_sub += (n_frames + chunk_size - 1) // chunk_size
                    if _video_needs_deferred_geometry(job):
                        n_sub += 1
                else:
                    n_sub += 1
            print(f"🎨 {step}Multi-view rendering → {y_dir} ({n_sub} subprocesses)")
            try:
                from .core import util
            except (ImportError, ValueError):
                from core import util  # type: ignore
            for view_name in views_to_run:
                print(f"🔄 [{util.log_timestamp()}] Subprocess rendering: {view_name}")
                spec = auto_specs.get(view_name) or (job.get("custom_view_specs") or {}).get(view_name)
                if _should_chunk_video_view(view_name, spec, job):
                    _spawn_video_view_workers(job_path, view_name, spec, job)
                else:
                    _spawn_worker_view(job_path, view_name)
    elif not views_to_run and views == "auto":
        print("⏭️  Auto views not generated (backend does not support bpy)")
    elif not views_to_run and video:
        print("⏭️  Video views not generated (backend does not support bpy)")
    elif not views_to_run and normalized_topdown:
        print("⏭️  No views specified; skipping multi-view rendering")

    need_holo_post = holo_geometry and (export_glb or export_point_cloud or export_voxel)
    if need_holo_post:
        print(f"🔄 Subprocess exporting full-scene geometry → {y_dir}")
        _spawn_worker_post(job_path)

    print(f"✅ Render complete! Output directory: {os.path.abspath(y_dir)}")
    return y_dir, standard_ssl, floor_result


ALL_VIEWS = [
    "topdown", "front", "behind", "left", "right",
    "leftfront", "rightfront", "leftbehind", "rightbehind",
    "left_seq",
]


def _parse_ssl_id(ssl_id: str) -> Tuple[str, int]:
    """Parse ``310449449_4`` -> (``310449449``, 4). Room index is 0-based line number in JSONL."""
    if not ssl_id or "_" not in ssl_id:
        raise ValueError(
            f"Invalid --ssl_id {ssl_id!r}; expected format {{design_id}}_{{room_index}}, e.g. 310449449_4"
        )
    design_id, room_str = ssl_id.rsplit("_", 1)
    if not design_id or not room_str.isdigit():
        raise ValueError(
            f"Invalid --ssl_id {ssl_id!r}; expected format {{design_id}}_{{room_index}}, e.g. 310449449_4"
        )
    return design_id, int(room_str)


def _read_jsonl_line(file_path: str, line_number: int) -> Dict[str, Any]:
    """Read line ``line_number`` (0-based) from a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == line_number:
                return json.loads(line.strip())
    raise ValueError(f"Line number {line_number} is out of file range")


def _find_scene_in_jsonl(
    jsonl_path: str, ssl_id: str, room_index: int
) -> Tuple[Dict[str, Any], int]:
    """Find a room by ``room.id``; fall back to legacy line index."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            scene_data = json.loads(line.strip())
            room_id = (scene_data.get("room") or {}).get("id")
            if room_id == ssl_id:
                return scene_data, i

    scene_data = _read_jsonl_line(jsonl_path, room_index)
    return scene_data, room_index


def _load_scene_text_from_collection(ssl_id: str, collection_dir: str) -> Tuple[str, str]:
    """Load one room JSON line from ``{collection_dir}/{design_id}.jsonl``."""
    design_id, room_index = _parse_ssl_id(ssl_id)
    jsonl_path = os.path.join(collection_dir, f"{design_id}.jsonl")
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(
            f"JSONL not found for --ssl_id {ssl_id!r}: {jsonl_path}"
        )

    try:
        scene_data, line_index = _find_scene_in_jsonl(jsonl_path, ssl_id, room_index)
    except ValueError as exc:
        raise ValueError(
            f"Room {ssl_id!r} not found in {jsonl_path} (legacy index {room_index})"
        ) from exc

    room = scene_data.get("room") or {}
    if line_index != room_index:
        print(
            f"📂 Loaded scene from {jsonl_path} line {line_index} "
            f"by room.id={ssl_id!r} (legacy index {room_index})"
        )
    else:
        print(
            f"📂 Loaded scene from {jsonl_path} line {line_index} "
            f"({room.get('room_type', 'unknown')})"
        )

    scene_text = json.dumps(scene_data, ensure_ascii=False)
    return scene_text, jsonl_path


def _read_scene_input(args) -> Tuple[str, Optional[str], Optional[str]]:
    """Load scene text. Returns (input_text, source_path, asset_id_timestamp)."""
    if getattr(args, "ssl_id", None) is not None:
        scene_text, jsonl_path = _load_scene_text_from_collection(args.ssl_id, args.ssl_collection_dir)
        return scene_text, jsonl_path, args.ssl_id
    if args.ssl_str is not None:
        return args.ssl_str, None, None
    if args.ssl == "-":
        import sys
        text = sys.stdin.read()
        if not text.strip():
            raise ValueError("Empty scene input from stdin (--ssl -)")
        return text, None, None
    with open(args.ssl, "r", encoding="utf-8") as f:
        return f.read(), args.ssl, None


def _prepare_ssl_and_dirs(args) -> tuple:
    """Read SSL/JSON input and resolve output directory plus asset/texture paths."""
    input_text, source_path, asset_id_timestamp = _read_scene_input(args)

    if args.output is not None:
        output_dir = args.output
    elif getattr(args, "ssl_id", None) is not None:
        output_dir = os.path.join(os.getcwd(), "render_output", args.ssl_id)
    elif source_path is not None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(source_path)), "render_output")
    else:
        output_dir = os.path.join(os.getcwd(), "render_output")

    asset_dir = args.assets
    if asset_dir is None:
        search_roots = []
        if source_path is not None:
            search_roots.append(os.path.dirname(os.path.abspath(source_path)))
        search_roots.append(os.getcwd())
        for root in search_roots:
            for candidate in ("assets", "Assets", "nano_gen_asset"):
                path = os.path.join(root, candidate)
                if os.path.isdir(path):
                    asset_dir = path
                    break
            if asset_dir is not None:
                break

    texture_dir = args.texture
    if texture_dir is None:
        search_roots = []
        if source_path is not None:
            search_roots.append(os.path.dirname(os.path.abspath(source_path)))
        search_roots.append(os.getcwd())
        for root in search_roots:
            candidate = os.path.join(root, "texture")
            if os.path.isdir(candidate):
                texture_dir = candidate
                break

    if "asset_id=" not in input_text and "mesh_id=" in input_text:
        if asset_id_timestamp is not None:
            timestamp = asset_id_timestamp
        elif source_path is not None:
            timestamp = os.path.basename(os.path.dirname(source_path))
        else:
            timestamp = None

        if timestamp is not None:
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
        description="Render multi-view images from SSL/JSON scene files and export GLB/point clouds, etc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (equivalent to common commands in SpatialFactory/scripts/render_scene.py):

  python /data-nas/data/experiments/mushui/projects/utils/fast-scene/scenebuilder/render_ssl.py \\
    --ssl /data-nas/data/experiments/mushui/projects/SpatialFactory/benchmark/data/Balcony/310449449_4/ssl.txt \\
    --views topdown left_seq \\
    --output /data-nas/data/experiments/mushui/projects/SpatialFactory/benchmark/data/Balcony/310449449_4/out9 \\
    --glb --assets /data-nas/data/dataset/qunhe/Manycore-Future/simplified \\
    --ply --visible_geometry --semantic --depth --pano
""",
    )
    parser.add_argument(
        "--ssl",
        help="Path to SSL/JSON scene file, or '-' to read scene text from stdin",
    )
    parser.add_argument(
        "--ssl_str",
        default=None,
        help="Inline SSL or JSON scene string (e.g. one line from JSONL); mutually exclusive with --ssl / --ssl_id",
    )
    parser.add_argument(
        "--ssl_id",
        default=None,
        help="Room id like 310449449_4 (design_id + 0-based room index); requires --ssl_collection_dir",
    )
    parser.add_argument(
        "--ssl_collection_dir",
        default=None,
        help="Directory of per-design JSONL files, e.g. /root/datasets/manycore/spatiallm_raw",
    )
    parser.add_argument("--output", default=None, help="Output directory (default: render_output next to ssl file, cwd, or as given)")
    parser.add_argument(
        "--views", nargs="*", default=None, metavar="VIEW",
        help=f"View list ({', '.join(ALL_VIEWS)}); omit to skip preset views; "
             f"auto forces normalized_topdown + floor path and auto-generates views from path; "
             f"--camera_position / --look_at append custom views after views finish",
    )
    parser.add_argument(
        "--camera_position",
        default=None,
        help="JSON: [x,y,z] for single frame, or [[x,y,z], ...] for sequence; appends custom/custom_seq after views",
    )
    parser.add_argument(
        "--look_at",
        default=None,
        help="JSON: [x,y,z] or [[x,y,z], ...]; for sequences, a single point (shared) or same length as camera_position",
    )
    parser.add_argument(
        "--up_vector",
        default=None,
        help="JSON: [x,y,z] or [[x,y,z], ...]; defaults to [0,0,1] in render_view; auto [0,1,0] when view axis is collinear",
    )
    parser.add_argument("--backend", choices=["bpy", "pyrender"], default="bpy", help="Render backend")
    parser.add_argument("--texture", default=None, help="Texture directory")
    parser.add_argument("--assets", default=None, help="3D asset directory (auto-detected if omitted)")
    parser.add_argument(
        "--hole_assets",
        default=None,
        help="Door/window asset fallback directory; used when asset_id is missing under --assets, before config model_hole_path",
    )
    parser.add_argument("--glb", action="store_true", help="Export GLB")
    parser.add_argument("--ply", action="store_true", help="Export colored point cloud PLY")
    parser.add_argument("--visible_geometry", action="store_true",
                        help="Per-view frustum GLB/PLY/voxel (requires --glb, --ply, or --voxel)")
    parser.add_argument("--holo_geometry", action="store_true",
                        help="Root-level full-scene GLB/point cloud/voxel (requires --glb, --ply, or --voxel)")
    parser.add_argument("--voxel", action="store_true",
                        help="256³ colored occupancy voxel (per-view needs --visible_geometry, root needs --holo_geometry)")
    parser.add_argument("--semantic", action="store_true", help="Export semantic segmentation map")
    parser.add_argument("--depth", action="store_true", help="Export depth map and normal map")
    parser.add_argument("--pano", action="store_true",
                        help="For render_view perspectives, also export sibling dir {millis_timestamp}_pano (not for topdown)")
    parser.add_argument("--pano_resolution", type=int, default=4096,
                        help="Panorama horizontal resolution; only with --pano; height is half (default: 4096, i.e. 4096x2048)")
    parser.add_argument(
        "--pano_num",
        type=int,
        default=3,
        help="With --pano and --views auto: evenly sample this many auto_path singles for pano "
             "(default: 3; persp render still runs for all auto_path views)",
    )
    parser.add_argument("--width", type=int, default=1000,
                        help="Render width for views (default: 1000; same semantics as render_view)")
    parser.add_argument("--height", type=int, default=1000,
                        help="Render height for views (default: 1000; same semantics as render_view)")
    parser.add_argument("--manual_fov", type=float, default=None,
                        help="Vertical FOV in degrees; overrides auto_fov (same as render_view)")
    parser.add_argument("--no_auto_fov", action="store_true",
                        help="Disable auto FOV; use with --manual_fov for perspective views")
    parser.add_argument("--normalized_topdown", action="store_true",
                        help="Enable pixel-aligned SSL: sole output Y={output}_normalized, "
                             "Y/topdown_normalized path planning first, then render views")
    parser.add_argument("--no_floor_path", action="store_true",
                        help="With --normalized_topdown: skip floor path sampling under topdown_normalized")
    parser.add_argument("--samples", type=int, default=None, help="Blender sample count")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume; force re-render all views and exports",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Render center-look closed-loop video along the floor path "
             "(perspective 896×896; pano 1024×512 only with --pano). "
             "Use alone for video-only (normalized topdown + path planning, no sparse auto views), "
             "or with --views auto to append on top of sparse auto views",
    )
    parser.add_argument(
        "--video_frames",
        type=int,
        default=None,
        help="Fixed frame count per video trajectory (≥3). "
             "If omitted, frame count is computed from path arc length / --video_spacing",
    )
    parser.add_argument(
        "--video_spacing",
        type=float,
        default=0.2,
        help="Target arc-length spacing between video frames in meters (default: 0.2)",
    )
    parser.add_argument(
        "--video_chunk_size",
        type=int,
        default=None,
        help="Split long video_center_* trajectories into subprocesses of this many frames "
             "(default: 20; reduces OOM risk on heavy --semantic/--depth/--pano runs)",
    )

    args = parser.parse_args()
    input_modes = [name for name, val in (
        ("--ssl", args.ssl),
        ("--ssl_str", args.ssl_str),
        ("--ssl_id", args.ssl_id),
    ) if val is not None]
    if not input_modes:
        parser.error("one of --ssl, --ssl_str, or --ssl_id is required")
    if len(input_modes) > 1:
        parser.error(f"only one scene input flag allowed; got {', '.join(input_modes)}")
    if args.ssl_id is not None and not args.ssl_collection_dir:
        parser.error("--ssl_collection_dir is required when using --ssl_id")
    if args.ssl_collection_dir is not None and args.ssl_id is None:
        parser.error("--ssl_collection_dir requires --ssl_id")
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
            parser.error("--views auto cannot be combined with other views")
        views = list(args.views)

    if args.video:
        normalized_topdown = True
        floor_path = True
        if views is None:
            print("ℹ️  --video only → normalized_topdown + floor path + video (skip sparse auto views)")
        elif views == "auto":
            print("ℹ️  views=auto + --video → sparse auto views + video")
        else:
            parser.error("--video can be used alone (omit --views) or with --views auto")
    if args.video_spacing <= 0:
        parser.error("--video_spacing must be positive")
    if args.video_frames is not None and args.video_frames < 3:
        parser.error("--video_frames must be at least 3")
    if args.video_chunk_size is not None and args.video_chunk_size < 1:
        parser.error("--video_chunk_size must be at least 1")
    if args.pano_num < 1:
        parser.error("--pano_num must be at least 1")

    render_kwargs = dict(
        backend=args.backend,
        output_root=output_dir,
        retrieve_hole=True,
        asset_mode="none",
        asset_dir=asset_dir,
        hole_asset_dir=args.hole_assets,
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
        pano_num=args.pano_num,
        normalized_topdown=normalized_topdown,
        floor_path=floor_path,
        resume=not args.no_resume,
        width=args.width,
        height=args.height,
        auto_fov=not args.no_auto_fov,
        manual_fov=args.manual_fov,
        video=args.video,
        video_frame_spacing=args.video_spacing,
        video_frames=args.video_frames,
        video_frame_chunk_size=args.video_chunk_size,
    )
    if args.samples is not None:
        render_kwargs["samples"] = args.samples

    if args.camera_position is not None or args.look_at is not None:
        if args.camera_position is None:
            parser.error("--camera_position is required when using --look_at")
        render_kwargs["camera_position"] = _parse_camera_cli_arg(args.camera_position, "camera_position")
        if args.look_at is not None:
            render_kwargs["look_at"] = _parse_camera_cli_arg(args.look_at, "look_at")
        if args.up_vector is not None:
            render_kwargs["up_vector"] = _parse_camera_cli_arg(args.up_vector, "up_vector")

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
python /data/mushui/scenebuilder/render_ssl.py --ssl_id 246597155_2 --ssl_collection_dir /data/mushui/datasets/manycore/spatiallm_raw --output /data/mushui/scenebuilder/out/246597155_2/out3 --assets /data/mushui/datasets/Manycore-Future/simplified --hole_assets /data/mushui/datasets/Manycore-Future/holes   --pano --glb --ply --visible_geometry --semantic --depth --voxel --holo_geometry --video --video_frames 2
'''