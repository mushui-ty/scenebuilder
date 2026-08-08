"""render_ssl 断点续跑：按逻辑视角名检测产物是否完整，跳过已完成项。

时间戳仅出现在视角目录内的 PNG 文件名；目录名对 preset / auto 视角固定为
``front``、``left_seq``、``auto_path_0004`` 等，便于 resume。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROGRESS_FILENAME = ".render_progress.json"

_AUX_PNG_MARKERS = (
    "_depth",
    "_semantic",
    "_normal",
    "_lines",
    "_bbox_2d",
    "nav_mask",
    "floor_path",
    "topdown_floor_path",
)

_MILLIS_DIR_RE = re.compile(r"^\d{10,16}$")
_MILLIS_SEQ_DIR_RE = re.compile(r"^\d{10,16}_seq$")


def view_dir_name_for(view_name: str) -> str:
    """逻辑视角 → 输出子目录名（与 worker 中 view_dir_name 一致）。"""
    return view_name


def is_sequence_view(
    view_name: str,
    view_cameras: Optional[Dict[str, Any]] = None,
    auto_spec: Optional[Dict[str, Any]] = None,
) -> bool:
    if auto_spec is not None:
        return auto_spec.get("type") == "sequence"
    if view_cameras and view_name in view_cameras:
        cam = view_cameras[view_name]
        return isinstance(cam, list) and cam and isinstance(cam[0], (list, tuple))
    return view_name.endswith("_seq")


def expected_frame_count(
    view_name: str,
    view_cameras: Optional[Dict[str, Any]] = None,
    auto_spec: Optional[Dict[str, Any]] = None,
) -> int:
    if auto_spec is not None:
        if auto_spec.get("type") == "sequence":
            return len(auto_spec.get("camera_positions") or [])
        return 1
    if view_cameras and view_name in view_cameras:
        cam = view_cameras[view_name]
        if isinstance(cam, list) and cam and isinstance(cam[0], (list, tuple)):
            return len(cam)
    return 1


def _is_primary_png(name: str) -> bool:
    if not name.lower().endswith(".png"):
        return False
    return not any(marker in name for marker in _AUX_PNG_MARKERS)


def list_primary_frames(view_dir: str) -> List[str]:
    """返回带 ``*_camera_para.json`` 的主渲染 PNG 路径列表。"""
    if not os.path.isdir(view_dir):
        return []
    frames: List[str] = []
    for name in sorted(os.listdir(view_dir)):
        if not _is_primary_png(name):
            continue
        base = os.path.splitext(name)[0]
        para = os.path.join(view_dir, f"{base}_camera_para.json")
        png = os.path.join(view_dir, name)
        if os.path.isfile(para) and os.path.getsize(png) > 0:
            frames.append(png)
    return frames


def _rgb_depth_frame_ok(
    png_path: str,
    *,
    depth: bool,
) -> bool:
    view_dir = os.path.dirname(png_path)
    base = os.path.splitext(os.path.basename(png_path))[0]
    if depth:
        if not os.path.isfile(os.path.join(view_dir, f"{base}_depth.png")):
            return False
        if not os.path.isfile(os.path.join(view_dir, f"{base}_normal.png")):
            return False
    return True


def _semantic_frame_ok(png_path: str) -> bool:
    view_dir = os.path.dirname(png_path)
    base = os.path.splitext(os.path.basename(png_path))[0]
    if not os.path.isfile(os.path.join(view_dir, f"{base}_semantic.png")):
        return False
    if not os.path.isfile(os.path.join(view_dir, f"{base}_semantic.json")):
        return False
    index_path = os.path.join(view_dir, "semantic_masks", "index.json")
    if not os.path.isfile(index_path):
        return False
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    for item in payload.get("objects", []):
        mask_name = item.get("mask")
        if not mask_name:
            return False
        mask_path = os.path.join(view_dir, "semantic_masks", mask_name)
        if not os.path.isfile(mask_path):
            return False
    return True


def _frame_artifacts_ok(
    png_path: str,
    *,
    semantic: bool,
    depth: bool,
) -> bool:
    if not _rgb_depth_frame_ok(png_path, depth=depth):
        return False
    if semantic and not _semantic_frame_ok(png_path):
        return False
    return True


def _pano_complete(
    view_dir: str,
    *,
    semantic: bool,
    depth: bool,
) -> bool:
    parent = os.path.dirname(view_dir.rstrip(os.sep)) or view_dir
    pano_dir = os.path.join(parent, f"{os.path.basename(view_dir.rstrip(os.sep))}_pano")
    frames = list_primary_frames(pano_dir)
    if not frames:
        return False
    return all(_frame_artifacts_ok(p, semantic=semantic, depth=depth) for p in frames)


def _visible_glb_ok(view_dir: str) -> bool:
    return os.path.isfile(os.path.join(view_dir, "scene_visible.glb"))


def _visible_ply_ok(view_dir: str) -> bool:
    pc_dir = os.path.join(view_dir, "pointcloud")
    if not os.path.isdir(pc_dir):
        return False
    return any(name.endswith(".ply") for name in os.listdir(pc_dir))


def _voxel_ok(view_dir: str) -> bool:
    return os.path.isfile(os.path.join(view_dir, "voxel", "occupancy_world_meta.json"))


def _visibility_json_ok(view_dir: str) -> bool:
    return os.path.isfile(os.path.join(view_dir, "visibility.json"))


def _planar_faces_ok(view_dir: str) -> bool:
    if os.path.isfile(os.path.join(view_dir, "planar_faces.json")):
        return True
    return any(n.endswith("_lines.png") for n in os.listdir(view_dir))


def _visible_geometry_ok(
    view_dir: str,
    *,
    export_glb: bool,
    export_point_cloud: bool,
    export_voxel: bool,
) -> bool:
    if export_glb and not _visible_glb_ok(view_dir):
        return False
    if export_voxel and not _voxel_ok(view_dir):
        return False
    if export_point_cloud and not _visible_ply_ok(view_dir):
        return False
    return True


def _visible_geometry_artifacts_ok(
    view_dir: str,
    *,
    export_glb: bool,
    export_point_cloud: bool,
    export_voxel: bool,
) -> bool:
    if not _visibility_json_ok(view_dir):
        return False
    return _visible_geometry_ok(
        view_dir,
        export_glb=export_glb,
        export_point_cloud=export_point_cloud,
        export_voxel=export_voxel,
    )


def _render_frames_ok(
    view_dir: str,
    view_name: str,
    *,
    semantic: bool,
    depth: bool,
    n_expected: int = 1,
) -> bool:
    if view_name == "topdown":
        png = os.path.join(view_dir, "topdown.png")
        if not os.path.isfile(png) or os.path.getsize(png) <= 0:
            return False
        return _rgb_depth_frame_ok(png, depth=depth)

    frames = list_primary_frames(view_dir)
    if len(frames) < n_expected:
        return False
    return all(_rgb_depth_frame_ok(p, depth=depth) for p in frames)


def _semantic_frames_ok(
    view_dir: str,
    view_name: str,
    *,
    n_expected: int = 1,
) -> bool:
    if view_name == "topdown":
        png = os.path.join(view_dir, "topdown.png")
        if not os.path.isfile(png):
            return False
        return _semantic_frame_ok(png)
    frames = list_primary_frames(view_dir)
    if len(frames) < n_expected:
        return False
    return all(_semantic_frame_ok(p) for p in frames)


def _camera_para_ok(
    view_dir: str,
    view_name: str,
    *,
    n_expected: int = 1,
) -> bool:
    if view_name == "topdown":
        return os.path.isfile(os.path.join(view_dir, "topdown_camera_para.json"))
    return len(list_primary_frames(view_dir)) >= n_expected


def missing_view_artifacts(
    y_dir: str,
    view_name: str,
    job: Dict[str, Any],
    *,
    auto_spec: Optional[Dict[str, Any]] = None,
    progress: Optional[Dict[str, Any]] = None,
) -> set:
    """返回该视角仍缺失的产物键：render / glb / ply / planar / voxel / ssl / pano。"""
    missing: set = set()
    semantic = bool(job.get("semantic"))
    depth = bool(job.get("depth"))
    pano = bool(job.get("pano"))
    export_glb = bool(job.get("export_glb"))
    export_point_cloud = bool(job.get("export_point_cloud"))
    export_voxel = bool(job.get("export_voxel"))
    visible_geometry = bool(job.get("visible_geometry"))
    view_cameras = job.get("view_cameras") or {}

    view_dir = view_dir_for(y_dir, view_name, progress)
    n_expected = 1 if view_name == "topdown" else expected_frame_count(
        view_name, view_cameras, auto_spec
    )

    if not _render_frames_ok(
        view_dir, view_name, semantic=semantic, depth=depth, n_expected=n_expected
    ):
        missing.add("render")

    if semantic and not _semantic_frames_ok(
        view_dir, view_name, n_expected=n_expected
    ):
        missing.add("semantic")

    if not _camera_para_ok(view_dir, view_name, n_expected=n_expected):
        missing.add("camera_para")

    if not os.path.isfile(os.path.join(view_dir, "ssl_opencv.txt")):
        missing.add("ssl")

    if visible_geometry and export_glb and not _visible_glb_ok(view_dir):
        missing.add("glb")
    if visible_geometry and export_point_cloud and not _visible_ply_ok(view_dir):
        missing.add("ply")
    if visible_geometry and (export_glb or export_point_cloud or export_voxel):
        if not _visibility_json_ok(view_dir):
            if export_glb:
                missing.add("glb")
            if export_point_cloud:
                missing.add("ply")
            if export_voxel:
                missing.add("voxel")
    if export_point_cloud and not pano and not _planar_faces_ok(view_dir):
        missing.add("planar")
    if visible_geometry and export_voxel and not _voxel_ok(view_dir):
        missing.add("voxel")
    # topdown 视角 worker 会 pop pano，从不产出 pano 目录
    if pano and view_name != "topdown" and not _pano_complete(
        view_dir, semantic=semantic, depth=depth
    ):
        missing.add("pano")
    return missing


def apply_partial_resume_to_kwargs(job: Dict[str, Any], missing: set) -> Dict[str, Any]:
    """按缺失产物生成 partial render kwargs（含 skip_render 等）。"""
    visible_geometry = bool(job.get("visible_geometry"))
    want_glb = bool(job.get("export_glb"))
    want_ply = bool(job.get("export_point_cloud"))
    want_voxel = bool(job.get("export_voxel"))

    export_glb = want_glb and visible_geometry and ("glb" in missing)
    export_voxel = want_voxel and visible_geometry and ("voxel" in missing)
    export_visible_point_cloud = (
        want_ply and visible_geometry and ("ply" in missing)
    )
    export_planar_faces = want_ply and ("planar" in missing)

    return {
        "export_glb": export_glb,
        "export_voxel": export_voxel,
        "export_point_cloud": export_visible_point_cloud or export_planar_faces,
        "export_visible_point_cloud": export_visible_point_cloud,
        "export_planar_faces": export_planar_faces,
        "visible_geometry": visible_geometry,
        "render_semantic": (bool(job.get("semantic")) or visible_geometry) and ("semantic" in missing),
        "render_depth": bool(job.get("depth")) and ("render" in missing),
        "pano": bool(job.get("pano")) and ("pano" in missing),
        "pano_resolution": int(job.get("pano_resolution", 4096)),
        "skip_render": "render" not in missing,
        "write_ssl": "ssl" in missing,
        "write_camera_para": ("render" in missing) or ("camera_para" in missing),
    }


def view_dir_for(
    y_dir: str,
    view_name: str,
    progress: Optional[Dict[str, Any]] = None,
) -> str:
    if view_name == "topdown":
        return os.path.join(y_dir, "topdown")
    dir_name = view_dir_name_for(view_name)
    recorded = (progress or {}).get("views", {}).get(view_name, {}).get("view_dir")
    if recorded:
        return os.path.join(y_dir, recorded)
    return os.path.join(y_dir, dir_name)


def is_view_complete(
    y_dir: str,
    view_name: str,
    job: Dict[str, Any],
    *,
    auto_spec: Optional[Dict[str, Any]] = None,
    progress: Optional[Dict[str, Any]] = None,
) -> bool:
    """检测单个逻辑视角是否已按当前 job 开关完整渲染。"""
    if view_name != "topdown":
        view_dir = view_dir_for(y_dir, view_name, progress)
        if not os.path.isdir(view_dir):
            return False
    return len(missing_view_artifacts(
        y_dir, view_name, job, auto_spec=auto_spec, progress=progress
    )) == 0


def is_normalized_topdown_complete(
    y_dir: str,
    *,
    floor_path: bool,
    semantic: bool,
    depth: bool,
) -> bool:
    td = os.path.join(y_dir, "topdown_normalized")
    png = os.path.join(td, "topdown.png")
    if not os.path.isfile(png) or os.path.getsize(png) <= 0:
        return False
    if not os.path.isfile(os.path.join(td, "camera_para.json")):
        return False
    if floor_path:
        need_depth_sem = True
        if need_depth_sem:
            if not os.path.isfile(os.path.join(td, "topdown_depth.png")):
                return False
            if not os.path.isfile(os.path.join(td, "topdown_semantic.png")):
                return False
            if not os.path.isfile(os.path.join(td, "topdown_semantic.json")):
                return False
        if not os.path.isfile(os.path.join(td, "floor_path_ssl.txt")):
            return False
    else:
        if depth and not os.path.isfile(os.path.join(td, "topdown_depth.png")):
            return False
        if semantic and not os.path.isfile(os.path.join(td, "topdown_semantic.png")):
            return False
    return True


def is_post_export_complete(y_dir: str, job: Dict[str, Any]) -> bool:
    if not job.get("holo_geometry"):
        return True
    if job.get("export_glb"):
        glb = os.path.join(y_dir, "scene.glb")
        if not os.path.isfile(glb) or os.path.getsize(glb) <= 0:
            return False
    if job.get("export_point_cloud"):
        scene_ply = os.path.join(y_dir, "pointcloud", "scene_all.ply")
        if not os.path.isfile(scene_ply) or os.path.getsize(scene_ply) <= 0:
            return False
    if job.get("export_voxel") and not _voxel_ok(y_dir):
        return False
    if not os.path.isfile(os.path.join(y_dir, "data.json")):
        return False
    return True


def load_progress(y_dir: str) -> Dict[str, Any]:
    path = os.path.join(y_dir, PROGRESS_FILENAME)
    if not os.path.isfile(path):
        return {"version": 1, "views": {}, "post": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("views", {})
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"version": 1, "views": {}, "post": {}}


def save_progress(y_dir: str, progress: Dict[str, Any]) -> None:
    path = os.path.join(y_dir, PROGRESS_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def record_view_complete(
    y_dir: str,
    view_name: str,
    *,
    view_dir: str,
    main_pngs: Sequence[str],
) -> None:
    progress = load_progress(y_dir)
    rel_dir = os.path.relpath(view_dir, y_dir)
    progress["views"][view_name] = {
        "view_dir": rel_dir,
        "main_pngs": [os.path.relpath(p, y_dir) for p in main_pngs],
        "complete": True,
    }
    save_progress(y_dir, progress)


def record_post_complete(y_dir: str) -> None:
    progress = load_progress(y_dir)
    progress["post"] = {"complete": True}
    save_progress(y_dir, progress)


def load_auto_views_manifest(y_dir: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    path = os.path.join(y_dir, "auto_views.json")
    if not os.path.isfile(path):
        return {}, []
    with open(path, "r", encoding="utf-8") as f:
        specs = json.load(f)
    if not isinstance(specs, dict):
        return {}, []
    names = list(specs.keys())
    return specs, names


def load_floor_result_from_disk(y_dir: str, subdir: str = "topdown_normalized") -> Optional[Dict[str, Any]]:
    td = os.path.join(y_dir, subdir)
    json_path = os.path.join(td, "floor_path_points.json")
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("path_points_ssl"):
            return data
    ssl_txt = os.path.join(td, "floor_path_ssl.txt")
    if not os.path.isfile(ssl_txt):
        return None
    points: List[List[float]] = []
    with open(ssl_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                points.append([float(parts[0]), float(parts[1]), 0.0])
    if not points:
        return None
    return {"path_points_ssl": points, "method": "loaded_from_floor_path_ssl.txt"}


def filter_views_to_run(
    y_dir: str,
    view_names: List[str],
    job: Dict[str, Any],
    *,
    resume: bool,
) -> Tuple[List[str], List[str]]:
    """返回 (待渲染, 已跳过)。"""
    if not resume:
        return list(view_names), []
    progress = load_progress(y_dir)
    auto_specs = job.get("auto_view_specs") or {}
    todo: List[str] = []
    skipped: List[str] = []
    for name in view_names:
        spec = auto_specs.get(name)
        if is_view_complete(y_dir, name, job, auto_spec=spec, progress=progress):
            skipped.append(name)
        else:
            todo.append(name)
    return todo, skipped
