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


def _frame_artifacts_ok(
    png_path: str,
    *,
    semantic: bool,
    depth: bool,
) -> bool:
    view_dir = os.path.dirname(png_path)
    base = os.path.splitext(os.path.basename(png_path))[0]
    if depth:
        if not os.path.isfile(os.path.join(view_dir, f"{base}_depth.png")):
            return False
        if not os.path.isfile(os.path.join(view_dir, f"{base}_normal.png")):
            return False
    if semantic:
        if not os.path.isfile(os.path.join(view_dir, f"{base}_semantic.png")):
            return False
        if not os.path.isfile(os.path.join(view_dir, f"{base}_semantic.json")):
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


def _visible_geometry_ok(view_dir: str, *, export_glb: bool, export_point_cloud: bool) -> bool:
    if export_glb and not os.path.isfile(os.path.join(view_dir, "scene_visible.glb")):
        return False
    if export_point_cloud:
        pc_dir = os.path.join(view_dir, "pointcloud")
        if not os.path.isdir(pc_dir):
            return False
        has_ply = any(name.endswith(".ply") for name in os.listdir(pc_dir))
        if not has_ply and not os.path.isfile(os.path.join(view_dir, "planar_faces.json")):
            return False
    return True


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
    semantic = bool(job.get("semantic"))
    depth = bool(job.get("depth"))
    pano = bool(job.get("pano"))
    export_glb = bool(job.get("export_glb"))
    export_point_cloud = bool(job.get("export_point_cloud"))
    visible_geometry = bool(job.get("visible_geometry"))
    view_cameras = job.get("view_cameras") or {}

    if view_name == "topdown":
        view_dir = os.path.join(y_dir, "topdown")
        png = os.path.join(view_dir, "topdown.png")
        if not os.path.isfile(png) or os.path.getsize(png) <= 0:
            return False
        if not os.path.isfile(os.path.join(view_dir, "topdown_camera_para.json")):
            return False
        if not os.path.isfile(os.path.join(view_dir, "ssl_opencv.txt")):
            return False
        if not _frame_artifacts_ok(png, semantic=semantic, depth=depth):
            return False
        if visible_geometry and (export_glb or export_point_cloud):
            if not _visible_geometry_ok(view_dir, export_glb=export_glb, export_point_cloud=export_point_cloud):
                return False
        return True

    view_dir = view_dir_for(y_dir, view_name, progress)
    if not os.path.isdir(view_dir):
        return False

    n_expected = expected_frame_count(view_name, view_cameras, auto_spec)
    frames = list_primary_frames(view_dir)
    if len(frames) < n_expected:
        return False
    if not all(_frame_artifacts_ok(p, semantic=semantic, depth=depth) for p in frames):
        return False
    if not os.path.isfile(os.path.join(view_dir, "ssl_opencv.txt")):
        return False
    if export_point_cloud and not pano:
        if not os.path.isfile(os.path.join(view_dir, "planar_faces.json")):
            has_lines = any(n.endswith("_lines.png") for n in os.listdir(view_dir))
            if not has_lines:
                return False
    if pano and not _pano_complete(view_dir, semantic=semantic, depth=depth):
        return False
    if visible_geometry and (export_glb or export_point_cloud):
        if not _visible_geometry_ok(view_dir, export_glb=export_glb, export_point_cloud=export_point_cloud):
            return False
    return True


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
    if job.get("export_glb"):
        glb = os.path.join(y_dir, "scene.glb")
        if not os.path.isfile(glb) or os.path.getsize(glb) <= 0:
            return False
    if job.get("export_point_cloud"):
        scene_ply = os.path.join(y_dir, "pointcloud", "scene_all.ply")
        if not os.path.isfile(scene_ply) or os.path.getsize(scene_ply) <= 0:
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
