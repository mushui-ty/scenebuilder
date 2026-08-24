"""render_ssl resume: detect complete artifacts by logical view name and skip finished items.

Timestamps appear only in PNG filenames inside view dirs; directory names for preset/auto views are fixed
(e.g. ``front``, ``left_seq``, ``auto_path_0004``) for resume.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from . import util_data
except ImportError:
    import util_data  # type: ignore

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
    """Logical view → output subdirectory name (same as view_dir_name in worker)."""
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
    """Return primary render PNG paths that have a matching ``*_camera_para.json``."""
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


def _pano_dir_for(view_dir: str) -> str:
    parent = os.path.dirname(view_dir.rstrip(os.sep)) or view_dir
    return os.path.join(parent, f"{os.path.basename(view_dir.rstrip(os.sep))}_pano")


def _pano_complete(
    view_dir: str,
    *,
    semantic: bool,
    depth: bool,
) -> bool:
    frames = list_primary_frames(_pano_dir_for(view_dir))
    if not frames:
        return False
    return all(_frame_artifacts_ok(p, semantic=semantic, depth=depth) for p in frames)


def _pano_planar_ok(view_dir: str, *, n_expected: int = 1) -> bool:
    frames = list_primary_frames(_pano_dir_for(view_dir))
    if len(frames) < n_expected:
        return False
    return all(_planar_frame_ok(p) for p in frames)


def _visible_glb_ok(view_dir: str) -> bool:
    merged = util_data.visible_glb_path(view_dir)
    if os.path.isfile(merged) and os.path.getsize(merged) > 1024:
        return True
    # Legacy: GLB at view root (pre glb/ subdir).
    legacy = os.path.join(view_dir, "scene_visible.glb")
    if os.path.isfile(legacy) and os.path.getsize(legacy) > 1024:
        return True
    manifest_path = util_data.visible_glb_manifest_path(view_dir)
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        objects = manifest.get("objects") or []
        if not objects:
            return False
        glb_root = util_data.geometry_glb_dir(view_dir)
        for item in objects:
            rel = item.get("path")
            if not rel:
                return False
            part = os.path.join(glb_root, rel)
            if not os.path.isfile(part) or os.path.getsize(part) <= 0:
                return False
        return True
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def _visible_ply_ok(view_dir: str) -> bool:
    merged = util_data.visible_merged_ply_path(view_dir)
    if os.path.isfile(merged) and os.path.getsize(merged) > 0:
        return True
    pc_dir = util_data.geometry_pointcloud_dir(view_dir)
    if os.path.isdir(pc_dir):
        for _root, _dirs, files in os.walk(pc_dir):
            if any(name.endswith(".ply") for name in files):
                return True
    # Legacy: merged PLY at view root, or per-object PLY scattered under view_dir.
    scene_ply = os.path.join(view_dir, "scene_visible.ply")
    if os.path.isfile(scene_ply) and os.path.getsize(scene_ply) > 0:
        return True
    return False


def _voxel_ok(view_dir: str) -> bool:
    return os.path.isfile(os.path.join(view_dir, "voxel", "occupancy_world_meta.json"))


def _visibility_json_ok(view_dir: str) -> bool:
    return os.path.isfile(os.path.join(view_dir, "visibility.json"))


def _planar_frame_ok(png_path: str) -> bool:
    view_dir = os.path.dirname(png_path)
    base = os.path.splitext(os.path.basename(png_path))[0]
    if os.path.isfile(os.path.join(view_dir, f"{base}_planar_faces.json")):
        return True
    return os.path.isfile(os.path.join(view_dir, f"{base}_lines.png"))


def _planar_faces_ok(view_dir: str) -> bool:
    if os.path.isfile(os.path.join(view_dir, "planar_faces.json")):
        return True
    if not os.path.isdir(view_dir):
        return False
    return any(n.endswith("_lines.png") for n in os.listdir(view_dir))


def _planar_frames_ok(
    view_dir: str,
    view_name: str,
    *,
    n_expected: int = 1,
) -> bool:
    if view_name == "topdown":
        png = os.path.join(view_dir, "topdown.png")
        if not os.path.isfile(png):
            return False
        return _planar_frame_ok(png)
    frames = list_primary_frames(view_dir)
    if len(frames) < n_expected:
        return False
    return all(_planar_frame_ok(p) for p in frames)


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


def _is_video_view(auto_spec: Optional[Dict[str, Any]] = None) -> bool:
    return bool((auto_spec or {}).get("video"))


def _is_tangent_video_view(auto_spec: Optional[Dict[str, Any]] = None) -> bool:
    if not _is_video_view(auto_spec):
        return False
    trajectory = (auto_spec or {}).get("video_trajectory")
    if trajectory == "tangent":
        return True
    if trajectory is None and (auto_spec or {}).get("pano_only"):
        return True
    return False


def _is_center_video_view(auto_spec: Optional[Dict[str, Any]] = None) -> bool:
    return _is_video_view(auto_spec) and (auto_spec or {}).get("video_trajectory") == "center"


def _auto_path_needs_pano(
    view_name: str,
    job: Dict[str, Any],
    auto_spec: Optional[Dict[str, Any]] = None,
) -> bool:
    """Whether this auto_path single view should render panorama artifacts."""
    if not bool(job.get("pano")):
        return False
    if not view_name.startswith("auto_path_") or view_name.endswith("_seq"):
        return False
    if auto_spec is not None and "render_pano" in auto_spec:
        return bool(auto_spec.get("render_pano"))
    try:
        from .auto_views import select_auto_path_pano_view_names
    except ImportError:
        from auto_views import select_auto_path_pano_view_names  # type: ignore
    auto_specs = job.get("auto_view_specs") or {}
    selected = select_auto_path_pano_view_names(
        auto_specs,
        int(job.get("pano_num", 3)),
    )
    return view_name in selected


def _artifact_view_dir(view_dir: str, auto_spec: Optional[Dict[str, Any]] = None) -> str:
    """Primary geometry directory (perspective sequence root for all video views)."""
    return view_dir


def missing_view_artifacts(
    y_dir: str,
    view_name: str,
    job: Dict[str, Any],
    *,
    auto_spec: Optional[Dict[str, Any]] = None,
    progress: Optional[Dict[str, Any]] = None,
) -> set:
    """Return artifact keys still missing for this view: render / glb / ply / planar / voxel / ssl / pano."""
    missing: set = set()
    is_video = _is_video_view(auto_spec)
    is_tangent_video = _is_tangent_video_view(auto_spec)
    is_center_video = _is_center_video_view(auto_spec)
    semantic = bool(job.get("semantic"))
    depth = bool(job.get("depth"))
    pano = bool(job.get("pano"))
    export_glb = bool(job.get("export_glb"))
    export_point_cloud = bool(job.get("export_point_cloud"))
    export_voxel = bool(job.get("export_voxel"))
    visible_geometry = bool(job.get("visible_geometry"))
    view_cameras = job.get("view_cameras") or {}

    view_dir = view_dir_for(y_dir, view_name, progress)
    artifact_dir = _artifact_view_dir(view_dir, auto_spec)
    n_expected = 1 if view_name == "topdown" else expected_frame_count(
        view_name, view_cameras, auto_spec
    )

    if is_tangent_video:
        pano_only = bool((auto_spec or {}).get("pano_only"))
        if pano_only:
            if not _pano_complete(view_dir, semantic=semantic, depth=depth):
                missing.add("pano")
        else:
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
            if not _pano_complete(view_dir, semantic=False, depth=False):
                missing.add("pano")
    elif is_center_video:
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
        if pano and not _pano_complete(view_dir, semantic=semantic, depth=depth):
            missing.add("pano")
    else:
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

    ssl_dir = view_dir
    if is_tangent_video and bool((auto_spec or {}).get("pano_only")):
        ssl_dir = _pano_dir_for(view_dir)
    if not os.path.isfile(os.path.join(ssl_dir, "ssl_opencv.txt")):
        missing.add("ssl")

    if visible_geometry and export_glb and not _visible_glb_ok(artifact_dir):
        missing.add("glb")
    if visible_geometry and export_point_cloud and not _visible_ply_ok(artifact_dir):
        missing.add("ply")
    if export_point_cloud and not is_video and not _planar_frames_ok(
        view_dir, view_name, n_expected=n_expected
    ):
        missing.add("planar")
    if visible_geometry and export_voxel and not _voxel_ok(artifact_dir):
        missing.add("voxel")
    if (
        export_point_cloud
        and (
            is_tangent_video
            or (is_center_video and pano)
        )
        and not _pano_planar_ok(view_dir, n_expected=n_expected)
    ):
        missing.add("pano_planar")
    # topdown view worker pops pano and never produces a pano directory
    if pano and view_name != "topdown" and not is_video:
        if _auto_path_needs_pano(view_name, job, auto_spec):
            if not _pano_complete(view_dir, semantic=semantic, depth=depth):
                missing.add("pano")
            elif export_point_cloud and not _pano_planar_ok(view_dir, n_expected=n_expected):
                missing.add("pano_planar")
    return missing


def apply_partial_resume_to_kwargs(job: Dict[str, Any], missing: set) -> Dict[str, Any]:
    """Build partial render kwargs from missing artifacts (includes skip_render, etc.)."""
    visible_geometry = bool(job.get("visible_geometry"))
    want_glb = bool(job.get("export_glb"))
    want_ply = bool(job.get("export_point_cloud"))
    want_voxel = bool(job.get("export_voxel"))

    export_glb = want_glb and visible_geometry and ("glb" in missing)
    export_voxel = want_voxel and visible_geometry and ("voxel" in missing)
    export_visible_point_cloud = (
        want_ply and visible_geometry and ("ply" in missing)
    )
    export_planar_faces = want_ply and (
        ("planar" in missing) or ("pano_planar" in missing)
    )

    return {
        "export_glb": export_glb,
        "export_voxel": export_voxel,
        "export_point_cloud": export_visible_point_cloud or export_planar_faces,
        "export_visible_point_cloud": export_visible_point_cloud,
        "export_planar_faces": export_planar_faces,
        "visible_geometry": visible_geometry,
        "render_semantic": (bool(job.get("semantic")) or visible_geometry) and ("semantic" in missing),
        "render_depth": bool(job.get("depth")) and ("render" in missing),
        "pano": bool(job.get("pano")) and (
            ("pano" in missing) or ("pano_planar" in missing)
        ),
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
    """Check whether a single logical view is fully rendered per current job flags."""
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
        glb = util_data.holo_glb_path(y_dir)
        if not os.path.isfile(glb) or os.path.getsize(glb) <= 0:
            legacy = os.path.join(y_dir, "scene.glb")
            if not os.path.isfile(legacy) or os.path.getsize(legacy) <= 0:
                return False
    if job.get("export_point_cloud"):
        scene_ply = util_data.holo_scene_all_ply_path(y_dir)
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
    """Return (views to render, views skipped)."""
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
