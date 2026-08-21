"""Per-element multi-view visibility sidecars for sequence visible geometry exports."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import util
    from . import util_data
except ImportError:
    import util  # type: ignore
    import util_data  # type: ignore


def visible_viewvis_point_path(view_dir: str) -> str:
    return os.path.join(util_data.geometry_pointcloud_dir(view_dir), "scene_visible_viewvis_point.npz")


def visible_viewvis_object_path(view_dir: str) -> str:
    return os.path.join(util_data.geometry_pointcloud_dir(view_dir), "scene_visible_viewvis_object.npz")


def visible_glb_viewvis_point_tri_path(view_dir: str) -> str:
    return os.path.join(util_data.geometry_glb_dir(view_dir), "scene_visible_viewvis_point_tri.npz")


def visible_glb_viewvis_object_tri_path(view_dir: str) -> str:
    return os.path.join(util_data.geometry_glb_dir(view_dir), "scene_visible_viewvis_object_tri.npz")


def build_frame_infos(view_dir: str, image_stamps: Sequence[str]) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    for stamp in image_stamps:
        png_path = os.path.join(view_dir, f"{stamp}.png")
        depth_path = os.path.join(view_dir, f"{stamp}_depth.png")
        camera_para_path = os.path.join(view_dir, f"{stamp}_camera_para.json")
        depth_scale = None
        if os.path.isfile(camera_para_path):
            try:
                with open(camera_para_path, "r", encoding="utf-8") as f:
                    depth_scale = json.load(f).get("depth_scale")
            except (json.JSONDecodeError, OSError):
                depth_scale = None
        frames.append(
            {
                "image_stamp": stamp,
                "png_path": png_path,
                "depth_path": depth_path if os.path.isfile(depth_path) else None,
                "camera_para_path": camera_para_path,
                "depth_scale": float(depth_scale) if depth_scale is not None else None,
            }
        )
    return frames


def load_depth_meters(depth_path: str, depth_scale: float) -> Optional[np.ndarray]:
    if not depth_path or not os.path.isfile(depth_path) or depth_scale is None or depth_scale <= 0:
        return None
    try:
        depth_u16 = util.read_image_array(depth_path)
        return util.decode_depth_uint16(depth_u16, float(depth_scale))
    except Exception:
        return None


def project_points_to_camera(
    points: np.ndarray,
    scene,
    camera_obj,
    *,
    margin: float = 0.002,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, z, in_frustum) arrays aligned with Blender render/depth maps."""
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view

    pts = np.asarray(points, dtype=float)
    n = len(pts)
    u = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)
    z = np.zeros(n, dtype=float)
    in_frustum = np.zeros(n, dtype=bool)
    for i, p in enumerate(pts):
        ui, vi, zi = world_to_camera_view(scene, camera_obj, mathutils.Vector(p.tolist()))
        u[i] = float(ui)
        v[i] = float(vi)
        z[i] = float(zi)
        in_frustum[i] = (
            zi > 0.0
            and (-margin <= ui <= 1.0 + margin)
            and (-margin <= vi <= 1.0 + margin)
        )
    return u, v, z, in_frustum


def depth_occlusion_visible(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    in_frustum: np.ndarray,
    depth_m: np.ndarray,
    *,
    tolerance_m: float = 0.05,
) -> np.ndarray:
    """Point-level visibility: in frustum and not occluded according to rendered depth."""
    visible = np.zeros(len(u), dtype=bool)
    if depth_m is None or depth_m.size == 0:
        return visible
    h, w = depth_m.shape
    idx = np.flatnonzero(in_frustum)
    if len(idx) == 0:
        return visible
    px = np.clip(np.floor(u[idx] * w), 0, w - 1).astype(int)
    py = np.clip(np.floor((1.0 - v[idx]) * h), 0, h - 1).astype(int)
    rendered = depth_m[py, px]
    point_z = z[idx]
    valid = rendered > 0.0
    tol = np.maximum(float(tolerance_m), 0.02 * np.maximum(point_z, 0.0))
    match = valid & (np.abs(point_z - rendered) <= tol)
    visible[idx[match]] = True
    return visible


def compute_object_level_point_matrix(
    object_indices: np.ndarray,
    object_visible: np.ndarray,
    frustum_masks: Sequence[np.ndarray],
) -> np.ndarray:
    """Object-level per-point visibility: object passes gate AND point in frustum."""
    n_pts = len(object_indices)
    n_cams = len(frustum_masks)
    out = np.zeros((n_pts, n_cams), dtype=np.uint8)
    for ki, frustum in enumerate(frustum_masks):
        obj_ok = object_visible[object_indices, ki].astype(bool)
        out[:, ki] = (obj_ok & frustum).astype(np.uint8)
    return out


def compute_point_level_matrix(
    frustum_masks: Sequence[np.ndarray],
    depth_visible_masks: Sequence[np.ndarray],
    ray_visible_masks: Sequence[Optional[np.ndarray]],
    depth_available: Sequence[bool],
) -> np.ndarray:
    n_pts = len(frustum_masks[0])
    n_cams = len(frustum_masks)
    out = np.zeros((n_pts, n_cams), dtype=np.uint8)
    for ki in range(n_cams):
        if depth_available[ki]:
            vis = depth_visible_masks[ki]
        elif ray_visible_masks[ki] is not None:
            vis = ray_visible_masks[ki]
        else:
            vis = np.zeros(n_pts, dtype=bool)
        out[:, ki] = vis.astype(np.uint8)
    return out


def write_viewvis_sidecars(
    view_dir: str,
    point_matrix: np.ndarray,
    object_matrix: np.ndarray,
    camera_frames: Sequence[Dict[str, Any]],
    *,
    ply_relpath: str = "pointcloud/scene_visible.ply",
) -> Dict[str, Any]:
    os.makedirs(util_data.geometry_pointcloud_dir(view_dir), exist_ok=True)
    point_path = visible_viewvis_point_path(view_dir)
    object_path = visible_viewvis_object_path(view_dir)
    frame_names = [str(f.get("image_stamp", f"frame_{i}")) for i, f in enumerate(camera_frames)]
    np.savez_compressed(point_path, visibility=point_matrix.astype(np.uint8), camera_frames=np.asarray(frame_names))
    np.savez_compressed(object_path, visibility=object_matrix.astype(np.uint8), camera_frames=np.asarray(frame_names))

    meta = {
        "visibility_type_point": "point_occlusion",
        "visibility_type_object": "object_level_frustum",
        "shape": [int(point_matrix.shape[0]), int(point_matrix.shape[1])],
        "camera_frames": frame_names,
        "ply_path": ply_relpath,
        "opencv_ply_path": ply_relpath.replace(".ply", "_opencv.ply"),
        "viewvis_point": os.path.relpath(point_path, view_dir),
        "viewvis_object": os.path.relpath(object_path, view_dir),
        "notes": (
            "Row i aligns with line i+1 of scene_visible.ply / scene_visible_opencv.ply. "
            "Column j is visibility from camera_frames[j]."
        ),
    }
    meta_path = os.path.join(util_data.geometry_pointcloud_dir(view_dir), "scene_visible_viewvis.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def write_mesh_viewvis_sidecars(
    view_dir: str,
    point_matrix: np.ndarray,
    object_matrix: np.ndarray,
    camera_frames: Sequence[Dict[str, Any]],
    *,
    glb_relpath: str = "glb/scene_visible.glb",
) -> Dict[str, Any]:
    os.makedirs(util_data.geometry_glb_dir(view_dir), exist_ok=True)
    point_path = visible_glb_viewvis_point_tri_path(view_dir)
    object_path = visible_glb_viewvis_object_tri_path(view_dir)
    frame_names = [str(f.get("image_stamp", f"frame_{i}")) for i, f in enumerate(camera_frames)]
    np.savez_compressed(point_path, visibility=point_matrix.astype(np.uint8), camera_frames=np.asarray(frame_names))
    np.savez_compressed(object_path, visibility=object_matrix.astype(np.uint8), camera_frames=np.asarray(frame_names))

    meta = {
        "element_type": "triangle",
        "visibility_type_point": "point_occlusion",
        "visibility_type_object": "object_level_frustum",
        "shape": [int(point_matrix.shape[0]), int(point_matrix.shape[1])],
        "camera_frames": frame_names,
        "glb_path": glb_relpath,
        "opencv_glb_path": glb_relpath.replace(".glb", "_opencv.glb"),
        "viewvis_point": os.path.relpath(point_path, view_dir),
        "viewvis_object": os.path.relpath(object_path, view_dir),
        "notes": (
            "Row i is triangle i in scene_visible.glb face export order (same as visible_entries loop). "
            "Sample position is triangle centroid. Column j is visibility from camera_frames[j]."
        ),
    }
    meta_path = os.path.join(util_data.geometry_glb_dir(view_dir), "scene_visible_viewvis_tri.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def compute_viewvis_matrices(
    points: np.ndarray,
    object_indices: np.ndarray,
    object_visible: np.ndarray,
    camera_states: Sequence[Tuple[Any, Any]],
    frame_infos: Sequence[Dict[str, Any]],
    scene,
    *,
    project_fn: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = project_points_to_camera,
    ray_visible_fn: Optional[
        Callable[[np.ndarray, Any, Any, np.ndarray], np.ndarray]
    ] = None,
    depth_tolerance_m: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n_cams = len(camera_states)
    frustum_masks: List[np.ndarray] = []
    depth_visible_masks: List[np.ndarray] = []
    ray_visible_masks: List[Optional[np.ndarray]] = []
    depth_available: List[bool] = []

    for ki, (camera_obj, _transparent) in enumerate(camera_states):
        u, v, z, in_frustum = project_fn(points, scene, camera_obj)
        frustum_masks.append(in_frustum)

        frame = frame_infos[ki] if ki < len(frame_infos) else {}
        depth_m = None
        if frame.get("depth_path") and frame.get("depth_scale"):
            depth_m = load_depth_meters(frame["depth_path"], float(frame["depth_scale"]))
        has_depth = depth_m is not None
        depth_available.append(has_depth)

        depth_vis = (
            depth_occlusion_visible(u, v, z, in_frustum, depth_m, tolerance_m=depth_tolerance_m)
            if has_depth
            else np.zeros(len(points), dtype=bool)
        )
        depth_visible_masks.append(depth_vis)

        ray_vis = None
        if not has_depth and ray_visible_fn is not None:
            ray_vis = ray_visible_fn(points, camera_obj, _transparent, in_frustum)
        ray_visible_masks.append(ray_vis)

    object_matrix = compute_object_level_point_matrix(object_indices, object_visible, frustum_masks)
    point_matrix = compute_point_level_matrix(
        frustum_masks, depth_visible_masks, ray_visible_masks, depth_available
    )
    meta = {
        "depth_frames": int(sum(depth_available)),
        "raycast_frames": sum(1 for rv in ray_visible_masks if rv is not None),
    }
    return point_matrix, object_matrix, meta
