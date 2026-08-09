"""Geometry export: view SSL coordinates + OpenCV camera-coordinate copies (+X right, +Y down, +Z forward)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

CameraPose = Dict[str, List[float]]
SslRef = Tuple[List[float], List[float]]

DEFAULT_RENDER_VIEW_UP = (0.0, 0.0, 1.0)
TOPDOWN_FALLBACK_UP = (0.0, 1.0, 0.0)
_COLLINEAR_EPS = 1e-6


def resolve_render_view_up_vector(
    camera_position,
    look_at_target,
    up_vector=None,
) -> List[float]:
    """Resolve the up direction for render_view.

    Defaults to ``[0,0,1]`` when ``up_vector`` is not set; if the view direction is
    collinear with ``[0,0,1]`` (e.g. top-down), falls back to ``[0,1,0]`` from
    ``topdown_view`` to avoid forward/up degeneracy.
    Returns ``up_vector`` unchanged when explicitly provided.
    """
    if up_vector is not None:
        return [float(x) for x in up_vector]

    cam = np.asarray(camera_position, dtype=float)
    look = np.asarray(look_at_target, dtype=float)
    forward = look - cam
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < _COLLINEAR_EPS:
        return list(DEFAULT_RENDER_VIEW_UP)

    forward = forward / forward_norm
    default_up = np.asarray(DEFAULT_RENDER_VIEW_UP, dtype=float)
    if abs(float(np.dot(forward, default_up))) > 1.0 - _COLLINEAR_EPS:
        return list(TOPDOWN_FALLBACK_UP)
    return list(DEFAULT_RENDER_VIEW_UP)


def opencv_duplicate_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}_opencv{ext}"


def make_export_camera_pose(
    camera_position,
    look_at_target,
    world_up=(0.0, 0.0, 1.0),
) -> CameraPose:
    return {
        "camera_position": [float(x) for x in camera_position],
        "look_at_target": [float(x) for x in look_at_target],
        "world_up": [float(x) for x in world_up],
    }


def _view_ssl_params(camera_position, look_at_target):
    try:
        from .util_data import view_ssl_transform_params
    except ImportError:
        from util_data import view_ssl_transform_params  # type: ignore
    return view_ssl_transform_params(camera_position, look_at_target)


def _transform_xyz_view_ssl(xyz, origin_xy, theta: float):
    try:
        from .util_data import _transform_xyz_view_ssl
    except ImportError:
        from util_data import _transform_xyz_view_ssl  # type: ignore
    return _transform_xyz_view_ssl(xyz, origin_xy, theta)


def _transform_direction_view_ssl(xyz, theta: float):
    try:
        from .util_data import _transform_direction_view_ssl
    except ImportError:
        from util_data import _transform_direction_view_ssl  # type: ignore
    return _transform_direction_view_ssl(xyz, theta)


def _reference_view_camera_pose(world_camera, world_look_at, origin_xy, theta: float):
    try:
        from .util_data import reference_view_camera_pose
    except ImportError:
        from util_data import reference_view_camera_pose  # type: ignore
    return reference_view_camera_pose(world_camera, world_look_at, origin_xy, theta)


def view_ssl_to_blender_gltf_vertices(points: np.ndarray) -> np.ndarray:
    """View SSL Z-up (x,y,z) → Blender mesh vertices so glTF export matches PLY.

    Blender Z-up exports to glTF Y-up: (bx, by, bz) → (bx, bz, -by).
    Target glTF = view SSL, so bpy vertices should be (vx, -vz, vy).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        return np.array([pts[0], -pts[2], pts[1]], dtype=float)
    return np.column_stack([pts[..., 0], -pts[..., 2], pts[..., 1]])


def build_view_ssl_transform_matrix(origin_xy, theta: float) -> np.ndarray:
    """4×4 transform from world/scene SSL → view SSL (same as transform_points_to_view_ssl)."""
    origin_xy = np.asarray(origin_xy, dtype=float)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    T = np.eye(4, dtype=float)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = -c * origin_xy[0] + s * origin_xy[1]
    T[1, 3] = -s * origin_xy[0] - c * origin_xy[1]
    return T


def resolve_visible_ssl_frames(
    ssl_world_camera,
    ssl_world_look_at,
    ssl_world_up=(0.0, 0.0, 1.0),
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[CameraPose]]:
    if ssl_world_camera is None or ssl_world_look_at is None:
        return None, None, None
    return build_ply_export_frames(
        ssl_world_camera,
        ssl_world_look_at,
        ssl_world_up if ssl_world_up is not None else [0.0, 0.0, 1.0],
    )


def build_ply_export_frames(
    world_camera,
    look_at_target,
    world_up=(0.0, 0.0, 1.0),
) -> Tuple[np.ndarray, float, CameraPose]:
    """Build view SSL params and first-frame OpenCV pose for PLY export from world camera pose (both in SSL)."""
    origin_xy, theta = _view_ssl_params(world_camera, look_at_target)
    ref_cam, ref_look = _reference_view_camera_pose(
        world_camera, look_at_target, origin_xy, theta
    )
    up = np.asarray(world_up if world_up is not None else [0.0, 0.0, 1.0], dtype=float)
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-12:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up = up / up_norm
    up_ssl = _transform_direction_view_ssl(up, theta)
    opencv_pose = make_export_camera_pose(ref_cam, ref_look, up_ssl)
    return np.asarray(origin_xy, dtype=float), float(theta), opencv_pose


def transform_points_to_view_ssl(
    points: np.ndarray,
    origin_xy,
    theta: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3)
    out = np.empty_like(pts)
    for i, p in enumerate(pts):
        out[i] = _transform_xyz_view_ssl(p, origin_xy, theta)
    return out


def transform_normals_to_view_ssl(
    normals: np.ndarray,
    theta: float,
) -> np.ndarray:
    nrm = np.asarray(normals, dtype=float)
    if nrm.size == 0:
        return nrm.reshape(0, 3)
    out = np.empty_like(nrm)
    for i, n in enumerate(nrm):
        out[i] = _transform_direction_view_ssl(n, theta)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norms, out=np.zeros_like(out), where=norms > 1e-12)


def build_blender_camera_matrix(
    camera_position,
    look_at_target,
    world_up=(0.0, 0.0, 1.0),
) -> np.ndarray:
    """Same 4×4 camera matrix as scenebuilder_bpy._build_view_camera_matrix_and_fov (column basis: right, up, -forward)."""
    camera_position = np.asarray(camera_position, dtype=float)
    look_at_target = np.asarray(look_at_target, dtype=float)
    up_vector = np.asarray(world_up if world_up is not None else [0.0, 0.0, 1.0], dtype=float)
    if np.linalg.norm(up_vector) < 1e-6:
        up_vector = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up_vector = up_vector / np.linalg.norm(up_vector)

    forward = look_at_target - camera_position
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([0.0, 0.0, -1.0], dtype=float)
    else:
        forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up_vector)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(forward, fallback_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            fallback_up = np.array([1.0, 0.0, 0.0], dtype=float)
            right = np.cross(forward, fallback_up)
            right_norm = np.linalg.norm(right)
    right = right / max(right_norm, 1e-6)
    up = np.cross(right, forward)

    mat = np.eye(4, dtype=float)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -forward
    mat[:3, 3] = camera_position
    return mat


def opencv_rotation_from_pose(
    camera_position,
    look_at_target,
    world_up=(0.0, 0.0, 1.0),
) -> np.ndarray:
    """3×3 rotation R from view SSL / world → OpenCV camera coords, with p_cam = R @ (p - eye).

    OpenCV: +X image right, +Y image down, +Z along view ray into the scene.
    Matches the Blender render camera matrix (extracted directly from matrix columns).
    """
    mat = build_blender_camera_matrix(camera_position, look_at_target, world_up)
    return opencv_rotation_from_blender_matrix(mat)


def opencv_transform_matrix(camera_pose: CameraPose) -> np.ndarray:
    eye = np.asarray(camera_pose["camera_position"], dtype=float)
    R = opencv_rotation_from_pose(
        eye,
        camera_pose["look_at_target"],
        camera_pose.get("world_up", [0.0, 0.0, 1.0]),
    )
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    return T


def transform_points_to_opencv(
    points: np.ndarray,
    camera_pose: CameraPose,
) -> np.ndarray:
    eye = np.asarray(camera_pose["camera_position"], dtype=float)
    R = opencv_rotation_from_pose(
        eye,
        camera_pose["look_at_target"],
        camera_pose.get("world_up", [0.0, 0.0, 1.0]),
    )
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return (R @ (pts - eye).T).T


def transform_normals_to_opencv(
    normals: np.ndarray,
    camera_pose: CameraPose,
) -> np.ndarray:
    R = opencv_rotation_from_pose(
        camera_pose["camera_position"],
        camera_pose["look_at_target"],
        camera_pose.get("world_up", [0.0, 0.0, 1.0]),
    )
    nrm = np.asarray(normals, dtype=float)
    if nrm.size == 0:
        return nrm.reshape(0, 3)
    out = (R @ nrm.T).T
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norms, out=np.zeros_like(out), where=norms > 1e-12)


def write_ply(
    path: str,
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    *,
    ssl_origin_xy: Optional[np.ndarray] = None,
    ssl_theta: Optional[float] = None,
    opencv_camera_pose: Optional[CameraPose] = None,
    points_in_view_ssl: bool = False,
    camera_pose: Optional[CameraPose] = None,
) -> None:
    """Write PLY: main file in view SSL; optional _opencv copy (first-frame OpenCV camera frame).

    - By default input points are in Blender world coords and are transformed to view SSL before export.
    - If points_in_view_ssl=True, input is already in view SSL and no SSL transform is applied.
    - opencv_camera_pose should be in view SSL (first-frame reference camera at (0,0,h)).
    """
    if opencv_camera_pose is None and camera_pose is not None:
        opencv_camera_pose = camera_pose

    pts = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)

    if not points_in_view_ssl and ssl_origin_xy is not None and ssl_theta is not None:
        pts = transform_points_to_view_ssl(pts, ssl_origin_xy, ssl_theta)
        nrm = transform_normals_to_view_ssl(nrm, ssl_theta)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, normal, color in zip(pts, nrm, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )

    if opencv_camera_pose is not None:
        opencv_path = opencv_duplicate_path(path)
        pts_o = transform_points_to_opencv(pts, opencv_camera_pose)
        nrm_o = transform_normals_to_opencv(nrm, opencv_camera_pose)
        write_ply(
            opencv_path,
            pts_o,
            colors,
            nrm_o,
            points_in_view_ssl=True,
        )
        print(f"✅ OpenCV point cloud copy: {opencv_path}")


def gltf_yup_to_blender_zup_matrix() -> np.ndarray:
    """glTF Y-up (Blender-exported GLB) → Blender Z-up / SSL world coordinates.

    Inverse of ``view_ssl_to_blender_gltf_vertices``:
    Blender (bx, by, bz) exports to glTF as (gx, gy, gz) = (bx, bz, -by).
    """
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def gltf_yup_to_blender_zup(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return np.column_stack([pts[..., 0], -pts[..., 2], pts[..., 1]])


def opencv_transform_matrix_for_gltf_vertices(camera_pose: CameraPose) -> np.ndarray:
    """OpenCV transform for already-exported glTF Y-up GLB vertices (first restore to Blender world coords)."""
    return opencv_transform_matrix(camera_pose) @ gltf_yup_to_blender_zup_matrix()


def export_glb_opencv_copy(src_path: str, camera_pose: CameraPose) -> Optional[str]:
    """Generate OpenCV copy from main GLB (fallback: restore glTF Y-up to Blender world coords first)."""
    if not os.path.isfile(src_path):
        return None
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Exporting OpenCV GLB copy requires trimesh") from exc

    dst_path = opencv_duplicate_path(src_path)
    loaded = trimesh.load(src_path, force="scene")
    transform = opencv_transform_matrix_for_gltf_vertices(camera_pose)
    if isinstance(loaded, trimesh.Scene):
        loaded.apply_transform(transform)
    elif isinstance(loaded, trimesh.Trimesh):
        loaded = loaded.copy()
        loaded.apply_transform(transform)
        scene = trimesh.Scene()
        scene.add_geometry(loaded)
        loaded = scene
    else:
        raise RuntimeError(f"Unrecognized GLB type: {type(loaded)}")

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    loaded.export(dst_path)
    print(f"✅ OpenCV GLB copy: {dst_path}")
    return dst_path


def opencv_rotation_from_blender_matrix(camera_matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(camera_matrix, dtype=float)
    right = mat[:3, 0]
    down = -mat[:3, 1]
    forward = -mat[:3, 2]
    return np.stack([right, down, forward], axis=0)


def build_opencv_c2w_matrix(
    camera_position,
    look_at_target,
    world_up=(0.0, 0.0, 1.0),
) -> np.ndarray:
    """4×4 c2w from OpenCV camera frame → SSL/Blender world frame (ScanNet / OpenSpatial style).

    Column vectors are camera +X (right), +Y (down), +Z (forward/depth) in world frame; translation t is optical center in meters.
    """
    mat = build_blender_camera_matrix(camera_position, look_at_target, world_up)
    right = mat[:3, 0]
    down = -mat[:3, 1]
    forward = -mat[:3, 2]
    c2w = np.eye(4, dtype=float)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = np.asarray(camera_position, dtype=float)
    return c2w


def camera_pose_from_matrix(camera_matrix: np.ndarray) -> CameraPose:
    mat = np.asarray(camera_matrix, dtype=float)
    eye = mat[:3, 3]
    forward = -mat[:3, 2]
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-12:
        forward = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        forward = forward / forward_norm
    look = eye + forward
    return make_export_camera_pose(eye, look, mat[:3, 1])
