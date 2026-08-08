"""Colored occupancy voxel grid export (256³) + optional PLY preview for visualization."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_RESOLUTION = 256
VOXEL_FORMAT = "colored_occupancy_voxel_grid"
VOXEL_VERSION = 1
# 大三角形 surface splat 上限；过大时单视角体素化会明显变慢
MAX_SAMPLES_PER_TRIANGLE = 4096
BRUTEFORCE_VOXEL_LIMIT = 4096

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


if _HAS_NUMBA:

    @njit(cache=True)
    def _point_in_triangle_3d_numba(point, tri):
        v0 = tri[0]
        v1 = tri[1]
        v2 = tri[2]
        u = v1 - v0
        v = v2 - v0
        w = point - v0
        d00 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
        d01 = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
        d11 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        d20 = w[0] * u[0] + w[1] * u[1] + w[2] * u[2]
        d21 = w[0] * v[0] + w[1] * v[1] + w[2] * v[2]
        denom = d00 * d11 - d01 * d01
        eps = 1e-8
        if abs(denom) < eps:
            return False
        v_coord = (d11 * d20 - d01 * d21) / denom
        w_coord = (d00 * d21 - d01 * d20) / denom
        return v_coord >= -eps and w_coord >= -eps and (v_coord + w_coord) <= 1.0 + eps

    @njit(cache=True)
    def _triangle_plane_normal_numba(tri):
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]) ** 0.5
        if norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return normal / norm

    @njit(cache=True)
    def _voxel_intersects_triangle_numba(i, j, k, tri, corner_min, voxel_size):
        center = corner_min + (np.array([i, j, k], dtype=np.float64) + 0.5) * voxel_size
        half = voxel_size * 0.5
        voxel_min = center - half
        voxel_max = center + half
        tri_mean = (tri[0] + tri[1] + tri[2]) / 3.0
        if (
            tri_mean[0] >= voxel_min[0]
            and tri_mean[0] <= voxel_max[0]
            and tri_mean[1] >= voxel_min[1]
            and tri_mean[1] <= voxel_max[1]
            and tri_mean[2] >= voxel_min[2]
            and tri_mean[2] <= voxel_max[2]
        ):
            return True
        normal = _triangle_plane_normal_numba(tri)
        nlen = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]) ** 0.5
        if nlen < 1e-12:
            return False
        diff = center - tri[0]
        plane_dist = abs(diff[0] * normal[0] + diff[1] * normal[1] + diff[2] * normal[2])
        if plane_dist > voxel_size * 0.8660254:
            return False
        proj = center - (diff[0] * normal[0] + diff[1] * normal[1] + diff[2] * normal[2]) * normal
        if _point_in_triangle_3d_numba(proj, tri):
            return True
        for sx in (0.0, 1.0):
            for sy in (0.0, 1.0):
                for sz in (0.0, 1.0):
                    p = voxel_min + np.array([sx, sy, sz], dtype=np.float64) * voxel_size
                    if _point_in_triangle_3d_numba(p, tri):
                        return True
        return False

    @njit(cache=True)
    def _bruteforce_triangle_voxels_numba(tri, color, corner_min, voxel_size, resolution, occupancy, rgb):
        tri_min = np.empty(3, dtype=np.float64)
        tri_max = np.empty(3, dtype=np.float64)
        for axis in range(3):
            tri_min[axis] = min(tri[0, axis], tri[1, axis], tri[2, axis])
            tri_max[axis] = max(tri[0, axis], tri[1, axis], tri[2, axis])
        idx_lo = np.floor((tri_min - corner_min) / voxel_size).astype(np.int64)
        idx_hi = np.floor((tri_max - corner_min) / voxel_size).astype(np.int64)
        for axis in range(3):
            if idx_lo[axis] < 0:
                idx_lo[axis] = 0
            if idx_hi[axis] >= resolution:
                idx_hi[axis] = resolution - 1
        for i in range(int(idx_lo[0]), int(idx_hi[0]) + 1):
            for j in range(int(idx_lo[1]), int(idx_hi[1]) + 1):
                for k in range(int(idx_lo[2]), int(idx_hi[2]) + 1):
                    if not _voxel_intersects_triangle_numba(i, j, k, tri, corner_min, voxel_size):
                        continue
                    occupancy[i, j, k] = 1
                    rgb[i, j, k, 0] = color[0]
                    rgb[i, j, k, 1] = color[1]
                    rgb[i, j, k, 2] = color[2]


def compute_grid_params(
    points: np.ndarray,
    resolution: int = DEFAULT_RESOLUTION,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Return center, span, voxel_size, corner_min for a cubic grid covering points."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        raise ValueError("Cannot build voxel grid from empty geometry")
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    extents = max_pt - min_pt
    span = float(max(extents.max(), 1e-6))
    center = (min_pt + max_pt) * 0.5
    voxel_size = span / float(resolution)
    corner_min = center - span * 0.5
    return center, span, voxel_size, corner_min


def _point_in_triangle_3d(point: np.ndarray, tri: np.ndarray, eps: float = 1e-8) -> bool:
    v0, v1, v2 = tri
    u = v1 - v0
    v = v2 - v0
    w = point - v0
    d00 = float(np.dot(u, u))
    d01 = float(np.dot(u, v))
    d11 = float(np.dot(v, v))
    d20 = float(np.dot(w, u))
    d21 = float(np.dot(w, v))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < eps:
        return False
    v_coord = (d11 * d20 - d01 * d21) / denom
    w_coord = (d00 * d21 - d01 * d20) / denom
    return v_coord >= -eps and w_coord >= -eps and (v_coord + w_coord) <= 1.0 + eps


def _triangle_plane_normal(tri: np.ndarray) -> np.ndarray:
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        return np.zeros(3, dtype=float)
    return normal / norm


def _voxel_index_bounds(
    tri: np.ndarray,
    corner_min: np.ndarray,
    voxel_size: float,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    idx_lo = np.floor((tri.min(axis=0) - corner_min) / voxel_size).astype(int)
    idx_hi = np.floor((tri.max(axis=0) - corner_min) / voxel_size).astype(int)
    idx_lo = np.clip(idx_lo, 0, resolution - 1)
    idx_hi = np.clip(idx_hi, 0, resolution - 1)
    return idx_lo, idx_hi


def _voxel_intersects_triangle(
    i: int,
    j: int,
    k: int,
    tri: np.ndarray,
    corner_min: np.ndarray,
    voxel_size: float,
) -> bool:
    center = corner_min + (np.array([i, j, k], dtype=float) + 0.5) * voxel_size
    half = voxel_size * 0.5
    voxel_min = center - half
    voxel_max = center + half

    if np.all(tri.mean(axis=0) >= voxel_min) and np.all(tri.mean(axis=0) <= voxel_max):
        return True

    normal = _triangle_plane_normal(tri)
    if float(np.linalg.norm(normal)) < 1e-12:
        return False

    plane_dist = abs(float(np.dot(center - tri[0], normal)))
    if plane_dist > voxel_size * 0.8660254:
        return False

    proj = center - np.dot(center - tri[0], normal) * normal
    if _point_in_triangle_3d(proj, tri):
        return True

    for sx, sy, sz in (
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0),
    ):
        p = voxel_min + np.array([sx, sy, sz], dtype=float) * voxel_size
        if _point_in_triangle_3d(p, tri):
            return True
    return False


def _triangle_areas(tris: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) * 0.5


def _sample_points_on_triangles(
    tris: np.ndarray,
    sample_counts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    total = int(sample_counts.sum())
    if total <= 0:
        return np.empty((0, 3), dtype=float)

    tri_idx = np.repeat(np.arange(len(tris)), sample_counts)
    r1 = np.sqrt(rng.random(total))
    r2 = rng.random(total)
    chosen = tris[tri_idx]
    v0 = chosen[:, 0]
    v1 = chosen[:, 1]
    v2 = chosen[:, 2]
    return (1.0 - r1)[:, None] * v0 + (r1 * (1.0 - r2))[:, None] * v1 + (r1 * r2)[:, None] * v2


def _splat_points_to_grid(
    points: np.ndarray,
    point_colors: np.ndarray,
    corner_min: np.ndarray,
    voxel_size: float,
    resolution: int,
    occupancy: np.ndarray,
    rgb: np.ndarray,
) -> None:
    if len(points) == 0:
        return
    idx = np.floor((points - corner_min) / voxel_size).astype(np.int64)
    np.clip(idx, 0, resolution - 1, out=idx)
    linear = idx[:, 0] * resolution * resolution + idx[:, 1] * resolution + idx[:, 2]
    flat_occ = occupancy.ravel()
    flat_rgb = rgb.reshape(-1, 3)
    flat_occ[linear] = 1
    flat_rgb[linear] = point_colors


def _bruteforce_triangle_voxels(
    tri: np.ndarray,
    color: np.ndarray,
    corner_min: np.ndarray,
    voxel_size: float,
    resolution: int,
    occupancy: np.ndarray,
    rgb: np.ndarray,
) -> None:
    if _HAS_NUMBA:
        _bruteforce_triangle_voxels_numba(
            np.asarray(tri, dtype=np.float64),
            np.asarray(color, dtype=np.uint8),
            np.asarray(corner_min, dtype=np.float64),
            float(voxel_size),
            int(resolution),
            occupancy,
            rgb,
        )
        return
    idx_lo, idx_hi = _voxel_index_bounds(tri, corner_min, voxel_size, resolution)
    for i in range(int(idx_lo[0]), int(idx_hi[0]) + 1):
        for j in range(int(idx_lo[1]), int(idx_hi[1]) + 1):
            for k in range(int(idx_lo[2]), int(idx_hi[2]) + 1):
                if not _voxel_intersects_triangle(i, j, k, tri, corner_min, voxel_size):
                    continue
                occupancy[i, j, k] = 1
                rgb[i, j, k] = color


def voxelize_colored_triangles(
    triangles: np.ndarray,
    colors: np.ndarray,
    resolution: int = DEFAULT_RESOLUTION,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Voxelize triangle soup into dense occupancy + RGB arrays."""
    tris = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(tris) == 0:
        raise ValueError("No triangles to voxelize")
    if len(tris) != len(cols):
        raise ValueError("triangles/colors length mismatch")

    center, span, voxel_size, corner_min = compute_grid_params(tris.reshape(-1, 3), resolution)
    occupancy = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    rgb = np.zeros((resolution, resolution, resolution, 3), dtype=np.uint8)

    voxel_face = max(voxel_size * voxel_size, 1e-12)
    areas = _triangle_areas(tris)
    rng = np.random.default_rng(0)

    splat_counts: List[int] = []
    splat_tris: List[np.ndarray] = []
    splat_colors: List[np.ndarray] = []

    for tri, color, area in zip(tris, cols, areas):
        idx_lo, idx_hi = _voxel_index_bounds(tri, corner_min, voxel_size, resolution)
        bbox_voxels = int(np.prod(idx_hi - idx_lo + 1))
        if bbox_voxels <= BRUTEFORCE_VOXEL_LIMIT:
            _bruteforce_triangle_voxels(
                tri, color, corner_min, voxel_size, resolution, occupancy, rgb
            )
            continue
        n_samples = int(
            max(8, min(MAX_SAMPLES_PER_TRIANGLE, np.ceil(float(area) / voxel_face)))
        )
        splat_counts.append(n_samples)
        splat_tris.append(tri)
        splat_colors.append(color)

    if splat_counts:
        counts = np.asarray(splat_counts, dtype=np.int32)
        all_pts = _sample_points_on_triangles(np.asarray(splat_tris, dtype=float), counts, rng)
        tri_idx = np.repeat(np.arange(len(splat_tris)), counts)
        all_cols = cols[tri_idx]
        _splat_points_to_grid(
            all_pts, all_cols, corner_min, voxel_size, resolution, occupancy, rgb
        )

    geom = {
        "center": center,
        "span": span,
        "voxel_size": voxel_size,
        "corner_min": corner_min,
        "resolution": resolution,
        "occupied_count": int(occupancy.sum()),
    }
    return occupancy, rgb, geom


def build_voxel_meta(
    geom: Dict[str, Any],
    *,
    coordinate_space: str,
    world_up: Sequence[float],
    world_up_space: str,
    data_path: str,
    compressed: bool = False,
) -> Dict[str, Any]:
    res = int(geom["resolution"])
    center = np.asarray(geom["center"], dtype=float)
    voxel_size = float(geom["voxel_size"])
    span = float(geom["span"])
    corner_min = np.asarray(geom["corner_min"], dtype=float)
    return {
        "format": VOXEL_FORMAT,
        "version": VOXEL_VERSION,
        "data_path": os.path.basename(data_path),
        "compressed": compressed,
        "resolution": [res, res, res],
        "coordinate_space": coordinate_space,
        "center": [float(x) for x in center],
        "voxel_size": voxel_size,
        "span": span,
        "corner_min": [float(x) for x in corner_min],
        "world_up": [float(x) for x in world_up],
        "world_up_space": world_up_space,
        "index_origin": "corner_min",
        "index_order": "ijk",
        "index_to_point": "corner_min + (i + 0.5, j + 0.5, k + 0.5) * voxel_size",
        "array_layout": "occupancy uint8 [Ni,Nj,Nk]; rgb uint8 [Ni,Nj,Nk,3]",
        "empty_voxel_occupancy": 0,
        "empty_voxel_rgb_ignored": True,
        "occupied_count": int(geom.get("occupied_count", 0)),
        "voxelization": "hybrid_bruteforce_or_surface_splat",
    }


def write_voxel_bundle(
    base_path: str,
    occupancy: np.ndarray,
    rgb: np.ndarray,
    meta: Dict[str, Any],
    *,
    compress: bool = False,
) -> None:
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
    # 256³ dense 数组 gzip 压缩仍偏慢；默认不压缩以保证渲染流水线速度
    if compress:
        np.savez_compressed(base_path, occupancy=occupancy, rgb=rgb)
    else:
        np.savez(base_path, occupancy=occupancy, rgb=rgb)
    meta_path = f"{os.path.splitext(base_path)[0]}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def export_voxel_preview_ply(
    ply_path: str,
    occupancy: np.ndarray,
    rgb: np.ndarray,
    corner_min: np.ndarray,
    voxel_size: float,
    *,
    max_points: int = 200_000,
) -> int:
    """Write occupied voxel centers as colored PLY (MeshLab / CloudCompare / Blender)."""
    idx = np.argwhere(occupancy > 0)
    if len(idx) == 0:
        return 0
    if len(idx) > max_points:
        step = max(1, len(idx) // max_points)
        idx = idx[::step]
    centers = corner_min + (idx.astype(float) + 0.5) * voxel_size
    colors = rgb[idx[:, 0], idx[:, 1], idx[:, 2]]
    os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(centers)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(
        f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])}"
        for pt, col in zip(centers, colors)
    )
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(centers)


def _transform_triangles_to_opencv(tris: np.ndarray, camera_pose: Dict[str, Any]) -> np.ndarray:
    try:
        from . import geometry_opencv as geo_cv
    except ImportError:
        import geometry_opencv as geo_cv  # type: ignore
    flat = tris.reshape(-1, 3)
    flat_o = geo_cv.transform_points_to_opencv(flat, camera_pose)
    return flat_o.reshape(tris.shape)


def export_colored_voxel_grids(
    output_dir: str,
    triangles: np.ndarray,
    colors: np.ndarray,
    *,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
    camera_pose: Optional[Dict[str, Any]] = None,
    resolution: int = DEFAULT_RESOLUTION,
    write_preview_ply: bool = True,
    compress: bool = False,
) -> Dict[str, Any]:
    """Export world + optional OpenCV camera voxel grids under output_dir/voxel/."""
    t0 = time.perf_counter()
    tris = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(tris) == 0:
        print("⚠️ 体素导出跳过：没有三角形")
        return {}

    voxel_dir = os.path.join(output_dir, "voxel")
    os.makedirs(voxel_dir, exist_ok=True)
    summary: Dict[str, Any] = {"voxel_dir": voxel_dir, "grids": {}}

    t_vox = time.perf_counter()
    occ_w, rgb_w, geom_w = voxelize_colored_triangles(tris, cols, resolution=resolution)
    vox_w_s = time.perf_counter() - t_vox

    world_path = os.path.join(voxel_dir, "occupancy_world.npz")
    meta_w = build_voxel_meta(
        geom_w,
        coordinate_space="world_ssl",
        world_up=world_up,
        world_up_space="world_ssl",
        data_path=os.path.basename(world_path),
        compressed=compress,
    )
    t_io = time.perf_counter()
    write_voxel_bundle(world_path, occ_w, rgb_w, meta_w, compress=compress)
    io_w_s = time.perf_counter() - t_io
    summary["grids"]["world"] = meta_w
    print(
        f"✅ 体素(世界系): {world_path} 占用 {meta_w['occupied_count']} 格 "
        f"(体素化 {vox_w_s:.1f}s, 写盘 {io_w_s:.1f}s)"
    )

    if write_preview_ply:
        preview_w = os.path.join(voxel_dir, "occupancy_world_preview.ply")
        n_prev = export_voxel_preview_ply(
            preview_w, occ_w, rgb_w, geom_w["corner_min"], geom_w["voxel_size"]
        )
        summary["grids"]["world"]["preview_ply"] = os.path.basename(preview_w)
        summary["grids"]["world"]["preview_points"] = n_prev
        if n_prev:
            print(f"✅ 体素预览 PLY(世界系): {preview_w} ({n_prev} 点)")

    if camera_pose is not None:
        try:
            from . import geometry_opencv as geo_cv
        except ImportError:
            import geometry_opencv as geo_cv  # type: ignore

        t_vox = time.perf_counter()
        tris_o = _transform_triangles_to_opencv(tris, camera_pose)
        up = np.asarray(world_up, dtype=float)
        up_o = geo_cv.transform_normals_to_opencv(up.reshape(1, 3), camera_pose)[0]
        occ_o, rgb_o, geom_o = voxelize_colored_triangles(tris_o, cols, resolution=resolution)
        vox_o_s = time.perf_counter() - t_vox

        opencv_path = os.path.join(voxel_dir, "occupancy_opencv.npz")
        meta_o = build_voxel_meta(
            geom_o,
            coordinate_space="opencv_camera",
            world_up=up_o,
            world_up_space="opencv_camera",
            data_path=os.path.basename(opencv_path),
            compressed=compress,
        )
        t_io = time.perf_counter()
        write_voxel_bundle(opencv_path, occ_o, rgb_o, meta_o, compress=compress)
        io_o_s = time.perf_counter() - t_io
        summary["grids"]["opencv"] = meta_o
        print(
            f"✅ 体素(相机系): {opencv_path} 占用 {meta_o['occupied_count']} 格 "
            f"(体素化 {vox_o_s:.1f}s, 写盘 {io_o_s:.1f}s)"
        )

        if write_preview_ply:
            preview_o = os.path.join(voxel_dir, "occupancy_opencv_preview.ply")
            n_prev_o = export_voxel_preview_ply(
                preview_o, occ_o, rgb_o, geom_o["corner_min"], geom_o["voxel_size"]
            )
            summary["grids"]["opencv"]["preview_ply"] = os.path.basename(preview_o)
            summary["grids"]["opencv"]["preview_points"] = n_prev_o
            if n_prev_o:
                print(f"✅ 体素预览 PLY(相机系): {preview_o} ({n_prev_o} 点)")

    summary_path = os.path.join(voxel_dir, "metadata_voxel.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    total_s = time.perf_counter() - t0
    print(f"⏱️  体素导出总耗时: {total_s:.1f}s")
    return summary
