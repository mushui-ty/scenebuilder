"""2D mask visibility ratio: full projection mask (no occlusion) vs semantic visible mask."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SEMANTIC_MASKS_DIR = "semantic_masks"

STRUCTURAL_CATEGORIES = frozenset({"floor", "ceiling", "walls", "doors", "windows"})


def semantic_masks_dir(view_dir: str) -> str:
    return os.path.join(view_dir, SEMANTIC_MASKS_DIR)


def semantic_mask_relpath(category: str, object_id: str) -> str:
    return f"{category}_{object_id}.png"


def semantic_mask_path(view_dir: str, category: str, object_id: str) -> str:
    return os.path.join(semantic_masks_dir(view_dir), semantic_mask_relpath(category, object_id))


def semantic_masks_index_path(view_dir: str) -> str:
    return os.path.join(semantic_masks_dir(view_dir), "index.json")


def count_mask_pixels(mask_path: str) -> int:
    import imageio

    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return int(np.count_nonzero(np.asarray(mask, dtype=np.uint8)))


def binary_mask_from_isolated_semantic(
    semantic_rgb: np.ndarray,
    target_color: Tuple[int, int, int],
    background: Tuple[int, int, int] = (0, 0, 0),
    max_dist_sq: float = 0,
) -> np.ndarray:
    """Extract binary mask from isolated per-object semantic image (black background + single emission color).

    EEVEE anti-aliases object edges; film dither may add 1–2 gray levels to background.
    Two-color nearest-neighbor split (background / target): background noise → black, edge blend → object.
    When max_dist_sq>0, drop pixels too far from nearest color (legacy compat; default 0 disables).
    """
    rgb = np.asarray(semantic_rgb[..., :3], dtype=np.uint8)
    h, w = rgb.shape[:2]
    palette = np.asarray([background, target_color], dtype=np.float32)
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    dist_sq = np.sum((pixels[:, None, :] - palette[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dist_sq, axis=1)
    mask = nearest == 1
    if float(max_dist_sq) > 0:
        min_dist = dist_sq[np.arange(pixels.shape[0]), nearest]
        mask &= min_dist <= float(max_dist_sq)
    return mask.reshape(h, w)


def load_overall_pixels_map(view_dir: str) -> Dict[Tuple[str, str], int]:
    index_path = semantic_masks_index_path(view_dir)
    if not os.path.isfile(index_path):
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mapping: Dict[Tuple[str, str], int] = {}
    for item in payload.get("objects", []):
        category = str(item.get("category", ""))
        object_id = str(item.get("id", ""))
        mapping[(category, object_id)] = int(item.get("overall_pixels", 0))
    return mapping


def visibility_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    cfg = config or {}
    return {
        "ratio_if_visible": float(cfg.get("ratio_if_visible", 0.1)),
        "min_overall_pixels": int(cfg.get("min_overall_pixels", 16)),
    }


def build_entity_color_map(
    semantic_objects: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str], Tuple[int, int, int]]:
    """(category, object_id) -> RGB, matching semantic pass colors."""
    mapping: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
    for obj in semantic_objects:
        category = str(obj.get("category", ""))
        color = obj.get("color")
        if not color or len(color) != 3:
            continue
        rgb = (int(color[0]), int(color[1]), int(color[2]))
        if category in ("floor", "ceiling"):
            mapping[(category, category)] = rgb
            continue
        entity_id = obj.get("entity_id")
        if entity_id is None:
            entity_id = obj.get("label")
        if entity_id is None:
            continue
        mapping[(category, str(entity_id))] = rgb
    return mapping


def iter_record_triangles(records: Iterable[Dict[str, Any]]) -> Iterable[np.ndarray]:
    for record in records:
        for tri in record.get("triangles", []):
            yield np.asarray(tri, dtype=float)


def _clip4_to_pixel(clip4: np.ndarray, width: int, height: int) -> Tuple[float, float]:
    w = float(clip4[3])
    if w <= 1e-8:
        return float("nan"), float("nan")
    ndc_x = float(clip4[0]) / w
    ndc_y = float(clip4[1]) / w
    ndc_z = float(clip4[2]) / w
    if ndc_z < 0.0:
        return float("nan"), float("nan")
    px = (ndc_x * 0.5 + 0.5) * width
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * height
    return px, py


def _lerp_xyz(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    return (
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    )


def _clip_triangle_to_screen_tris(
    triangle: np.ndarray,
    proj: np.ndarray,
    modelview: np.ndarray,
    width: int,
    height: int,
    world_depth_fn: Callable[[np.ndarray], float],
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
    """Clip-space cull + clip projection to pixels + world_to_camera_view depth (aligned with semantic)."""
    from .util_bpy import _clip_homogeneous_polygon_against_plane, render_frustum_clip_planes

    tri = np.asarray(triangle, dtype=float)
    payload = []
    for idx in range(3):
        ph = np.array([tri[idx, 0], tri[idx, 1], tri[idx, 2], 1.0], dtype=float)
        payload.append({
            "clip": proj @ (modelview @ ph),
            "world": tri[idx],
        })
    polygon = payload
    for plane_normal, plane_offset in render_frustum_clip_planes():
        polygon = _clip_homogeneous_polygon_against_plane(polygon, plane_normal, plane_offset)
        if len(polygon) < 3:
            return []

    screen_tris: List[
        Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
    ] = []
    for i in range(1, len(polygon) - 1):
        items = [polygon[0], polygon[i], polygon[i + 1]]
        pts: List[Tuple[float, float, float]] = []
        for item in items:
            px, py = _clip4_to_pixel(item["clip"], width, height)
            if px != px or py != py:
                break
            depth = world_depth_fn(np.asarray(item["world"], dtype=float))
            if depth <= 0.0 or depth != depth:
                break
            pts.append((px, py, depth))
        else:
            if len(pts) < 3:
                continue
            clipped = _clip_polygon_2d_to_image_rect(pts, width, height)
            if len(clipped) < 3:
                continue
            for j in range(1, len(clipped) - 1):
                screen_tris.append((clipped[0], clipped[j], clipped[j + 1]))
    return screen_tris


def _clip_polygon_2d_to_image_rect(
    points: Sequence[Tuple[float, float, float]],
    width: int,
    height: int,
) -> List[Tuple[float, float, float]]:
    """Clip a depth-aware 2D polygon to [0, width] x [0, height]."""
    if len(points) < 3:
        return []

    def clip_halfspace(pts, inside_fn, intersect_fn):
        if not pts:
            return []
        out: List[Tuple[float, float, float]] = []
        prev = pts[-1]
        prev_in = inside_fn(prev)
        for curr in pts:
            curr_in = inside_fn(curr)
            if curr_in != prev_in:
                out.append(intersect_fn(prev, curr))
            if curr_in:
                out.append(curr)
            prev = curr
            prev_in = curr_in
        return out

    def intersect_x(a, b, x_val):
        denom = b[0] - a[0]
        t = 0.0 if abs(denom) <= 1e-12 else (x_val - a[0]) / denom
        t = float(np.clip(t, 0.0, 1.0))
        return _lerp_xyz(a, b, t)

    def intersect_y(a, b, y_val):
        denom = b[1] - a[1]
        t = 0.0 if abs(denom) <= 1e-12 else (y_val - a[1]) / denom
        t = float(np.clip(t, 0.0, 1.0))
        return _lerp_xyz(a, b, t)

    polygon = list(points)
    polygon = clip_halfspace(
        polygon,
        lambda p: p[0] >= 0.0,
        lambda a, b: intersect_x(a, b, 0.0),
    )
    polygon = clip_halfspace(
        polygon,
        lambda p: p[0] <= float(width),
        lambda a, b: intersect_x(a, b, float(width)),
    )
    polygon = clip_halfspace(
        polygon,
        lambda p: p[1] >= 0.0,
        lambda a, b: intersect_y(a, b, 0.0),
    )
    polygon = clip_halfspace(
        polygon,
        lambda p: p[1] <= float(height),
        lambda a, b: intersect_y(a, b, float(height)),
    )
    return polygon


def _iter_overall_mask_pixel_tris(
    records: Iterable[Dict[str, Any]],
    width: int,
    height: int,
    *,
    clip_mats: Tuple[np.ndarray, np.ndarray],
    world_depth_fn: Callable[[np.ndarray], float],
) -> Iterable[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
    proj, modelview = clip_mats
    for tri in iter_record_triangles(records):
        yield from _clip_triangle_to_screen_tris(
            tri, proj, modelview, width, height, world_depth_fn
        )


def _rasterize_triangle_zbuffer(
    mask: np.ndarray,
    depth_buf: np.ndarray,
    tri: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]],
    width: int,
    height: int,
) -> None:
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = tri
    min_x = max(0, int(np.floor(min(x0, x1, x2))))
    max_x = min(width - 1, int(np.ceil(max(x0, x1, x2))))
    min_y = max(0, int(np.floor(min(y0, y1, y2))))
    max_y = min(height - 1, int(np.ceil(max(y0, y1, y2))))
    if min_x > max_x or min_y > max_y:
        return

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) <= 1e-12:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float64)
    ys = np.arange(min_y, max_y + 1, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
    w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
    if not np.any(inside):
        return

    zz = w0 * z0 + w1 * z1 + w2 * z2
    iy, ix = np.nonzero(inside)
    gx = min_x + ix
    gy = min_y + iy
    zvals = zz[inside]
    current = depth_buf[gy, gx]
    closer = zvals < current
    if not np.any(closer):
        return
    gx = gx[closer]
    gy = gy[closer]
    zvals = zvals[closer]
    depth_buf[gy, gx] = zvals
    mask[gy, gx] = True


def _rasterize_overall_mask_zbuffer(
    pixel_tris: Iterable[
        Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
    ],
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    depth_buf = np.full((height, width), np.inf, dtype=np.float64)
    for tri in pixel_tris:
        _rasterize_triangle_zbuffer(mask, depth_buf, tri, width, height)
    return mask


def rasterize_overall_mask(
    records: Iterable[Dict[str, Any]],
    width: int,
    height: int,
    *,
    clip_mats: Tuple[np.ndarray, np.ndarray],
    world_depth_fn: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Per-object Z-buffer projection mask: same mesh as semantic, ignores occlusion by other objects.

    Pixel/depth projection matches Blender: clip projection + world_to_camera_view depth.
    """
    if width <= 0 or height <= 0:
        return np.zeros((max(height, 0), max(width, 0)), dtype=bool)

    pixel_tris = list(_iter_overall_mask_pixel_tris(
        records, width, height, clip_mats=clip_mats, world_depth_fn=world_depth_fn,
    ))
    if not pixel_tris:
        return np.zeros((height, width), dtype=bool)

    return _rasterize_overall_mask_zbuffer(pixel_tris, width, height)


def rasterize_triangle_pixel_count(
    records: Iterable[Dict[str, Any]],
    width: int,
    height: int,
    *,
    clip_mats: Tuple[np.ndarray, np.ndarray],
    world_depth_fn: Callable[[np.ndarray], float],
) -> int:
    """Project in-frustum triangles to image plane; return occupied pixel count (ignores occlusion by other objects)."""
    return int(np.count_nonzero(rasterize_overall_mask(
        records, width, height, clip_mats=clip_mats, world_depth_fn=world_depth_fn,
    )))


def count_semantic_color_pixels(
    semantic_rgb: np.ndarray,
    color: Tuple[int, int, int],
    overall_mask: Optional[np.ndarray] = None,
) -> int:
    rgb = np.asarray(semantic_rgb[..., :3], dtype=np.uint8)
    target = np.asarray(color, dtype=np.uint8)
    sem = np.all(rgb == target, axis=-1)
    if overall_mask is not None:
        sem &= np.asarray(overall_mask, dtype=bool)
    return int(np.count_nonzero(sem))


def evaluate_visibility(
    *,
    category: str,
    in_frustum: bool,
    overall_pixels: int,
    visible_pixels: int,
    ratio_if_visible: float,
    min_overall_pixels: int,
) -> Dict[str, Any]:
    """in_frustum: geometry remains after clip-space cull (same as README appendix B frustum clip).

    min_overall_pixels is only for visibility ratio / export thresholds; it no longer defines in_frustum.
    """
    has_projection = overall_pixels >= min_overall_pixels
    if not in_frustum or not has_projection or overall_pixels <= 0:
        ratio = 0.0
    else:
        ratio = float(visible_pixels) / float(overall_pixels)

    mostly_occluded = in_frustum and has_projection and ratio < ratio_if_visible
    export_geometry = in_frustum and has_projection

    if not export_geometry:
        in_merged_export = False
    elif category in STRUCTURAL_CATEGORIES:
        in_merged_export = visible_pixels > 0
    else:
        in_merged_export = visible_pixels > 0 and not mostly_occluded

    return {
        "overall_pixels": int(overall_pixels),
        "visible_pixels": int(visible_pixels),
        "visibility_ratio": round(ratio, 6),
        "in_frustum": bool(in_frustum),
        "mostly_occluded": bool(mostly_occluded),
        "export_geometry": bool(export_geometry),
        "in_merged_export": bool(in_merged_export),
    }


def append_mostly_occluded_suffix(path: str) -> str:
    root, ext = os.path.splitext(path)
    if root.endswith("_mostly_occluded"):
        return path
    return f"{root}_mostly_occluded{ext}"
