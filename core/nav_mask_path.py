"""Top-down depth-based nav mask floor closed-loop path sampling."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon

try:
    from .floor_path_utils import (
        MIN_PATH_POINTS,
        compute_path_point_inward_xy,
        draw_path_ring,
        pixel_points_to_ssl_ground,
    )
except ImportError:
    from floor_path_utils import (  # type: ignore
        MIN_PATH_POINTS,
        compute_path_point_inward_xy,
        draw_path_ring,
        pixel_points_to_ssl_ground,
    )

DEFAULT_NAV_MASK_INSET_M = 0.15
DEFAULT_POINT_SPACING_M = 0.3
DEFAULT_DEPTH_TOLERANCE_M = 0.05
TRAJECTORY_STEP_PX = 1.0

STRUCTURE_CATEGORIES = frozenset({"wall", "walls", "door", "doors", "window", "windows"})
FLOOR_CATEGORIES = frozenset({"floor"})


def _normalize_category(name: str) -> str:
    return name.strip().lower().rstrip("0123456789")


def _build_semantic_palette(meta: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """Build semantic palette (including background) for nearest-neighbor pixel assignment."""
    palette: List[Tuple[int, int, int]] = []
    categories: List[str] = []
    seen: set[Tuple[Tuple[int, int, int], str]] = set()

    bg = meta.get("background", [0, 0, 0])
    palette.append((int(bg[0]), int(bg[1]), int(bg[2])))
    categories.append("background")

    def _add(color: Tuple[int, int, int], category: str) -> None:
        norm = _normalize_category(category)
        key = (color, norm)
        if key in seen:
            return
        seen.add(key)
        palette.append(color)
        categories.append(norm)

    for obj in meta.get("objects", []):
        color = obj.get("color")
        if not color or len(color) != 3:
            continue
        _add((int(color[0]), int(color[1]), int(color[2])), str(obj.get("category", "")))

    for cat, color in meta.get("category_colors", {}).items():
        if not color or len(color) != 3:
            continue
        if _normalize_category(str(cat)) == "background":
            continue
        _add((int(color[0]), int(color[1]), int(color[2])), str(cat))

    return np.asarray(palette, dtype=np.float32), categories


def build_semantic_category_masks(
    semantic_png_path: str,
    semantic_json_path: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract wall/door/window mask and floor mask from semantic image.

    Cycles semantic boundaries have anti-aliased intermediate colors; exact RGB match fails.
    Assign each pixel to nearest palette color from JSON, then group by category.
    """
    if not os.path.isfile(semantic_png_path):
        raise FileNotFoundError(semantic_png_path)
    if not os.path.isfile(semantic_json_path):
        raise FileNotFoundError(semantic_json_path)

    with open(semantic_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    img = imageio.imread(semantic_png_path)
    rgb = img[..., :3] if img.ndim == 3 else np.stack([img] * 3, axis=-1)
    palette, categories = _build_semantic_palette(meta)
    if len(palette) <= 1:
        raise RuntimeError("Semantic JSON missing valid entity colors")

    h, w = rgb.shape[:2]
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    dist_sq = np.sum((pixels[:, None, :] - palette[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dist_sq, axis=1)

    structure_flat = np.zeros(h * w, dtype=bool)
    floor_flat = np.zeros(h * w, dtype=bool)
    for idx, cat in enumerate(categories):
        if cat in STRUCTURE_CATEGORIES:
            structure_flat[nearest == idx] = True
        elif cat in FLOOR_CATEGORIES:
            floor_flat[nearest == idx] = True

    structure = structure_flat.reshape(h, w)
    floor = floor_flat.reshape(h, w)
    structure = _clean_category_mask(structure, min_component_area=16)
    floor = _clean_category_mask(floor, min_component_area=16)

    info = {
        "match_mode": "nearest_color",
        "palette_size": int(len(palette)),
        "structure_pixels": int(structure.sum()),
        "floor_pixels": int(floor.sum()),
        "semantic_png": semantic_png_path,
        "semantic_json": semantic_json_path,
    }
    return structure, floor, info


def _clean_category_mask(mask: np.ndarray, *, min_component_area: int = 16) -> np.ndarray:
    """Remove tiny isolated pixels from category mask (anti-aliasing mis-assignment)."""
    labeled, count = ndimage.label(mask)
    if count <= 0:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, range(1, count + 1))
    cleaned = np.zeros_like(mask, dtype=bool)
    for idx, size in enumerate(sizes, start=1):
        if size >= min_component_area:
            cleaned[labeled == idx] = True
    return cleaned


def combine_nav_mask(
    depth_mask: np.ndarray,
    structure_mask: np.ndarray,
    floor_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Final nav mask = (depth_mask - structure_mask) ∪ floor_mask."""
    depth_mask = depth_mask.astype(bool)
    structure_mask = structure_mask.astype(bool)
    floor_mask = floor_mask.astype(bool)
    subtracted = depth_mask & ~structure_mask
    combined = subtracted | floor_mask
    info = {
        "depth_pixels": int(depth_mask.sum()),
        "structure_pixels": int(structure_mask.sum()),
        "floor_pixels": int(floor_mask.sum()),
        "depth_minus_structure_pixels": int(subtracted.sum()),
        "combined_pixels": int(combined.sum()),
    }
    return combined, info


def _reference_floor_depth_m(camera_para: Dict[str, Any]) -> float:
    """Distance (m) from top-down camera to ground look_at, i.e. expected floor depth."""
    cam = np.asarray(camera_para["camera_position"], dtype=float)
    look = np.asarray(camera_para["look_at_target"], dtype=float)
    return float(np.linalg.norm(cam - look))


def build_nav_mask_from_depth(
    depth_png_path: str,
    camera_para: Dict[str, Any],
    *,
    tolerance_m: float = DEFAULT_DEPTH_TOLERANCE_M,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Pixels with depth within camera-to-ground distance ± tolerance are walkable."""
    if not os.path.isfile(depth_png_path):
        raise FileNotFoundError(depth_png_path)
    depth_scale = camera_para.get("depth_scale")
    if depth_scale is None:
        raise ValueError("camera_para.json missing depth_scale; render topdown_depth.png first")

    try:
        from . import util
    except ImportError:
        import util  # type: ignore

    depth_u16 = imageio.imread(depth_png_path)
    depth_m = util.decode_depth_uint16(depth_u16, float(depth_scale))
    ref_depth = _reference_floor_depth_m(camera_para)
    tol = float(tolerance_m)
    valid = depth_m > 0
    nav_mask = valid & (np.abs(depth_m - ref_depth) <= tol)

    info = {
        "source": "depth",
        "width": int(nav_mask.shape[1]),
        "height": int(nav_mask.shape[0]),
        "nav_pixels": int(nav_mask.sum()),
        "valid_depth_pixels": int(valid.sum()),
        "reference_depth_m": ref_depth,
        "depth_tolerance_m": tol,
        "depth_min_m": float(depth_m[valid].min()) if valid.any() else None,
        "depth_max_m": float(depth_m[valid].max()) if valid.any() else None,
        "depth_scale": float(depth_scale),
    }
    return nav_mask, info


def _largest_component(mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    labeled, count = ndimage.label(mask)
    meta: Dict[str, Any] = {"component_count": int(count)}
    if count <= 0:
        return mask.astype(bool), meta
    sizes = ndimage.sum(mask, labeled, range(1, count + 1))
    if len(sizes) == 0:
        return mask.astype(bool), meta
    best = int(1 + np.argmax(sizes))
    meta["largest_component_pixels"] = int(sizes[best - 1])
    meta["largest_component_index"] = best
    return (labeled == best), meta


def _contour_to_xy(contour_yx: np.ndarray) -> np.ndarray:
    return np.column_stack([contour_yx[:, 1], contour_yx[:, 0]])


def _mask_to_polygon_with_holes(mask: np.ndarray) -> Optional[Polygon]:
    """Build polygon with holes from binary mask (outer contour + interior holes)."""
    try:
        from skimage.measure import find_contours
    except ImportError as exc:
        raise RuntimeError("scikit-image required (skimage.measure.find_contours)") from exc

    contours = find_contours(mask.astype(float), 0.5)
    contours_xy = [_contour_to_xy(c) for c in contours if len(c) >= 3]
    if not contours_xy:
        return None

    def _poly_area(xy: np.ndarray) -> float:
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return abs(float(poly.area))

    contours_xy.sort(key=_poly_area, reverse=True)
    ext_poly = Polygon(contours_xy[0])
    if not ext_poly.is_valid:
        ext_poly = ext_poly.buffer(0)
    if ext_poly.is_empty:
        return None

    holes: List[List[Tuple[float, float]]] = []
    for xy in contours_xy[1:]:
        hole_poly = Polygon(xy)
        if not hole_poly.is_valid:
            hole_poly = hole_poly.buffer(0)
        if hole_poly.is_empty:
            continue
        if ext_poly.contains(hole_poly.centroid):
            holes.append(list(hole_poly.exterior.coords))

    poly = Polygon(ext_poly.exterior.coords, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if not poly.is_empty else None


def _inset_boundary_ring(
    poly: Polygon,
    inset_px: float,
) -> Tuple[LineString, Dict[str, Any]]:
    """Inset a polygon with holes inward to get a single boundary-following closed loop."""
    meta: Dict[str, Any] = {"inset_px": float(inset_px)}
    if poly.is_empty or poly.area <= 1e-6:
        raise ValueError("nav mask polygon invalid")

    meta["boundary_length_px"] = float(poly.exterior.length)
    meta["boundary_area_px2"] = float(poly.area)
    meta["hole_count"] = len(poly.interiors)

    inset_geom = poly.buffer(-float(inset_px), join_style=2)
    if inset_geom.is_empty:
        inset_geom = poly.buffer(-float(inset_px) * 0.5, join_style=2)
        meta["inset_fallback_half"] = True
    if inset_geom.is_empty:
        inset_geom = poly
        meta["inset_fallback_zero"] = True

    if inset_geom.geom_type == "MultiPolygon":
        inset_geom = max(inset_geom.geoms, key=lambda g: g.area)
        meta["inset_multipolygon_largest_only"] = True
    if inset_geom.geom_type != "Polygon":
        raise ValueError(f"Unexpected inset geometry type: {inset_geom.geom_type}")

    ring = LineString(inset_geom.exterior.coords)
    meta["inset_ring_length_px"] = float(ring.length)
    return ring, meta


def _snap_pixel_to_mask(
    mask: np.ndarray,
    x: float,
    y: float,
    *,
    max_radius: int = 48,
) -> Optional[List[int]]:
    xi, yi = int(round(x)), int(round(y))
    h, w = mask.shape
    if 0 <= xi < w and 0 <= yi < h and mask[yi, xi]:
        return [xi, yi]
    for r in range(1, max_radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                nx, ny = xi + dx, yi + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]:
                    return [nx, ny]
    return None


def _segment_in_mask(p0: List[int], p1: List[int], mask: np.ndarray) -> bool:
    try:
        from skimage.draw import line as sk_line
    except ImportError as exc:
        raise RuntimeError("scikit-image required (skimage.draw.line)") from exc

    rr, cc = sk_line(int(p0[1]), int(p0[0]), int(p1[1]), int(p1[0]))
    h, w = mask.shape
    rr = np.clip(rr, 0, h - 1)
    cc = np.clip(cc, 0, w - 1)
    return bool(mask[rr, cc].all())


def _ring_distances(start: float, end: float, length: float, forward: bool) -> List[float]:
    if length <= 1e-9:
        return []
    if forward:
        if end >= start:
            span = end - start
        else:
            span = (length - start) + end
        n = max(int(math.ceil(span / TRAJECTORY_STEP_PX)), 1)
        return [start + span * i / n for i in range(n + 1)]
    if start >= end:
        span = start - end
    else:
        span = start + (length - end)
    n = max(int(math.ceil(span / TRAJECTORY_STEP_PX)), 1)
    return [start - span * i / n for i in range(n + 1)]


def _ring_arc_in_mask(
    ring: LineString,
    mask: np.ndarray,
    p0: List[int],
    p1: List[int],
) -> List[List[int]]:
    """Walk ring arc from p0→p1; prefer direction that stays fully inside mask."""
    length = float(ring.length)
    if length <= 1e-9:
        return [p0, p1]

    d0 = float(ring.project(Point(p0)))
    d1 = float(ring.project(Point(p1)))
    best: List[List[int]] = []

    for forward in (True, False):
        dists = _ring_distances(d0, d1, length, forward)
        arc: List[List[int]] = []
        ok = True
        for d in dists:
            pt = ring.interpolate(d % length)
            snapped = _snap_pixel_to_mask(mask, pt.x, pt.y)
            if snapped is None:
                ok = False
                break
            if not arc or arc[-1] != snapped:
                arc.append(snapped)
        if not ok or len(arc) < 2:
            continue
        if all(_segment_in_mask(arc[i], arc[i + 1], mask) for i in range(len(arc) - 1)):
            if not best or len(arc) < len(best):
                best = arc
    return best if best else [p0, p1]


def _repair_closed_trajectory(
    trajectory: List[List[int]],
    ring: LineString,
    mask: np.ndarray,
) -> List[List[int]]:
    if len(trajectory) < 2:
        return trajectory

    changed = True
    while changed:
        changed = False
        fixed: List[List[int]] = [trajectory[0]]
        for nxt in trajectory[1:]:
            if not _segment_in_mask(fixed[-1], nxt, mask):
                arc = _ring_arc_in_mask(ring, mask, fixed[-1], nxt)
                if len(arc) > 2:
                    fixed.extend(arc[1:-1])
                    changed = True
                if not _segment_in_mask(fixed[-1], nxt, mask):
                    mid_arc = _ring_arc_in_mask(ring, mask, fixed[-1], nxt)
                    if len(mid_arc) > 2:
                        fixed.extend(mid_arc[1:-1])
                        changed = True
            if fixed[-1] != nxt:
                fixed.append(nxt)
        if not _segment_in_mask(fixed[-1], fixed[0], mask):
            arc = _ring_arc_in_mask(ring, mask, fixed[-1], fixed[0])
            if len(arc) > 2:
                fixed.extend(arc[1:-1])
                changed = True
        trajectory = fixed
    return trajectory


def _build_trajectory_in_mask(
    ring: LineString,
    mask: np.ndarray,
    *,
    step_px: float = TRAJECTORY_STEP_PX,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    length = float(ring.length)
    meta: Dict[str, Any] = {"trajectory_step_px": float(step_px)}
    if length <= 1e-6:
        return [], meta

    n = max(int(math.ceil(length / step_px)), MIN_PATH_POINTS * 4)
    trajectory: List[List[int]] = []
    skipped = 0
    for i in range(n):
        pt = ring.interpolate(i * length / n)
        snapped = _snap_pixel_to_mask(mask, pt.x, pt.y)
        if snapped is None:
            skipped += 1
            continue
        if not trajectory or trajectory[-1] != snapped:
            trajectory.append(snapped)

    trajectory = _repair_closed_trajectory(trajectory, ring, mask)
    meta["trajectory_points"] = len(trajectory)
    meta["trajectory_skipped"] = skipped
    return trajectory, meta


def _polyline_lengths(points: List[List[int]]) -> Tuple[List[float], float]:
    if len(points) < 2:
        return [0.0], 0.0
    cum = [0.0]
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        total += math.hypot(dx, dy)
        cum.append(total)
    return cum, total


def _extract_waypoints_from_trajectory(
    trajectory: List[List[int]],
    spacing_px: float,
    *,
    min_points: int = MIN_PATH_POINTS,
) -> List[List[int]]:
    if len(trajectory) < 2:
        return trajectory

    closed = trajectory + [trajectory[0]]
    cum, total = _polyline_lengths(closed)
    if total <= 1e-6:
        return trajectory[: max(min_points, 1)]

    n_pts = max(int(min_points), int(math.ceil(total / spacing_px)))
    targets = [total * i / n_pts for i in range(n_pts)]
    waypoints: List[List[int]] = []
    j = 0
    for t in targets:
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        waypoints.append(list(closed[j]))
    deduped: List[List[int]] = []
    for pt in waypoints:
        if not deduped or deduped[-1] != pt:
            deduped.append(pt)
    return deduped


def _validate_path_in_mask(
    points: List[List[int]],
    mask: np.ndarray,
    *,
    closed: bool = True,
) -> Dict[str, Any]:
    h, w = mask.shape
    bad_points: List[int] = []
    bad_segments: List[int] = []
    for i, (x, y) in enumerate(points):
        if not (0 <= x < w and 0 <= y < h and mask[y, x]):
            bad_points.append(i)
    seq = points + ([points[0]] if closed and points else [])
    for i in range(len(seq) - 1):
        if not _segment_in_mask(seq[i], seq[i + 1], mask):
            bad_segments.append(i)
    return {
        "valid": not bad_points and not bad_segments,
        "bad_points": bad_points,
        "bad_segments": bad_segments,
    }


def sample_path_on_nav_mask(
    nav_mask: np.ndarray,
    pixel2real_ratio: float,
    *,
    inset_m: float = DEFAULT_NAV_MASK_INSET_M,
    point_spacing_m: float = DEFAULT_POINT_SPACING_M,
    min_points: int = MIN_PATH_POINTS,
) -> Tuple[List[List[int]], List[List[int]], np.ndarray, Dict[str, Any]]:
    """Sample closed loop path in largest connected mask (waypoints + trajectory strictly inside mask)."""
    ratio = float(pixel2real_ratio)
    if ratio <= 0:
        raise ValueError("pixel2real_ratio must be > 0")

    mask, comp_meta = _largest_component(nav_mask.astype(bool))
    sample_meta: Dict[str, Any] = {"component": comp_meta}
    if int(mask.sum()) < min_points:
        return [], [], mask, sample_meta

    poly = _mask_to_polygon_with_holes(mask)
    if poly is None:
        return [], [], mask, sample_meta

    inset_px = float(inset_m) / ratio
    spacing_px = float(point_spacing_m) / ratio
    ring, inset_meta = _inset_boundary_ring(poly, inset_px)
    sample_meta.update(inset_meta)

    ring_len = float(ring.length)
    if ring_len <= 1e-6:
        return [], [], mask, sample_meta

    trajectory, traj_meta = _build_trajectory_in_mask(ring, mask)
    sample_meta.update(traj_meta)
    if len(trajectory) < min_points:
        return [], [], mask, sample_meta

    waypoints = _extract_waypoints_from_trajectory(
        trajectory, spacing_px, min_points=min_points
    )
    traj_check = _validate_path_in_mask(trajectory, mask, closed=True)
    wp_check = _validate_path_in_mask(waypoints, mask, closed=True)
    sample_meta["trajectory_validation"] = traj_check
    sample_meta["waypoints_validation"] = wp_check
    sample_meta["point_spacing_m"] = float(point_spacing_m)
    sample_meta["inset_m"] = float(inset_m)
    sample_meta["spacing_px"] = spacing_px
    sample_meta["num_waypoints"] = len(waypoints)
    sample_meta["ring_length_m"] = ring_len * ratio

    if not traj_check["valid"]:
        raise RuntimeError(f"trajectory not fully inside nav mask: {traj_check}")

    return waypoints, trajectory, mask, sample_meta


def save_nav_mask_png(nav_mask: np.ndarray, output_path: str) -> None:
    vis = np.zeros((*nav_mask.shape, 3), dtype=np.uint8)
    vis[nav_mask] = (0, 220, 80)
    imageio.imwrite(output_path, vis)


def run_nav_mask_floor_path(
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sample floor closed-loop path on normalized top-down dir using depth+semantic nav mask."""
    config = config or {}
    image_path = os.path.join(output_dir, "topdown.png")
    camera_para_path = os.path.join(output_dir, "camera_para.json")
    depth_png = os.path.join(output_dir, "topdown_depth.png")
    semantic_png = os.path.join(output_dir, "topdown_semantic.png")
    semantic_json = os.path.join(output_dir, "topdown_semantic.json")

    for path in (image_path, camera_para_path, depth_png, semantic_png, semantic_json):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    with open(camera_para_path, "r", encoding="utf-8") as f:
        camera_para = json.load(f)
    ratio = float(camera_para["pixel2real_ratio"])
    width, height = camera_para.get("image_size", [1000, 1000])

    inset_m = float(config.get("nav_mask_inset_m", DEFAULT_NAV_MASK_INSET_M))
    spacing_m = float(config.get("nav_mask_point_spacing_m", DEFAULT_POINT_SPACING_M))
    tol_m = float(config.get("nav_mask_depth_tolerance_m", DEFAULT_DEPTH_TOLERANCE_M))
    print(
        f"🗺️  Depth+semantic nav mask path: inset={inset_m}m, spacing={spacing_m}m, "
        f"depth_tol=±{tol_m}m, pixel2real_ratio={ratio:.6f}"
    )

    depth_mask, depth_info = build_nav_mask_from_depth(
        depth_png, camera_para, tolerance_m=tol_m
    )
    depth_mask_path = os.path.join(output_dir, "nav_mask_depth.png")
    save_nav_mask_png(depth_mask, depth_mask_path)

    structure_mask, floor_mask, semantic_info = build_semantic_category_masks(
        semantic_png, semantic_json
    )
    structure_path = os.path.join(output_dir, "nav_mask_structure.png")
    floor_path = os.path.join(output_dir, "nav_mask_floor.png")
    save_nav_mask_png(structure_mask, structure_path)
    save_nav_mask_png(floor_mask, floor_path)

    nav_mask_raw, combine_info = combine_nav_mask(depth_mask, structure_mask, floor_mask)
    nav_mask_path = os.path.join(output_dir, "nav_mask.png")
    save_nav_mask_png(nav_mask_raw, nav_mask_path)

    mask_largest, _ = _largest_component(nav_mask_raw)
    largest_path = os.path.join(output_dir, "nav_mask_largest.png")
    save_nav_mask_png(mask_largest, largest_path)

    points_px, trajectory_px, _, sample_meta = sample_path_on_nav_mask(
        nav_mask_raw,
        ratio,
        inset_m=inset_m,
        point_spacing_m=spacing_m,
    )
    if len(points_px) < MIN_PATH_POINTS:
        raise RuntimeError(
            f"nav mask sampling failed: only {len(points_px)} points (minimum {MIN_PATH_POINTS} required)"
        )

    points_ssl = pixel_points_to_ssl_ground(points_px, ratio)
    inward_xy, inward_meta = compute_path_point_inward_xy(points_px, output_dir, ratio)

    json_path = os.path.join(output_dir, "floor_path_points.json")
    ssl_txt_path = os.path.join(output_dir, "floor_path_ssl.txt")
    inward_txt_path = os.path.join(output_dir, "floor_path_inward.txt")
    overlay_path = os.path.join(output_dir, "topdown_floor_path.png")

    mask_info = {
        **depth_info,
        "semantic": semantic_info,
        "combine": combine_info,
    }

    result = {
        "method": "depth_semantic_nav_mask",
        "path_points_px": points_px,
        "path_trajectory_px": trajectory_px,
        "path_points_ssl": points_ssl,
        "path_inward_xy": inward_xy,
        "pixel2real_ratio": ratio,
        "camera_para": camera_para,
        "nav_mask_depth_path": depth_mask_path,
        "nav_mask_structure_path": structure_path,
        "nav_mask_floor_path": floor_path,
        "nav_mask_path": nav_mask_path,
        "nav_mask_largest_path": largest_path,
        "nav_mask_info": mask_info,
        "nav_mask_sample": sample_meta,
        "path_inward": inward_meta,
        "inset_m": inset_m,
        "point_spacing_m": spacing_m,
        "depth_tolerance_m": tol_m,
        "image_size": [width, height],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(ssl_txt_path, "w", encoding="utf-8") as f:
        for x, y, z in points_ssl:
            f.write(f"{x} {y} {z}\n")
    with open(inward_txt_path, "w", encoding="utf-8") as f:
        for ix, iy in inward_xy:
            f.write(f"{ix} {iy}\n")

    draw_path_ring(
        image_path, points_px, overlay_path, trajectory_px=trajectory_px
    )
    print(
        f"✅ Depth floor path: {len(points_px)} waypoints, "
        f"{len(trajectory_px)} trajectory pts → {json_path}"
    )
    return result
