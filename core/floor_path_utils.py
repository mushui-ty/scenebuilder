"""地板路径通用工具：像素↔SSL、墙内线 snap、可视化。"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

MIN_PATH_POINTS = 8


def _resolve_ssl_path(output_dir: str) -> Optional[str]:
    cand = os.path.join(output_dir, "ssl.txt")
    if os.path.isfile(cand):
        return cand
    parent = os.path.dirname(output_dir.rstrip(os.sep))
    if parent:
        alt = os.path.join(parent, "ssl.txt")
        if os.path.isfile(alt):
            return alt
    return None


def _pixel_to_ssl_xy(px: float, py: float, ratio: float) -> tuple[float, float]:
    try:
        from .util_data import image_xy_to_ssl_xy
    except ImportError:
        from util_data import image_xy_to_ssl_xy  # type: ignore
    return image_xy_to_ssl_xy(float(px) * ratio, float(py) * ratio)


def _ssl_xy_to_pixel(sx: float, sy: float, ratio: float) -> tuple[float, float]:
    r = float(ratio)
    if r <= 0:
        return float(sx), float(sy)
    try:
        from .util_data import ssl_xy_to_image_xy
    except ImportError:
        from util_data import ssl_xy_to_image_xy  # type: ignore
    ix, iy = ssl_xy_to_image_xy(float(sx), float(sy))
    return ix / r, iy / r


def _build_room_ring_from_walls(walls: List[Dict[str, Any]]) -> List[tuple[float, float]]:
    ring: List[tuple[float, float]] = []
    for i, wall in enumerate(walls):
        p = (float(wall["p"][0]), float(wall["p"][1]))
        q = (float(wall["q"][0]), float(wall["q"][1]))
        if i == 0:
            ring.append(p)
        ring.append(q)
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring.pop()
    return ring


def _point_in_polygon(point: tuple[float, float], polygon: List[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, len(polygon) + 1):
        p2x, p2y = polygon[i % len(polygon)]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _wall_inward_direction(
    x1: float, y1: float, x2: float, y2: float, room_ring: List[tuple[float, float]]
) -> tuple[float, float]:
    import math

    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 0.0
    unit_x, unit_y = dx / length, dy / length
    perp_x, perp_y = -unit_y, unit_x
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    test1 = (cx + perp_x * 0.1, cy + perp_y * 0.1)
    test2 = (cx - perp_x * 0.1, cy - perp_y * 0.1)
    in1 = _point_in_polygon(test1, room_ring)
    in2 = _point_in_polygon(test2, room_ring)
    if not in1 and in2:
        return -perp_x, -perp_y
    if in1 and not in2:
        return perp_x, perp_y
    return -perp_x, -perp_y


def _point_to_segment(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float]:
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        dist = ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        return dist, x1, y1
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    fx = x1 + t * dx
    fy = y1 + t * dy
    dist = ((px - fx) ** 2 + (py - fy) ** 2) ** 0.5
    return dist, fx, fy


def _load_walls_with_inward(ssl_path: str) -> tuple[List[Dict[str, Any]], List[tuple[float, float]]]:
    try:
        from .util_data import parse_scene_input
    except ImportError:
        from util_data import parse_scene_input  # type: ignore

    with open(ssl_path, "r", encoding="utf-8") as f:
        scene = parse_scene_input(f.read())
    raw_walls = scene.get("wall") or []
    ring = _build_room_ring_from_walls(raw_walls)
    walls: List[Dict[str, Any]] = []
    for wall in raw_walls:
        x1, y1 = float(wall["p"][0]), float(wall["p"][1])
        x2, y2 = float(wall["q"][0]), float(wall["q"][1])
        inward = _wall_inward_direction(x1, y1, x2, y2, ring)
        walls.append({"p": (x1, y1), "q": (x2, y2), "inward": inward})
    return walls, ring


def _nearest_wall_inward_at_ssl(
    sx: float, sy: float, walls: List[Dict[str, Any]]
) -> tuple[float, float]:
    best_dist = float("inf")
    best_inward = (0.0, 0.0)
    for wall in walls:
        x1, y1 = wall["p"]
        x2, y2 = wall["q"]
        dist, _, _ = _point_to_segment(sx, sy, x1, y1, x2, y2)
        if dist < best_dist:
            best_dist = dist
            best_inward = wall["inward"]
    return best_inward


def _nearest_wall_info_at_ssl(
    sx: float, sy: float, walls: List[Dict[str, Any]]
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    best_dist = float("inf")
    best_foot = (sx, sy)
    best_inward = (0.0, 0.0)
    for wall in walls:
        x1, y1 = wall["p"]
        x2, y2 = wall["q"]
        dist, fx, fy = _point_to_segment(sx, sy, x1, y1, x2, y2)
        if dist < best_dist:
            best_dist = dist
            best_foot = (fx, fy)
            best_inward = wall["inward"]
    return best_dist, best_foot, best_inward


def _enforce_ssl_point_margin(
    sx: float,
    sy: float,
    walls: List[Dict[str, Any]],
    room_ring: List[tuple[float, float]],
    margin_m: float,
) -> tuple[float, float]:
    eps = max(1e-3, float(margin_m) * 0.01)
    target = float(margin_m) + eps
    for _ in range(12):
        dist, foot, inward = _nearest_wall_info_at_ssl(sx, sy, walls)
        inside = _point_in_polygon((sx, sy), room_ring)
        if inside and dist >= margin_m:
            return sx, sy
        ix, iy = inward
        sx = foot[0] + ix * target
        sy = foot[1] + iy * target
    return sx, sy


def _snap_ssl_point_to_room(
    sx: float,
    sy: float,
    walls: List[Dict[str, Any]],
    room_ring: List[tuple[float, float]],
    margin_m: float,
) -> tuple[float, float, bool]:
    if not walls or len(room_ring) < 3:
        return sx, sy, False

    ox, oy = sx, sy
    changed = False
    for _ in range(8):
        best_dist = float("inf")
        best_foot = (sx, sy)
        best_inward = (0.0, 0.0)
        for wall in walls:
            x1, y1 = wall["p"]
            x2, y2 = wall["q"]
            dist, fx, fy = _point_to_segment(sx, sy, x1, y1, x2, y2)
            if dist < best_dist:
                best_dist = dist
                best_foot = (fx, fy)
                best_inward = wall["inward"]

        inside = _point_in_polygon((sx, sy), room_ring)
        if inside and best_dist >= margin_m - 1e-6:
            break

        ix, iy = best_inward
        sx = best_foot[0] + ix * margin_m
        sy = best_foot[1] + iy * margin_m
        changed = True

    if abs(sx - ox) > 1e-9 or abs(sy - oy) > 1e-9:
        changed = True
    return sx, sy, changed


def _finalize_point_pixel(
    sx: float,
    sy: float,
    walls: List[Dict[str, Any]],
    room_ring: List[tuple[float, float]],
    margin_m: float,
    ratio: float,
) -> List[int]:
    sx, sy = _enforce_ssl_point_margin(sx, sy, walls, room_ring, margin_m)
    npx = int(round(_ssl_xy_to_pixel(sx, sy, ratio)[0]))
    npy = int(round(_ssl_xy_to_pixel(sx, sy, ratio)[1]))

    for _ in range(40):
        rsx, rsy = _pixel_to_ssl_xy(npx, npy, ratio)
        dist, _, inward = _nearest_wall_info_at_ssl(rsx, rsy, walls)
        if _point_in_polygon((rsx, rsy), room_ring) and dist >= margin_m - 1e-4:
            break
        if not _point_in_polygon((rsx, rsy), room_ring):
            sx, sy = _enforce_ssl_point_margin(rsx, rsy, walls, room_ring, margin_m)
            npx = int(round(_ssl_xy_to_pixel(sx, sy, ratio)[0]))
            npy = int(round(_ssl_xy_to_pixel(sx, sy, ratio)[1]))
            continue
        bx, by = _ssl_xy_to_pixel(rsx + inward[0] * 0.01, rsy + inward[1] * 0.01, ratio)
        ax, ay = _ssl_xy_to_pixel(rsx, rsy, ratio)
        dpx = 0 if abs(bx - ax) < 1e-6 else (1 if bx > ax else -1)
        dpy = 0 if abs(by - ay) < 1e-6 else (1 if by > ay else -1)
        if dpx == 0 and dpy == 0:
            sx, sy = _enforce_ssl_point_margin(rsx, rsy, walls, room_ring, margin_m + 0.01)
            npx = int(round(_ssl_xy_to_pixel(sx, sy, ratio)[0]))
            npy = int(round(_ssl_xy_to_pixel(sx, sy, ratio)[1]))
            continue
        npx += dpx
        npy += dpy

    return [npx, npy]


def snap_path_points_to_room(
    points_px: List[List[int]],
    output_dir: str,
    pixel2real_ratio: float,
    *,
    wall_margin_m: float,
) -> tuple[List[List[int]], Dict[str, Any]]:
    """保证每个点在墙环内，且离最近墙内线 ≥ margin（米）。"""
    meta: Dict[str, Any] = {
        "snapped": False,
        "wall_margin_m": wall_margin_m,
        "points_moved": 0,
    }
    if not points_px:
        return points_px, meta

    ssl_path = _resolve_ssl_path(output_dir)
    if not ssl_path:
        meta["skip_reason"] = "ssl.txt not found"
        return points_px, meta

    walls, ring = _load_walls_with_inward(ssl_path)
    if len(ring) < 3:
        meta["skip_reason"] = "room ring unavailable"
        return points_px, meta

    ratio = float(pixel2real_ratio)
    moved = 0
    out: List[List[int]] = []
    for px, py in points_px:
        sx, sy = _pixel_to_ssl_xy(px, py, ratio)
        nsx, nsy, changed = _snap_ssl_point_to_room(sx, sy, walls, ring, wall_margin_m)
        if abs(nsx - sx) > 1e-9 or abs(nsy - sy) > 1e-9:
            changed = True
        pt = _finalize_point_pixel(nsx, nsy, walls, ring, wall_margin_m, ratio)
        rsx, rsy = _pixel_to_ssl_xy(pt[0], pt[1], ratio)
        if abs(rsx - sx) > 1e-9 or abs(rsy - sy) > 1e-9:
            changed = True
        if changed:
            moved += 1
        out.append(pt)
    meta["snapped"] = True
    meta["points_moved"] = moved
    meta["ssl_path"] = ssl_path
    return out, meta


def compute_path_point_inward_xy(
    points_px: List[List[int]],
    output_dir: str,
    pixel2real_ratio: float,
    *,
    round_decimals: int = 6,
) -> tuple[List[List[float]], Dict[str, Any]]:
    meta: Dict[str, Any] = {"computed": False}
    if not points_px:
        return [], meta

    ssl_path = _resolve_ssl_path(output_dir)
    if not ssl_path:
        meta["skip_reason"] = "ssl.txt not found"
        return [], meta

    walls, ring = _load_walls_with_inward(ssl_path)
    if len(ring) < 3:
        meta["skip_reason"] = "room ring unavailable"
        return [], meta

    ratio = float(pixel2real_ratio)
    vectors: List[List[float]] = []
    for px, py in points_px:
        sx, sy = _pixel_to_ssl_xy(px, py, ratio)
        ix, iy = _nearest_wall_inward_at_ssl(sx, sy, walls)
        vectors.append([round(ix, round_decimals), round(iy, round_decimals)])

    meta["computed"] = True
    meta["ssl_path"] = ssl_path
    return vectors, meta


def pixel_points_to_ssl_ground(
    points_px: List[List[int]],
    pixel2real_ratio: float,
    *,
    round_decimals: int = 4,
) -> List[List[float]]:
    try:
        from .util_data import image_xy_to_ssl_xy
    except ImportError:
        from util_data import image_xy_to_ssl_xy  # type: ignore

    ratio = float(pixel2real_ratio)
    ssl_points = []
    for px, py in points_px:
        sx, sy = image_xy_to_ssl_xy(px * ratio, py * ratio)
        ssl_points.append([round(sx, round_decimals), round(sy, round_decimals), 0.0])
    return ssl_points


def draw_path_ring(
    image_path: str,
    points_px: List[List[int]],
    output_path: str,
    *,
    trajectory_px: Optional[List[List[int]]] = None,
) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_pts = trajectory_px if trajectory_px else points_px
    if line_pts:
        closed = line_pts + [line_pts[0]]
        draw.line([tuple(p) for p in closed], fill=(0, 220, 80, 230), width=4)
    for idx, (x, y) in enumerate(points_px):
        r = 8
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=(255, 60, 60, 255),
            outline=(255, 255, 255, 255),
            width=2,
        )
        draw.text((x + 10, y - 10), str(idx), fill=(255, 255, 0, 255))
    Image.alpha_composite(img, overlay).convert("RGB").save(output_path)
