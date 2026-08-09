"""Planar inner-surface vertex extraction, frustum clip, JSON export, and line overlay."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import util, util_bpy
except ImportError:
    import util  # type: ignore
    import util_bpy  # type: ignore

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore




def clip_polygon_to_render_frustum(
    vertices: np.ndarray,
    clip_mats: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Blender calc_matrix_camera homogeneous clip-space cull (consistent with visible geometry)."""
    proj, modelview = clip_mats
    return util_bpy.clip_polygon_to_render_frustum(vertices, proj, modelview)


def _as3(v) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size == 2:
        return np.array([arr[0], arr[1], 0.0], dtype=float)
    return arr[:3].astype(float)


def compute_wall_inner_quad(wall: Dict[str, Any], wall_height: float) -> np.ndarray:
    """Inner wall surface quad (in SSL, p/q are inner wall base points)."""
    s = np.asarray(wall["s"], dtype=float)[:2]
    e = np.asarray(wall["e"], dtype=float)[:2]
    h = float(wall_height)
    return np.array(
        [
            [s[0], s[1], 0.0],
            [e[0], e[1], 0.0],
            [e[0], e[1], h],
            [s[0], s[1], h],
        ],
        dtype=float,
    )


def compute_opening_inner_quad(
    opening: Dict[str, Any],
    wall: Dict[str, Any],
    inner_offset: float = 0.001,
) -> np.ndarray:
    """Door/window inner surface quad; logic matches util.create_door_or_window_mesh."""
    center = opening["center"]
    width = float(opening["width"])
    height = float(opening["height"])

    s = np.asarray(wall["s"], dtype=float)[:2]
    e = np.asarray(wall["e"], dtype=float)[:2]
    orientation = np.asarray(wall["orientation"], dtype=float)[:2]

    wall_dir = e - s
    wall_length = np.linalg.norm(wall_dir)
    if wall_length < 1e-9:
        return np.empty((0, 3), dtype=float)
    wall_dir_norm = wall_dir / wall_length

    center_2d = np.asarray(center[:2], dtype=float)
    along_wall = float(np.dot(center_2d - s, wall_dir_norm))

    half_width = width / 2.0
    half_height = height / 2.0
    z_bottom = float(center[2]) - half_height
    z_top = float(center[2]) + half_height
    pos_left = along_wall - half_width
    pos_right = along_wall + half_width

    vertices = np.array(
        [
            [s[0] + pos_left * wall_dir_norm[0], s[1] + pos_left * wall_dir_norm[1], z_bottom],
            [s[0] + pos_right * wall_dir_norm[0], s[1] + pos_right * wall_dir_norm[1], z_bottom],
            [s[0] + pos_right * wall_dir_norm[0], s[1] + pos_right * wall_dir_norm[1], z_top],
            [s[0] + pos_left * wall_dir_norm[0], s[1] + pos_left * wall_dir_norm[1], z_top],
        ],
        dtype=float,
    )
    if inner_offset:
        offset_vec = np.array([orientation[0], orientation[1], 0.0], dtype=float) * inner_offset
        vertices += offset_vec
    return vertices


def compute_floor_polygon(vertices_2d: Sequence[Sequence[float]]) -> np.ndarray:
    """Floor top face (z=0)."""
    return np.array([[_as3(v)[0], _as3(v)[1], 0.0] for v in vertices_2d], dtype=float)


def compute_ceiling_polygon(vertices_2d: Sequence[Sequence[float]], z_max: float) -> np.ndarray:
    """Ceiling bottom face (z=z_max)."""
    z = float(z_max)
    return np.array([[_as3(v)[0], _as3(v)[1], z] for v in vertices_2d], dtype=float)


def _camera_basis(camera_pos: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam = np.asarray(camera_pos, dtype=float)
    ref = np.asarray(reference, dtype=float)
    view_dir = ref - cam
    norm = np.linalg.norm(view_dir)
    if norm < 1e-9:
        view_dir = np.array([0.0, 0.0, -1.0], dtype=float)
    else:
        view_dir = view_dir / norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(view_dir, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(view_dir, world_up)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, view_dir)
    up = up / (np.linalg.norm(up) + 1e-9)
    return view_dir, right, up


def _start_vertex_index(vertices: np.ndarray, eps: float = 1e-9) -> int:
    """Pick ring start vertex: minimum z → y → x."""
    idx = 0
    for i in range(1, len(vertices)):
        vi, vj = vertices[i], vertices[idx]
        if vi[2] < vj[2] - eps:
            idx = i
        elif abs(vi[2] - vj[2]) <= eps:
            if vi[1] < vj[1] - eps:
                idx = i
            elif abs(vi[1] - vj[1]) <= eps and vi[0] < vj[0] - eps:
                idx = i
    return idx


def _signed_area_in_camera_plane(vertices: np.ndarray, camera_pos: np.ndarray) -> float:
    """Signed area in camera view plane; >0 means CCW as seen from camera."""
    verts = np.asarray(vertices, dtype=float)
    if len(verts) < 3:
        return 0.0
    centroid = verts.mean(axis=0)
    cam = np.asarray(camera_pos, dtype=float)
    _, basis_r, basis_u = _camera_basis(cam, centroid)
    rel = verts - centroid
    xs = np.array([np.dot(r, basis_r) for r in rel])
    ys = np.array([np.dot(r, basis_u) for r in rel])
    signed = 0.0
    m = len(verts)
    for i in range(m):
        j = (i + 1) % m
        signed += xs[i] * ys[j] - xs[j] * ys[i]
    return float(signed)


def finalize_ring_vertices(
    vertices: np.ndarray,
    camera_pos: np.ndarray,
) -> np.ndarray:
    """Preserve input boundary order (topology); ensure CCW facing camera; rotate to canonical start."""
    verts = np.asarray(vertices, dtype=float)
    if len(verts) < 3:
        return verts

    ordered = verts.copy()
    if _signed_area_in_camera_plane(ordered, camera_pos) < 0:
        ordered = ordered[::-1]

    start = _start_vertex_index(ordered)
    return np.roll(ordered, -start, axis=0)


def make_bpy_view_context(scene, camera_obj, width: int, height: int) -> Dict[str, Any]:
    """Build Blender frustum clip matrices and pixel projection (world_to_camera_view)."""
    from bpy_extras.object_utils import world_to_camera_view
    import mathutils

    clip_mats = util_bpy.build_render_frustum_clip_mats(scene, camera_obj, width, height)

    def to_pixel(co: np.ndarray) -> List[float]:
        u, v, z = world_to_camera_view(scene, camera_obj, mathutils.Vector(co.tolist()))
        if z <= 0:
            return [float("nan"), float("nan")]
        px = float(u) * width
        py = (1.0 - float(v)) * height
        return [px, py]

    def world_depth(co: np.ndarray) -> float:
        _u, _v, z = world_to_camera_view(scene, camera_obj, mathutils.Vector(co.tolist()))
        return float(z)

    return {
        "clip_mats": clip_mats,
        "to_pixel": to_pixel,
        "world_depth": world_depth,
        "camera_pos": np.array(camera_obj.matrix_world.translation, dtype=float),
    }


def make_bpy_pano_view_context(scene, camera_obj, width: int, height: int) -> Dict[str, Any]:
    """Build equirectangular panorama pixel projection; no frustum clip matrix."""
    world_to_camera = camera_obj.matrix_world.inverted()

    def to_pixel(co: np.ndarray) -> List[float]:
        import mathutils

        local = world_to_camera @ mathutils.Vector(co.tolist())
        direction = np.asarray([local.x, local.y, local.z], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            return [float("nan"), float("nan")]
        direction /= norm
        lon = float(np.arctan2(direction[0], -direction[2]))
        lat = float(np.arcsin(np.clip(direction[1], -1.0, 1.0)))
        u = (lon / (2.0 * np.pi) + 0.5) % 1.0
        v = 0.5 - lat / np.pi
        return [u * width, float(np.clip(v, 0.0, 1.0)) * height]

    return {
        "clip_mats": None,
        "to_pixel": to_pixel,
        "camera_pos": np.array(camera_obj.matrix_world.translation, dtype=float),
    }


def make_bpy_project_fn(scene, camera_obj, width: int, height: int):
    """Legacy compatibility wrapper."""
    ctx = make_bpy_view_context(scene, camera_obj, width, height)
    return None, ctx["to_pixel"]


def collect_planar_objects(
    context: Dict[str, Any],
    *,
    align_height: bool = True,
    show_wall: bool = True,
    show_door: bool = True,
    show_window: bool = True,
    include_floor: bool = True,
    include_ceiling: bool = True,
) -> List[Dict[str, Any]]:
    """Collect inner-surface rings for walls/doors/windows/floor/ceiling."""
    z_max = float(context["meta"]["z_max"])
    room_vertices = context["meta"]["vertices"]
    objects: List[Dict[str, Any]] = []

    if include_floor and room_vertices:
        objects.append(
            {
                "category": "floor",
                "id": "floor",
                "loops": [{"role": "outer", "vertices_3d": compute_floor_polygon(room_vertices)}],
            }
        )

    if include_ceiling and room_vertices and z_max > 0:
        objects.append(
            {
                "category": "ceiling",
                "id": "ceiling",
                "loops": [{"role": "outer", "vertices_3d": compute_ceiling_polygon(room_vertices, z_max)}],
            }
        )

    for wall_id, wall in context["walls"].items():
        wall_height = float(wall["height"])
        if wall_height <= 0:
            continue
        h = z_max if align_height else wall_height

        if show_wall:
            loops = [{"role": "outer", "vertices_3d": compute_wall_inner_quad(wall, h)}]
            if show_door:
                for door_id, door in wall.get("doors", {}).items():
                    hole = compute_opening_inner_quad(door, wall, inner_offset=0.0)
                    if len(hole) == 4:
                        loops.append({"role": "hole", "opening_type": "door", "opening_id": door_id, "vertices_3d": hole})
            if show_window:
                for window_id, window in wall.get("windows", {}).items():
                    hole = compute_opening_inner_quad(window, wall, inner_offset=0.0)
                    if len(hole) == 4:
                        loops.append(
                            {"role": "hole", "opening_type": "window", "opening_id": window_id, "vertices_3d": hole}
                        )
            objects.append({"category": "walls", "id": wall_id, "loops": loops})

        if show_door:
            for door_id, door in wall.get("doors", {}).items():
                quad = compute_opening_inner_quad(door, wall)
                if len(quad) == 4:
                    objects.append(
                        {
                            "category": "doors",
                            "id": door_id,
                            "wall_id": wall_id,
                            "loops": [{"role": "outer", "vertices_3d": quad}],
                        }
                    )
        if show_window:
            for window_id, window in wall.get("windows", {}).items():
                quad = compute_opening_inner_quad(window, wall)
                if len(quad) == 4:
                    objects.append(
                        {
                            "category": "windows",
                            "id": window_id,
                            "wall_id": wall_id,
                            "loops": [{"role": "outer", "vertices_3d": quad}],
                        }
                    )
    return objects


def _vertex_occluded_flags(
    vertices: np.ndarray,
    pixels: List[List[float]],
    category: str,
    obj_id: str,
    occlusion_fn: Optional[Callable[[str, str, np.ndarray], int]],
) -> List[int]:
    flags: List[int] = []
    for v, px in zip(vertices, pixels):
        if len(px) < 2 or not np.isfinite(px[0]) or not np.isfinite(px[1]):
            flags.append(1)
        elif occlusion_fn is None:
            flags.append(0)
        else:
            flags.append(int(occlusion_fn(category, obj_id, v)))
    return flags


def process_planar_objects_for_view(
    objects: List[Dict[str, Any]],
    clip_mats: Tuple[np.ndarray, np.ndarray],
    to_pixel: Callable[[np.ndarray], List[float]],
    camera_pos: np.ndarray,
    occlusion_fn: Optional[Callable[[str, str, np.ndarray], int]] = None,
) -> List[Dict[str, Any]]:
    """Frustum clip + preserve boundary order + CCW + pixel coordinates."""
    result: List[Dict[str, Any]] = []
    for obj in objects:
        color_key = f"line:{obj['category']}:{obj['id']}"
        line_color = list(util.semantic_entity_color(color_key))
        processed_loops = []
        for loop in obj["loops"]:
            clipped = clip_polygon_to_render_frustum(loop["vertices_3d"], clip_mats)
            if len(clipped) < 3:
                continue
            ordered = finalize_ring_vertices(clipped, camera_pos)
            pixels = [to_pixel(v) for v in ordered]
            entry = {
                "role": loop.get("role", "outer"),
                "vertices_3d": ordered.tolist(),
                "vertices_2d_px": pixels,
                "occluded": _vertex_occluded_flags(
                    ordered, pixels, obj["category"], obj["id"], occlusion_fn
                ),
            }
            if loop.get("opening_type"):
                entry["opening_type"] = loop["opening_type"]
            if loop.get("opening_id"):
                entry["opening_id"] = loop["opening_id"]
            processed_loops.append(entry)
        if not processed_loops:
            continue
        out = {
            "category": obj["category"],
            "id": obj["id"],
            "line_color": line_color,
            "loops": processed_loops,
        }
        if obj.get("wall_id"):
            out["wall_id"] = obj["wall_id"]
        result.append(out)
    return result


def process_planar_objects_for_pano(
    objects: List[Dict[str, Any]],
    to_pixel: Callable[[np.ndarray], List[float]],
    camera_pos: np.ndarray,
    occlusion_fn: Optional[Callable[[str, str, np.ndarray], int]] = None,
) -> List[Dict[str, Any]]:
    """No frustum clip; project full planar rings onto equirectangular panorama."""
    result: List[Dict[str, Any]] = []
    for obj in objects:
        color_key = f"line:{obj['category']}:{obj['id']}"
        line_color = list(util.semantic_entity_color(color_key))
        processed_loops = []
        for loop in obj["loops"]:
            vertices = np.asarray(loop["vertices_3d"], dtype=float)
            if len(vertices) < 3:
                continue
            ordered = finalize_ring_vertices(vertices, camera_pos)
            pixels = [to_pixel(v) for v in ordered]
            entry = {
                "role": loop.get("role", "outer"),
                "vertices_3d": ordered.tolist(),
                "vertices_2d_px": pixels,
                "occluded": _vertex_occluded_flags(
                    ordered, pixels, obj["category"], obj["id"], occlusion_fn
                ),
            }
            if loop.get("opening_type"):
                entry["opening_type"] = loop["opening_type"]
            if loop.get("opening_id"):
                entry["opening_id"] = loop["opening_id"]
            processed_loops.append(entry)
        if not processed_loops:
            continue
        out = {
            "category": obj["category"],
            "id": obj["id"],
            "line_color": line_color,
            "loops": processed_loops,
        }
        if obj.get("wall_id"):
            out["wall_id"] = obj["wall_id"]
        result.append(out)
    return result


def draw_lines_overlay(
    image_path: str,
    objects: List[Dict[str, Any]],
    output_path: str,
    line_width: int = 2,
    break_seams: bool = False,
) -> None:
    """Draw per-object vertex line loops on a copy of the render image (one color per object)."""
    if Image is None or ImageDraw is None:
        print("⚠️ PIL unavailable; skipping planar_faces line overlay")
        return
    if not os.path.exists(image_path):
        print(f"⚠️ Render image not found; skipping line overlay: {image_path}")
        return

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    image_width = img.size[0]
    for obj in objects:
        color = tuple(obj["line_color"]) + (255,)
        for loop in obj["loops"]:
            pts = loop.get("vertices_2d_px") or []
            valid = [(p[0], p[1]) for p in pts if len(p) >= 2 and np.isfinite(p[0]) and np.isfinite(p[1])]
            if len(valid) < 2:
                continue
            closed = valid + [valid[0]]
            if break_seams:
                for p0, p1 in zip(closed, closed[1:]):
                    if image_width > 0 and abs(p1[0] - p0[0]) > image_width / 2:
                        continue
                    draw.line([p0, p1], fill=color, width=line_width, joint="curve")
            else:
                draw.line(closed, fill=color, width=line_width, joint="curve")
            r = max(2, line_width)
            for x, y in valid:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path)
    print(f"✅ Planar vertex line overlay: {output_path}")


def export_planar_faces_for_view(
    context: Dict[str, Any],
    output_path: str,
    clip_mats: Tuple[np.ndarray, np.ndarray],
    to_pixel: Callable[[np.ndarray], List[float]],
    camera_pos: np.ndarray,
    width: int,
    height: int,
    *,
    align_height: bool = True,
    show_wall: bool = True,
    show_door: bool = True,
    show_window: bool = True,
    include_floor: bool = True,
    include_ceiling: bool = True,
    occlusion_fn: Optional[Callable[[str, str, np.ndarray], int]] = None,
    panoramic: bool = False,
) -> Tuple[str, str]:
    """Export {basename}_planar_faces.json and {basename}_lines.png."""
    view_dir = os.path.dirname(output_path) or "."
    base = os.path.splitext(os.path.basename(output_path))[0]
    json_path = os.path.join(view_dir, f"{base}_planar_faces.json")
    lines_path = os.path.join(view_dir, f"{base}_lines.png")

    raw_objects = collect_planar_objects(
        context,
        align_height=align_height,
        show_wall=show_wall,
        show_door=show_door,
        show_window=show_window,
        include_floor=include_floor,
        include_ceiling=include_ceiling,
    )
    if panoramic:
        objects = process_planar_objects_for_pano(
            raw_objects, to_pixel, camera_pos, occlusion_fn=occlusion_fn
        )
    else:
        objects = process_planar_objects_for_view(
            raw_objects, clip_mats, to_pixel, camera_pos, occlusion_fn=occlusion_fn
        )

    payload = {
        "image": {
            "path": output_path,
            "width": width,
            "height": height,
            "lines_overlay": lines_path,
            "projection": "equirectangular" if panoramic else "perspective",
        },
        "objects": objects,
    }
    os.makedirs(view_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✅ Planar inner-surface vertex JSON: {json_path}")

    draw_lines_overlay(output_path, objects, lines_path, break_seams=panoramic)
    return json_path, lines_path
