#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-generate render_view camera poses from floor path (views=auto)."""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

AUTO_VIEW_WIDTH = 500
AUTO_VIEW_HEIGHT = 500
TOPDOWN_VIEW_WIDTH = 1000
TOPDOWN_VIEW_HEIGHT = 1000
AUTO_PATH_TOPDOWN_NAME = "auto_path_topdown"
AUTO_PATH_PANO_NUM_DEFAULT = 3
PATH_SAMPLE_STRIDE = 4
PATH_SAMPLE_MID_MIN = 40
PATH_SAMPLE_MID_MAX = 100
PATH_SAMPLE_TARGET_MID = 15
PATH_SAMPLE_TARGET_LARGE = 20
WORLD_UP = (0.0, 0.0, 1.0)
DEFAULT_AUTO_PATH_ROLL_MAX_DEG = 40.0
DEFAULT_AUTO_PATH_ROLL_STD_DEG = 16.0
DEFAULT_TOPDOWN_PITCH_MAX_DEG = 20.0
DEFAULT_TOPDOWN_YAW_MAX_DEG = 20.0
DEFAULT_TOPDOWN_ROLL_MAX_DEG = 180.0


def _evenly_spaced_indices(n_points: int, target_count: int) -> List[int]:
    """Uniformly sample target_count indices in [0, n_points-1] (includes first, tries to include last)."""
    if n_points <= 0:
        return []
    target_count = max(1, int(target_count))
    if target_count >= n_points:
        return list(range(n_points))
    if target_count == 1:
        return [0]
    indices = [
        int(round(i * (n_points - 1) / (target_count - 1)))
        for i in range(target_count)
    ]
    deduped: List[int] = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped


def list_auto_path_single_view_names(specs: Dict[str, Any]) -> List[str]:
    """Sorted auto_path single-frame view names (excludes ``*_seq`` and ``auto_path_topdown``)."""
    return sorted(
        n for n in specs
        if n.startswith("auto_path_")
        and not n.endswith("_seq")
        and n != AUTO_PATH_TOPDOWN_NAME
    )


def select_auto_path_pano_view_names(
    specs: Dict[str, Any],
    pano_num: int = AUTO_PATH_PANO_NUM_DEFAULT,
) -> List[str]:
    """Pick ``pano_num`` evenly spaced auto_path singles for panorama (+ optional pano geometry)."""
    names = list_auto_path_single_view_names(specs)
    if not names:
        return []
    indices = _evenly_spaced_indices(len(names), max(1, int(pano_num)))
    return [names[i] for i in indices]


def apply_auto_path_pano_flags(
    specs: Dict[str, Any],
    pano_num: int = AUTO_PATH_PANO_NUM_DEFAULT,
) -> List[str]:
    """Set ``render_pano`` on auto_path single specs; return selected view names."""
    selected = set(select_auto_path_pano_view_names(specs, pano_num))
    for name in list_auto_path_single_view_names(specs):
        specs[name]["render_pano"] = name in selected
    return sorted(selected)


def sample_path_indices(n_points: int, stride: int = PATH_SAMPLE_STRIDE) -> List[int]:
    """Sample indices along closed path (includes 0).

    - n < 40: every ``stride`` (default 4)
    - 40 <= n <= 100: uniformly sample 15
    - n > 100: uniformly sample 20
    """
    if n_points <= 0:
        return []
    if n_points < PATH_SAMPLE_MID_MIN:
        return list(range(0, n_points, max(1, int(stride))))
    if n_points <= PATH_SAMPLE_MID_MAX:
        return _evenly_spaced_indices(n_points, PATH_SAMPLE_TARGET_MID)
    return _evenly_spaced_indices(n_points, PATH_SAMPLE_TARGET_LARGE)


def _rotate_vector(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.asarray(v, dtype=float)
    axis = axis / norm
    v = np.asarray(v, dtype=float)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def scene_bbox_center(context: Dict[str, Any]) -> List[float]:
    """Scene 3D bounding box center (SSL coordinates)."""
    meta = context["meta"]
    cx, cy = float(meta["center"][0]), float(meta["center"][1])
    z_max = float(meta["z_max"])
    for box in context.get("boxes", {}).values():
        center = box.get("center") or [0.0, 0.0, 0.0]
        scale = box.get("scale") or [0.0, 0.0, 0.0]
        z_top = float(center[2]) + float(scale[2]) / 2.0
        z_max = max(z_max, z_top)
    return [cx, cy, z_max / 2.0]


def random_camera_z(
    rng: random.Random,
    wall_z_max: float,
    lo: float = 1.5,
    hi: float = 2.5,
) -> float:
    """Sample camera height, ensuring it stays below max wall height."""
    cap = min(float(hi), float(wall_z_max) - 1e-3)
    floor = float(lo)
    if cap <= floor:
        cap = max(floor, float(wall_z_max) * 0.85)
    while True:
        z = rng.uniform(floor, cap)
        if z < float(wall_z_max):
            return z


def random_fov_deg(rng: random.Random, lo: float = 60.0, hi: float = 90.0) -> float:
    return rng.uniform(float(lo), float(hi))


def _forward_unit(camera_pos: Sequence[float], look_at: Sequence[float]) -> np.ndarray:
    delta = np.asarray(look_at, dtype=float) - np.asarray(camera_pos, dtype=float)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-9:
        return np.array([0.0, 0.0, -1.0], dtype=float)
    return delta / dist


def _look_distance(camera_pos: Sequence[float], look_at: Sequence[float], default: float = 5.0) -> float:
    dist = float(np.linalg.norm(np.asarray(look_at, dtype=float) - np.asarray(camera_pos, dtype=float)))
    return dist if dist > 1e-6 else default


def adjust_look_at_pitch(
    camera_pos: Sequence[float],
    look_at: Sequence[float],
    pitch_deg: float,
    world_up: Sequence[float] = WORLD_UP,
) -> List[float]:
    """Adjust pitch (degrees) along ray AB; return new look_at."""
    eye = np.asarray(camera_pos, dtype=float)
    target = np.asarray(look_at, dtype=float)
    dist = _look_distance(camera_pos, look_at)
    forward = _forward_unit(camera_pos, look_at)
    up = np.asarray(world_up, dtype=float)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-9:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        right = right / right_norm
    forward_new = _rotate_vector(forward, right, math.radians(float(pitch_deg)))
    forward_new = forward_new / max(float(np.linalg.norm(forward_new)), 1e-9)
    return (eye + forward_new * dist).tolist()


def adjust_look_at_yaw(
    camera_pos: Sequence[float],
    look_at: Sequence[float],
    yaw_deg: float,
    world_up: Sequence[float] = WORLD_UP,
) -> List[float]:
    """Yaw (degrees) about world_up; positive = turn left when looking left-to-right; return new look_at."""
    eye = np.asarray(camera_pos, dtype=float)
    dist = _look_distance(camera_pos, look_at)
    forward = _forward_unit(camera_pos, look_at)
    up = np.asarray(world_up, dtype=float)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    forward_new = _rotate_vector(forward, up, math.radians(float(yaw_deg)))
    forward_new = forward_new / max(float(np.linalg.norm(forward_new)), 1e-9)
    return (eye + forward_new * dist).tolist()


def _resolve_base_up_for_roll(
    forward: np.ndarray,
    base_up: Sequence[float] = WORLD_UP,
) -> np.ndarray:
    """Pick a reference up for roll when ``forward`` is nearly parallel to ``base_up``."""
    up = np.asarray(base_up, dtype=float)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    fwd = np.asarray(forward, dtype=float)
    if abs(float(np.dot(fwd, up))) > 1.0 - 1e-6:
        for alt in (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
            if abs(float(np.dot(fwd, alt))) <= 1.0 - 1e-6:
                return alt
    return up


def sample_roll_perturbation_deg(
    rng: random.Random,
    max_deg: float,
    std_deg: Optional[float] = None,
) -> float:
    """Sample roll (degrees) in [-max_deg, max_deg], truncated normal centered at 0."""
    max_deg = float(max_deg)
    if max_deg <= 0.0:
        return 0.0
    std = float(std_deg if std_deg is not None else max_deg * 0.4)
    std = max(std, 1e-6)
    for _ in range(64):
        roll = float(rng.gauss(0.0, std))
        if -max_deg <= roll <= max_deg:
            return roll
    return float(max(-max_deg, min(max_deg, roll)))


def apply_roll_to_up_vector(
    camera_pos: Sequence[float],
    look_at: Sequence[float],
    roll_deg: float,
    base_up: Sequence[float] = WORLD_UP,
) -> List[float]:
    """Roll camera about view axis k; keep camera/look fixed, return perturbed up_vector."""
    roll_deg = float(roll_deg)
    forward = _forward_unit(camera_pos, look_at)
    up_ref = _resolve_base_up_for_roll(forward, base_up)
    up_perp = up_ref - forward * float(np.dot(up_ref, forward))
    perp_norm = float(np.linalg.norm(up_perp))
    if perp_norm < 1e-9:
        up_perp = np.cross(forward, np.array([1.0, 0.0, 0.0], dtype=float))
        perp_norm = float(np.linalg.norm(up_perp))
        if perp_norm < 1e-9:
            up_perp = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=float))
            perp_norm = float(np.linalg.norm(up_perp))
    up_perp = up_perp / max(perp_norm, 1e-9)
    if abs(roll_deg) < 1e-9:
        return up_perp.tolist()
    rolled = _rotate_vector(up_perp, forward, math.radians(roll_deg))
    norm = float(np.linalg.norm(rolled))
    if norm < 1e-9:
        return up_perp.tolist()
    return (rolled / norm).tolist()


def _load_auto_path_roll_config() -> Tuple[float, float]:
    try:
        from .config_utils import load_config
    except ImportError:
        from config_utils import load_config  # type: ignore
    cfg = load_config()
    max_deg = float(cfg.get("auto_path_roll_max_deg", DEFAULT_AUTO_PATH_ROLL_MAX_DEG))
    std_deg = float(cfg.get("auto_path_roll_std_deg", DEFAULT_AUTO_PATH_ROLL_STD_DEG))
    return max_deg, std_deg


def _load_topdown_perturb_config() -> Tuple[float, float, float, float, float, float]:
    try:
        from .config_utils import load_config
    except ImportError:
        from config_utils import load_config  # type: ignore
    cfg = load_config()

    def _pair(max_key: str, std_key: str, default_max: float) -> Tuple[float, float]:
        max_deg = float(cfg.get(max_key, default_max))
        std_deg = float(cfg.get(std_key, max_deg * 0.4))
        return max_deg, std_deg

    pitch_max, pitch_std = _pair(
        "topdown_pitch_max_deg", "topdown_pitch_std_deg", DEFAULT_TOPDOWN_PITCH_MAX_DEG
    )
    yaw_max, yaw_std = _pair(
        "topdown_yaw_max_deg", "topdown_yaw_std_deg", DEFAULT_TOPDOWN_YAW_MAX_DEG
    )
    roll_max, roll_std = _pair(
        "topdown_roll_max_deg", "topdown_roll_std_deg", DEFAULT_TOPDOWN_ROLL_MAX_DEG
    )
    return pitch_max, pitch_std, yaw_max, yaw_std, roll_max, roll_std


def _camera_frame_axes(
    camera_pos: Sequence[float],
    look_at: Sequence[float],
    up_vector: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OpenCV camera axes in world frame: forward (+Z), right (+X), down (+Y)."""
    forward = _forward_unit(camera_pos, look_at)
    up = np.asarray(up_vector, dtype=float)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-9:
        right = np.cross(forward, np.array([1.0, 0.0, 0.0], dtype=float))
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-9:
            right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=float))
            right_norm = float(np.linalg.norm(right))
    right = right / max(right_norm, 1e-9)
    down = np.cross(forward, right)
    down = down / max(float(np.linalg.norm(down)), 1e-9)
    return forward, right, down


def adjust_look_at_camera_axis(
    camera_pos: Sequence[float],
    look_at: Sequence[float],
    axis: Sequence[float],
    angle_deg: float,
) -> List[float]:
    """Rotate look direction about ``axis`` through ``camera_pos``; camera position fixed."""
    eye = np.asarray(camera_pos, dtype=float)
    dist = _look_distance(camera_pos, look_at)
    forward = _forward_unit(camera_pos, look_at)
    forward_new = _rotate_vector(forward, axis, math.radians(float(angle_deg)))
    forward_new = forward_new / max(float(np.linalg.norm(forward_new)), 1e-9)
    return (eye + forward_new * dist).tolist()


def apply_topdown_pose_perturbations(
    camera_pos: Sequence[float],
    look_at: Sequence[float],
    up_vector: Sequence[float],
    rng: random.Random,
    *,
    pitch_max_deg: Optional[float] = None,
    pitch_std_deg: Optional[float] = None,
    yaw_max_deg: Optional[float] = None,
    yaw_std_deg: Optional[float] = None,
    roll_max_deg: Optional[float] = None,
    roll_std_deg: Optional[float] = None,
) -> Tuple[List[float], List[float], float, float, float]:
    """Camera-local pitch (right) → yaw (down) → roll (forward); return look, up, angles."""
    if any(
        v is None
        for v in (pitch_max_deg, pitch_std_deg, yaw_max_deg, yaw_std_deg, roll_max_deg, roll_std_deg)
    ):
        cfg = _load_topdown_perturb_config()
        pitch_max_deg = cfg[0] if pitch_max_deg is None else pitch_max_deg
        pitch_std_deg = cfg[1] if pitch_std_deg is None else pitch_std_deg
        yaw_max_deg = cfg[2] if yaw_max_deg is None else yaw_max_deg
        yaw_std_deg = cfg[3] if yaw_std_deg is None else yaw_std_deg
        roll_max_deg = cfg[4] if roll_max_deg is None else roll_max_deg
        roll_std_deg = cfg[5] if roll_std_deg is None else roll_std_deg

    cam = list(camera_pos)
    look = list(look_at)
    up = list(up_vector)

    _, right, _ = _camera_frame_axes(cam, look, up)
    pitch_deg = sample_roll_perturbation_deg(rng, pitch_max_deg, pitch_std_deg)
    look = adjust_look_at_camera_axis(cam, look, right, pitch_deg)

    _, _, down = _camera_frame_axes(cam, look, up)
    yaw_deg = sample_roll_perturbation_deg(rng, yaw_max_deg, yaw_std_deg)
    look = adjust_look_at_camera_axis(cam, look, down, yaw_deg)

    roll_deg = sample_roll_perturbation_deg(rng, roll_max_deg, roll_std_deg)
    up = apply_roll_to_up_vector(cam, look, roll_deg, base_up=up)
    return look, up, float(pitch_deg), float(yaw_deg), float(roll_deg)


def topdown_camera_para_path(y_dir: str) -> str:
    return os.path.join(y_dir, "topdown", "topdown_camera_para.json")


def load_topdown_camera_para(y_dir: str) -> Optional[Dict[str, Any]]:
    path = topdown_camera_para_path(y_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    for key in ("camera_position", "look_at_target", "world_up"):
        if key not in data:
            return None
    return data


def build_auto_path_topdown_spec(
    y_dir: str,
    rng: Optional[random.Random] = None,
) -> Optional[Dict[str, Any]]:
    """Build ``auto_path_topdown`` view spec from completed ``topdown/topdown_camera_para.json``."""
    para = load_topdown_camera_para(y_dir)
    if para is None:
        return None
    rng = rng or random.Random()
    cam = [float(x) for x in para["camera_position"]]
    look = [float(x) for x in para["look_at_target"]]
    up = [float(x) for x in para["world_up"]]
    look, up, pitch_deg, yaw_deg, roll_deg = apply_topdown_pose_perturbations(cam, look, up, rng)
    fov_y = para.get("fov_y")
    manual_fov = math.degrees(float(fov_y)) if fov_y is not None else None
    spec: Dict[str, Any] = {
        "type": "single",
        "camera_position": cam,
        "look_at_target": look,
        "up_vector": up,
        "width": TOPDOWN_VIEW_WIDTH,
        "height": TOPDOWN_VIEW_HEIGHT,
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        "roll_deg": roll_deg,
        "source": "topdown_camera_para",
    }
    if manual_fov is not None:
        spec["manual_fov"] = float(manual_fov)
    return spec


def ensure_auto_path_topdown_spec(
    y_dir: str,
    *,
    resume: bool = True,
    rng: Optional[random.Random] = None,
) -> Optional[Dict[str, Any]]:
    """Load or sample ``auto_path_topdown`` spec; persist to ``auto_views.json`` when created."""
    path = os.path.join(y_dir, "auto_views.json")
    specs: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            specs = loaded

    if resume and AUTO_PATH_TOPDOWN_NAME in specs:
        return specs[AUTO_PATH_TOPDOWN_NAME]

    spec = build_auto_path_topdown_spec(y_dir, rng=rng)
    if spec is None:
        return None

    specs[AUTO_PATH_TOPDOWN_NAME] = spec
    write_auto_views_manifest(y_dir, specs)
    return spec


def try_add_auto_path_topdown_to_manifest(
    y_dir: str,
    specs: Dict[str, Dict[str, Any]],
    names: List[str],
    *,
    resume: bool = True,
) -> bool:
    """Ensure manifest contains ``auto_path_topdown`` when topdown camera_para exists."""
    if resume and AUTO_PATH_TOPDOWN_NAME in specs:
        if AUTO_PATH_TOPDOWN_NAME not in names:
            names.insert(0, AUTO_PATH_TOPDOWN_NAME)
        return True
    resolved = ensure_auto_path_topdown_spec(y_dir, resume=resume)
    if resolved is None:
        return False
    specs[AUTO_PATH_TOPDOWN_NAME] = resolved
    if AUTO_PATH_TOPDOWN_NAME not in names:
        names.insert(0, AUTO_PATH_TOPDOWN_NAME)
    return True


def build_single_auto_view_spec(
    path_index: int,
    path_point: Sequence[float],
    bbox_center: Sequence[float],
    wall_z_max: float,
    rng: random.Random,
    *,
    width: int = AUTO_VIEW_WIDTH,
    height: int = AUTO_VIEW_HEIGHT,
    roll_max_deg: Optional[float] = None,
    roll_std_deg: Optional[float] = None,
) -> Dict[str, Any]:
    if roll_max_deg is None or roll_std_deg is None:
        cfg_max, cfg_std = _load_auto_path_roll_config()
        roll_max_deg = cfg_max if roll_max_deg is None else roll_max_deg
        roll_std_deg = cfg_std if roll_std_deg is None else roll_std_deg
    x, y = float(path_point[0]), float(path_point[1])
    z_cam = random_camera_z(rng, wall_z_max)
    camera_pos = [x, y, z_cam]
    look_base = list(bbox_center)
    pitch = rng.uniform(-40.0, 5.0)
    look_at = adjust_look_at_pitch(camera_pos, look_base, pitch)
    roll_deg = sample_roll_perturbation_deg(rng, roll_max_deg, roll_std_deg)
    up_vector = apply_roll_to_up_vector(camera_pos, look_at, roll_deg)
    fov = random_fov_deg(rng)
    name = f"auto_path_{path_index:04d}"
    return {
        "type": "single",
        "path_index": int(path_index),
        "camera_position": camera_pos,
        "look_at_target": look_at,
        "up_vector": up_vector,
        "manual_fov": float(fov),
        "pitch_deg": float(pitch),
        "roll_deg": float(roll_deg),
        "width": int(width),
        "height": int(height),
        "_view_name": name,
    }


def build_sequence_auto_view_spec(
    path_index: int,
    path_point: Sequence[float],
    bbox_center: Sequence[float],
    rng: random.Random,
    *,
    width: int = AUTO_VIEW_WIDTH,
    height: int = AUTO_VIEW_HEIGHT,
    roll_max_deg: Optional[float] = None,
    roll_std_deg: Optional[float] = None,
) -> Dict[str, Any]:
    if roll_max_deg is None or roll_std_deg is None:
        cfg_max, cfg_std = _load_auto_path_roll_config()
        roll_max_deg = cfg_max if roll_max_deg is None else roll_max_deg
        roll_std_deg = cfg_std if roll_std_deg is None else roll_std_deg
    x, y = float(path_point[0]), float(path_point[1])
    camera_pos = [x, y, 1.5]
    look_center = list(bbox_center)
    yaw_left = rng.uniform(10.0, 40.0)
    yaw_right = rng.uniform(0.0, 40.0)
    look_left = adjust_look_at_yaw(camera_pos, look_center, yaw_left)
    look_right = adjust_look_at_yaw(camera_pos, look_center, -yaw_right)
    look_targets = [look_left, look_center, look_right]
    up_vectors: List[List[float]] = []
    roll_degs: List[float] = []
    for look_at in look_targets:
        roll_deg = sample_roll_perturbation_deg(rng, roll_max_deg, roll_std_deg)
        up_vectors.append(apply_roll_to_up_vector(camera_pos, look_at, roll_deg))
        roll_degs.append(roll_deg)
    fov = random_fov_deg(rng)
    name = f"auto_path_{path_index:04d}_seq"
    return {
        "type": "sequence",
        "path_index": int(path_index),
        "camera_positions": [camera_pos, camera_pos, camera_pos],
        "look_at_targets": look_targets,
        "up_vectors": up_vectors,
        "manual_fov": float(fov),
        "yaw_left_deg": float(yaw_left),
        "yaw_right_deg": float(yaw_right),
        "roll_degs": roll_degs,
        "width": int(width),
        "height": int(height),
        "_view_name": name,
    }


def _closest_path_index(points: Sequence[Sequence[float]], center_xy: Sequence[float]) -> int:
    """Return index of the path point whose XY is closest to ``center_xy``."""
    if not points:
        return 0
    cx, cy = float(center_xy[0]), float(center_xy[1])
    best_idx = 0
    best_dist = float("inf")
    for idx, pt in enumerate(points):
        dx = float(pt[0]) - cx
        dy = float(pt[1]) - cy
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def build_auto_views_from_path(
    path_points_ssl: Sequence[Sequence[float]],
    context: Dict[str, Any],
    *,
    rng: Optional[random.Random] = None,
    stride: int = PATH_SAMPLE_STRIDE,
    width: int = AUTO_VIEW_WIDTH,
    height: int = AUTO_VIEW_HEIGHT,
    pano_num: int = AUTO_PATH_PANO_NUM_DEFAULT,
    roll_max_deg: Optional[float] = None,
    roll_std_deg: Optional[float] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Build all view specs for auto mode. Returns (name→spec, render order name list)."""
    rng = rng or random.Random()
    if roll_max_deg is None or roll_std_deg is None:
        cfg_max, cfg_std = _load_auto_path_roll_config()
        roll_max_deg = cfg_max if roll_max_deg is None else roll_max_deg
        roll_std_deg = cfg_std if roll_std_deg is None else roll_std_deg
    points = [list(p) for p in path_points_ssl]
    n = len(points)
    if n == 0:
        return {}, []

    bbox_center = scene_bbox_center(context)
    wall_z_max = float(context["meta"]["z_max"])
    specs: Dict[str, Dict[str, Any]] = {}
    names: List[str] = []

    for idx in sample_path_indices(n, stride=stride):
        pt = points[idx]
        spec = build_single_auto_view_spec(
            idx, pt, bbox_center, wall_z_max, rng,
            width=width, height=height,
            roll_max_deg=roll_max_deg, roll_std_deg=roll_std_deg,
        )
        name = spec.pop("_view_name")
        specs[name] = spec
        names.append(name)

    seq_idx = _closest_path_index(points, bbox_center)
    seq_spec = build_sequence_auto_view_spec(
        seq_idx, points[seq_idx], bbox_center, rng,
        width=width, height=height,
        roll_max_deg=roll_max_deg, roll_std_deg=roll_std_deg,
    )
    seq_name = seq_spec.pop("_view_name")
    specs[seq_name] = seq_spec
    names.append(seq_name)

    apply_auto_path_pano_flags(specs, pano_num)

    return specs, names


def write_auto_views_manifest(output_dir: str, specs: Dict[str, Dict[str, Any]]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "auto_views.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2, ensure_ascii=False)
    return path
