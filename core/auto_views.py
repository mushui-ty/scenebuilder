#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据地板路径自动生成 render_view 相机位姿（views=auto）。"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

AUTO_VIEW_WIDTH = 1000
AUTO_VIEW_HEIGHT = 1000
PATH_SAMPLE_STRIDE = 4
WORLD_UP = (0.0, 0.0, 1.0)


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
    """场景 3D 包围盒中心（SSL 坐标）。"""
    meta = context["meta"]
    cx, cy = float(meta["center"][0]), float(meta["center"][1])
    z_max = float(meta["z_max"])
    for box in context.get("boxes", {}).values():
        center = box.get("center") or [0.0, 0.0, 0.0]
        scale = box.get("scale") or [0.0, 0.0, 0.0]
        z_top = float(center[2]) + float(scale[2]) / 2.0
        z_max = max(z_max, z_top)
    return [cx, cy, z_max / 2.0]


def sample_path_indices(n_points: int, stride: int = PATH_SAMPLE_STRIDE) -> List[int]:
    """从闭环路径点中按 stride 采样索引（含 0）。"""
    if n_points <= 0:
        return []
    return list(range(0, n_points, max(1, int(stride))))


def random_camera_z(
    rng: random.Random,
    wall_z_max: float,
    lo: float = 0.5,
    hi: float = 2.5,
) -> float:
    """采样相机高度，保证小于墙高最大值。"""
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
    """沿射线 AB 调整俯仰角（度），返回新的 look_at。"""
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
    """绕 world_up 偏航（度，正=从左往右看时向左转），返回新的 look_at。"""
    eye = np.asarray(camera_pos, dtype=float)
    dist = _look_distance(camera_pos, look_at)
    forward = _forward_unit(camera_pos, look_at)
    up = np.asarray(world_up, dtype=float)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    forward_new = _rotate_vector(forward, up, math.radians(float(yaw_deg)))
    forward_new = forward_new / max(float(np.linalg.norm(forward_new)), 1e-9)
    return (eye + forward_new * dist).tolist()


def build_single_auto_view_spec(
    path_index: int,
    path_point: Sequence[float],
    bbox_center: Sequence[float],
    wall_z_max: float,
    rng: random.Random,
) -> Dict[str, Any]:
    x, y = float(path_point[0]), float(path_point[1])
    z_cam = random_camera_z(rng, wall_z_max)
    camera_pos = [x, y, z_cam]
    look_base = list(bbox_center)
    pitch = rng.uniform(-40.0, 5.0)
    look_at = adjust_look_at_pitch(camera_pos, look_base, pitch)
    fov = random_fov_deg(rng)
    name = f"auto_path_{path_index:04d}"
    return {
        "type": "single",
        "path_index": int(path_index),
        "camera_position": camera_pos,
        "look_at_target": look_at,
        "up_vector": list(WORLD_UP),
        "manual_fov": float(fov),
        "pitch_deg": float(pitch),
        "width": AUTO_VIEW_WIDTH,
        "height": AUTO_VIEW_HEIGHT,
        "_view_name": name,
    }


def build_sequence_auto_view_spec(
    path_index: int,
    path_point: Sequence[float],
    bbox_center: Sequence[float],
    rng: random.Random,
) -> Dict[str, Any]:
    x, y = float(path_point[0]), float(path_point[1])
    camera_pos = [x, y, 1.5]
    look_center = list(bbox_center)
    yaw_left = rng.uniform(10.0, 40.0)
    yaw_right = rng.uniform(0.0, 40.0)
    look_left = adjust_look_at_yaw(camera_pos, look_center, yaw_left)
    look_right = adjust_look_at_yaw(camera_pos, look_center, -yaw_right)
    fov = random_fov_deg(rng)
    name = f"auto_path_{path_index:04d}_seq"
    return {
        "type": "sequence",
        "path_index": int(path_index),
        "camera_positions": [camera_pos, camera_pos, camera_pos],
        "look_at_targets": [look_left, look_center, look_right],
        "up_vectors": [list(WORLD_UP)] * 3,
        "manual_fov": float(fov),
        "yaw_left_deg": float(yaw_left),
        "yaw_right_deg": float(yaw_right),
        "width": AUTO_VIEW_WIDTH,
        "height": AUTO_VIEW_HEIGHT,
        "_view_name": name,
    }


def build_auto_views_from_path(
    path_points_ssl: Sequence[Sequence[float]],
    context: Dict[str, Any],
    *,
    rng: Optional[random.Random] = None,
    stride: int = PATH_SAMPLE_STRIDE,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """生成 auto 模式下的全部视角 spec。返回 (name→spec, 渲染顺序名列表)。"""
    rng = rng or random.Random()
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
        spec = build_single_auto_view_spec(idx, pt, bbox_center, wall_z_max, rng)
        name = spec.pop("_view_name")
        specs[name] = spec
        names.append(name)

    seq_idx = rng.randrange(n)
    seq_spec = build_sequence_auto_view_spec(seq_idx, points[seq_idx], bbox_center, rng)
    seq_name = seq_spec.pop("_view_name")
    specs[seq_name] = seq_spec
    names.append(seq_name)

    return specs, names


def write_auto_views_manifest(output_dir: str, specs: Dict[str, Dict[str, Any]]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "auto_views.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2, ensure_ascii=False)
    return path
