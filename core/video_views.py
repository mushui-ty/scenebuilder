#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smooth closed-loop camera trajectories for --video."""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.interpolate import CubicSpline
except ImportError:  # pragma: no cover
    CubicSpline = None  # type: ignore

from .auto_views import (
    AUTO_VIEW_HEIGHT,
    AUTO_VIEW_WIDTH,
    WORLD_UP,
    random_fov_deg,
    scene_bbox_center,
)

VIDEO_FRAME_SPACING_M = 0.2
VIDEO_MIN_FRAMES = 3
VIDEO_HEIGHT_LO = 1.3
VIDEO_HEIGHT_HI = 1.8
VIDEO_CENTER_WIDTH = 896
VIDEO_CENTER_HEIGHT = 896
VIDEO_PANO_RESOLUTION = 1024
VIDEO_TANGENT_PANO_RESOLUTION = VIDEO_PANO_RESOLUTION  # legacy alias
VIDEO_TANGENT_LOOK_DIST = 3.0
VIDEO_TANGENT_SMOOTH_WINDOW = 9
VIDEO_TANGENT_CENTER_BIAS = 0.35


def _ring_xy(path_points_ssl: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray([[float(p[0]), float(p[1])] for p in path_points_ssl], dtype=float)
    if len(pts) < 3:
        raise ValueError("video path requires at least 3 waypoints")
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
        pts = pts[:-1]
    return pts


def closed_path_length_m(path_points_ssl: Sequence[Sequence[float]]) -> float:
    ring = _ring_xy(path_points_ssl)
    closed = np.vstack([ring, ring[:1]])
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))


def compute_video_frame_count(
    path_points_ssl: Sequence[Sequence[float]],
    *,
    spacing_m: float = VIDEO_FRAME_SPACING_M,
    min_frames: int = VIDEO_MIN_FRAMES,
) -> int:
    """Pick frame count so adjacent samples are ~spacing_m apart on the closed path."""
    spacing_m = max(1e-3, float(spacing_m))
    total = closed_path_length_m(path_points_ssl)
    if total <= 1e-9:
        raise ValueError("floor path length is zero")
    return max(int(min_frames), int(round(total / spacing_m)))


def _arc_length_params(ring: np.ndarray) -> Tuple[np.ndarray, float]:
    closed = np.vstack([ring, ring[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-9:
        raise ValueError("floor path length is zero")
    return cum[:-1] / total, total


def _resample_xy_by_arc_length(xy: np.ndarray, n_samples: int) -> np.ndarray:
    """Resample closed polyline xy uniformly in arc length (n_samples points, endpoint=False)."""
    n_samples = max(3, int(n_samples))
    closed = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-9:
        raise ValueError("sampled path length is zero")
    targets = np.linspace(0.0, total, n_samples, endpoint=False)
    xs = np.interp(targets, cum, closed[:, 0])
    ys = np.interp(targets, cum, closed[:, 1])
    return np.column_stack([xs, ys])


def _periodic_cubic_samples(ring: np.ndarray, n_samples: int) -> np.ndarray:
    """Sample n_samples XY points uniformly in arc length on a closed ring."""
    n_samples = max(3, int(n_samples))
    u_verts, _total = _arc_length_params(ring)

    if CubicSpline is not None:
        ring_ext = np.vstack([ring, ring[:1]])
        u_ext = np.concatenate([u_verts, [1.0]])
        cs_x = CubicSpline(u_ext, ring_ext[:, 0], bc_type="periodic")
        cs_y = CubicSpline(u_ext, ring_ext[:, 1], bc_type="periodic")
        dense_u = np.linspace(0.0, 1.0, max(n_samples * 16, 256), endpoint=False)
        dense_xy = np.column_stack([cs_x(dense_u), cs_y(dense_u)])
        return _resample_xy_by_arc_length(dense_xy, n_samples)

    closed = np.vstack([ring, ring[:1]])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    total = cum[-1]
    targets = np.linspace(0.0, total, n_samples, endpoint=False)
    xs = np.interp(targets, cum, closed[:, 0])
    ys = np.interp(targets, cum, closed[:, 1])
    return np.column_stack([xs, ys])


def _periodic_tangents(xy: np.ndarray) -> np.ndarray:
    n = len(xy)
    tangents = np.zeros((n, 2), dtype=float)
    for i in range(n):
        prev_pt = xy[(i - 1) % n]
        next_pt = xy[(i + 1) % n]
        delta = next_pt - prev_pt
        norm = float(np.linalg.norm(delta))
        tangents[i] = delta / norm if norm > 1e-9 else np.array([1.0, 0.0])
    return tangents


def _smooth_periodic_vectors(vectors: np.ndarray, window: int) -> np.ndarray:
    n = len(vectors)
    if n == 0:
        return vectors
    window = max(3, int(window) | 1)
    half = window // 2
    out = np.zeros_like(vectors, dtype=float)
    for i in range(n):
        acc = np.zeros(2, dtype=float)
        for k in range(-half, half + 1):
            acc += vectors[(i + k) % n]
        norm = float(np.linalg.norm(acc))
        out[i] = acc / norm if norm > 1e-9 else vectors[i]
    return out


def _camera_height(rng: random.Random) -> float:
    return float(rng.uniform(VIDEO_HEIGHT_LO, VIDEO_HEIGHT_HI))


def build_video_trajectory_tangent(
    path_points_ssl: Sequence[Sequence[float]],
    n_frames: int,
    camera_z: float,
    look_at_center: Sequence[float],
    *,
    look_dist: float = VIDEO_TANGENT_LOOK_DIST,
    smooth_window: int = VIDEO_TANGENT_SMOOTH_WINDOW,
    center_bias: float = VIDEO_TANGENT_CENTER_BIAS,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Tangent-dominant look direction with smooth bias toward room center."""
    xy = _periodic_cubic_samples(_ring_xy(path_points_ssl), n_frames)
    tangents = _smooth_periodic_vectors(_periodic_tangents(xy), smooth_window)
    center_xy = np.array([float(look_at_center[0]), float(look_at_center[1])], dtype=float)
    center_z = float(look_at_center[2])
    bias = float(np.clip(center_bias, 0.0, 1.0))

    raw_dirs = np.zeros((n_frames, 2), dtype=float)
    for i in range(n_frames):
        cam_xy = xy[i]
        to_center = center_xy - cam_xy
        center_dist = float(np.linalg.norm(to_center))
        if center_dist > 1e-6:
            to_center /= center_dist
        else:
            to_center = tangents[i]
        blended = (1.0 - bias) * tangents[i] + bias * to_center
        norm = float(np.linalg.norm(blended))
        raw_dirs[i] = blended / norm if norm > 1e-9 else tangents[i]

    look_dirs = _smooth_periodic_vectors(raw_dirs, smooth_window)

    camera_positions: List[List[float]] = []
    look_at_targets: List[List[float]] = []
    for i in range(n_frames):
        cam = [float(xy[i, 0]), float(xy[i, 1]), float(camera_z)]
        direction = look_dirs[i]
        look = [
            cam[0] + float(direction[0]) * float(look_dist),
            cam[1] + float(direction[1]) * float(look_dist),
            (1.0 - bias) * float(camera_z) + bias * center_z,
        ]
        camera_positions.append(cam)
        look_at_targets.append(look)
    return camera_positions, look_at_targets


def build_video_trajectory_center(
    path_points_ssl: Sequence[Sequence[float]],
    n_frames: int,
    camera_z: float,
    look_at_center: Sequence[float],
) -> Tuple[List[List[float]], List[List[float]]]:
    xy = _periodic_cubic_samples(_ring_xy(path_points_ssl), n_frames)
    center = [float(look_at_center[0]), float(look_at_center[1]), float(look_at_center[2])]
    camera_positions: List[List[float]] = []
    look_at_targets: List[List[float]] = []
    for i in range(n_frames):
        cam = [float(xy[i, 0]), float(xy[i, 1]), float(camera_z)]
        camera_positions.append(cam)
        look_at_targets.append(list(center))
    return camera_positions, look_at_targets


def _base_video_spec(
    *,
    trajectory: str,
    n_frames: int,
    camera_z: float,
    fov: float,
    cam_positions: List[List[float]],
    look_targets: List[List[float]],
    width: int,
    height: int,
    frame_spacing_m: float,
    video_pano: bool,
    view_name: str,
    pano_only: bool = False,
    pano_resolution: Optional[int] = None,
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "type": "sequence",
        "video": True,
        "video_trajectory": trajectory,
        "video_pano": bool(video_pano),
        "video_frames": int(n_frames),
        "video_frame_spacing_m": float(frame_spacing_m),
        "camera_height_m": camera_z,
        "camera_positions": cam_positions,
        "look_at_targets": look_targets,
        "up_vectors": [list(WORLD_UP)] * int(n_frames),
        "manual_fov": float(fov),
        "width": int(width),
        "height": int(height),
        "_view_name": view_name,
    }
    if pano_only:
        spec["pano_only"] = True
    if pano_resolution is not None:
        spec["pano_resolution"] = int(pano_resolution)
    return spec


def build_video_view_specs(
    path_points_ssl: Sequence[Sequence[float]],
    context: Dict[str, Any],
    n_frames: int,
    rng: random.Random,
    *,
    width: int = AUTO_VIEW_WIDTH,
    height: int = AUTO_VIEW_HEIGHT,
    frame_spacing_m: float = VIDEO_FRAME_SPACING_M,
    center_width: int = VIDEO_CENTER_WIDTH,
    center_height: int = VIDEO_CENTER_HEIGHT,
    pano_resolution: int = VIDEO_PANO_RESOLUTION,
) -> Dict[str, Dict[str, Any]]:
    """Build center-look video sequence spec (perspective; pano when --pano)."""
    bbox_center = scene_bbox_center(context)
    center_z = _camera_height(rng)
    center_fov = random_fov_deg(rng)
    center_cam, center_look = build_video_trajectory_center(
        path_points_ssl, n_frames, center_z, bbox_center
    )
    center_name = f"video_center_{n_frames}"
    return {
        center_name: _base_video_spec(
            trajectory="center",
            n_frames=n_frames,
            camera_z=center_z,
            fov=center_fov,
            cam_positions=center_cam,
            look_targets=center_look,
            width=center_width,
            height=center_height,
            frame_spacing_m=frame_spacing_m,
            video_pano=False,
            view_name=center_name,
            pano_resolution=pano_resolution,
        ),
    }


_LEGACY_VIDEO_RE = re.compile(r"^video_\d+$")


def video_manifest_complete(names: Sequence[str]) -> bool:
    """True when a center-look video trajectory is present."""
    return any(n.startswith("video_center_") for n in names)


def video_specs_current(
    specs: Dict[str, Dict[str, Any]],
    names: Sequence[str],
) -> bool:
    """True when cached video specs match current resolution / trajectory settings."""
    if not video_manifest_complete(names):
        return False
    if any(n.startswith("video_tangent_") for n in names):
        return False
    for name in names:
        if not name.startswith("video_center_"):
            continue
        spec = specs.get(name) or {}
        if int(spec.get("width", 0)) != VIDEO_CENTER_WIDTH:
            return False
        if int(spec.get("height", 0)) != VIDEO_CENTER_HEIGHT:
            return False
        if int(spec.get("pano_resolution", 0)) != VIDEO_PANO_RESOLUTION:
            return False
        if spec.get("pano_only"):
            return False
    return True


def strip_legacy_video_entries(
    specs: Dict[str, Dict[str, Any]],
    names: List[str],
) -> None:
    """Drop legacy single ``video_{N}`` entries when upgrading to tangent+center."""
    legacy = [n for n in names if _LEGACY_VIDEO_RE.match(n)]
    for name in legacy:
        specs.pop(name, None)
        names.remove(name)


def build_video_views_from_path(
    path_points_ssl: Sequence[Sequence[float]],
    context: Dict[str, Any],
    *,
    frame_spacing_m: float = VIDEO_FRAME_SPACING_M,
    n_frames: Optional[int] = None,
    rng: Optional[random.Random] = None,
    width: int = AUTO_VIEW_WIDTH,
    height: int = AUTO_VIEW_HEIGHT,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Build center-look video sequence spec. Returns (name→spec, render order)."""
    rng = rng or random.Random()
    spacing_m = max(1e-3, float(frame_spacing_m))
    n_frames = max(VIDEO_MIN_FRAMES, int(n_frames)) if n_frames is not None else compute_video_frame_count(
        path_points_ssl, spacing_m=spacing_m
    )
    specs = build_video_view_specs(
        path_points_ssl,
        context,
        n_frames,
        rng,
        width=width,
        height=height,
        frame_spacing_m=spacing_m,
    )
    for spec in specs.values():
        spec.pop("_view_name", None)
    center_name = f"video_center_{n_frames}"
    return specs, [center_name]
