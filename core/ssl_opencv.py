"""Export ssl_opencv.txt: scene geometry in OpenCV camera coordinates."""

from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

try:
    from . import geometry_opencv as geo_cv
    from .util_data import _ssl_fmt_list, _ssl_fmt_num
except ImportError:
    import geometry_opencv as geo_cv  # type: ignore
    from util_data import _ssl_fmt_list, _ssl_fmt_num  # type: ignore

SSL_OPENCV_FILENAME = "ssl_opencv.txt"
OA_FRONT_LOCAL = np.array([0.0, -1.0, 0.0], dtype=float)
OA_LEFT_LOCAL = np.array([1.0, 0.0, 0.0], dtype=float)
OA_TOP_LOCAL = np.array([0.0, 0.0, 1.0], dtype=float)
OA_STANDARD_CAM = np.column_stack(
    [
        np.array([0.0, 0.0, -1.0], dtype=float),  # front -> camera -Z
        np.array([1.0, 0.0, 0.0], dtype=float),   # left  -> camera +X
        np.array([0.0, -1.0, 0.0], dtype=float),  # top   -> camera -Y
    ]
)


def rotation_matrix_to_semantic_pose(rot_cam: np.ndarray) -> List[float]:
    """Return pose=[roll, pitch, yaw] as intrinsic XYZ from the OA standard pose."""
    rot_cam = np.asarray(rot_cam, dtype=float).reshape(3, 3)
    actual_oa_cam = np.column_stack(
        [
            rot_cam @ OA_FRONT_LOCAL,
            rot_cam @ OA_LEFT_LOCAL,
            rot_cam @ OA_TOP_LOCAL,
        ]
    )
    relative = OA_STANDARD_CAM.T @ actual_oa_cam
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        roll, pitch, yaw = SciRotation.from_matrix(relative).as_euler("XYZ", degrees=False)
    return [float(roll), float(pitch), float(yaw)]


def build_opencv_w2c_matrix(
    camera_position,
    look_at_target,
    world_up=(0.0, 0.0, 1.0),
) -> np.ndarray:
    c2w = geo_cv.build_opencv_c2w_matrix(camera_position, look_at_target, world_up)
    return np.linalg.inv(c2w)


def transform_point_world_to_opencv(
    point,
    camera_position,
    look_at_target,
    world_up=(0.0, 0.0, 1.0),
) -> np.ndarray:
    w2c = build_opencv_w2c_matrix(camera_position, look_at_target, world_up)
    p = np.asarray(point, dtype=float).reshape(3)
    return (w2c @ np.concatenate([p, [1.0]]))[:3]


def transform_direction_world_to_opencv(
    direction,
    camera_position,
    look_at_target,
    world_up=(0.0, 0.0, 1.0),
) -> np.ndarray:
    w2c = build_opencv_w2c_matrix(camera_position, look_at_target, world_up)
    d = np.asarray(direction, dtype=float).reshape(3)
    out = w2c[:3, :3] @ d
    norm = float(np.linalg.norm(out))
    if norm > 1e-12:
        out = out / norm
    return out


def bbox_world_to_opencv_pose(
    center,
    scale,
    angle_z_deg: float,
    camera_position,
    look_at_target,
    world_up=(0.0, 0.0, 1.0),
) -> Tuple[List[float], List[float], List[float]]:
    """World SSL bbox -> OpenCV center + scale + pose=[azimuth,polar,rotation]."""
    center_w = np.asarray(center, dtype=float).reshape(3)
    angle_rad = np.radians(float(angle_z_deg))
    rot_world = SciRotation.from_euler("z", angle_rad, degrees=False).as_matrix()
    transform_world = np.eye(4, dtype=float)
    transform_world[:3, :3] = rot_world
    transform_world[:3, 3] = center_w

    c2w = geo_cv.build_opencv_c2w_matrix(camera_position, look_at_target, world_up)
    transform_cam = np.linalg.inv(c2w) @ transform_world

    cam_center = transform_cam[:3, 3]
    rot_cam = transform_cam[:3, :3]
    pose = rotation_matrix_to_semantic_pose(rot_cam)
    scale_list = [float(v) for v in scale]
    return (
        cam_center.tolist(),
        scale_list,
        [float(v) for v in pose],
    )


def build_opencv_ssl_context(
    context: Dict[str, Any],
    camera_position,
    look_at_target,
    world_up=(0.0, 0.0, 1.0),
) -> Tuple[Dict[str, Any], List[float]]:
    """Deep-copy context and transform geometry into OpenCV camera coordinates."""
    out = copy.deepcopy(context)
    world_up_vec = transform_direction_world_to_opencv(
        [0.0, 0.0, 1.0],
        camera_position,
        look_at_target,
        world_up,
    )

    for wall in out.get("walls", {}).values():
        p_cam = transform_point_world_to_opencv(
            [wall["s"][0], wall["s"][1], 0.0],
            camera_position,
            look_at_target,
            world_up,
        )
        q_cam = transform_point_world_to_opencv(
            [wall["e"][0], wall["e"][1], 0.0],
            camera_position,
            look_at_target,
            world_up,
        )
        wall["s"] = p_cam.tolist()
        wall["e"] = q_cam.tolist()
        for door in wall.get("doors", {}).values():
            door["center"] = transform_point_world_to_opencv(
                door["center"], camera_position, look_at_target, world_up
            ).tolist()
        for window in wall.get("windows", {}).values():
            window["center"] = transform_point_world_to_opencv(
                window["center"], camera_position, look_at_target, world_up
            ).tolist()

    for box in out.get("boxes", {}).values():
        center_cam, scale_cam, pose = bbox_world_to_opencv_pose(
            box["center"],
            box["scale"],
            float(box.get("angle_z", 0.0)),
            camera_position,
            look_at_target,
            world_up,
        )
        box["center"] = center_cam
        box["scale"] = scale_cam
        box["pose"] = pose
        box.pop("angle_z", None)

    return out, world_up_vec.tolist()


def format_opencv_ssl(context: Dict[str, Any], world_up_cam: Sequence[float]) -> str:
    lines: List[str] = []
    room_type = context.get("meta", {}).get("scene_type", "unknown")
    lines.append(
        f'Room(room_type="{room_type}", world_up={_ssl_fmt_list(world_up_cam)})'
    )

    for wall_id, wall in context.get("walls", {}).items():
        wall_label = wall.get("label", wall_id)
        parts = [
            f'label="{wall_label}"',
            f'p={_ssl_fmt_list(wall["s"])}',
            f'q={_ssl_fmt_list(wall["e"])}',
            f'height={_ssl_fmt_num(wall["height"])}',
        ]
        wall_caption = wall.get("caption")
        if wall_caption:
            parts.append(f'caption="{wall_caption}"')
        lines.append(f'Wall({", ".join(parts)})')

    for wall in context.get("walls", {}).values():
        for door in wall.get("doors", {}).values():
            parts = [
                f'label="{door.get("label", "door")}"',
                f'center={_ssl_fmt_list(door["center"])}',
                f'width={_ssl_fmt_num(door["width"])}',
                f'height={_ssl_fmt_num(door["height"])}',
            ]
            caption = door.get("caption")
            if caption:
                parts.append(f'caption="{caption}"')
            wall_label = door.get("wall")
            if wall_label:
                parts.append(f'wall="{wall_label}"')
            if door.get("asset_id") is not None:
                parts.append(f'asset_id="{door["asset_id"]}"')
            lines.append(f'Door({", ".join(parts)})')

    for wall in context.get("walls", {}).values():
        for window in wall.get("windows", {}).values():
            parts = [
                f'label="{window.get("label", "window")}"',
                f'center={_ssl_fmt_list(window["center"])}',
                f'width={_ssl_fmt_num(window["width"])}',
                f'height={_ssl_fmt_num(window["height"])}',
            ]
            caption = window.get("caption")
            if caption:
                parts.append(f'caption="{caption}"')
            wall_label = window.get("wall")
            if wall_label:
                parts.append(f'wall="{wall_label}"')
            if window.get("asset_id") is not None:
                parts.append(f'asset_id="{window["asset_id"]}"')
            lines.append(f'Window({", ".join(parts)})')

    for box in context.get("boxes", {}).values():
        box_label = box.get("label", "object0")
        parts = [
            f'label="{box_label}"',
            f'center={_ssl_fmt_list(box["center"])}',
            f'pose={_ssl_fmt_list(box["pose"])}',
            f'scale={_ssl_fmt_list(box["scale"])}',
        ]
        caption = box.get("caption")
        if caption:
            parts.append(f'caption="{caption}"')
        if box.get("asset_id") is not None:
            parts.append(f'asset_id="{box["asset_id"]}"')
        lines.append(f'Bbox({", ".join(parts)})')

    return "\n".join(lines) + "\n"


def write_opencv_ssl(
    context: Dict[str, Any],
    output_dir: str,
    camera_position,
    look_at_target,
    world_up=(0.0, 0.0, 1.0),
) -> str:
    """Write ``ssl_opencv.txt`` under ``output_dir`` (OpenCV camera reference frame)."""
    os.makedirs(output_dir, exist_ok=True)
    opencv_ctx, world_up_cam = build_opencv_ssl_context(
        context,
        camera_position,
        look_at_target,
        world_up,
    )
    ssl_text = format_opencv_ssl(opencv_ctx, world_up_cam)
    ssl_path = os.path.join(output_dir, SSL_OPENCV_FILENAME)
    with open(ssl_path, "w", encoding="utf-8") as f:
        f.write(ssl_text)
    print(f"✅ OpenCV SSL 已导出: {ssl_path}")
    return ssl_path
