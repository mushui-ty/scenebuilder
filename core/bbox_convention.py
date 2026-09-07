"""SceneBuilder vs OpenSpatial 3D bbox conventions in OpenCV camera frame."""

from __future__ import annotations

import warnings
from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

BBOX_DECIMALS = 2

# SceneBuilder semantic face-to-face pose in OpenCV camera frame (data.md §8.7)
_SCENEBUILDER_R0 = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]], dtype=float
)
_SCENEBUILDER_Q = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float
)
_SCENEBUILDER_S = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
)
_SCENEBUILDER_QS_INV = np.linalg.inv(_SCENEBUILDER_Q @ _SCENEBUILDER_S)


def round_bbox_float(value: float) -> float:
    return float(round(float(value), BBOX_DECIMALS))


def round_bbox_values(values: Sequence[float]) -> List[float]:
    return [round_bbox_float(v) for v in values]


def scenebuilder_cam_to_openspatial(
    center: Sequence[float],
    scale: Sequence[float],
    roll: float,
    pitch: float,
    yaw: float,
) -> Tuple[List[float], List[float], List[float]]:
    """SceneBuilder camera bbox -> OpenSpatial camera bbox (center, scale, pose)."""
    r_actual = _SCENEBUILDER_R0 @ SciRotation.from_euler(
        "XYZ", [roll, pitch, yaw], degrees=False
    ).as_matrix()
    r_os = r_actual @ _SCENEBUILDER_QS_INV
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        os_roll, os_pitch, os_yaw = SciRotation.from_matrix(r_os).as_euler("zxy", degrees=False)
    xl, yl, zl = float(scale[0]), float(scale[2]), float(scale[1])
    center_out = round_bbox_values(center)
    scale_out = round_bbox_values([xl, yl, zl])
    pose_out = round_bbox_values([os_roll, os_pitch, os_yaw])
    return center_out, scale_out, pose_out


def openspatial_cam_to_scenebuilder(
    center: Sequence[float],
    xl: float,
    yl: float,
    zl: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> Tuple[List[float], List[float], List[float]]:
    """OpenSpatial camera bbox -> SceneBuilder camera bbox (center, scale, pose)."""
    r_os = SciRotation.from_euler("zxy", [roll, pitch, yaw], degrees=False).as_matrix()
    r_actual = r_os @ _SCENEBUILDER_Q @ _SCENEBUILDER_S
    r_delta = _SCENEBUILDER_R0.T @ r_actual
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        sb_roll, sb_pitch, sb_yaw = SciRotation.from_matrix(r_delta).as_euler("XYZ", degrees=False)
    scale = round_bbox_values([xl, zl, yl])
    center_out = round_bbox_values(center)
    pose_out = round_bbox_values([sb_roll, sb_pitch, sb_yaw])
    return center_out, scale, pose_out


def is_already_openspatial(
    center: Sequence[float],
    scale: Sequence[float],
    pose: Sequence[float],
) -> bool:
    """Fixed-point test: f(g(x)) ~= x means x is already OpenSpatial."""
    sb_center, sb_scale, sb_pose = openspatial_cam_to_scenebuilder(
        center,
        float(scale[0]),
        float(scale[1]),
        float(scale[2]),
        float(pose[0]),
        float(pose[1]),
        float(pose[2]),
    )
    os_center, os_scale, os_pose = scenebuilder_cam_to_openspatial(
        sb_center, sb_scale, sb_pose[0], sb_pose[1], sb_pose[2]
    )

    def _close(a: Sequence[float], b: Sequence[float]) -> bool:
        eps = 10 ** (-BBOX_DECIMALS)
        return all(abs(float(x) - float(y)) <= eps for x, y in zip(a, b))

    return _close(center, os_center) and _close(scale, os_scale) and _close(pose, os_pose)
