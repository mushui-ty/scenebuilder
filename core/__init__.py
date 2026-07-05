"""Core rendering engine and shared utilities for fast_scene."""

from .config_utils import (
    ASSETS_DIR,
    CONFIG_PATH,
    PACKAGE_DIR,
    PROJECT_ROOT,
    load_config,
    resolve_package_path,
)
from .fast_scene import SceneCtx
from .fast_scene_bpy import BpySceneCtx

__all__ = [
    "SceneCtx",
    "BpySceneCtx",
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "CONFIG_PATH",
    "ASSETS_DIR",
    "load_config",
    "resolve_package_path",
]
