"""Load config.yaml and resolve package-relative asset paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_DIR.parent
CONFIG_PATH = PACKAGE_DIR / "config.yaml"
ASSETS_DIR = PACKAGE_DIR / "assets"

# Keys whose values are filesystem paths; relative entries resolve against PACKAGE_DIR.
_PATH_KEYS = (
    "floor_texture_path",
    "ceiling_texture_path",
    "wall_texture_path",
    "door_texture_path",
    "window_texture_path",
    "floor_blender_texture_path",
    "ceiling_blender_texture_path",
    "wall_blender_texture_path",
    "hdri_path",
    "model_path",
    "model_simplified_path",
    "model_generate_path",
    "model_hole_path",
)


def resolve_package_path(path: Optional[str]) -> Optional[str]:
    if not path or not isinstance(path, str):
        return path
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((PACKAGE_DIR / candidate).resolve())


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    for key in _PATH_KEYS:
        if key in config:
            config[key] = resolve_package_path(config[key])
    return config
