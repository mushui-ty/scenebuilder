"""Random bpy texture / HDRI selection for render_ssl (auto_texture mode)."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .config_utils import ASSETS_DIR, PACKAGE_DIR
except ImportError:
    from config_utils import ASSETS_DIR, PACKAGE_DIR  # type: ignore

TEXTURE_SELECTION_FILENAME = "texture_selection.json"

WALL_CEILING_POOL = ASSETS_DIR / "textures" / "wall_ceiling"
FLOOR_POOL = ASSETS_DIR / "textures" / "floor"
HDRI_POOL = ASSETS_DIR / "HDRIs"


def _discover_blend_materials(pool_dir: Path) -> List[str]:
    """Return absolute paths to .blend material files under pool_dir."""
    if not pool_dir.is_dir():
        return []
    paths: List[str] = []
    for entry in sorted(pool_dir.iterdir()):
        if not entry.is_dir():
            continue
        preferred = entry / f"{entry.name}.blend"
        if preferred.is_file():
            paths.append(str(preferred.resolve()))
            continue
        for candidate in sorted(entry.glob("*.blend")):
            if candidate.is_file():
                paths.append(str(candidate.resolve()))
                break
    return paths


def _discover_hdri_files(pool_dir: Path) -> List[str]:
    if not pool_dir.is_dir():
        return []
    paths: List[str] = []
    for pattern in ("*.exr", "*.hdr", "*.EXR", "*.HDR"):
        paths.extend(str(p.resolve()) for p in sorted(pool_dir.glob(pattern)) if p.is_file())
    # Stable dedupe while preserving order.
    seen = set()
    unique: List[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def pick_random_scene_textures() -> Dict[str, str]:
    wall_pool = _discover_blend_materials(WALL_CEILING_POOL)
    floor_pool = _discover_blend_materials(FLOOR_POOL)
    hdri_pool = _discover_hdri_files(HDRI_POOL)
    if not wall_pool:
        raise FileNotFoundError(f"No wall/ceiling .blend materials under {WALL_CEILING_POOL}")
    if not floor_pool:
        raise FileNotFoundError(f"No floor .blend materials under {FLOOR_POOL}")
    if not hdri_pool:
        raise FileNotFoundError(f"No HDRI files under {HDRI_POOL}")

    wall = random.choice(wall_pool)
    ceiling = random.choice(wall_pool)
    floor = random.choice(floor_pool)
    hdri = random.choice(hdri_pool)
    return {
        "wall_blender_texture_path": wall,
        "ceiling_blender_texture_path": ceiling,
        "floor_blender_texture_path": floor,
        "hdri_path": hdri,
    }


def save_texture_selection(y_dir: str, scene_textures: Dict[str, str]) -> str:
    os.makedirs(y_dir, exist_ok=True)
    path = os.path.join(y_dir, TEXTURE_SELECTION_FILENAME)
    payload = {"auto_texture": True, **scene_textures}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_texture_selection(y_dir: str) -> Optional[Dict[str, str]]:
    path = os.path.join(y_dir, TEXTURE_SELECTION_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    keys = (
        "wall_blender_texture_path",
        "ceiling_blender_texture_path",
        "floor_blender_texture_path",
        "hdri_path",
    )
    picked = {k: data[k] for k in keys if data.get(k)}
    return picked or None


def _legacy_scene_without_auto_textures(y_dir: str) -> bool:
    """Scene started before auto_texture: keep config.yaml paths on resume."""
    job_path = os.path.join(y_dir, ".render_job.json")
    if os.path.isfile(job_path):
        with open(job_path, "r", encoding="utf-8") as f:
            job = json.load(f)
        if isinstance(job, dict) and job.get("scene_textures"):
            return False
        return True

    markers = (
        os.path.join(y_dir, "topdown_normalized", "topdown.png"),
        os.path.join(y_dir, ".render_progress.json"),
    )
    if any(os.path.isfile(p) for p in markers):
        return True
    topdown_dir = os.path.join(y_dir, "topdown")
    if os.path.isdir(topdown_dir):
        try:
            if any(name.lower().endswith(".png") for name in os.listdir(topdown_dir)):
                return True
        except OSError:
            pass
    return False


def resolve_scene_textures(
    y_dir: str,
    *,
    backend: str,
    auto_texture: bool,
    texture_dir: Optional[str],
    gen_texture: bool,
    resume: bool,
) -> Optional[Dict[str, str]]:
    """Return bpy texture paths for this scene, or None to use config defaults."""
    if texture_dir and os.path.isdir(texture_dir):
        return None
    if gen_texture:
        return None
    if backend != "bpy":
        return None

    if not resume:
        selection_path = os.path.join(y_dir, TEXTURE_SELECTION_FILENAME)
        if os.path.isfile(selection_path):
            try:
                os.remove(selection_path)
            except OSError:
                pass

    if resume:
        saved = load_texture_selection(y_dir)
        if saved:
            return saved
        job_path = os.path.join(y_dir, ".render_job.json")
        if os.path.isfile(job_path):
            with open(job_path, "r", encoding="utf-8") as f:
                job = json.load(f)
            if isinstance(job, dict) and job.get("scene_textures"):
                save_texture_selection(y_dir, job["scene_textures"])
                return dict(job["scene_textures"])

    if not auto_texture:
        return None

    if resume and _legacy_scene_without_auto_textures(y_dir):
        return None

    picked = pick_random_scene_textures()
    save_texture_selection(y_dir, picked)
    return picked


def apply_scene_textures_to_ctx(ctx: Any, scene_textures: Optional[Dict[str, str]]) -> None:
    if not scene_textures:
        return
    wall = scene_textures.get("wall_blender_texture_path")
    floor = scene_textures.get("floor_blender_texture_path")
    ceiling = scene_textures.get("ceiling_blender_texture_path")
    hdri = scene_textures.get("hdri_path")

    if wall and os.path.isfile(wall) and hasattr(ctx, "set_wall_blender_texture_path"):
        ctx.set_wall_blender_texture_path(wall)
    if floor and os.path.isfile(floor) and hasattr(ctx, "set_floor_blender_texture_path"):
        ctx.set_floor_blender_texture_path(floor)
    if ceiling and os.path.isfile(ceiling) and hasattr(ctx, "set_ceiling_blender_texture_path"):
        ctx.set_ceiling_blender_texture_path(ceiling)
    if hdri and os.path.isfile(hdri) and hasattr(ctx, "set_hdri_path"):
        ctx.set_hdri_path(hdri)


def log_scene_textures(scene_textures: Optional[Dict[str, str]], *, source: str) -> None:
    if not scene_textures:
        return
    print(f"🎨 Scene textures ({source}):")
    for key in (
        "wall_blender_texture_path",
        "ceiling_blender_texture_path",
        "floor_blender_texture_path",
        "hdri_path",
    ):
        path = scene_textures.get(key)
        if path:
            try:
                rel = os.path.relpath(path, str(PACKAGE_DIR))
            except ValueError:
                rel = path
            print(f"   {key}: {rel}")
