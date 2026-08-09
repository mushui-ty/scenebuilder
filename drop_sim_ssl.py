"""
Physics drop simulation: read SSL with supported_by, run gravity sim, export updated SSL.

Usage:
    from scenebuilder.drop_sim_ssl import drop_sim_ssl
    new_ssl = drop_sim_ssl(ssl_text, asset_dir="/path/to/assets")
"""

import os
import re
from typing import Optional

try:
    from .core.util_data import parse_ssl_to_json
except (ImportError, ValueError):
    from core.util_data import parse_ssl_to_json  # type: ignore


PASSIVE_SUPPORTS = {"floor", "wall", "ceiling"}


def drop_sim_ssl(
    ssl_text: str,
    asset_dir: Optional[str] = None,
    sim_time: float = 1.0,
) -> str:
    """Run physics drop simulation and return updated SSL text.

    Steps:
      1. Parse SSL (including supported_by attributes)
      2. Mark active: supported_by not in {floor, wall, ceiling} and not a chip → active
      3. If no active objects → return original SSL (strip supported_by)
      4. Build scene + run drop_sim
      5. Export updated SSL (without supported_by)

    Args:
        ssl_text: SSL text with supported_by attributes
        asset_dir: Asset directory (GLB files)
        sim_time: Simulation duration (seconds)

    Returns:
        Updated SSL text (no supported_by, z corrected)
    """
    # 1. Parse
    scene_json = parse_ssl_to_json(ssl_text)

    # 2. Determine active
    has_active = False
    for bbox in scene_json["bbox"]:
        supported_by = bbox.get("supported_by", "floor")
        # Chip objects (doors/windows embedded in walls) are always passive
        is_chip = bbox.get("scale", [1, 1, 1])[1] == 0  # b=0 marks a chip
        if supported_by not in PASSIVE_SUPPORTS and not is_chip:
            bbox["active"] = True
            has_active = True
        else:
            bbox["active"] = False

    # 3. No active objects → return SSL with supported_by stripped
    if not has_active:
        print("⚡ drop_sim_ssl: no active objects; skipping physics simulation")
        return _strip_supported_by(ssl_text)

    print(f"⚡ drop_sim_ssl: found {sum(1 for b in scene_json['bbox'] if b.get('active'))} active object(s)")

    # 4. Build scene
    try:
        from .core.scenebuilder_bpy import BpySceneCtx
    except (ImportError, ValueError):
        from core.scenebuilder_bpy import BpySceneCtx  # type: ignore

    room_type = scene_json["room"].get("room_type", "unknown")
    ctx = BpySceneCtx(room_type, asset_dir)

    ctx.add_walls(scene_json["wall"])
    if scene_json["door"]:
        ctx.add_doors(scene_json["door"])
    if scene_json["window"]:
        ctx.add_windows(scene_json["window"])

    # add_boxes reads active flag
    # Also pass supported_by into drop_sim penetration checks
    for bbox in scene_json["bbox"]:
        ctx.add_box(
            center=bbox["center"],
            angle_z=bbox["angle_z"],
            scale=bbox["scale"],
            label=bbox.get("label"),
            caption=bbox.get("caption"),
            asset_id=bbox.get("asset_id"),
            active=bbox.get("active", False),
        )
        # Store supported_by in context box_data for drop_sim penetration checks
        # add_box returns box_id; the last added is the newest
        last_box_id = list(ctx.context["boxes"].keys())[-1]
        if bbox.get("supported_by"):
            ctx.context["boxes"][last_box_id]["supported_by"] = bbox["supported_by"]

    # Build scene geometry
    geometry_mode = "gltf" if asset_dir else "bbox"
    ctx.construct_scene(geometry_mode=geometry_mode, show_ceiling=False)

    # 5. Run physics simulation
    ctx.drop_sim(sim_time=sim_time)

    # 6. Export updated SSL
    #    export_ssl writes to file; read back and return
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx.export_ssl(tmpdir)
        ssl_path = os.path.join(tmpdir, "ssl.txt")
        with open(ssl_path, "r", encoding="utf-8") as f:
            new_ssl = f.read()

    return new_ssl


def _strip_supported_by(ssl_text: str) -> str:
    """Remove supported_by attributes from SSL text."""
    # Match , supported_by="xxx" or supported_by="xxx",
    return re.sub(r',?\s*supported_by="[^"]*"', '', ssl_text)
