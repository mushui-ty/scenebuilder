#!/usr/bin/env python3
"""Split merged visible PLY/GLB by per-view viewvis sidecars.

Example:
  python split_viewvis_geometry.py \\
    /path/to/left_seq/pointcloud/scene_visible_opencv.ply view

  python split_viewvis_geometry.py \\
    /path/to/left_seq/glb/scene_visible.glb object

Input geometry must live next to the exported viewvis NPZ sidecars (same ``pointcloud/``
or ``glb/`` folder). Mode ``view`` uses point-level occlusion visibility; mode
``object`` uses object-level + frustum visibility. Outputs ``{stem}_by_{mode}/`` beside
the input file with one PLY/GLB per camera frame plus ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys

try:
    from core.viewvis_split import split_geometry_by_views
except ImportError:
    from scenebuilder.core.viewvis_split import split_geometry_by_views  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split scene_visible PLY/GLB by per-view viewvis NPZ sidecars.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "geometry",
        help="Path to scene_visible(.ply|.glb) or scene_visible_opencv(.ply|.glb)",
    )
    parser.add_argument(
        "mode",
        choices=["view", "object"],
        help="view = point-level occlusion NPZ; object = object-level frustum NPZ",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (default: {input_dir}/{stem}_by_{mode}/)",
    )
    args = parser.parse_args()

    manifest = split_geometry_by_views(args.geometry, args.mode, output_dir=args.output)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    for view in manifest["views"]:
        kept = view.get("kept", view.get("kept_faces"))
        total = view.get("total", view.get("total_faces"))
        print(f"✅ {view['frame']}: {view['path']} ({kept}/{total})")
    print(f"📂 Output: {manifest['output_dir']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)
