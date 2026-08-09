# SceneBuilder

**English** | [中文](README.zh-CN.md)

A repository for **3D scene build**, **render**, **path planning** (for coverage video rendering), and **geometry export** — based on **bpy / pyrender**, from **SSL** structured scene format.

SceneBuilder render example

Build indoor scenes from SSL text (walls, doors, windows, furniture), then render multi-view images, semantic/depth maps, panoramas, export GLB/PLY/voxels, and plan floor paths for auto camera trajectories.

## Roadmap


| Step  | Status         | Goal                                                                                                         |
| ----- | -------------- | ------------------------------------------------------------------------------------------------------------ |
| **1** | ✅ Done         | Render & geometry export toolkit (`render_ssl`, `topdown_view`, `render_view`, GLB/PLY/voxel/semantic/depth) |
| **2** | 🚧 In progress | Release **scene example datasets**                                                                           |
| **3** | ⏳ Planned      | **Canonical asset retrieve** system                                                                          |
| **4** | ⏳ Planned      | **Asset generation** pipeline integration                                                                    |




## Features

- **Dual backend**: Blender (`bpy`, recommended) or Pyrender (lightweight)
- **Views**: topdown, preset cameras, custom camera, `--views auto` path-driven rendering
- **Exports**: RGB, depth, normals, semantic masks, panorama, visible GLB/PLY, 256³ voxels
- **Path planning**: pixel-aligned topdown + nav-mask floor loop for coverage video
- **Coordinates**: SSL world space + OpenCV camera copies (`ssl_opencv.txt`, `c2w`, `intrinsic`)



## Installation

**Python 3.11** (required for latest `bpy`).

```bash
pip install numpy shapely scipy imageio pyyaml
pip install bpy          # recommended
# or: pip install pyrender trimesh

cd scenebuilder
pip install -e .
```

Headless Pyrender on Linux: set `PYOPENGL_PLATFORM=egl` and install Mesa/EGL libs — see [Full Documentation](docs/doc.en.md#1-installation-and-dependencies).

## Quick Start



### CLI — batch render

```bash
python render_ssl.py \
  --ssl path/to/ssl.txt \
  --views auto \
  --output out \
  --assets path/to/assets \
  --glb --ply --visible_geometry --semantic --depth --pano
```



### Python — single topdown

```python
from scenebuilder.core.util_data import parse_scene_input
from scenebuilder.core.scenebuilder_bpy import BpySceneCtx

with open("scene.ssl", encoding="utf-8") as f:
    scene = parse_scene_input(f.read())

ctx = BpySceneCtx(scene["room"]["room_type"], asset_dir="path/to/assets")
ctx.add_walls(scene["wall"])
ctx.add_boxes(scene["bbox"])
ctx.normalize_scene_data()
ctx.topdown_view("output", rebuild=True)  # → output/topdown/topdown.png
```



### Python — custom camera

```python
meta = ctx.context["meta"]
center, z_max = meta["center"], meta["z_max"]

ctx.render_view(
    "output",
    camera_position=[center[0], center[1] - meta["span"][1] / 3, z_max * 5 / 6],
    look_at_target=[center[0], center[1], z_max / 2],
    rebuild=True,
)
```

Custom cameras can also be passed via CLI: `--camera_position`, `--look_at`, optional `--up_vector`. If view direction is collinear with default up `[0,0,1]`, `render_view` falls back to `[0,1,0]` automatically.

## Documentation


| Topic                                                               | Link                                                        |
| ------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Full reference** (API, SSL coords, outputs, depth/semantic/voxel) | [docs/doc.en.md](docs/doc.en.md)                            |
| SSL coordinate system                                               | [§2](docs/doc.en.md#2-ssl-coordinate-system-and-entities)   |
| `render_ssl` / CLI flags                                            | [§4.3](docs/doc.en.md#43-render_ssl--render_sslpy)          |
| Output layout & `c2w`                                               | [§6](docs/doc.en.md#6-output-layout-and-coordinate-systems) |
| `--views auto` & floor path                                         | [§5](docs/doc.en.md#5-advanced-workflows)                   |
| 中文文档                                                                | [docs/doc.zh-CN.md](docs/doc.zh-CN.md)                      |




## Project Layout

```
scenebuilder/
├── render_ssl.py          # CLI & high-level batch API
├── core/
│   ├── scenebuilder_bpy.py
│   ├── scenebuilder.py
│   ├── auto_views.py
│   └── nav_mask_path.py
├── assets/                # example images & furniture GLBs
├── config.yaml
└── docs/
    ├── doc.en.md          # full documentation (English)
    └── doc.zh-CN.md       # 完整文档（中文）
```



## License

See [LICENSE](LICENSE).