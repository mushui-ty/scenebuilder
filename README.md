# SceneBuilder

**English** | [中文](README.zh-CN.md)

A repository for **3D scene build**, **space planning for coverage video rendering**, **render**, and **geometry export** — based on **bpy / pyrender**, from **SSL** structured scene format.

<p align="center">
  <img src="assets/image.png" alt="SceneBuilder render example" width="800">
</p>

Build indoor scenes from SSL text (walls, doors, windows, furniture), then render multi-view images, semantic/depth maps, panoramas, export GLB/PLY/voxels, and plan floor paths for auto camera trajectories.

## Roadmap


| Step  | Status         | Goal                                                                                                         |
| ----- | -------------- | ------------------------------------------------------------------------------------------------------------ |
| **1** | ✅ Done         | Render & geometry export toolkit (`render_ssl`, `topdown_view`, `render_view`, GLB/PLY/voxel/semantic/depth) |
| **2** | 🚧 In progress | Release **structured scene data sets**                                                                           |
| **3** | ⏳ Planned      | **Canonical asset retrieve** system                                                                          |
| **4** | ⏳ Planned      | **Asset generation** pipeline integration                                                                    |




## Features

- **Dual backend**: Blender (`bpy`, recommended) or Pyrender (lightweight)
- **Views**: topdown, preset cameras, custom camera, `--views auto` path-driven rendering
- **Exports**: RGB, depth, normals, semantic masks, panorama, visible GLB/PLY, 256³ voxels
- **Path planning**: pixel-aligned topdown + nav-mask floor loop for coverage video
- **Coordinates**: SSL world space + OpenCV camera copies (`ssl_opencv.txt`, `c2w`, `intrinsic`); **`ssl_opencv.txt` Bbox uses OpenSpatial 9-parameter convention** (zxy Euler, `[xl, yl, zl]` scale) for 3D Grounding alignment — see [§6.4.4](docs/doc.en.md#644-bbox-convention-openspatial-9-parameter--intrinsic-zxy)



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

## Advanced — `render_ssl` batch pipeline

For data production and benchmark runs, `render_ssl.py` is the high-level entry point. One invocation can:

- **Normalize** the scene to a pixel-aligned topdown frame and **plan a floor coverage path**
- **Render auto cameras** along that path (`--views auto`), plus **`--video` smooth closed-loop trajectories**, preset / custom views
- **Export geometry**: per-view visible GLB / PLY / 256³ voxels, and full-scene holo exports at the output root
- **Export supervision**: semantic masks, metric depth + normals, equirectangular panoramas
- **Resume** interrupted jobs (`--no-resume` to force a full rerun)

### Simple usage

**CLI** — scene input (choose one of `--ssl`, `--ssl_str`, `--ssl_id`):

```bash
# file path (SSL or JSON)
python render_ssl.py --ssl path/to/scene.ssl --output out --views topdown

# inline string (e.g. one line from JSONL)
python render_ssl.py --ssl_str '{"wall":[],"bbox":[],"room":{"room_type":"bedroom"}}' \
  --output out --views topdown

# stdin (pipe one JSONL line)
sed -n '1p' scenes.jsonl | python render_ssl.py --ssl - --output out --views auto

# JSONL collection room id (310449449_4 = line index 4 in 310449449.jsonl, 0-based)
python render_ssl.py --ssl_id 310449449_4 \
  --ssl_collection_dir /root/datasets/manycore/spatiallm_raw \
  --output out --views topdown
```

**Python** — first argument is always `input_text` (SSL or JSON **string**); read from a file, or pass text directly:

```python
from scenebuilder.render_ssl import render_ssl

# 1) from file path (read SSL / JSON file, then pass content)
scene_path = "path/to/scene.ssl"  # or scene.json
with open(scene_path, encoding="utf-8") as f:
    render_ssl(f.read(), output_root="out", views=["topdown"])

# 2) from input_text (string already in memory)
ssl_text = """
Room(id="D54g", room_type="bedroom")
Wall(id="0", room_id="D54g", p=[0,0,0], q=[3,0,0], height=2.8)
"""
render_ssl(ssl_text, output_root="out", views=["topdown"])

# JSON / JSONL line as string works too
render_ssl('{"wall":[],"bbox":[],"room":{"room_type":"bedroom"}}', output_root="out", views="auto")
```

### Full example — all export switches

Example — turn on the main export switches:

```bash
python render_ssl.py \
  --ssl path/to/ssl.txt \
  --output out \
  --assets path/to/assets \
  --hole_assets path/to/holes \
  --visible_geometry --holo_geometry \
  --glb --ply --voxel \
  --semantic --depth --pano \
  --views auto \
  --width 1000 --height 1000
```

**Camera / resolution (`--width`, `--height`, `--manual_fov`, `--no_auto_fov`)**: Same semantics as `render_view`. Default output size is **1000×1000** with auto FOV. `manual_fov` is the **long-axis FOV** under Blender `sensor_fit=AUTO` (see [§4.0](docs/doc.en.md#40-resolution-and-fov-blender-backend)). **`topdown_normalized/` is fixed at 1000×1000** and is not affected by `--width`/`--height`. Auto views keep randomized FOV unless `--manual_fov` is set.

**Output layout** (`out_normalized/` with `--views auto`):

```
out_normalized/
├── ssl.txt, data.json                # normalized scene (when --normalized_topdown / --views auto)
├── glb/scene.glb                     # --holo_geometry --glb (full scene)
├── pointcloud/scene_all.ply          # --holo_geometry --ply (full scene)
├── voxel/                            # --holo_geometry --voxel (256³ occupancy)
├── topdown_normalized/               # pixel-aligned topdown + floor path (1000² by default)
│   ├── topdown.png
│   ├── camera_para.json
│   ├── topdown_depth.png             # [--depth] (also used for path planning)
│   ├── topdown_semantic.*            # [--semantic]
│   ├── nav_mask*.png                 # floor path planning masks
│   ├── floor_path_ssl.txt
│   └── topdown_floor_path.png
├── auto_views.json                   # auto camera manifest
├── topdown/                          # regular topdown (--width × --height, default 1000×1000)
│   ├── topdown.png
│   ├── topdown_depth.png             # [--depth]
│   ├── topdown_semantic.*            # [--semantic]
│   ├── planar_faces.json             # [--ply]
│   ├── glb/                          # [--visible_geometry --glb]
│   │   ├── scene_visible.glb
│   │   ├── scene_visible_opencv.glb
│   │   ├── scene_visible_viewvis_point_tri.npz   # sequence: per-triangle occlusion vs each camera
│   │   ├── scene_visible_viewvis_object_tri.npz  # sequence: per-triangle object-level vs each camera
│   │   └── scene_visible_viewvis_tri.json
│   ├── pointcloud/                   # [--visible_geometry --ply]
│   │   ├── scene_visible.ply
│   │   ├── scene_visible_opencv.ply
│   │   ├── floor/, walls/, boxes/    # (+ ceiling/, doors/, windows/ when present)
│   │   └── …/*_visible.ply, *_opencv.ply
│   └── voxel/                        # [--visible_geometry --voxel]
├── auto_path_0000/                   # auto single-frame view
│   ├── {timestamp}.png
│   ├── {timestamp}_depth.png         # [--depth]
│   ├── {timestamp}_semantic.*        # [--semantic]
│   ├── {timestamp}_camera_para.json
│   ├── glb/scene_visible.glb         # [--visible_geometry --glb]
│   ├── glb/scene_visible_opencv.glb
│   ├── pointcloud/scene_visible.ply  # [--visible_geometry --ply] merged (world SSL)
│   ├── pointcloud/scene_visible_opencv.ply
│   ├── pointcloud/floor/, walls/, boxes/
│   ├── voxel/                        # [--visible_geometry --voxel]
│   └── ssl_opencv.txt
├── auto_path_0000_pano/              # [--pano] sibling panorama for auto_path_0000
│   ├── {timestamp}_pano.png          # equirectangular (--pano_resolution × half)
│   └── {timestamp}_pano_camera_para.json
├── auto_path_0008_seq/               # auto three-frame sequence
│   ├── {timestamp}_000.png … _002.png
│   ├── {timestamp}_*_depth.png       # [--depth]
│   ├── glb/scene_visible.glb
│   ├── glb/scene_visible_viewvis_*_tri.npz  # [--visible_geometry --glb]
│   ├── pointcloud/scene_visible.ply         # [--visible_geometry --ply]
│   ├── pointcloud/scene_visible_viewvis_*.npz
│   └── …
├── auto_path_0008_seq_pano/        # [--pano] panorama for sequence (first-frame ref)
├── video_{N}/                      # [--video] tangent trajectory root (usually no perspective RGB)
├── video_{N}_pano/                 # [--video] pano sequence (~0.2m arc spacing; N auto-computed)
│   ├── {timestamp}.png …
│   └── …
```

## Video trajectories (`--video`)

`--video` builds **one** smooth tangent closed-loop sequence along the floor path from `topdown_normalized/`. It renders **pano RGB only** (equirectangular); frame count is computed from path arc length (default **~0.2 m spacing**, tunable via `--video_spacing`). Independent of sparse auto views (`auto_path_*`).

| Mode | Command | Pipeline |
| ---- | ------- | -------- |
| **Video only** | `--video` (omit `--views`) | `topdown_normalized` + path planning → `video_{N}_pano/` only; **skips** sparse auto and regular `topdown/` |
| **Auto + video** | `--views auto --video` | Keeps sparse auto + `topdown/` + appends video |

Directories: `video_{N}/` (sequence root, usually empty of perspective RGB) + `video_{N}_pano/` (pano frames). Trajectory manifest → `Y/auto_views.json`; resume reuses it.

**RGB**: regardless of `--pano` / `--depth` / `--semantic`, video views output **panorama images only**. **Geometry exports** (`--glb`, `--ply`, `--visible_geometry`, `--voxel`, …) still follow CLI flags and land under `video_{N}_pano/`.

```bash
python render_ssl.py --ssl scene.txt --output out --video \
  --glb --ply --visible_geometry --video_spacing 0.2
```

See [§5.1.1 Video trajectories](docs/doc.en.md#511-video-trajectories--video).

Full directory reference: [§6 Output layout](docs/doc.en.md#6-output-layout-and-coordinate-systems).

Custom cameras are also supported via `--camera_position`, `--look_at`, and optional `--up_vector` (auto-fallback to `[0,1,0]` when the view direction is collinear with default up `[0,0,1]`).

## Split merged visible geometry by camera (`split_viewvis_geometry.py`)

Multi-frame sequences (`*_seq/`, e.g. `left_seq`) export a **union** `scene_visible.ply` / `scene_visible.glb` plus `viewvis` NPZ sidecars (one row per point or triangle, one column per camera frame). See [§7.1.2](docs/doc.en.md#712-multi-view-sequence-visibility-sidecars-viewvis) for how those matrices are produced.

Use **`split_viewvis_geometry.py`** to split the merged file into **one PLY/GLB per frame**, keeping only elements visible from that camera:

```bash
# Point cloud (world or OpenCV copy — same row order)
python split_viewvis_geometry.py left_seq/pointcloud/scene_visible_opencv.ply view
python split_viewvis_geometry.py left_seq/pointcloud/scene_visible.ply object

# Mesh (world or OpenCV GLB)
python split_viewvis_geometry.py left_seq/glb/scene_visible_opencv.glb view
python split_viewvis_geometry.py left_seq/glb/scene_visible.glb object
```

| Mode | NPZ used | Meaning |
| ---- | -------- | ------- |
| **`view`** | `scene_visible_viewvis_point*.npz` | **Point-level occlusion** — keep samples in frustum and not occluded (depth / ray cast) |
| **`object`** | `scene_visible_viewvis_object*.npz` | **Object-level export rule** — keep all samples belonging to objects visible from that camera (in frustum) |

The script auto-detects sidecars in the same folder (`pointcloud/` or `glb/`). Output: `{input_stem}_by_{mode}/` next to the input file, with `{camera_frame}.ply` or `.glb` per frame and `manifest.json` (kept counts). Implementation: `core/viewvis_split.py`.

Optional: `-o /path/to/output_dir` to override the default output folder.

## Documentation

**Full manuals:** [English](docs/doc.en.md) · [中文](docs/doc.zh-CN.md)

Read in this order:

| Step | Topic | English | 中文 |
| ---- | ----- | ------- | ---- |
| 1 | Installation & dependencies | [§1](docs/doc.en.md#1-installation-and-dependencies) | [§1](docs/doc.zh-CN.md#1-安装与依赖) |
| 2 | SSL format & coordinates | [§2](docs/doc.en.md#2-ssl-coordinate-system-and-entities) | [§2](docs/doc.zh-CN.md#2-ssl-坐标系与实体约定) |
| 3 | Quick start & examples | [§3](docs/doc.en.md#3-quick-start) | [§3](docs/doc.zh-CN.md#3-快速开始) |
| 4 | API reference (`topdown_view`, `render_view`, `render_ssl`) | [§4](docs/doc.en.md#4-api-reference) | [§4](docs/doc.zh-CN.md#4-api-参考) |
| 5 | Auto views, video trajectories & floor path | [§5](docs/doc.en.md#5-advanced-workflows) | [§5](docs/doc.zh-CN.md#5-高级工作流) |
| 6 | Output layout, `c2w`, exports | [§6](docs/doc.en.md#6-output-layout-and-coordinate-systems) | [§6](docs/doc.zh-CN.md#6-输出目录与坐标系) |
| 7 | Semantic / depth / voxel details | [§7](docs/doc.en.md#7-export-artifacts) | [§7](docs/doc.zh-CN.md#7-导出产物详解) |

## License

This project is licensed under the [MIT License](LICENSE).