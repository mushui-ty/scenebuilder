# SceneBuilder — Full Documentation

**English** | [中文](doc.zh-CN.md)

> Full API reference, coordinate systems, output directories, and export artifacts. For a quick overview, see [README (English)](../README.md) / [README（中文）](../README.zh-CN.md).

Build 3D scenes from SSL / JSON descriptions (walls, doors, windows, furniture), with dual-backend rendering via **Blender (bpy)** and **Pyrender**, multi-view export, semantic maps, depth maps, path planning, panoramas, point clouds, visible voxels, and mesh export.

## Table of Contents


| Section | Contents |
| ------ | ----------------------------------------------------------------- |
| **§1** | Installation and Dependencies |
| **§2** | SSL Coordinate System and Entity Conventions |
| **§3** | Quick Start (minimal example + `normalized_topdown`) |
| **§4** | API Reference (including `normalized_topdown_view` / `render_normalized_topdown`) |
| **§5** | Advanced Workflows (auto views, pixel alignment, floor path) |
| **§6** | Output Directories and Coordinate Systems (c2w, ssl_opencv) |
| **§7** | Export Artifacts (visible geometry, PLY, voxels, semantic, depth) |
| **§8** | Configuration and Backend Comparison |
| **§9** | Asset Processing (`get_mesh`) |
| **Appendix** | More examples, implementation notes |


---



## 1. Installation and Dependencies



### Python Version Requirement: 3.11

The latest version of `bpy` cannot be installed on versions other than 3.11.

### Core Dependencies

```bash
pip install numpy shapely scipy imageio pyyaml
```



### Rendering Backends


| Backend | Installation | Use Cases |
| ------------ | ------------------------------ | ------------------------- |
| **bpy (recommended)** | `pip install bpy` | PBR rendering, semantic maps, Cycles depth, visible geometry |
| **pyrender** | `pip install pyrender trimesh` | Lightweight OpenGL rendering, environments without bpy |


**Additional dependencies for Pyrender on headless servers:**

```bash
conda install -c conda-forge libstdcxx-ng=12 -y
apt install -y libegl1-mesa-dev libgles2-mesa-dev mesa-utils xvfb \
  libgl1-mesa-glx libglu1-mesa libxrender1 libxext6
export PYOPENGL_PLATFORM=egl
```

**System libraries for Blender (as needed):**

```bash
apt install -y libxi6 libxrender1 libxrandr2 libxfixes3 libxcursor1 libxinerama1 \
  libxxf86vm1 libgl1-mesa-glx libglu1-mesa libxkbcommon0 libgtk-3-0
```



### Install into Python Environment

```bash
cd scenebuilder
pip install -e . --config-settings editable_mode=strict
```

---



## 2. SSL Coordinate System and Entity Conventions

All SSL parsing, scene `context`, rendering, and exports (including `ssl.txt` and point cloud naming) share the same **world coordinate system**, with units in **meters (m)**.

### 2.1 Axes

When viewing a room from above (topdown):


| Axis | Positive Direction | Direction in Top-Down View |
| ----- | --- | -------- |
| **X** | +X | Right |
| **Y** | +Y | **Up** |
| **Z** | +Z | Vertical up (height) |


- The ground lies in the **XY plane**; **Z = 0** is typically the ground height.
- Furniture orientation `**angle_z**`: at `0°` the object faces **−Y** (toward the **bottom** of the top-down view); increasing angles rotate **counterclockwise** around the **Z axis** (right-hand rule).

```
Top-down view (+Y up, +X right):

        +Y
         ↑
         |
    −X ←─┼─→ +X
         |
         ↓
        −Y     orientation at angle_z=0° →
```



### 2.2 Entity Fields

**Room**

```ssl
Room(room_type="balcony")
```

**Wall** — A wall segment is a line on the ground, extruded vertically upward:

```ssl
Wall(label="wall0", p=[x1, y1, 0], q=[x2, y2, 0], height=2.7)
```


| Field | Meaning |
| -------- | --------------------------- |
| `p`, `q` | Wall segment start and end; the third dimension can be `0`; only XY is used |
| `height` | Wall height, from `z=0` to `z=height` |


**Door / Window** — Door/window openings attached to a wall:

```ssl
Door(label="door0", center=[x, y, z], width=2.32, height=2.3, wall="wall2", asset_id="61064792")
Window(label="window0", center=[x, y, z], width=2.37, height=2.2, wall="wall0")
```


| Field | Meaning |
| ---------- | -------------------------------------------------------------- |
| `center` | **Geometric center** of the opening (3D). For example, `height=2.3, z=1.15` means the center is at mid-height, with the bottom edge near `z ≈ 0` |
| `width` | Width along the **wall direction** |
| `height` | Height along the **Z axis** |
| `wall` | `label` of the parent wall (e.g. `wall0`) |
| `asset_id` | Optional; if missing, the opening is still cut but no door/window model is loaded |


If `center` XY is not on the wall line at load time, it is snapped to the perpendicular foot on the nearest wall.

**Bbox** — Furniture or oriented bounding box:

```ssl
Bbox(label="sidetable0", center=[1.96, 0.21, 0.31], angle_z=180, scale=[0.42, 0.42, 0.63], asset_id="56056912")
```


| Field | Meaning |
| ---------- | ------------------------------------------------------ |
| `center` | **Geometric center** of the rotated OBB |
| `angle_z` | Rotation around Z axis (degrees), positive = counterclockwise |
| `scale` | **Full length** along local XYZ axes (not half-length). For floor-standing objects, typically `center.z ≈ scale.z / 2` |
| `asset_id` | Corresponds to `{asset_id}.glb`; if missing, the bbox is removed |


The SSL bbox **local coordinate system** convention: local +X / +Y / +Z correspond to the three OBB axes `scale[0]` / `scale[1]` / `scale[2]` respectively; semantic "forward/facing" is local **−Y**. Therefore at `angle_z=0°` local axes align with SSL world axes, while object forward is world **−Y**; as `angle_z` increases, the local frame and forward direction rotate counterclockwise around SSL +Z together.

GLB loading pipeline: center model → scale by `scale` → rotate by `angle_z` around Z → translate to `center`.

### 2.3 Relationship to Rendering Views

`render_ssl` default perspective cameras (e.g. `front`) are placed on the **−Y side** (`y = center_y − span_y/3`), looking toward the room center, i.e. looking in **+Y** into the room. This is consistent with "+Y at the top of the top-down view, `angle_z=0°` facing −Y".

The `topdown` camera looks down at the XY plane from **+Z**.

### 2.4 Standard SSL Output

After normalization via `normalize_scene_data()`, the output `ssl.txt` uses canonical `label` values (`wall0`, `door0`, `sidetable0` …), removing original `id` / `room_id`; field semantics follow the conventions above.

---



## 3. Quick Start

The examples below are based on the **SSL world coordinate system** (see §2). The underlying rendering class is `BpySceneCtx` (`core/scenebuilder_bpy.py`); `SceneCtx` (pyrender backend) has a similar signature, but some advanced exports are bpy-only.

### 3.1 Simplest Example: `topdown_view`

Load a scene and render a single top-down view only, without depth/semantic/GLB or other advanced features.

```python
from scenebuilder.core.util_data import parse_scene_input
from scenebuilder.core.scenebuilder_bpy import BpySceneCtx

SSL = "path/to/ssl.txt"
ASSETS = "/path/to/assets"

with open(SSL, encoding="utf-8") as f:
    scene = parse_scene_input(f.read())

ctx = BpySceneCtx(scene["room"]["room_type"], asset_dir=ASSETS)
ctx.add_walls(scene["wall"])
if scene.get("door"):
    ctx.add_doors(scene["door"])
if scene.get("window"):
    ctx.add_windows(scene["window"])
ctx.add_boxes(scene["bbox"])
ctx.normalize_scene_data()

# output is a directory → actual PNG written to {output}/topdown/topdown.png
ctx.topdown_view("output", rebuild=True)
```



### 3.2 Simplest Example: `render_view`

On the same `ctx`, specify camera position and look-at target to render a single perspective image.

```python
meta = ctx.context["meta"]
center = meta["center"]
z_cam = meta["z_max"] * 5 / 6

# output is a directory → actual PNG written to {output}/{timestamp}/{timestamp}.png
ctx.render_view(
    "output",
    camera_position=[center[0], center[1] - meta["span"][1] / 3, z_cam],
    look_at_target=[center[0], center[1], z_cam / 2],
    rebuild=True,
)
```

All coordinates are in the **SSL world frame** `[x, y, z]` (meters): the camera is offset toward the −Y side from the room center, at height approximately `5/6 * z_max`, looking at the middle of the room.

### 3.3 Advanced Example: Batch Multi-View + Full Export (`render_ssl.py`)

For benchmarks / data production, use the CLI or `render_ssl()` to complete multi-view rendering, GLB, point clouds, semantic, depth, panorama, etc. in one pass.

```bash
python /data-nas/data/experiments/mushui/projects/utils/fast-scene/scenebuilder/render_ssl.py \
  --ssl /data-nas/data/experiments/mushui/projects/SpatialFactory/benchmark/data/Office/325148303_0/render_output/ssl.txt \
  --views auto \
  --output /data-nas/data/experiments/mushui/projects/SpatialFactory/benchmark/data/Office/325148303_0/out2 \
  --glb \
  --assets /data-nas/data/dataset/qunhe/Manycore-Future/simplified \
  --ply --visible_geometry --voxel --semantic --depth --pano
```

Equivalent Python call:

```python
from scenebuilder.render_ssl import render_ssl

with open(SSL, encoding="utf-8") as f:
    ssl_text = f.read()

y_dir, standard_ssl, floor_result = render_ssl(
    ssl_text,
    output_root="out2",
    views="auto",
    asset_dir="/data-nas/data/dataset/qunhe/Manycore-Future/simplified",
    export_glb=True,
    export_point_cloud=True,
    export_voxel=True,
    visible_geometry=True,
    semantic=True,
    depth=True,
    pano=True,
)
```

`--views auto` **forces** `normalized_topdown=True` (see §3.4); for static multi-view use `--views topdown front left_seq`, etc.; see §4.4 for whether normalization is combined.

### 3.4 Pixel-Aligned Top-Down View (`normalized_topdown`)

Unlike the regular `topdown_view` in §3.1 (1024², SSL world coordinates unchanged): **translate the entire scene SSL** so the top-left of the top-down image corresponds to ground `(0,0)`, output directory is `{output}_normalized/`, main artifact in `topdown_normalized/`. Commonly used for floor path planning and SpatialFactory pixel coordinate alignment.

**CLI (normalization + path only, no other views):**

```bash
python scenebuilder/render_ssl.py --ssl path/to/ssl.txt \
  --normalized_topdown \
  --output path/to/out \
  --assets path/to/assets
# To skip floor path sampling: add --no_floor_path
```

**Python (recommended wrapper** `render_normalized_topdown`**):**

```python
from scenebuilder.render_ssl import render_normalized_topdown

with open(SSL, encoding="utf-8") as f:
    ssl_text = f.read()

result = render_normalized_topdown(
    ssl_text,
    output_dir="out",              # actual output root → out_normalized/
    backend="bpy",
    asset_dir=ASSETS,
    floor_path=True,               # default True: depth+semantic then sample closed-loop path
)
# result["path_points_ssl"] — path points (pixel-aligned SSL ground coordinates)
# result["topdown_normalized_dir"] — .../out_normalized/topdown_normalized/
```

**Direct Context call** (when you already have `ctx` and the scene has been `normalize_scene_data`):

```python
ctx.normalized_topdown_view(
    "out/topdown_normalized",
    render_depth=True,             # required for floor_path
    render_semantic=True,
)
```

See §4.4 comparison table for differences from §3.1 `topdown_view`; see **§5.2** for the nav_mask path pipeline.

See **§4** for parameters; **§6** for output directories; **§7** for export flags.

---



## 4. API Reference


| Need | Entry Point |
| ----------------- | ------------------------------------------------------------ |
| Resolution / FOV semantics (Blender) | **§4.0** (long-axis FOV, `sensor_fit=AUTO`, focal length) |
| Single top-down view (regular SSL) | `BpySceneCtx.topdown_view` (§4.1) |
| Pixel-aligned top-down + optional floor path | `render_normalized_topdown()` / `--normalized_topdown` (§4.4) |
| Custom camera perspective | `BpySceneCtx.render_view` (§4.2) |
| Batch multi-view / benchmark | `render_ssl()` or `render_ssl.py` (§4.3) |
| Auto path-driven views | `render_ssl(..., views="auto")` (§5.1, includes normalized_topdown) |




### 4.0 Resolution and FOV (Blender backend)

Applies to perspective views rendered via **bpy** in `topdown_view`, `render_view`, and `render_ssl` (not equirectangular `pano` cameras).

#### Specifiable imaging parameters

| Parameter | Role |
| --------- | ---- |
| `width` / `height` | Output pixel resolution, written to `scene.render.resolution_x/y` independently |
| `auto_fov` | Auto-compute field of view from scene bounding box (degrees → radians for Blender) |
| `manual_fov` | Manually specify one FOV angle (**degrees**); overrides `auto_fov` when not `None` |

Camera setup (`BpySceneCtx._create_render_camera`):

```python
camera_data.lens_unit = 'FOV'
camera_data.angle = fov_rad   # from manual_fov or auto_fov
# default sensor_fit = 'AUTO' (not explicitly changed)
scene.render.resolution_x = width
scene.render.resolution_y = height
```

#### `manual_fov` is the **long-axis** FOV, not always "vertical FOV"

With Blender `sensor_fit=AUTO`, `camera_data.angle` applies to the image **long axis**; the short-axis FOV is derived from aspect ratio. Both axes share the same focal length \(f_x = f_y\):

| Aspect | Long axis | `manual_fov` means | Short-axis FOV |
| ------ | --------- | ------------------ | -------------- |
| width ≥ height (e.g. 1280×720) | horizontal | **horizontal FOV** | vertical FOV smaller, derived from aspect |
| height > width (e.g. 720×1280) | vertical | **vertical FOV** | horizontal FOV smaller, derived from aspect |
| 1:1 (e.g. 1000×1000) | — | horizontal FOV = vertical FOV | — |

Implementation matches `util_bpy.camera_frustum_tangents` and `build_opencv_intrinsic_4x4` (§6.3.3):

- width ≥ height: `tan_x = tan(angle/2)`, `tan_y = tan_x / aspect`
- height > width: `tan_y = tan(angle/2)`, `tan_x = tan_y × aspect`
- Focal length: `fx = width / (2·tan_x)`, `fy = height / (2·tan_y)`, and **`fx = fy`**

**Mental model:** specify X/Y resolution + one FOV angle → long-axis FOV is fixed → focal length is computed → same focal length applies to the short axis → short-axis FOV follows. It is **not** "the same FOV applied to both X and Y".

The internal variable is often named `fov_y`, but for non-square landscape images it actually corresponds to Blender's `angle` (long-axis FOV)—do not read the name literally as "always vertical FOV".

#### PyRender backend difference

PyRender uses `PerspectiveCamera(yfov=..., aspectRatio=width/height)`, where `manual_fov` is **always vertical FOV**. For non-1:1 aspect ratios, PyRender and Blender FOV semantics may differ slightly; prefer `backend="bpy"` for data production.

#### Relation to pixel-aligned top-down

`topdown_normalized/` is **fixed at 1000×1000** and cannot be changed via API (SpatialFactory Stage 1 convention; `image_half=500` in `prepare_pixel_aligned_topdown_context` is tied to this). At 1:1, long-axis FOV equals horizontal = vertical FOV; the `fov` in §5.2 `pixel2real_ratio = camera_z × tan(fov/2) / 500` is that angle.

#### Default resolution by API (not all 1000×1000)

| API | Default resolution | Configurable? |
| --- | ------------------ | ------------- |
| `topdown_view` / `render_view` | **1024×1024** | Yes (`width` / `height`) |
| `render_ssl` views (incl. `topdown/`, auto) | **1000×1000** | Yes (`width` / `height` or CLI) |
| `topdown_normalized/` (pixel-aligned) | **1000×1000** | **No** (fixed) |




### 4.1 `topdown_view`

```python
ctx.topdown_view(
    output_path: str,              # directory → {dir}/topdown/topdown.png; or direct .png path
    width: int = 1024,
    height: int = 1024,
    geometry_mode: str = "gltf",   # geometry loading mode
    show_wall: bool = True,
    show_window: bool = True,
    show_door: bool = True,
    show_ceiling: bool = True,     # usually False for top-down
    up_vector: list = None,        # default [0,1,0]; camera "up" direction (SSL world frame)
    auto_fov: bool = True,         # auto FOV from scene bbox (long axis; see §4.0)
    manual_fov: float = None,      # specify long-axis FOV (degrees); overrides auto_fov (see §4.0)
    auto_transparent: bool = True, # auto-transparent occluding walls/ceiling/floor
    transparent_alpha: float = 0.0,
    render_depth: bool = False,    # also writes {basename}_depth.png + camera_para in same dir
    use_HDRI: bool = True,
    hdri_transparent_background: bool = True,
    visible_shadow: bool = True,
    lighting_type: Literal["area", "array", "none"] = "array",
    align_height: bool = True,     # align furniture to floor height
    rebuild: bool = False,         # True forces mesh rebuild
    export_glb: bool = False,
    glb_path: Optional[str] = None,
    export_point_cloud: bool = False,
    export_voxel: bool = False,
    visible_geometry: bool = False,# requires export_glb / export_point_cloud / export_voxel
    render_semantic: bool = False, # {basename}_semantic.png/json + bbox_2d
)
```


| Parameter | Description |
| ----------------------------------- | ------------------------------------------------------------------------------------- |
| `output_path` | When a directory: main image at `{dir}/topdown/topdown.png`; also writes `topdown_camera_para.json`, `ssl_opencv.txt` |
| `show_ceiling` | When `False`, ceiling is not rendered (common top-down config) |
| `up_vector` | Top-down default `[0,1,0]`; differs from `render_view` default `[0,0,1]` |
| `auto_fov` / `manual_fov` | Camera directly above scene; FOV determines visible range; `manual_fov` is **long-axis FOV** (§4.0) |
| `render_depth` | bpy: Cycles same-pass depth; `camera_para.json` includes `depth_scale` |
| `render_semantic` | Semantic map + JSON + bbox_2d overlay (see §7.3) |
| `export_glb` / `export_point_cloud` / `export_voxel` | Geometry type flags; with `visible_geometry=True` exports **current view** frustum subset; root full-scene requires `render_ssl(..., holo_geometry=True)` |
| `rebuild` | Recommended `True` on first call or after scene changes |


Camera position is auto-computed from scene `meta` (directly above center); **no** manual `camera_position` needed.

**Parameters shared with** `render_view` **(defaults per function signature):**


| Parameter | Default | Description |
| ----------------------------------- | ----------- | ------------------------------------ |
| `width` / `height` | `1024` | Resolution |
| `geometry_mode` | `"gltf"` | `"gltf"` / `"mixed"` / `"bbox"` |
| `show_wall/window/door/ceiling` | `True` | Element visibility; top-down often uses `show_ceiling=False` |
| `auto_fov` / `manual_fov` | auto / `None` | Long-axis FOV (degrees); equals horizontal = vertical at 1:1 (§4.0) |
| `auto_transparent` | `True` | Auto-transparent occluding walls/ceiling/floor |
| `use_HDRI` | `True` | Enable HDRI from config |
| `hdri_transparent_background` | `True` | Top-down default transparent background |
| `lighting_type` | `"array"` | `"array"` / `"area"` / `"none"` |
| `align_height` | `True` | Wall height alignment |
| `rebuild` | `False` | Recommended `True` after scene changes |
| `render_depth` | `False` | `{basename}_depth.png` + normal (bpy) |
| `render_semantic` | `False` | Semantic map (§7.3) |
| `export_glb` / `export_point_cloud` / `export_voxel` | `False` | Geometry type flags; per-view requires `visible_geometry=True`, root requires `holo_geometry=True` |
| `visible_geometry` | `False` | Per-view visible GLB/PLY/voxels (§7.1 / §7.5) |
| `holo_geometry` | `False` | Root full-scene GLB/point cloud/voxels (§7.1.1) |




### 4.2 `render_view`

```python
ctx.render_view(
    output_path: str,
    camera_position: list,         # required, SSL world frame [x,y,z]
    look_at_target: list = None,    # default scene center + z_max/2
    width: int = 1024,
    height: int = 1024,
    up_vector: list = None,         # default None → [0,0,1] (SSL +Z)
    auto_fov: bool = True,
    manual_fov: float = None,
    auto_transparent: bool = True,
    transparent_alpha: float = 0.0,
    geometry_mode: str = "gltf",
    render_depth: bool = False,
    show_wall / show_window / show_door / show_ceiling: bool = True,
    use_HDRI: bool = True,
    hdri_transparent_background: bool = False,
    visible_shadow: bool = True,
    lighting_type: Literal["area", "array", "none"] = "array",
    align_height: bool = True,
    rebuild: bool = False,
    export_glb / glb_path / export_point_cloud / export_voxel / visible_geometry: same as topdown_view,
    render_semantic: bool = False,
    pano: bool = False,             # also outputs sibling dir {basename}_pano/
    pano_resolution: int = 4096,     # panorama width; height = width/2
    view_dir_name: Optional[str] = None,  # fixed subdirectory name (e.g. "front")
)
```


| Parameter | Description |
| ----------------- | ------------------------------------------------------------------------ |
| `camera_position` | Camera center, SSL world coordinates (meters) |
| `look_at_target` | Look-at point; when omitted uses `[center_x, center_y, z_max/2]` |
| `up_vector` | Camera up direction, default SSL +Z `[0,0,1]` |
| `output_path` | When a directory → `{dir}/{view_dir_name or timestamp}/{timestamp}.png` |
| `view_dir_name` | Fixed output subdirectory; `render_ssl.py` passes `"front"` etc. for preset views |
| `pano` | Additionally renders equirectangular panorama in `{view_dir}_pano/` (only `render_view`, not topdown) |
| **Sequence mode** | When `camera_position` / `look_at_target` / `up_vector` are **lists of lists**, multi-frame sequence rendering is triggered automatically |


Output: `{basename}.png`, `{basename}_camera_para.json` (includes `c2w`, `intrinsic`, §6.3), `ssl_opencv.txt` in same directory (§6.4); optional depth / semantic / pano attachments (§7).

**Unique parameters:**


| Parameter | Description |
| -------------------------- | --------------------------------------------- |
| `camera_position` | Camera center, SSL world coordinates (meters), **required** |
| `look_at_target` | Look-at point; default `[center_x, center_y, z_max/2]` |
| `up_vector` | Camera up, default SSL +Z |
| `pano` / `pano_resolution` | Additional `{view_dir}_pano/` panorama (topdown has no pano) |
| `view_dir_name` | Fixed subdirectory name (e.g. `"front"`) |
| **Sequence mode** | Pass **lists of lists** for the three position parameters → multi-frame sequence, directory `{timestamp}_seq/` |


Remaining parameters same as §4.1 shared table; `hdri_transparent_background` defaults to `False`.

### 4.3 `render_ssl` / `render_ssl.py`

**Python signature:**

```python
def render_ssl(
    input_text: str,                    # SSL text or JSON string
    backend: str = "bpy",               # "bpy" | "pyrender"
    output_root: str = "output_ssl",
    image: Optional[str] = None,        # reference image when asset_mode=retrieve/generate
    retrieve_hole: bool = True,
    asset_mode: Literal["none", "retrieve", "generate"] = "none",
    outpaint_image_dir: Optional[str] = None,
    asset_dir: Optional[str] = None,
    gen_3d_model: Literal["hunyuan-3d-rapid", "hunyuan-3d-pro"] = "hunyuan-3d-pro",
    gen_texture: bool = False,
    texture_dir: Optional[str] = None,
    correct_tilt: bool = True,
    correct_yaw: bool = True,
    views: ViewsSpec = None,            # None=no render; list=specified views; "auto"=path-driven
    export_glb: bool = False,
    export_point_cloud: bool = False,
    export_voxel: bool = False,
    visible_geometry: bool = False,
    holo_geometry: bool = False,
    semantic: bool = False,
    depth: bool = False,
    pano: bool = False,
    pano_resolution: int = 4096,
    samples: Optional[int] = None,      # Blender Cycles sample count
    normalized_topdown: bool = False,
    floor_path: bool = True,
    normalized_topdown_show_ceiling: bool = False,
    width: int = 1000,                # per-view render width (§4.0); not topdown_normalized/
    height: int = 1000,               # per-view render height (§4.0)
    auto_fov: bool = True,
    manual_fov: Optional[float] = None,
    resume: bool = True,               # resume from checkpoint; CLI uses --no-resume to disable
) -> Tuple[str, str, Optional[dict]]
```

**CLI ↔ Python mapping:**


| CLI | Python | Description |
| ---------------------- | ------------------------- | ------------------------------------ |
| `--ssl PATH` | `input_text` (read file) | SSL or JSON scene |
| `--output DIR` | `output_root` | Default `{ssl_dir}/render_output` |
| `--views V ...` | `views` | Unspecified = no view rendering; `auto` used alone |
| `--backend bpy         | pyrender` | `backend` |
| `--assets DIR` | `asset_dir` | When omitted, tries `assets/` next to ssl, etc. |
| `--texture DIR` | `texture_dir` | Contains `floor/wall/ceiling_texture.png` |
| `--glb` | `export_glb=True` | Geometry flag (requires `--visible_geometry` or `--holo_geometry`) |
| `--ply` | `export_point_cloud=True` | Geometry flag + per-view `planar_faces` (latter independent of `--visible_geometry`) |
| `--visible_geometry` | `visible_geometry=True` | **Per-view** frustum GLB/PLY/voxels (requires at least one of `--glb` / `--ply` / `--voxel`) |
| `--holo_geometry` | `holo_geometry=True` | **Root** full-scene GLB/point cloud/voxels (requires at least one of `--glb` / `--ply` / `--voxel`) |
| `--voxel` | `export_voxel=True` | 256³ colored occupancy: per-view needs `--visible_geometry`, root needs `--holo_geometry` |
| `--semantic` | `semantic=True` | Per-view semantic map + JSON + bbox_2d |
| `--depth` | `depth=True` | Per-view depth + normal (bpy) |
| `--pano` | `pano=True` | Extra panorama per `render_view` view (topdown skipped) |
| `--pano_resolution N` | `pano_resolution` | Default 4096 |
| `--width W` | `width` | Per-view render width, default 1000 (§4.0) |
| `--height H` | `height` | Per-view render height, default 1000 (§4.0) |
| `--manual_fov DEG` | `manual_fov` | Long-axis FOV (degrees), overrides auto_fov (§4.0) |
| `--no_auto_fov` | `auto_fov=False` | Disable auto FOV; use with `--manual_fov` for perspective views |
| `--normalized_topdown` | `normalized_topdown=True` | Output Y=`{output}_normalized` |
| `--no_floor_path` | `floor_path=False` | Skip path sampling in `topdown_normalized/` |
| `--samples N` | `samples` | Blender sample count |
| `--no-resume` | `resume=False` | Disable resume, force re-render completed views |


**Preset view names** (`views` list elements): `topdown`, `front`, `behind`, `left`, `right`, `leftfront`, `rightfront`, `leftbehind`, `rightbehind`, `left_seq`. Camera positions are auto-computed from scene `meta` (similar rules to §3.2), `look_at` is `[center_x, center_y, z_max/2]`.


| Parameter | Description |
| ----------------------------------- | ------------------------------------------------------------------------------ |
| `views=None` | Only writes `data.json` / optional full-scene geometry (needs `holo_geometry`), **renders no views** |
| `views="auto"` | Forces `normalized_topdown=True` + `floor_path=True`; renders `topdown` + auto-generated path views |
| `normalized_topdown` | Sole output root `Y={output_root}_normalized`; pixel-align SSL first, then `Y/topdown_normalized/` |
| `export_glb` / `export_point_cloud` / `export_voxel` | Geometry type flags; **do not write alone**, require `visible_geometry` (per-view) or `holo_geometry` (root) |
| `visible_geometry` | Per-view frustum GLB/PLY/voxels (§7.1 / §7.5) |
| `holo_geometry` | Root full-scene GLB/point cloud/voxels (§7.1.1) |
| `semantic` / `depth` | Mapped to per-view `render_semantic` / `render_depth` on `topdown_view` / `render_view` |
| `asset_mode` | `"none"` uses existing SSL `asset_id` only; `retrieve`/`generate` require `image`, etc. |
| `gen_texture` | Generate floor/wall/ceiling textures internally when no external textures |


**bpy + depth behavior:** Each view renders in an independent subprocess (`worker_render_view`) to avoid Cycles memory leaks; root full-scene geometry is exported by `worker_render_post` (only when `--holo_geometry`).

**Return value:** `(y_dir, standard_ssl_text, floor_result)`. `floor_result` contains `path_points_ssl`, etc. when `normalized_topdown` and `floor_path=True`.


| Export Flag | Description |
| -------------------- | --------------------------------- |
| `export_glb` / `export_point_cloud` / `export_voxel` | Geometry type flags; require `visible_geometry` or `holo_geometry` |
| `visible_geometry` | Per-view frustum-cropped GLB/PLY/voxels (§7.1) |
| `holo_geometry` | Root full-scene GLB/point cloud/voxels (§7.1.1, `worker_render_post`) |
| `export_point_cloud` | Also: per-view `planar_faces` (§7.2, independent of `visible_geometry`) |
| `semantic` / `depth` | §7.3 / §7.4 |




### 4.4 Pixel-Aligned Top-Down View (`normalized_topdown`)

Three equivalent levels (low to high):


| Level | Usage |
| ------- | ----------------------------------------------------------------------- |
| Context | `ctx.normalized_topdown_view(output_dir, ...)` |
| Wrapper API | `render_normalized_topdown(input_text, output_dir, ...)` |
| General render | `render_ssl(..., normalized_topdown=True)` or CLI `--normalized_topdown` |


**Differences from** `topdown_view`**:


| | `topdown_view` (§4.1) | `normalized_topdown_view` / `--normalized_topdown` |
| ------------------ | ------------------------ | -------------------------------------------------- |
| SSL coordinates | Unchanged | **Translate** XY so image top-left ↔ ground `(0,0)` |
| Default resolution | 1024² | **Fixed 1000²** (not configurable) |
| Output directory | `{Y}/{view}/topdown.png` | `{Y}/topdown_normalized/topdown.png` |
| `camera_para.json` | SSL world frame | Includes `pixel2real_ratio`, image coordinate system (SpatialFactory) |
| Floor path | None | Auto nav_mask sampling when `floor_path=True` (§5.2) |




#### `render_normalized_topdown`

```python
def render_normalized_topdown(
    input_text: str,
    output_dir: str,              # actual Y = {output_dir}_normalized
    backend: str = "bpy",
    asset_dir: Optional[str] = None,
    texture_dir: Optional[str] = None,
    show_ceiling: bool = False,
    floor_path: bool = True,      # False → top-down only, no path sampling
    export_glb: bool = False,
    export_point_cloud: bool = False,
    ...
) -> Union[str, Dict[str, Any]]
```

Equivalent to `render_ssl(..., normalized_topdown=True, views=None)`. When `floor_path=True` and successful, returns dict with `path_points_ssl`, `topdown_normalized_dir`, `floor_path`, etc.; otherwise returns `y_dir` string.

#### `render_ssl` `normalized_topdown`-Related Parameters


| Parameter / CLI | Description |
| --------------------------------------------- | ------------------------------------------------------------------------------------- |
| `normalized_topdown` / `--normalized_topdown` | Output root `Y={output_root}_normalized`; pixel-align first; subsequent views/GLB/point cloud use normalized SSL |
| `floor_path` / `--no_floor_path` | Whether to run nav_mask path in `topdown_normalized/` (default True) |
| `topdown_normalized/` resolution | **Fixed 1000×1000**, not configurable (SpatialFactory Stage 1) |
| `normalized_topdown_show_ceiling` | Whether to render ceiling in normalization pass |
| `views="auto"` | **Implicitly** `normalized_topdown=True` + `floor_path=True`, plus extra `topdown/` + auto views (§5.1) |


Can be **combined** with `--views topdown front left_seq`: normalize first, then render regular multi-view under `Y/` (coordinate system is already normalized SSL).

#### `BpySceneCtx.normalized_topdown_view`

```python
ctx.normalized_topdown_view(
    output_dir: str,               # writes topdown.png / ssl.txt / camera_para.json inside dir
    show_ceiling: bool = False,
    render_depth: bool = False,    # required for floor_path
    render_semantic: bool = False,
    align: Optional[dict] = None,  # pass when already pixel-aligned, skip second translation
    write_ssl: bool = True,
    round_decimals: int = 2,
    ...
)
```

Resolution is **fixed at 1000×1000**; no `width` / `height` parameters.

Translation computed by `prepare_pixel_aligned_topdown_context`; when `align` is non-empty the scene is already in normalized SSL (`render_ssl` internal path). **Does not support** `visible_geometry` / `export_glb` (stripped from kwargs).

Pixel alignment formula, nav_mask pipeline, config keys: see **§5.2**.

---



## 5. Advanced Workflows



### 5.1 Auto Views (`--views auto`)

Equivalent to **`--normalized_topdown` + floor path + regular topdown + auto-generated views from path** (currently bpy only).

```bash
python scenebuilder/render_ssl.py --ssl scene.txt \
  --views auto --output out \
  --glb --ply --visible_geometry --voxel --semantic --depth --pano \
  --assets path/to/assets
# → Y = out_normalized/
```



#### Pipeline

1. **Step 1–2**: Same as §5.2 — pixel-aligned SSL normalization + `topdown_normalized/` floor path planning
  Path output `path_points_ssl`: closed-loop ground waypoints `0…n-1` (first and last connected)
2. **Step 3**:
  - Regular `**Y/topdown/**` top-down view (1024², same as `--views topdown`; **not** the pixel-aligned top-down from Step 2)
  - `core/auto_views.py` generates cameras from path points and renders in subprocesses; manifest written to `Y/auto_views.json`



#### Two Auto Render Types

**Type A — Single frame (multiple** `render_view`** calls)**

- Path point sampling (`sample_path_indices`):
  - **n < 40**: indices `0, 4, 8, …` (stride=4)
  - **40 ≤ n ≤ 100**: **15 evenly spaced** waypoints on the loop (e.g. n=100 ≈ every 7 points)
  - **n > 100**: **20 evenly spaced** waypoints
- Camera position: `(x, y, z)`, `x,y` from waypoint; `z ∈ [0.5, 2.5]` random, and `<` max wall height
- `world_up = (0, 0, 1)`; initial `look_at` = scene 3D bbox center
- Pitch: adjusted along ray AB, random **−40° ~ 5°**
- FOV: random **60° ~ 90°**; resolution **1000×1000**
- One output directory per waypoint `auto_path_{index}/` (e.g. `auto_path_0000/`)

**Type B — Three-frame sequence (one** `render_view`** call)**

- **Randomly** pick one point `(x, y, 0)` from `n` waypoints, index `k`
- Fixed camera position `(x, y, 1.5)`; initial `look_at` = bbox center
- Three-frame `look_at` (same camera position):
  - Frame 1: yaw left **10° ~ 40°**
  - Frame 2: center (no yaw)
  - Frame 3: yaw right **0° ~ 40°**
- FOV: random **60° ~ 90°**; resolution **1000×1000**
- Output directory `**auto_path_{k:04d}_seq/**`



#### Artifacts

In auto mode, Step 3 `**topdown/**`, `**auto_path_***`, `**auto_path_*_seq**` all use the same `worker_render_view`; **all CLI flags apply**:


| CLI | Auto single frame | Auto sequence | Auto `topdown/` |
| ------------------------------ | -------------- | ------- | ----------------- |
| `--depth` | ✅ | ✅ per frame | ✅ |
| `--semantic` | ✅ | ✅ per frame | ✅ |
| `--pano` | ✅ | ✅ per frame | ❌ |
| `--glb` + `--visible_geometry` | ✅ | ✅ multi-frame union | ✅ |
| `--ply` + `--visible_geometry` | ✅ | ✅ multi-frame union | ✅ |
| `--voxel` + `--visible_geometry` | ✅ | ✅ multi-frame union | ✅ |
| `--ply` (no visible_geometry) | ✅ planar_faces | ✅ per frame | ✅ |


Root `**scene.glb` / `pointcloud/scene_all.ply` / `voxel/**` exported by `worker_render_post` (full scene, requires `--holo_geometry`).

#### Output Example

```
out_normalized/
├── topdown/                     # Step 3 regular top-down (1024²)
├── auto_views.json
├── auto_path_0000/ …
└── auto_path_0012_seq/          # three-frame sequence
```

> Manual sequences (e.g. `left_seq`) still use directory name `{timestamp}_seq/`.



### 5.2 Pixel Alignment and Floor Path (Implementation Details)

API and quick examples: see **§3.4**, **§4.4**. This section explains alignment rules and the nav_mask pipeline.

```python
def render_normalized_topdown(
    input_text: str,
    output_dir: str,
    backend: str = "bpy",
    asset_dir: Optional[str] = None,
    texture_dir: Optional[str] = None,
    show_ceiling: bool = False,
    floor_path: bool = True,   # when True: render depth+semantic and call nav_mask_path
    ...
) -> Union[str, Dict[str, Any]]
```

`topdown_normalized/` output is **fixed at 1000×1000**; not configurable via parameters.

Underlying render: `BpySceneCtx.normalized_topdown_view()` / `SceneCtx.normalized_topdown_view()` (`core/util_data.prepare_pixel_aligned_topdown_context` performs XY translation).

### Pixel Alignment Rules

1. Compute topdown camera and FOV (same as regular top-down; at default 1000×1000 this is long-axis FOV, see §4.0)
2. `pixel2real_ratio = camera_z × tan(fov/2) / 500` (at 1:1, fov equals horizontal = vertical FOV)
3. **Translate entire scene SSL** so camera ground projection lands at image pixel `(ratio×500, ratio×500)` (image coordinate system)
4. Render 1000×1000 top-down → image top-left `(0,0)` corresponds to ground `(0,0)`
5. `camera_para.json` uses **image coordinate system** + `pixel2real_ratio` (SpatialFactory compatible)



### Floor Path Pipeline (`core/nav_mask_path.py`)

When `floor_path=True` (default), after rendering automatically calls `run_nav_mask_floor_path(output_dir, config)`:

```
topdown.png + topdown_depth.png + topdown_semantic.*
        ↓
① depth_mask     — pixels within camera-to-floor distance ± tol
② structure_mask — walls/doors/windows in semantic map
③ floor_mask     — floor in semantic map
        ↓
nav_mask = (depth_mask − structure_mask) ∪ floor_mask
        ↓
Sample closed-loop path along largest component inner boundary (default 8–20 points)
        ↓
floor_path_points.json / floor_path_ssl.txt / topdown_floor_path.png
```

**Nav mask merge formula** (`combine_nav_mask`):

```
nav_mask = (depth_mask \ structure_mask) ∪ floor_mask
```

i.e. remove walls/doors/windows from depth-walkable region, then union semantic floor region.

Output directory tree: see **§6**.

### Configuration (`config.yaml`)


| Key | Default | Description |
| ---------------------------- | ------ | -------------------------- |
| `nav_mask_depth_tolerance_m` | `0.05` | Depth mask: camera-to-floor distance ± this value (meters) |
| `nav_mask_inset_m`           | `0.15` | Path sampling: inset from nav mask inner boundary (meters) |
| `nav_mask_point_spacing_m`   | `0.3`  | Closed-loop path point spacing (meters) |


---



## 6. Output Directories and Coordinate Systems



### 6.1 Directory Tree

**Single output root Y** (all artifacts under Y):


| `--normalized_topdown` | Y |
| ---------------------- | ---------------------- |
| No | `{output}/` |
| Yes | `{output}_normalized/` |


`--normalized_topdown` is a **global setting**: normalize SSL first; all subsequent rendering (views, GLB, point cloud) uses that coordinate system.

**Pipeline (with** `--normalized_topdown`**):**

1. Set Y = `{output}_normalized`, write normalized `ssl.txt` + `data.json`
2. `Y/topdown_normalized/`: 1000² pixel-aligned top-down + floor path (automatic)
3. `Y/topdown/`, `Y/{timestamp}/`, `Y/{timestamp}_seq/` …: render per `--views` or `auto`
4. `Y/scene.glb`, `Y/pointcloud/`, `Y/voxel/`: full-scene export (requires `--holo_geometry`)

```
Y/                                    # {output} or {output}_normalized
├── ssl.txt
├── data.json
├── scene.glb                         # [--holo_geometry + --glb]
├── pointcloud/                       # [--holo_geometry + --ply]
├── voxel/                            # [--holo_geometry + --voxel]
├── topdown_normalized/               # [--normalized_topdown] pixel-aligned + path planning
│   ├── topdown.png                   # 1000×1000
│   ├── camera_para.json              # image coords + pixel2real_ratio
│   ├── topdown_depth.png             # [floor_path]
│   ├── topdown_semantic.*
│   ├── nav_mask*.png                 # [floor_path]
│   ├── floor_path_ssl.txt            # [floor_path]
│   └── topdown_floor_path.png        # [floor_path]
├── auto_views.json                   # [--views auto]
├── auto_path_0000/ …                 # [--views auto] single frame
├── auto_path_0008_seq/               # [--views auto] three-frame sequence
├── {timestamp}_seq/                  # left_seq etc. manual sequences
├── topdown/                          # [--views topdown] 1024² regular top-down
│   ├── topdown.png
│   ├── topdown_depth.png             # [--depth]
│   ├── topdown_semantic.*            # [--semantic]
│   ├── planar_faces.json             # [--ply]
│   ├── pointcloud/                   # [--visible_geometry + --ply]
│   └── voxel/                        # [--visible_geometry + --voxel]
└── {millisecond_timestamp}/                      # single camera view
    ├── …
    ├── scene_visible.glb             # [--visible_geometry + --glb]
    ├── pointcloud/                   # [--visible_geometry + --ply]
    ├── voxel/                        # [--visible_geometry + --voxel]
    │   ├── occupancy_world.npz
    │   ├── occupancy_world_meta.json
    │   ├── occupancy_world_preview.ply
    │   ├── occupancy_opencv.npz
    │   ├── occupancy_opencv_meta.json
    │   ├── occupancy_opencv_preview.ply
    │   └── metadata_voxel.json
    └── …
```

```bash
# Normalization + auto path views
python scenebuilder/render_ssl.py --ssl path/to/ssl.txt \
  --views auto --output path/to/out --glb --assets path/to/assets

# Normalization + multi-view
python scenebuilder/render_ssl.py --ssl path/to/ssl.txt \
  --normalized_topdown --views topdown left_seq \
  --output path/to/out --glb --assets path/to/assets
# → sole output: path/to/out_normalized/
# left_seq → {timestamp}_seq/
```

---



### 6.2 Geometry Export Coordinate Systems (World SSL and OpenCV)

Each view is constructed/rendered in world (or normalized) SSL. With `--visible_geometry` and `--glb` / `--ply` / `--voxel`, **per-view directory** visible GLB/PLY/voxels **primary files are world SSL** (voxels: `occupancy_world.npz`); `*_opencv.*` copies are also written (OpenCV camera frame, see `core/geometry_opencv.py`). Root full-scene geometry (`scene.glb`, `pointcloud/scene_all.ply`, `voxel/`) requires `--holo_geometry`, no frustum cropping.


| Coordinate System | Applicable Files | Origin and Axes |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| **World SSL** | Root `ssl.txt`, `data.json`; `worker_render_post` (`--holo_geometry`) `scene.glb`, `pointcloud/scene_all.ply`, `voxel/occupancy_world.npz`; `{view}/pointcloud/*.ply` (no `_opencv`), `scene_visible.ply`, `scene_visible.glb` | Same as §2: +X right, +Y up, +Z height |
| **OpenCV camera frame** | `*_opencv.ply` / `*_opencv.glb` for each geometry file above | Origin: camera center; +X: image right; +Y: image down; +Z: into scene (depth positive) |


**Conventions:**

- Primary **PLY** vertices and normals are Blender world coordinates (= SSL Z-up, or normalized world after `--normalized_topdown`); attributes are `x y z nx ny nz red green blue`.
- Primary **GLB** and `**_opencv.glb**` both written via Blender `export_scene.gltf`, auto **Z-up → glTF Y-up**: `(gx, gy, gz) = (bx, bz, -by)`. Primary GLB mesh is world SSL; `_opencv.glb` mesh is OpenCV-transformed `(ox, oy, oz)`, on disk `(ox, oz, -oy)`; read back OpenCV coords with `gltf_yup_to_blender_zup`.
- `**_opencv.glb**`: after primary GLB export, mesh vertices transformed in Blender with same `transform_points_to_opencv` as PLY, then second glTF export (**preserves materials/UV**), no extra axis compensation.
- `**_opencv.ply**`: world points and normals transformed separately to OpenCV:
  - Points: `p_cam = R @ (p_world - eye)`
  - Normals: `n_cam = normalize(R @ n_world)` (rotate only, no translation)
  - ASCII vertices are OpenCV `(ox, oy, oz, nx, ny, nz)` (no glTF axis transform)
- OpenCV axes: +X right, +Y down, +Z into scene.
- `**left_seq**` and other multi-frame sequences: OpenCV copies use **first frame** camera as reference; visible geometry is union of all frame frustums.
- `metadata_visible.json` / `metadata.json` may record `"path_opencv": "scene_visible_opencv.ply"` etc. for merged point clouds.
- `planar_faces.json`, depth/semantic PNGs still use world SSL or pixel coords aligned with render image.
- Per-view directories **no longer** write `{view}/ssl.txt` (scene SSL only at output root Y); each view dir auto-writes `**ssl_opencv.txt**` (OpenCV camera frame, see §6.4).

`**camera_para.json` (per view)** records camera pose, depth/normal metadata, and **ScanNet / OpenSpatial compatible c2w extrinsics and pinhole intrinsics** (see §6.3). Example:

```json
{
  "camera_position": [wx, wy, wz],
  "look_at_target": [wx, wy, wz],
  "world_up": [0, 0, 1],
  "aspectRatio": 1.0,
  "fov_y": 1.2,
  "image_size": [1024, 1024],
  "camera_convention": "opencv",
  "world_convention": "ssl_z_up",
  "c2w": [[...4×4...]],
  "intrinsic": [[...4×4...]],
  "intrinsic_model": "pinhole_no_distortion",
  "depth_unit": "meter",
  "depth_scale": 5811.535562,
  "is_metric_depth": true
}
```

`camera_position` / `look_at_target` are SSL world coordinates (meters) for human readability; **for geometric projection prefer** `c2w` **+** `intrinsic`.

---



### 6.3 Camera Extrinsics c2w and Intrinsics intrinsic (ScanNet / OpenSpatial Compatible)

Each `{basename}_camera_para.json` (and SSL branch of `topdown_normalized/camera_para.json`) writes **4×4 c2w** and **4×4 pinhole K**, directly usable via `np.array(json["c2w"])`, equivalent to ScanNet `{frame}_pose.txt` / `{frame}_intrinsic.txt`.

#### 6.3.1 Coordinate Conventions


| Coordinate System | Axes | Usage |
| ------------------ | -------------------------- | ---------------------------------------- |
| **World (SSL)** | +X right, +Y up (top-down), +Z height; ground ∥ XY | Same frame as `ssl.txt`, 3D bbox, primary PLY point cloud |
| **Camera (OpenCV)** | +X right, +Y down, +Z forward (depth positive) | Depth map, normal map `{view}_normal.png`, `*_opencv.ply` |


`world_convention: "ssl_z_up"`, `camera_convention: "opencv"`.

#### 6.3.2 Extrinsics `c2w` (camera → world)

```
c2w = | R11  R12  R13  tx |
      | R21  R22  R23  ty |
      | R31  R32  R33  tz |
      |  0    0    0    1 |
```

- **R (3×3)**: OpenCV camera axes expressed in **SSL world frame**; columns are camera +X(right), +Y(down), +Z(forward).
- **t = [tx, ty, tz]ᵀ**: Camera center in SSL world frame (**meters**), same as `camera_position`.
- **Meaning**: Transform OpenCV camera point to world: `P_world = c2w @ [X_cam, Y_cam, Z_cam, 1]ᵀ`.
- **Inverse**: `w2c = inv(c2w)`, `P_cam = w2c @ P_world` (4D homogeneous).

Construction: exported consistently with `camera_position`, `look_at_target`, `world_up` and Blender render camera (`geometry_opencv.build_opencv_c2w_matrix`). **No** ScanNet-style `axis_align_matrix` — if input SSL is normalized, c2w is in normalized SSL world frame.

#### 6.3.3 Intrinsics `intrinsic` (pinhole K, no distortion)

```
intrinsic = | fx   0   cx   0 |
            |  0  fy   cy   0 |
            |  0   0    1   0 |
            |  0   0    0   1 |
```


| Element | Meaning |
| ---------------------- | ----------------- |
| `fx = intrinsic[0][0]` | Focal length in x (pixels) |
| `fy = intrinsic[1][1]` | Focal length in y (pixels) |
| `cx = intrinsic[0][2]` | Principal point u (pixels, image top-left origin) |
| `cy = intrinsic[1][2]` | Principal point v (pixels) |


Consistent with Blender `sensor_fit=AUTO` and current `image_size` (§4.0): `manual_fov` / `auto_fov` supplies long-axis angle; `tan_x` / `tan_y` follow aspect ratio; `fx = width / (2·tan_x)`, `fy = height / (2·tan_y)`, **`fx = fy`**; `cx = width/2`, `cy = height/2` (same rules as `util_bpy.camera_frustum_tangents`).

**Depth back-projection (camera frame, meters):**

```
X_cam = (u - cx) * depth_m / fx
Y_cam = (v - cy) * depth_m / fy
Z_cam = depth_m
```

where `depth_m = depth_pixel / depth_scale` (`depth_scale` see §7.4.1).

**Project world point to pixels:**

```
P_cam = w2c @ P_world                         # 4D homogeneous
uv_h  = intrinsic @ P_cam
u, v  = uv_h[0] / uv_h[2], uv_h[1] / uv_h[2]
```

Full chain: `pixel = K @ w2c @ P_world` (same as ScanNet `3dbox_filter` for 3D box projection validation).

#### 6.3.4 Companion Fields


| Field | Typical Value | Description |
| ----------------- | ------------------------- | -------------------------- |
| `depth_scale` | dynamic float | depth PNG ÷ `depth_scale` = meters |
| `depth_unit` | `"meter"` | Depth unit |
| `is_metric_depth` | `true` | Metric depth (when `--depth`) |
| `image_size` | `[W, H]` | Same resolution as RGB/depth/normal |
| `intrinsic_model` | `"pinhole_no_distortion"` | No distortion |


Panorama `{view}_pano.png` `camera_para` includes `c2w`, but `**intrinsic` omitted** (`projection: equirectangular`), pinhole K not applicable.

#### 6.3.5 Python Read Example

```python
import json
import numpy as np

with open("topdown_camera_para.json") as f:
    cam = json.load(f)

c2w = np.array(cam["c2w"], dtype=float)          # (4, 4)
K = np.array(cam["intrinsic"], dtype=float)       # (4, 4)
w2c = np.linalg.inv(c2w)

depth_m = imageio.imread("topdown_depth.png").astype(float) / cam["depth_scale"]
# pixel (u, v) + depth_m[v, u] → P_cam → P_world = (c2w @ [X,Y,Z,1])[:3]
```

In `topdown_normalized/camera_para.json`, `camera_position` may be in **image coordinates** (`coordinate_system: "image"`), but `**c2w` / `intrinsic` still based on SSL world `camera_position_ssl**` (same frame as scene geometry).

---



### 6.4 Per-View OpenCV SSL (`ssl_opencv.txt`)

Each view directory (including `topdown/`, `auto_path_0012/`, `auto_path_0023_seq/`, `auto_path_0012_pano/`, etc.) auto-writes `**ssl_opencv.txt**` after `topdown_view` / `render_view` render completes: transforms root `ssl.txt` geometry to **OpenCV camera coordinate system** (consistent with §6.3 `c2w` / depth / 3D Grounding).

#### 6.4.1 Reference Camera


| Render Mode | OpenCV Reference Frame |
| ----------------------------------- | ----------------------------------- |
| Single frame `render_view` / `topdown_view` | Camera pose of that render |
| Camera sequence `left_seq` / `auto_path_*_seq` | **First frame** camera (all frames and `_pano` subdirs share same reference) |
| `topdown_normalized/` | Pixel-aligned SSL top-down camera |


Reference camera is the same camera described by `c2w` in `{basename}_camera_para.json`; `c2w` is **OpenCV camera → SSL world**, `w2c = inv(c2w)` is **SSL world → OpenCV camera**.

#### 6.4.2 Coordinate System

- **OpenCV camera frame**: origin = optical center; +X right, +Y down, +Z forward (depth positive).
- All point coordinates in **meters**.
- `world_up`: original SSL world vertical up `[0, 0, 1]` transformed to current OpenCV camera frame (usually not `[0,0,1]`).



#### 6.4.3 Field Changes Relative to Root `ssl.txt`


| Entity | World SSL (`ssl.txt`) | OpenCV SSL (`ssl_opencv.txt`) |
| ----------------- | ----------------------------------------- | ---------------------------------------------------- |
| **Room** | `Room(room_type="...")` | Adds `world_up=[a,b,c]` |
| **Wall** | `p`, `q` SSL world XY (third dim often 0) | `p`, `q` OpenCV camera **3D** endpoints |
| **Door / Window** | `center` SSL world coords | `center` OpenCV camera coords |
| **Bbox** | `center` + `angle_z` (deg, around SSL +Z) + `scale` | `center` (camera frame) + `pose=[roll,pitch,yaw]` + `scale` (unchanged) |


Scalar dimensions `width` / `height` (doors/windows), wall `height`, bbox `scale` **unchanged** (local geometric lengths, meters).

#### 6.4.4 Bbox `pose` Convention (Standard Face-to-Face Pose + Intrinsic XYZ Rotation)

```ssl
Bbox(label="dining table0", center=[0.5, -0.1, 2.3], pose=[0.3, -0.05, 0.1], scale=[3.05, 0.93, 1.21], asset_id="17116819")
```



##### Field Meanings


| Field | Meaning |
| --------------------------- | -------------------------------------- |
| `pose = [roll, pitch, yaw]` | **Intrinsic XYZ** Euler angles from standard face-to-face pose to actual pose, in **radians** |
| `scale = [xl, yl, zl]` | Full length along box **local axes** (meters), not projected width on camera axes |
| `roll` | Step 1, intrinsic rotation around current OA/local **X(front)** |
| `pitch` | Step 2, intrinsic rotation around rotated OA/local **Y(left)** |
| `yaw` | Step 3, intrinsic rotation around rotated OA/local **Z(top)** |


Here `pose` is a physically meaningful Euler angle: first define an OA/local standard frame face-to-face with the observer, then rotate from that standard pose along moving axes `X -> Y -> Z` to the object's actual pose in the current OpenCV camera frame.

SSL bbox semantic axes:


| Semantic Axis | Bbox Local Axis |
| ------- | ---------- |
| `front` | `local -Y` |
| `left`  | `local +X` |
| `back`  | `local +Y` |
| `top`   | `local +Z` |


OpenCV camera frame: `+X` right, `+Y` down, `+Z` forward. Therefore:

Standard OA pose:

- `front -> camera -Z`: object front faces us.
- `left -> camera +X`: object left projects to image right.
- `top -> camera -Y`: object top projects to image top.

These three axes form a right-handed system: `front x left = top`, i.e. `camera -Z x camera +X = camera -Y`.

Standard pose matrix:

```python
R0 = [front0, left0, top0]
   = [camera -Z, camera +X, camera -Y]
```

Actual pose matrix:

```python
R_actual = [front_cam, left_cam, top_cam]
```

Relative rotation:

```python
R_delta = R0.T @ R_actual
pose = as_euler("XYZ", R_delta) = [roll, pitch, yaw]
```

Recovery:

```python
R_actual = R0 @ Rotation.from_euler("XYZ", pose).as_matrix()
```

SciPy uppercase `"XYZ"` means intrinsic rotation: from standard local frame, rotate sequentially around current `X(front)`, then rotated `Y(left)`, then rotated `Z(top)`.

##### Euler Principal Values and Gimbal Lock

This repo uses SciPy `"XYZ"` principal value convention for unique text representation:


| Component | Principal Range |
| ------- | ------------- |
| `roll`  | `[-π, π]`     |
| `pitch` | `[-π/2, π/2]` |
| `yaw`   | `[-π, π]`     |


When `pitch` approaches `+π/2` or `-π/2`, gimbal lock occurs: final 3D pose is still unique, but `roll` and `yaw` Euler allocation is not unique. This repo uses canonical solution:

- Fix `yaw = 0`
- Merge remaining freedom into `roll`

For example in topdown view, many floor objects have `pitch=π/2`; printer example can be canonically `pose=[π/2, π/2, 0]`. For training/eval, prefer rotation matrix or `front/left/top` axis directions recovered from `pose` for error metrics, not direct L1/L2 on Euler angles.

##### How `pose` Is Computed from World SSL

1. World OBB: `angle_z` → `R_world_box` rotation around SSL +Z only, translation `t_world = center`, size `scale`.
2. `T_world_box = [R_world_box | t_world]` means **bbox local → SSL world**; `T_cam_box = inv(c2w) @ T_world_box` means **bbox local → OpenCV camera**.
3. Extract semantic axes from `R_cam_box = T_cam_box[:3,:3]`:

```python
front_cam = R_cam_box @ [0, -1, 0]
left_cam  = R_cam_box @ [1,  0, 0]
top_cam   = R_cam_box @ [0,  0, 1]
```

4. Decompose `pose=[roll,pitch,yaw]` from `R_delta = R0.T @ [front_cam,left_cam,top_cam]`; `scale` unchanged.

Here `R_world_box = Rz(angle_z)` rotates local axes to SSL world axes. This does not contradict "`angle_z=0°` object faces −Y": forward vector is not local +X but `R_world_box @ [0, -1, 0]`.

This representation preserves full 3D pose via Euler angles with a clear physical zero pose. E.g. object front facing camera with `left/top` aligned to image right/top → `pose=[0,0,0]`; topdown printer extreme overhead pose can be interpreted as intrinsic rotation from face-to-face standard to actual `front/left/top` axes.

#### 6.4.5 Example Snippet

World SSL (root):

```ssl
Room(room_type="dining room")
Wall(label="wall0", p=[0.0, 0.0, 0], q=[5.0, 0.0, 0], height=2.7)
Bbox(label="dining table0", center=[5.1, 2.77, 0.6], angle_z=0, scale=[3.05, 0.93, 1.21], asset_id="17116819")
```

Same view `ssl_opencv.txt` (illustrative values):

```ssl
Room(room_type="dining room", world_up=[0.12, -0.98, 0.05])
Wall(label="wall0", p=[1.2, 0.3, 4.5], q=[-0.8, 0.3, 2.1], height=2.7)
Bbox(label="dining table0", center=[0.5, -0.1, 2.3], pose=[0.42, -0.15, 0.02], scale=[3.05, 0.93, 1.21], asset_id="17116819")
```



#### 6.4.6 Output Paths


| Directory | `ssl_opencv.txt` |
| ------------------------ | ---------------- |
| `Y/topdown/`             | ✅ |
| `Y/topdown_normalized/`  | ✅ |
| `Y/auto_path_0012/`      | ✅ |
| `Y/auto_path_0023_seq/`  | ✅ (first-frame camera frame) |
| `Y/auto_path_0012_pano/` | ✅ (same reference camera as corresponding perspective view) |
| `Y/ssl.txt` (root)       | ❌ remains world SSL |


Implementation: `core/ssl_opencv.py`; called by `BpySceneCtx.write_opencv_ssl_for_view()` / `SceneCtx.write_opencv_ssl_for_view()` during render save.

**Three types of point clouds:**


| Path | Coordinate System | Contents |
| ------------------------------ | ---------------- | ----------------------------------- |
| `output_root/pointcloud/`      | World SSL | **All** objects in scene (`--holo_geometry --ply`, post stage) |
| `{view}/pointcloud/*.ply`        | World SSL | **Visible** objects for that view (requires `--visible_geometry --ply`) |
| `{view}/pointcloud/*_opencv.ply` | OpenCV (that view/first-frame camera) | Same geometry as above, in OpenCV camera frame |
| `output_root/voxel/`           | World SSL | Full-scene 256³ occupancy (`--holo_geometry --voxel`) |
| `{view}/voxel/occupancy_*.npz`   | World SSL / OpenCV camera | Visible 256³ occupancy for that view (`--visible_geometry --voxel`) |
| `{view}/planar_faces.json`       | World SSL (3D vertices) | Wall/door/window/floor/ceiling inner surfaces (requires `--ply`) |


**Naming and context conventions (consistent with** `ssl.txt`**, context keys, point cloud filenames):**

- Floor / ceiling: `floor/floor.ply`, `ceiling/ceiling.ply` (with visible geometry: `floor/floor_visible.ply`, `floor/floor_visible_cutted.ply`, etc., same rules as walls/furniture)
- Walls: `wall0`, `wall1`, … → `walls/wall0.ply` (no `asset_id`)
- Doors/windows: `door0`, `window0`, … → `doors/door0_{asset_id}.ply`; if `asset_id` missing in asset dir **keep hole**, remove `asset_id`, filename and SSL omit `asset_id`
- Furniture: `sidetable0`, `armchair0`, … → `boxes/sidetable0_{asset_id}.ply`; if `asset_id` missing **delete that bbox**

Visible geometry suffixes: `_visible` (fully in frustum), `_visible_cutted` (frustum clipped, see below).

#### `_cutted` Suffix Meaning

`**_cutted` means the object was "cut by" the current camera frustum** — the object is still exported, but mesh/point cloud retains only the in-frustum portion; filename uses `_cutted` to mark "original object not fully in frame".


| Suffix | Meaning | Typical Scenario |
| ----------------- | --------------------------------------------------- | ----------------------- |
| `_visible`        | Object **≥ 95%** of triangle vertices in render viewport `[0,1]×[0,1]`, treated as fully in frame | Table in room center, wall segment fully in view |
| `_visible_cutted` | Above "fully in viewport" triangle ratio **< 95%**, frustum boundary cuts object | Window at frame edge, half wall cropped, furniture corner only |


**Determination and export are two steps:**

1. **Occlusion culling (object level)**: fully blocked by other objects → **entire object not exported** (no `_visible` or `_cutted`).
2. **Frustum clipping (triangle level)**: retained objects clipped by Blender render frustum; if geometry remains after clip, export.
3. **Naming**: based on pre-clip mesh "fully in viewport" face ratio ≥ 95% → `_visible`, else `_visible_cutted`.

Example:

```
floor/floor_visible_cutted.ply       # floor frustum clipped
floor/floor_visible_cutted_opencv.ply
ceiling/ceiling_visible.ply          # when ceiling visible (e.g. camera looking up)
walls/wall0_visible.ply          # entire wall mostly in frame
walls/wall2_visible_cutted.ply   # part of wall outside frame, frustum cut
walls/wall2_visible_cutted_opencv.ply  # same, OpenCV copy
```

Each object in `metadata_visible.json` has `"frustum_cutted": true/false` and `"frustum_in_view_ratio"` (0–1), consistent with filename.

**Note:** `_cutted` describes **frustum clipping only**, unrelated to occlusion; fully occluded objects simply do not appear in exports.

---



## 7. Export Artifacts



### 7.1 Visible Geometry (`visible_geometry`)

With `--visible_geometry` and at least one of `--glb`, `--ply`, or `--voxel`, each view directory exports visible GLB / point cloud / voxels; geometry files also have `_opencv` copies (see §6.2). Voxels detailed in **§7.5**.

**Note:** Voxels are **not** read back from GLB; they share the same in-memory visible triangles as GLB; `--voxel` can be used alone (no `--glb` required).

### Processing Pipeline (in order)

1. **Occlusion visibility (object level, based on full mesh)**
  - Ray cast on sampled surface points (transparent walls/ceiling/floor can pass through)
  - **Fully occluded** → discard object
  - **Partially or fully visible** → keep, proceed to next step
2. **Frustum clipping (triangle level)**
  - Use Blender `calc_matrix_camera` projection matrix, clip geometry in clip space
  - Remove triangles or parts outside frustum
  - No geometry after clip → discard object
3. **Naming rules (**`_visible` **/** `_cutted`**)**
  - See **`_cutted` Suffix Meaning** above
  - Only faces with **all three vertices in** viewport `[0,1]×[0,1]` count as "in frustum"
  - Ratio **≥ 95%** → `_visible`; **< 95%** → `_visible_cutted`
  - `metadata_visible.json` includes `"frustum_cutted"` and `"frustum_in_view_ratio"`



### Applicable Categories

floor, ceiling, walls, doors, windows, boxes all participate in visibility determination and frustum clipping.

---

### 7.1.1 Full-Scene Geometry (`holo_geometry`)

With `--holo_geometry` and at least one of `--glb`, `--ply`, or `--voxel`, export **complete scene** geometry at output root `Y/` (no occlusion/frustum clipping), written uniformly by `worker_render_post` after all views complete:

| Flag Combination | Root Artifacts |
| -------- | ---------- |
| `--holo_geometry --glb` | `scene.glb` |
| `--holo_geometry --ply` | `pointcloud/scene_all.ply` + per-object PLY |
| `--holo_geometry --voxel` | `voxel/occupancy_world.npz` etc. (world SSL only, no OpenCV copy) |

Independent from `--visible_geometry`: enable either or both (per-view visible + root full-scene).

```bash
# Per-view visible geometry only
--glb --ply --visible_geometry --voxel

# Root full-scene only
--glb --ply --holo_geometry --voxel

# Both
--glb --ply --visible_geometry --holo_geometry --voxel
```

---

### 7.2 Planar Inner-Surface Vertices (`export_point_cloud` / `--ply`)

With `--ply`, **each view** automatically exports after render (no separate flag):


| File | Description |
| ------------------- | -------------------- |
| `planar_faces.json` | Wall/door/window/floor/ceiling inner surface polygon vertices |
| `{view}_lines.png`  | Render image copy with vertex connections drawn (one color per object) |




### Geometry Definition

Consistent with scenebuilder mesh construction logic:

- **Wall**: SSL `p`/`q` are inner wall bottom points, height `align_height ? z_max : wall.height`; with door/window holes JSON includes `outer` loop and `hole` loops
- **Door/Window**: Inner face four vertices from `center`, `width`, `height`, and parent wall (same as `create_door_or_window_mesh`)
- **Floor**: Room `vertices` polygon @ z=0
- **Ceiling**: Room `vertices` polygon @ z=z_max (only when `show_ceiling=True` and built)



### Frustum Clipping and Vertex Order

- Polygons clipped in **Blender** `calc_matrix_camera` **homogeneous clip space** (same as visible geometry), interpolate world coords along edges to stay coplanar
- **Preserve boundary topology order** (floor/ceiling follow room `vertices` loop; walls/doors/windows follow mesh definition order; frustum clip does not re-sort by angle)
- If CW from camera view, reverse overall; then rotate start to vertex with min z → min y → min x
- Also record 3D coords, projected pixel coords `vertices_2d_px`, and occlusion flag `occluded` (1=occluded not visible in render, 0=visible)



### JSON Structure Example

```json
{
  "image": {"path": ".../topdown.png", "width": 1024, "height": 1024, "lines_overlay": ".../topdown_lines.png"},
  "objects": [
    {
      "category": "walls",
      "id": "0",
      "color_rgb": [255, 128, 64],
      "loops": [
        {"role": "outer", "vertices_3d": [[...], ...], "vertices_2d_px": [[px, py], ...], "occluded": [0, 0, 1, 0]},
        {"role": "hole", "opening_type": "window", "opening_id": "0", "vertices_3d": [...], "vertices_2d_px": [...]}
      ]
    }
  ]
}
```

**Note:** Planar vertex export is independent of `visible_geometry` — `--ply` alone suffices; visible point cloud `{view}/pointcloud/` still requires `--visible_geometry`.

---



### 7.3 Semantic Map (`semantic`)

With `--semantic`, each view (perspective / top-down / panorama, same rules) additionally outputs:


| File | Description |
| ---------------------- | ------------------------------------------ |
| `{view}_semantic.png`  | Entity-colored semantic segmentation map |
| `{view}_semantic.json` | Entity metadata, `color` ↔ mask, `pixel_num`, `bbox_2d` |
| `{view}_bbox_2d.png`   | Visualization of boxes and `label` on **original color render** |




#### 7.3.1 Rendering Mechanism

- Each entity unique color: internal key `entity:{category}:{label}` hashed via HSV to RGB; unique within scene (precise indexing)
- Colors guaranteed distinct from background `(0,0,0)`
- bpy: proxy mesh + **opaque Emission** material per entity, `view_transform=Raw`
- **Semantic-specific anti-aliasing off**: `cycles.samples=1`, `filter_width=0.01`, denoising off (EEVEE: `taa_render_samples=1`) so PNG pixels match JSON `color` **channel-exact**
- Semantic channel **does not preserve** transparent materials like glass: transparent objects occlude behind geometry as solid in semantic map
- Indexing: `pixel == color` exact match (no nearest neighbor); for legacy semantic maps with AA deviation, call `attach_semantic_bbox_2d(..., max_dist_sq=12)` for compatibility



#### 7.3.2 `label` Correspondence to SSL

`label` in `semantic.json` **equals normalized SSL** `label` (not original input `id` / `room_id`).


| Type | Normalization Rule | Example |
| --------------- | --------------------------- | ----------------------------- |
| Wall            | `wall{index}`               | `wall0`                       |
| Door            | global increment `door{n}`  | `door0`                       |
| Window          | global increment `window{n}`| `window0`                     |
| Bbox            | `slug(label/class)+ordinal` | `sidetable0`                  |
| floor / ceiling | fixed strings               | `floor` / `ceiling` (no SSL row) |


Inclusion in JSON depends on **successful entry into** `mesh_nodes`, not merely having `asset_id`:


| Case (default `geometry_mode=gltf`) | In `semantic.json`? |
| --------------------------------------- | ------------------- |
| Door/window/furniture: missing/invalid `asset_id`, not in `mesh_nodes` | **No** |
| Door/window/furniture: valid `asset_id` loaded into `mesh_nodes` | **Yes** (regardless of occlusion in current view) |
| Wall / floor / ceiling (no `asset_id`, but mesh built) | **Yes** |


Additional notes:

- Furniture `asset_id` set but asset missing → **entire row deleted** at normalization, not in scene or JSON
- Door/window invalid `asset_id` → keep wall hole, no model loaded → **not in** JSON
- In `mixed` / `bbox` mode, furniture without `asset_id` may fallback to box in `mesh_nodes`, thus in JSON



#### 7.3.3 Occlusion and Mask

- Semantic map colored by **opaque geometry depth**; pixels blocked by front mesh belong only to front entity
- **Occluded parts have no mask for that object**, even if occluder is transparent in color image (semantic channel treats as solid)
- Therefore: chair visible through glass in color image may still be fully blocked by glass door in semantic map



#### 7.3.4 `{view}_semantic.json` Fields

```json
{
  "semantic": true,
  "format": "per_entity_color_png",
  "background": [0, 0, 0],
  "objects": [
    {
      "category": "boxes",
      "label": "chair0",
      "color": [67, 222, 236],
      "pixel_num": 18432,
      "bbox_2d": [412, 256, 589, 701]
    },
    {
      "category": "boxes",
      "label": "table0",
      "color": [237, 64, 80],
      "pixel_num": 0
    }
  ]
}
```


| Field | Description |
| ----------- | ------------------------------------------------------------------------------------------------ |
| `category`  | `floor` / `ceiling` / `walls` / `doors` / `windows` / `boxes` |
| `label`     | Normalized entity name, matches SSL `label` |
| `color`     | RGB of entity in `*_semantic.png`, for mask lookup |
| `pixel_num` | Semantic mask pixel count for current view. Default **exact RGB match** on JSON `color` (with semantic AA off); `max_dist_sq>0` enables thresholded nearest neighbor for legacy maps |
| `bbox_2d`   | `[x1, y1, x2, y2]`, top-left `(0,0)`; axis-aligned bounding box of **all foreground pixels** (including multiple components); **only when** `pixel_num > 0`, otherwise omitted |


Notes:

- JSON `objects` list includes all loaded scene entities, **not indicating current view visibility**
- `pixel_num == 0` (fully invisible / fully occluded) → **no** `bbox_2d`
- Deprecated fields: `id` (use `label`), `color_key` (internal render only)



#### 7.3.5 `{view}_bbox_2d.png`

- Overlay detection boxes and `label` on original render (e.g. `topdown.png`, `{stamp}.png`)
- Box color uses entity `color`
- **Only draws objects with** `bbox_2d` (skip `pixel_num == 0` or missing `bbox_2d`)

---



### 7.4 Depth Map and Normal Map (`depth`)

With `--depth`, **bpy + Cycles** exports depth and normals in **same render pass** via compositor (no extra render time). `pyrender` backend exports depth only, no normals.

#### 7.4.1 Depth Map `{view}_depth.png`


| Item | Description |
| ---- | ---------------------------------------------------------------- |
| Format | **uint16 single-channel PNG** (not EXR) |
| Meaning | **Metric distance along +Z (view forward)** in **OpenCV camera frame** (consistent with Cycles Z pass / render clip) |
| Invalid pixels | `pixel == 0` (background, transparent, beyond clip, etc.); after decode `depth_m == 0` |


**bpy implementation:** Cycles compositor **Z pass** → temp EXR → uint16 PNG. On failure falls back to per-pixel `ray_cast` (**no** normal export then).

#### 7.4.1.1 File Pairing (Single Frame and Sequence)

Depth PNG pairs 1:1 with `**{basename}_camera_para.json**` in same directory (`basename` = color image filename without extension):


| Color Image | Depth Map | Camera Parameters |
| ------------- | ------------------- | -------------------------- |
| `topdown.png` | `topdown_depth.png` | `topdown_camera_para.json` |
| `{stamp}.png` | `{stamp}_depth.png` | `{stamp}_camera_para.json` |


When `render_view` receives **camera trajectory sequence**, all frames land in same `{timestamp}_seq/` (or `auto_path_XXXX_seq/`) directory, e.g.:

```
auto_path_0023_seq/
├── 1735123456789.png
├── 1735123456789_depth.png
├── 1735123456789_camera_para.json    # depth_scale applies to this frame only
├── 1735123456790.png
├── 1735123456790_depth.png
├── 1735123456790_camera_para.json
└── ssl_opencv.txt
```

**Important:** Sequences have **no** global unified `depth_scale`. Each frame computes encoding scale from **that frame's** max valid depth at render time, written to **that frame's** `*_camera_para.json`. When decoding frame *i* depth, must read frame *i* paired JSON, **cannot** use first or other frame's `depth_scale`.

#### 7.4.1.2 Encoding (Export)

1. Cycles Z pass yields metric depth `depth_m` (camera frame, OpenCV +Z depth consistent).
2. Take max valid pixel `d_max` where `depth_m > 0`; if none, `d_max` falls back to 1 m.
3. Set mapping upper bound `n = d_max × 1.5` (50% headroom to avoid far-point saturation).
4. Encoding scale: `depth_scale = 65535 / n` (implementation: `util.compute_depth_encode_scale`).
5. Store: `pixel = round(depth_m × depth_scale)`, clamp to `[0, 65535]`; invalid writes `0`.

Thus `depth_scale` is a **per-depth-map dynamic** parameter, varying with farthest visible distance in that frame; values may match across frames but **not guaranteed**.

#### 7.4.1.3 Decoding (Read)

Read `depth_scale` and `depth_unit` (always `"meter"`) from paired JSON:

```
depth_m = pixel / depth_scale     # pixel is uint16 depth PNG single channel
valid   = pixel > 0               # 0 means invalid, exclude from geometry
```

**Python (single frame):**

```python
import json
import imageio
import numpy as np
from scenebuilder.core import util

base = "1735123456789"  # same name as {base}.png
with open(f"{base}_camera_para.json") as f:
    cam = json.load(f)

depth_u16 = imageio.imread(f"{base}_depth.png")
depth_m = util.decode_depth_uint16(depth_u16, cam["depth_scale"])  # (H, W), meters, OpenCV +Z
valid = depth_u16 > 0
```

**Python (sequence batch, each frame its own scale):**

```python
import glob

for para_path in sorted(glob.glob("*_camera_para.json")):
    base = para_path.replace("_camera_para.json", "")
    with open(para_path) as f:
        cam = json.load(f)
    depth_m = util.decode_depth_uint16(
        imageio.imread(f"{base}_depth.png"),
        cam["depth_scale"],
    )
    # depth_m is metric depth in that frame's camera frame
```



#### 7.4.1.4 Relation to Back-Projection / `ssl_opencv.txt`

Decoded `depth_m(u, v)` is **+Z depth (meters)** of that pixel in OpenCV camera frame. With `intrinsic` from same-frame `{basename}_camera_para.json` (§6.3), back-project to camera 3D:

```
X_cam = (u - cx) * depth_m / fx
Y_cam = (v - cy) * depth_m / fy
Z_cam = depth_m
```

Then transform to SSL world via `c2w`: `P_world = c2w @ [X_cam, Y_cam, Z_cam, 1]ᵀ`. Geometry in same-dir `ssl_opencv.txt` (§6.4) is expressed in **first frame** (or single frame) OpenCV camera frame, consistent with depth back-projection.

#### 7.4.1.5 Depth Fields in `camera_para.json`

```json
{
  "depth_unit": "meter",
  "depth_scale": 5811.535562,
  "is_metric_depth": true
}
```


| Field | Description |
| ----------------- | --------------------------------------------------- |
| `depth_unit`      | Always `"meter"` |
| `depth_scale`     | **Applies only to** `{basename}_depth.png` **with same basename as this JSON** |
| `is_metric_depth` | `true` when depth exported |




#### 7.4.2 Normal Map `{view}_normal.png` (auto-exported with `--depth`)


| Item | Description |
| ---- | ------------------------------------------------------------------------- |
| Format | **uint8 RGB PNG** (3 channels, same resolution and pixel alignment as color) |
| Source | Blender Cycles **Normal pass** (compositor writes temp EXR, then PNG) |
| Coordinate System | **OpenCV camera frame** (same as `{view}_depth.png`; +X right, +Y down, +Z into scene) |
| Invalid pixels | Aligned with depth: invalid when corresponding `{view}_depth.png` pixel is `0` |


Same-directory `{basename}_camera_para.json` writes normal-related fields when exporting depth (`normal_space`, `normal_encode`, `normal_decode`, etc.); decode per JSON.

#### 7.4.2.1 What OpenCV Camera Frame Means

Normal components `(nx, ny, nz)` are unit vectors in **that view's OpenCV camera frame** (before export: Cycles Normal pass gets `n_cam = normalize(R @ n_world)`, same as `*_opencv.ply` normals and depth):


| Component | Positive Direction |
| ---- | ------------ |
| `nx` | +X, image **right** |
| `ny` | +Y, image **down** |
| `nz` | +Z, **into scene** along view (same as depth forward) |


**Consistent with depth:** Depth `{view}_depth.png` and normal `{view}_normal.png` **both in OpenCV camera frame**; directly usable for monocular normal supervision/prediction without manual `R` multiply.

**Difference from primary PLY:** Root or `{view}/pointcloud/*.ply` (no `_opencv`) point coords and normals remain **SSL world frame** (§6.2); only `_normal.png` and `*_opencv.ply` are camera frame.

#### 7.4.2.2 Normal Direction Definition

Each valid pixel stores: on the **frontmost visible surface** hit by that pixel's ray in the **same render as color**, the **geometric normal** Cycles uses for shading (Normal pass, not bump/normal map perturbation), transformed to OpenCV camera frame.

- **Semantics:** Unit vector `n = (nx, ny, nz)`, pointing along mesh outward normal of visible triangle (Blender computes from vertex winding), expressed **relative to current camera axes**.
- **View-dependent:** Same wall has different camera-frame normal components in different views; inner vs outer face can differ ~180° in same view.
- **Common intuition (varies with camera pose, values not fixed):**
  - Top-down floor top surface: normal roughly **away from camera** (OpenCV `nz` often negative, opposite world +Z).
  - Wall facing camera: normal mainly in `nx`/`ny` plane, small `|nz|`.
- **Furniture GLB:** Follows visible triangle orientation; consistent with `scene_visible_opencv.ply` normals if visible point cloud exported.

#### 7.4.2.3 Encoding and Decoding (Full Formulas)

Cycles Normal pass EXR is world-frame float vector (each component **[-1, 1]**); before PNG write transform to OpenCV camera frame, then linear map to uint8:

**Encoding (export,** `util_bpy._encode_opencv_normal_png_from_cycles_exr`**):**

```
n_cam = normalize(R @ n_world)     # R same as depth / *_opencv.ply
c = (n_cam + 1) / 2              # each component ∈ [-1, 1] → c ∈ [0, 1]
pixel_channel = round(c * 255)
```

**Decoding (read,** `util.decode_normal_opencv_uint8` **or per JSON formula):**

```
c = pixel_rgb / 255.0            # c ∈ [0, 1]
normal_opencv = c * 2.0 - 1.0  # back to [-1, 1]
# optional: re-normalize valid pixels to fix quantization error
```

**Examples (camera frame, view-dependent; illustrative):**


| OpenCV normal `normal_opencv` | Encoded RGB (approx) | Meaning (relative to current camera) |
| ------------------------- | ----------------- | ------------ |
| `(0, 0, +1)`              | `(128, 128, 255)` | Along +Z, into scene |
| `(0, 0, -1)`              | `(128, 128, 0)`   | Along −Z, toward camera |
| `(+1, 0, 0)`              | `(255, 128, 128)` | Image right |
| `(0, +1, 0)`              | `(128, 255, 128)` | Image down |


**Invalid / background:** No geometry or `depth==0` pixels may have arbitrary Normal pass values. **Must** filter with depth mask; do not infer normals on invalid pixels.

#### 7.4.2.4 Convert to SSL World Frame (Optional)

To align with `ssl.txt`, primary PLY, `planar_faces.json`, read `camera_position`, `look_at_target`, `world_up` from `{basename}_camera_para.json`, get `R` via `core/geometry_opencv.py` `opencv_rotation_from_pose`, then:

```
normal_world = (R.T @ normal_opencv.T).T   # then normalize
```

#### 7.4.2.5 Normal and Calibration Fields in `camera_para.json`

When exporting `--depth`, besides `depth_unit` / `depth_scale` / `**c2w` / `intrinsic**` (§6.3) also writes normal fields:

```json
{
  "depth_unit": "meter",
  "depth_scale": 5811.535562,
  "is_metric_depth": true,
  "image_size": [1024, 1024],
  "camera_convention": "opencv",
  "world_convention": "ssl_z_up",
  "c2w": [[1.0, 0.0, 0.0, 1.234], "..."],
  "intrinsic": [[577.59, 0.0, 512.0, 0.0], "..."],
  "intrinsic_model": "pinhole_no_distortion",
  "normal_space": "opencv_camera",
  "normal_axes": {
    "x": "+X image right",
    "y": "+Y image down",
    "z": "+Z along view into scene"
  },
  "normal_encoding": "uint8_rgb",
  "normal_encode": "uint8 = round(((normal_opencv + 1) / 2) * 255)",
  "normal_decode": "normal_opencv = (pixel_rgb / 255.0) * 2.0 - 1.0",
  "normal_invalid_mask": "depth_pixel == 0"
}
```

`normal_decode` and `normal_encode` are inverses; programs should read JSON first, avoid hardcoding inconsistent with docs.

#### 7.4.3 Python Decode Example (Depth + Normal)

Depth decoding see §7.4.1.3; here demonstrates same-frame paired read:

```python
import json
import imageio
import numpy as np
from scenebuilder.core import util

with open("topdown_camera_para.json") as f:
    cam = json.load(f)

depth_u16 = imageio.imread("topdown_depth.png")
depth_m = util.decode_depth_uint16(depth_u16, cam["depth_scale"])  # OpenCV +Z, meters

normal_u8 = imageio.imread("topdown_normal.png")
normal_cam = util.decode_normal_opencv_uint8(normal_u8)  # (H, W, 3), OpenCV camera frame

valid = depth_u16 > 0
nx = normal_cam[..., 0]  # +X image right
ny = normal_cam[..., 1]  # +Y image down
nz = normal_cam[..., 2]  # +Z into scene (same as depth)
normal_cam_valid = normal_cam.copy()
normal_cam_valid[~valid] = np.nan
```



#### 7.4.4 Other Notes

- Transparent occluders (auto-transparent walls/ceiling/floor) skipped in depth ray fallback path; Cycles pass path consistent with color (transparent areas show behind)
- **bpy + depth** still uses per-view independent subprocess (`render_ssl.worker_render_view`) to avoid compositor memory leaks

---


### 7.5 Voxels (`export_voxel` / `--voxel`)

Voxels have two paths, sharing 256³ colored occupancy format (see below), but different data sources:

| Path | Trigger | Output Directory | Data Source |
| ---- | -------- | -------- | ------ |
| **Per-view visible** | `--visible_geometry --voxel` | `{view}/voxel/` | Visible + frustum-clipped triangles; includes `occupancy_opencv.npz` |
| **Root full-scene** | `--holo_geometry --voxel` | `Y/voxel/` | Complete scene triangles; only `occupancy_world.npz` |

Per-view voxels are **not** read back from GLB; they share the same in-memory visible triangles as `scene_visible.glb`. Root voxels are sibling to `scene.glb`, exported by `worker_render_post`.

#### Per-View Visible Voxels (Trigger)

```text
--visible_geometry and --voxel
```

Voxels alone OK: `--visible_geometry --voxel` (no `--glb` / `--ply` required).

#### Root Full-Scene Voxels (Trigger)

```text
--holo_geometry and --voxel
```

#### Output Files


| File | Description |
| ---- | ---- |
| `occupancy_world.npz` | World SSL: `occupancy` uint8 `[256,256,256]` + `rgb` uint8 `[256,256,256,3]` |
| `occupancy_world_meta.json` | Grid geometry and indexing conventions (below) |
| `occupancy_world_preview.ply` | Occupied cell center colored point cloud for MeshLab / CloudCompare / Blender preview |
| `occupancy_opencv.npz` | OpenCV camera frame occupancy (same structure) |
| `occupancy_opencv_meta.json` | Camera frame meta (includes transformed `world_up`) |
| `occupancy_opencv_preview.ply` | Camera frame preview point cloud |
| `metadata_voxel.json` | Summary index |


#### Grid Definition

- **Resolution**: fixed `256³`
- **Cube span**: `span = max(x_extent, y_extent, z_extent)` (longest AABB axis in that frame)
- **Cell edge length**: `voxel_size = span / 256`
- **Center**: AABB geometric center; meta also gives `corner_min`
- **Index → 3D point** (voxel center):

```text
p(i,j,k) = corner_min + (i + 0.5, j + 0.5, k + 0.5) * voxel_size    # i,j,k ∈ [0,255]
```

- **Per cell**: `occupancy` 0=empty / 1=occupied; occupied `rgb` from triangle material/texture sample; empty RGB can be ignored

#### Main meta Fields

`format`, `version`, `coordinate_space`, `center`, `voxel_size`, `span`, `corner_min`, `world_up`, `world_up_space`, `index_origin`, `index_to_point`, `occupied_count`, `voxelization`

#### Read Example

```python
import json
import numpy as np

data = np.load("voxel/occupancy_world.npz")
occ = data["occupancy"]   # (256, 256, 256)
rgb = data["rgb"]         # (256, 256, 256, 3)

with open("voxel/occupancy_world_meta.json") as f:
    meta = json.load(f)

corner = np.array(meta["corner_min"])
vs = meta["voxel_size"]
i, j, k = 100, 200, 50
if occ[i, j, k]:
    center = corner + (np.array([i, j, k]) + 0.5) * vs
    color = rgb[i, j, k]
```

#### Visualization

| Method | Description |
| ---- | ---- |
| `*_preview.ply` | Auto-generated at export, open directly |
| Python | `np.load` + Open3D / matplotlib 3D scatter |
| Voxel cubes | Instantiate unit cube in Blender per meta |

**Size note**: Single npz uncompressed ~ **50–65 MB** (256³ dense); world + camerasets roughly doubles size. Default **no gzip** (`compressed: false` in meta) to avoid per-view wait; still exports `*_preview.ply`.

**Timing note** (`--voxel` per-view extra): voxelization ~5–30s + write ~3–10s; log prints `voxelization Xs, write Ys`.

---



## 8. Configuration and Backends



### 8.1 Configuration File (`config.yaml`)

Central management: canvas size, wall thickness, default FOV, HDRI path, furniture model library path, lighting intensity, nav_mask parameters (§5.2), etc. Readable via `BpySceneCtx` / `SceneCtx` `config` property.

### 8.2 Version Comparison


| Feature | scenebuilder (Pyrender) | scenebuilder_bpy (Blender) |
| -------- | --------------------- | ------------------------------------ |
| Render quality | Basic OpenGL | PBR (Eevee/Cycles) |
| Corner handling | Simple overlap | Miter joint correction |
| Occlusion handling | Simple clipping | Smart semi-transparent materials |
| Visible geometry | Supported (manual FOV plane) | Supported (`calc_matrix_camera` aligned with render) |
| Visible voxels | - | Supported (256³ colored occupancy, `--voxel`) |
| Planar inner-surface vertices | - | Supported (`--ply` auto-exports JSON + line overlay) |
| Semantic map | Supported | Supported (proxy + Raw) |
| Depth map | Pyrender depth buffer | Cycles Z pass + ray_cast fallback |
| Normal map | - | Cycles Normal pass (same render as `--depth`) |
| Depth + multi-view | Single process | Per-view independent subprocess (prevent memory leaks) |


---



## 9. Asset Processing (`get_mesh`)

`render_ssl` first parses SSL to JSON, then calls `get_mesh` for asset processing (retrieve or generate), finally passes to render engine.

- **Default**: `asset_mode="none"`, does not modify JSON, uses `asset_id` placeholders in SSL directly
- **Retrieve**: requires LanceDB build first (see below)



### 9.1 Retrieve Branch



#### Database Build

```bash
python build_lancedb.py
```

- Storage location: `manycore/` under current working directory
- `furniture` table first build ~16 hours (full Embedding)
- Requires `Qwen3VLEmbedder` path configuration (`util_data.py`)



#### Retrieve Logic

**Doors/Windows:** Match nearest model by `width`×`height` L2 distance.

**Furniture (Bbox):**

1. **With image** (`retrieve` + `image_path`): Group by `mesh_id`, bbox_2d crop + label/caption multimodal Embedding
2. **Without image**: label + caption text Embedding, batch parallel



### 9.2 Generate Branch (`asset_mode="generate"`)

- Requires `image_path`
- Group by `mesh_id` → crop → outpaint/super-resolution → 3D generation (Hunyuan, etc.)
- Models saved to `asset_dir` (default `/data-nas/data/dataset/qunhe/Manycore-Future/generate/`)

---



## Appendix A. More Usage Examples

One-liner commands equivalent to **§3.3**, **§5.2**:

```bash
# Full benchmark (same as §3.3)
python scenebuilder/render_ssl.py --ssl path/to/ssl.txt --views auto --output out \
  --glb --assets path/to/assets --ply --visible_geometry --voxel --semantic --depth --pano

# Pixel alignment + floor path only (§5.2)
python scenebuilder/render_ssl.py --ssl path/to/ssl.txt \
  --normalized_topdown --output out_normalized --assets path/to/assets

# Static multi-view
python scenebuilder/render_ssl.py --ssl path/to/ssl.txt \
  --views topdown left_seq --output out --glb --semantic --depth
```

---



## Appendix B. Frustum Clipping and Blender Native API (Implementation Notes)

On the bpy path, **visible geometry** (`visible_geometry`) and **planar inner-surface vertices** (`planar_faces.json` / `*_lines.png`) both depend on "what the current render camera sees and where clipped vertices land". The common root cause of repeated issues: **not using the projection matrix consistent with actual Cycles rendering**.

### Correct Approach (Current Implementation)

Aligned with Blender rendering, use:

1. `**camera.calc_matrix_camera(depsgraph, x, y, ...)**` to get projection matrix
2. Camera `matrix_world.inverted()` as modelview
3. Sutherland-Hodgman clip triangles/polygon loops in **homogeneous clip space**
4. When interpolating along edges, interpolate **both** `clip` **and** `world` so clipped intersection stays on original plane (wall/door/window/floor/ceiling)

Visible geometry triangle clipping (`scenebuilder_bpy._clip_triangle_to_render_frustum`) and planar vertex polygon clipping (`util_bpy.clip_polygon_to_render_frustum`) both follow this flow.

**Occlusion flag** `occluded` is unrelated to frustum; uses `scene.ray_cast` (same as visible geometry object-level occlusion), transparent walls/ceiling/floor can pass through.

### Wrong Approaches (Deprecated, Do Not Use)


| Approach | Problem |
| ------------------------------------------------------- | -------------------------------------- |
| Hand-written frustum planes / manual FOV cut planes | Misaligned with Blender render frustum; top-down/edge objects easily mislabeled cutted |
| `world_to_camera_view` for `(u,v,z)` then clip in screen space, 3D linear interpolation for intersections | Under perspective intersections **leave original plane**; partial in-frustum wall/ceiling vertex coords wrong, line overlay self-intersects |
| Re-sort polygon vertices by centroid angle | Breaks L-shaped/concave boundary topology (wrong floor order) |




### Other Independent Issues (Non-Frustum API)

- **Loop start rule**: canonical start is min z → min y → min x; rotate start only, do not reorder boundary
- **cutted statistics**: only when triangle **all three vertices in** viewport `[0,1]×[0,1]` count as "in frustum"; face ratio < 95% → `_cutted`



### Lessons Learned

- **Check if point in frustum (statistics/naming)**: can use `world_to_camera_view`, thresholds consistent with render viewport `[0,1]×[0,1]`, `z>0`
- **Clip geometry and find 3D intersections**: must use `**calc_matrix_camera` + clip space**, cannot back-project 3D from screen `(u,v)`
- **Export polygon vertex order**: preserve SSL/room `vertices` or mesh-defined boundary order, only overall CCW reversal + start rotation

Top-down rarely exposes old wrong clipping since objects fully in frustum; **perspective views (e.g. left) partial clipping** makes non-native API clipping issues very obvious.
