# SceneBuilder — 完整文档

[English](doc.en.md) | **中文**

> 完整 API、坐标系、输出目录与导出产物说明。首页概览见 [README（中文）](../README.zh-CN.md) / [README (English)](../README.md)。

从 SSL / JSON 描述（墙体、门窗、家具）构建 3D 场景，支持 **Blender (bpy)** 与 **Pyrender** 双后端渲染、多视角导出、语义图、深度图、路径规划、全景图、点云、可见体素、mesh 导出。

## 目录


| 章      | 内容                                                                |
| ------ | ----------------------------------------------------------------- |
| **§1** | 安装与依赖                                                             |
| **§2** | SSL 坐标系与实体约定                                                      |
| **§3** | 快速开始（最简示例 + `normalized_topdown`）                                 |
| **§4** | API 参考（含 `normalized_topdown_view` / `render_normalized_topdown`） |
| **§5** | 高级工作流（自动视角、像素对齐、地板路径）                                             |
| **§6** | 输出目录与坐标系（c2w、ssl_opencv）                                          |
| **§7** | 导出产物详解（可见几何、PLY、体素、语义、深度）                                            |
| **§8** | 配置与后端对比                                                           |
| **§9** | 资产处理（`get_mesh`）                                                  |
| **附录** | 更多示例、实现备忘                                                         |


---



## 1. 安装与依赖



### Python 版本要求: 3.11

不是 3.11 无法安装最新版本的 `bpy`。

### 基础依赖

```bash
pip install numpy shapely scipy imageio pyyaml
```



### 渲染后端


| 后端           | 安装                             | 适用场景                      |
| ------------ | ------------------------------ | ------------------------- |
| **bpy (推荐)** | `pip install bpy`              | PBR 渲染、语义图、Cycles 深度、可见几何 |
| **pyrender** | `pip install pyrender trimesh` | 轻量 OpenGL 渲染、无 bpy 环境     |


**Pyrender 无头服务器额外依赖：**

```bash
conda install -c conda-forge libstdcxx-ng=12 -y
apt install -y libegl1-mesa-dev libgles2-mesa-dev mesa-utils xvfb \
  libgl1-mesa-glx libglu1-mesa libxrender1 libxext6
export PYOPENGL_PLATFORM=egl
```

**Blender 版系统库（按需）：**

```bash
apt install -y libxi6 libxrender1 libxrandr2 libxfixes3 libxcursor1 libxinerama1 \
  libxxf86vm1 libgl1-mesa-glx libglu1-mesa libxkbcommon0 libgtk-3-0
```



### 安装到 Python 环境

```bash
cd scenebuilder
pip install -e . --config-settings editable_mode=strict
```

---



## 2. SSL 坐标系与实体约定

库内所有 SSL 解析、场景 `context`、渲染与导出（含 `ssl.txt`、点云命名）共用同一套**世界坐标系**，单位为**米 (m)**。

### 2.1 坐标轴

俯视房间（topdown）时：


| 轴     | 正方向 | 俯视图中的方向  |
| ----- | --- | -------- |
| **X** | +X  | 右        |
| **Y** | +Y  | **上**    |
| **Z** | +Z  | 竖直向上（高度） |


- 地面在 **XY 平面**，一般取 **Z = 0** 为地面高度。
- 家具朝向 `**angle_z**`：`0°` 时物体面向 **−Y**（俯视图**下方**）；绕 **Z 轴逆时针**旋转角度增大（右手定则）。

```
俯视图（+Y 在上，+X 在右）：

        +Y
         ↑
         |
    −X ←─┼─→ +X
         |
         ↓
        −Y     angle_z=0° 的朝向 →
```



### 2.2 实体字段

**Room**

```ssl
Room(room_type="balcony")
```

**Wall** — 墙段为地面上的线段，竖直向上拉伸：

```ssl
Wall(label="wall0", p=[x1, y1, 0], q=[x2, y2, 0], height=2.7)
```


| 字段       | 含义                          |
| -------- | --------------------------- |
| `p`, `q` | 墙段起点、终点；第三维写 `0` 即可，仅 XY 有效 |
| `height` | 墙高，从 `z=0` 到 `z=height`     |


**Door / Window** — 门洞 / 窗洞，挂在某面墙上：

```ssl
Door(label="door0", center=[x, y, z], width=2.32, height=2.3, wall="wall2", asset_id="61064792")
Window(label="window0", center=[x, y, z], width=2.37, height=2.2, wall="wall0")
```


| 字段         | 含义                                                             |
| ---------- | -------------------------------------------------------------- |
| `center`   | 洞口的**几何中心**（3D）。例如 `height=2.3, z=1.15` 表示中心在高度中间，底边约在 `z ≈ 0` |
| `width`    | 沿**墙走向**的宽度                                                    |
| `height`   | 沿 **Z 轴**的高度                                                   |
| `wall`     | 所属墙的 `label`（如 `wall0`）                                        |
| `asset_id` | 可选；缺失时仍挖洞，但不加载门/窗模型                                            |


加载时若 `center` 的 XY 不在墙线上，会吸附到最近墙的垂足。

**Bbox** — 家具或有向包围盒：

```ssl
Bbox(label="sidetable0", center=[1.96, 0.21, 0.31], angle_z=180, scale=[0.42, 0.42, 0.63], asset_id="56056912")
```


| 字段         | 含义                                                     |
| ---------- | ------------------------------------------------------ |
| `center`   | 旋转后 OBB 的**几何中心**                                      |
| `angle_z`  | 绕 Z 轴旋转角（度），逆时针为正                                      |
| `scale`    | 局部 XYZ 三轴上的**全长**（非半长）。落地物体通常 `center.z ≈ scale.z / 2` |
| `asset_id` | 对应 `{asset_id}.glb`；不存在时该 bbox 会被剔除                    |


SSL bbox 的**局部坐标系**约定为：局部 +X / +Y / +Z 分别对应 `scale[0]` / `scale[1]` / `scale[2]` 三条 OBB 轴；语义“前向/朝向”为局部 **−Y**。因此 `angle_z=0°` 时局部轴与 SSL 世界轴对齐，而物体前向为世界 **−Y**；`angle_z` 增大时，局部坐标系和前向一起绕 SSL +Z 逆时针旋转。

GLB 加载流程：模型居中 → 按 `scale` 缩放 → 绕 Z 转 `angle_z` → 平移到 `center`。

### 2.3 与渲染视角的关系

`render_ssl` 默认透视相机（如 `front`）放在 **−Y 一侧**（`y = center_y − span_y/3`），朝向房间中心，即朝 **+Y** 看进房间。这与「+Y 在俯视图上方、`angle_z=0°` 朝 −Y」一致。

`topdown` 相机从 **+Z** 向下俯视 XY 平面。

### 2.4 标准 SSL 输出

经 `normalize_scene_data()` 整理后，输出的 `ssl.txt` 使用规范 `label`（`wall0`, `door0`, `sidetable0` …），去掉原始 `id` / `room_id`；字段语义与上述约定相同。

---



## 3. 快速开始

以下示例均基于 **SSL 世界坐标**（见 §2）。底层渲染类为 `BpySceneCtx`（`core/scenebuilder_bpy.py`）；`SceneCtx`（pyrender 后端）签名类似，但部分高级导出仅 bpy 支持。

### 3.1 最简单示例：`topdown_view`

只加载场景、渲染一张俯视图，不开启深度/语义/GLB 等高级功能。

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

# output 为目录 → 实际 PNG 写入 {output}/topdown/topdown.png
ctx.topdown_view("output", rebuild=True)
```



### 3.2 最简单示例：`render_view`

在同一 `ctx` 上指定相机位置与注视点，渲染单张透视图。

```python
meta = ctx.context["meta"]
center = meta["center"]
z_cam = meta["z_max"] * 5 / 6

# output 为目录 → 实际 PNG 写入 {output}/{时间戳}/{时间戳}.png
ctx.render_view(
    "output",
    camera_position=[center[0], center[1] - meta["span"][1] / 3, z_cam],
    look_at_target=[center[0], center[1], z_cam / 2],
    rebuild=True,
)
```

坐标均为 **SSL 世界系** `[x, y, z]`（米）：相机在房间中心偏 −Y 一侧、高度约 `5/6 * z_max`，看向房间中部。

### 3.3 高级示例：批量多视角 + 全量导出（`render_ssl.py`）

Benchmark / 数据生产推荐走 CLI 或 `render_ssl()`，一次完成多视角、GLB、点云、语义、深度、全景等。

```bash
python /data-nas/data/experiments/mushui/projects/utils/fast-scene/scenebuilder/render_ssl.py \
  --ssl /data-nas/data/experiments/mushui/projects/SpatialFactory/benchmark/data/Office/325148303_0/render_output/ssl.txt \
  --views auto \
  --output /data-nas/data/experiments/mushui/projects/SpatialFactory/benchmark/data/Office/325148303_0/out2 \
  --glb \
  --assets /data-nas/data/dataset/qunhe/Manycore-Future/simplified \
  --ply --visible_geometry --voxel --semantic --depth --pano
```

等价 Python 调用：

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

`--views auto` 会**强制** `normalized_topdown=True`（见 §3.4）；静态多视角可用 `--views topdown front left_seq` 等，是否与规范化叠加见 §4.4。

### 3.4 像素对齐俯视图（`normalized_topdown`）

与 §3.1 的常规 `topdown_view`（1024²、SSL 世界坐标不变）不同：**平移整场景 SSL**，使俯视图图像左上角对应地面 `(0,0)`，输出目录为 `{output}_normalized/`，主产物在 `topdown_normalized/`。常用于地板路径规划、SpatialFactory 像素坐标对齐。

**命令行（仅规范化 + 路径，不渲染其它视角）：**

```bash
python scenebuilder/render_ssl.py --ssl path/to/ssl.txt \
  --normalized_topdown \
  --output path/to/out \
  --assets path/to/assets
# 不要地板路径采样：加 --no_floor_path
```

**Python（推荐封装** `render_normalized_topdown`**）：**

```python
from scenebuilder.render_ssl import render_normalized_topdown

with open(SSL, encoding="utf-8") as f:
    ssl_text = f.read()

result = render_normalized_topdown(
    ssl_text,
    output_dir="out",              # 实际输出根目录 → out_normalized/
    backend="bpy",
    asset_dir=ASSETS,
    floor_path=True,               # 默认 True：depth+semantic 后采样闭环路径
)
# result["path_points_ssl"] — 路径点（像素对齐 SSL 地面坐标）
# result["topdown_normalized_dir"] — .../out_normalized/topdown_normalized/
```

**底层 Context 直接调用**（已有 `ctx` 且场景已 `normalize_scene_data`）：

```python
ctx.normalized_topdown_view(
    "out/topdown_normalized",
    render_depth=True,             # floor_path 需要
    render_semantic=True,
)
```

与 §3.1 `topdown_view` 的区别见 §4.4 对照表；nav_mask 路径 pipeline 见 **§5.2**。

参数说明见 **§4**；输出目录见 **§6**；各导出开关见 **§7**。

---



## 4. API 参考


| 需求                | 入口                                                           |
| ----------------- | ------------------------------------------------------------ |
| 分辨率 / FOV 语义（Blender） | **§4.0**（长轴 FOV、`sensor_fit=AUTO`、焦距推导）                    |
| 单张俯视图（常规 SSL）     | `BpySceneCtx.topdown_view`（§4.1）                             |
| 像素对齐俯视图 + 可选地板路径  | `render_normalized_topdown()` / `--normalized_topdown`（§4.4） |
| 自定义相机透视图          | `BpySceneCtx.render_view`（§4.2）                              |
| 批量多视角 / benchmark | `render_ssl()` 或 `render_ssl.py`（§4.3）                       |
| auto 路径驱动视角       | `render_ssl(..., views="auto")`（§5.1，内含 normalized_topdown）  |
| video 闭环轨迹          | `render_ssl(..., video=True)` 或 `--video`（§5.1.1）              |




### 4.0 分辨率与 FOV（Blender 后端）

适用于 `topdown_view`、`render_view`、`render_ssl` 中通过 **bpy** 渲染的透视视角（不含 equirectangular 全景 `pano`）。

#### 可指定的成像参数

| 参数 | 作用 |
| ---- | ---- |
| `width` / `height` | 输出像素分辨率，分别写入 `scene.render.resolution_x/y`，**两轴独立** |
| `auto_fov` | 按场景包围盒自动计算视野角度（度 → 弧度后写入 Blender） |
| `manual_fov` | 手动指定一个 FOV 角度（**度**）；非 `None` 时覆盖 `auto_fov` |

相机创建时（`BpySceneCtx._create_render_camera`）：

```python
camera_data.lens_unit = 'FOV'
camera_data.angle = fov_rad   # 来自 manual_fov 或 auto_fov
# 默认 sensor_fit = 'AUTO'（未显式修改）
scene.render.resolution_x = width
scene.render.resolution_y = height
```

#### `manual_fov` 是长轴 FOV，不是固定「垂直 FOV」

Blender 在 `sensor_fit=AUTO` 下会把 `camera_data.angle` 作用在图像的**长轴**上；短轴 FOV 由宽高比自动推出。两轴共用同一焦距 \(f_x = f_y\)：

| 宽高比 | 长轴 | `manual_fov` 对应 | 短轴 FOV |
| ------ | ---- | ----------------- | -------- |
| 宽 ≥ 高（如 1280×720） | 水平 | **水平 FOV** | 垂直 FOV 更小，由 aspect 推出 |
| 高 > 宽（如 720×1280） | 垂直 | **垂直 FOV** | 水平 FOV 更小，由 aspect 推出 |
| 1:1（如 1000×1000） | — | 水平 FOV = 垂直 FOV | — |

实现与 `util_bpy.camera_frustum_tangents`、`build_opencv_intrinsic_4x4` 一致（§6.3.3）：

- 宽 ≥ 高：`tan_x = tan(angle/2)`，`tan_y = tan_x / aspect`
- 高 > 宽：`tan_y = tan(angle/2)`，`tan_x = tan_y × aspect`
- 焦距：`fx = width / (2·tan_x)`，`fy = height / (2·tan_y)`，且 **`fx = fy`**

**理解方式**：你指定 X/Y 分辨率 + 一个 FOV 角度 → 先确定长轴 FOV → 算出焦距 → 同一焦距作用到短轴 → 自动得到短轴 FOV。并非「同一个 FOV 同时赋给 X 和 Y」。

代码内部变量名常写作 `fov_y`，但在非正方形横图下它实际对应 Blender 的 `angle`（长轴 FOV），不要按字面理解为「永远是 Y 方向 FOV」。

#### PyRender 后端差异

PyRender 使用 `PerspectiveCamera(yfov=..., aspectRatio=width/height)`，此时 `manual_fov` **始终作为垂直 FOV**。非 1:1 宽高比时，PyRender 与 Blender 的 FOV 语义可能略有差异；数据生产建议优先使用 `backend="bpy"`。

#### 与像素对齐俯视的关系

`topdown_normalized/` **固定 1000×1000**，不可通过 API 修改（SpatialFactory Stage 1 约定；`prepare_pixel_aligned_topdown_context` 中 `image_half=500` 与之绑定）。正方形下长轴 FOV = 水平 = 垂直 FOV；§5.2 中 `pixel2real_ratio = camera_z × tan(fov/2) / 500` 的 `fov` 即该 FOV。

#### 各接口默认分辨率（并非全是 1000×1000）

| 接口 | 默认分辨率 | 可否改分辨率 |
| ---- | ---------- | ------------ |
| `topdown_view` / `render_view` | **1024×1024** | 可以（`width` / `height`） |
| `render_ssl` 各视角（含 `topdown/`、auto） | **1000×1000** | 可以（`width` / `height` 或 CLI） |
| `topdown_normalized/`（像素对齐） | **1000×1000** | **不可**（固定） |




### 4.1 `topdown_view`

```python
ctx.topdown_view(
    output_path: str,              # 目录 → {dir}/topdown/topdown.png；或直接 .png 路径
    width: int = 1024,
    height: int = 1024,
    geometry_mode: str = "gltf",   # 几何加载模式
    show_wall: bool = True,
    show_window: bool = True,
    show_door: bool = True,
    show_ceiling: bool = True,     # 俯视图通常设 False
    up_vector: list = None,        # 默认 [0,1,0]；相机「上」方向（SSL 世界系）
    auto_fov: bool = True,         # 按场景包围盒自动算 FOV（长轴，见 §4.0）
    manual_fov: float = None,      # 指定长轴 FOV（度）；非 None 时覆盖 auto_fov（见 §4.0）
    auto_transparent: bool = True, # 遮挡墙/顶/地自动透明
    transparent_alpha: float = 0.0,
    render_depth: bool = False,    # 同目录输出 {basename}_depth.png + camera_para
    use_HDRI: bool = True,
    hdri_transparent_background: bool = True,
    visible_shadow: bool = True,
    lighting_type: Literal["area", "array", "none"] = "array",
    align_height: bool = True,     # 家具贴地高度对齐
    rebuild: bool = False,         # True 强制重建 mesh
    export_glb: bool = False,
    glb_path: Optional[str] = None,
    export_point_cloud: bool = False,
    export_voxel: bool = False,
    visible_geometry: bool = False,# 需配合 export_glb / export_point_cloud / export_voxel
    render_semantic: bool = False, # {basename}_semantic.png/json + bbox_2d
)
```


| 参数                                  | 说明                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------- |
| `output_path`                       | 传目录时在 `{dir}/topdown/topdown.png` 写主图；同时写 `topdown_camera_para.json`、`ssl_opencv.txt` |
| `show_ceiling`                      | `False` 时不渲染天花板（常用俯视图配置）                                                              |
| `up_vector`                         | 俯视默认 `[0,1,0]`；与 `render_view` 默认 `[0,0,1]` 不同                                        |
| `auto_fov` / `manual_fov`           | 相机在场景正上方，FOV 决定可见范围；`manual_fov` 为**长轴 FOV**（§4.0）                         |
| `render_depth`                      | bpy：Cycles 同 pass 输出深度；`camera_para.json` 含 `depth_scale`                             |
| `render_semantic`                   | 语义图 + JSON + bbox_2d 叠加图（见 §7.3）                                                      |
| `export_glb` / `export_point_cloud` / `export_voxel` | 几何类型开关；`visible_geometry=True` 时导出**当前视角**视锥内部分；根目录全场景须 `render_ssl(..., holo_geometry=True)` |
| `rebuild`                           | 首次调用或场景变更后建议 `True`                                                                   |


相机位置由场景 `meta` 自动计算（中心正上方），**无需**手动传 `camera_position`。

**与** `render_view` **共有参数**（默认值以函数签名为准）：


| 参数                                  | 默认          | 说明                                   |
| ----------------------------------- | ----------- | ------------------------------------ |
| `width` / `height`                  | `1024`      | 分辨率                                  |
| `geometry_mode`                     | `"gltf"`    | `"gltf"` / `"mixed"` / `"bbox"`      |
| `show_wall/window/door/ceiling`     | `True`      | 各元素可见性；俯视常用 `show_ceiling=False`     |
| `auto_fov` / `manual_fov`           | 自动 / `None` | 长轴 FOV（度）；1:1 时等于水平 = 垂直（§4.0）   |
| `auto_transparent`                  | `True`      | 遮挡墙/顶/地自动透明                          |
| `use_HDRI`                          | `True`      | 启用 config 中 HDRI                     |
| `hdri_transparent_background`       | `True`      | 俯视默认背景透明                             |
| `lighting_type`                     | `"array"`   | `"array"` / `"area"` / `"none"`      |
| `align_height`                      | `True`      | 墙体高度对齐                               |
| `rebuild`                           | `False`     | 场景变更后建议 `True`                       |
| `render_depth`                      | `False`     | `{basename}_depth.png` + normal（bpy） |
| `render_semantic`                   | `False`     | 语义图（§7.3）                            |
| `export_glb` / `export_point_cloud` / `export_voxel` | `False`     | 几何类型开关；各视角须 `visible_geometry=True`，根目录须 `holo_geometry=True` |
| `visible_geometry`                  | `False`     | 各视角可见 GLB/PLY/体素（§7.1 / §7.5）                     |
| `holo_geometry`                     | `False`     | 根目录全场景 GLB/点云/体素（§7.1.1）                         |




### 4.2 `render_view`

```python
ctx.render_view(
    output_path: str,
    camera_position: list,         # 必填，SSL 世界系 [x,y,z]
    look_at_target: list = None,    # 默认场景 center + z_max/2
    width: int = 1024,
    height: int = 1024,
    up_vector: list = None,         # 默认 None → [0,0,1]（SSL +Z）
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
    export_glb / glb_path / export_point_cloud / export_voxel / visible_geometry: 同 topdown_view,
    render_semantic: bool = False,
    pano: bool = False,             # 额外输出兄弟目录 {basename}_pano/
    pano_resolution: int = 4096,     # 全景宽；高 = 宽/2
    view_dir_name: Optional[str] = None,  # 指定子目录名（如 "front"）
)
```


| 参数                | 说明                                                                       |
| ----------------- | ------------------------------------------------------------------------ |
| `camera_position` | 相机光心，SSL 世界坐标（米）                                                         |
| `look_at_target`  | 注视点；省略时用 `[center_x, center_y, z_max/2]`                                 |
| `up_vector`       | 相机 up 方向，默认 SSL +Z `[0,0,1]`                                             |
| `output_path`     | 传目录时 → `{dir}/{view_dir_name 或时间戳}/{时间戳}.png`                            |
| `view_dir_name`   | 固定输出子目录名；`render_ssl.py` 对预设视角传 `"front"` 等                              |
| `pano`            | 在 `{view_dir}_pano/` 额外渲染 equirectangular 全景（仅 `render_view`，topdown 无）  |
| **序列模式**          | `camera_position` / `look_at_target` / `up_vector` 传**列表的列表**时自动进入多帧序列渲染 |


输出：`{basename}.png`、`{basename}_camera_para.json`（含 `c2w`、`intrinsic`，§6.3）、同目录 `ssl_opencv.txt`（§6.4）；可选 depth / semantic / pano 附属文件（§7）。

**独有参数：**


| 参数                         | 说明                                            |
| -------------------------- | --------------------------------------------- |
| `camera_position`          | 相机光心，SSL 世界坐标（米），**必填**                       |
| `look_at_target`           | 注视点；默认 `[center_x, center_y, z_max/2]`        |
| `up_vector`                | 相机 up，默认 SSL +Z                               |
| `pano` / `pano_resolution` | 额外输出 `{view_dir}_pano/` 全景（topdown 无 pano）    |
| `view_dir_name`            | 固定子目录名（如 `"front"`）                           |
| **序列模式**                   | 三个位置参数传**列表的列表** → 多帧序列，目录 `{timestamp}_seq/` |


其余参数与 §4.1 共有表相同；`hdri_transparent_background` 默认 `False`。

### 4.3 `render_ssl` / `render_ssl.py`

**Python 签名：**

```python
def render_ssl(
    input_text: str,                    # SSL 文本或 JSON 字符串
    backend: str = "bpy",               # "bpy" | "pyrender"
    output_root: str = "output_ssl",
    image: Optional[str] = None,        # asset_mode=retrieve/generate 时的参考图
    retrieve_hole: bool = True,
    asset_mode: Literal["none", "retrieve", "generate"] = "none",
    outpaint_image_dir: Optional[str] = None,
    asset_dir: Optional[str] = None,
    hole_asset_dir: Optional[str] = None,
    gen_3d_model: Literal["hunyuan-3d-rapid", "hunyuan-3d-pro"] = "hunyuan-3d-pro",
    gen_texture: bool = False,
    texture_dir: Optional[str] = None,
    correct_tilt: bool = True,
    correct_yaw: bool = True,
    views: ViewsSpec = None,            # None=不渲染；list=指定视角；"auto"=路径驱动
    export_glb: bool = False,
    export_point_cloud: bool = False,
    export_voxel: bool = False,
    visible_geometry: bool = False,
    holo_geometry: bool = False,
    semantic: bool = False,
    depth: bool = False,
    pano: bool = False,
    pano_resolution: int = 4096,
    samples: Optional[int] = None,      # Blender Cycles 采样数
    normalized_topdown: bool = False,
    floor_path: bool = True,
    normalized_topdown_show_ceiling: bool = False,
    width: int = 1000,                # 各视角渲染宽（§4.0）；不含 topdown_normalized/
    height: int = 1000,               # 各视角渲染高（§4.0）
    auto_fov: bool = True,
    manual_fov: Optional[float] = None,
    resume: bool = True,               # 断点续跑；CLI 用 --no-resume 关闭
    video: bool = False,               # 沿地板路径渲染 video 轨迹（§5.1.1）
    video_frame_spacing: float = 0.2,  # video 帧弧长间距（米）
) -> Tuple[str, str, Optional[dict]]
```

**命令行 ↔ Python 对照：**


| CLI                    | Python                    | 说明                                   |
| ---------------------- | ------------------------- | ------------------------------------ |
| `--ssl PATH`           | `input_text`（读文件）         | SSL 或 JSON 场景                        |
| `--ssl_str TEXT`       | `input_text`（行内字符串）      | 与 `--ssl` / `--ssl_id` 二选一；例如 JSONL 一行              |
| `--ssl_id ID`          | —（CLI 读 JSONL）             | 如 `310449449_4`；须配合 `--ssl_collection_dir` |
| `--ssl_collection_dir DIR` | —                      | `{DIR}/{design_id}.jsonl` 第 `{room_index}` 行（0-based） |
| `--output DIR`         | `output_root`             | 默认 `{ssl_dir}/render_output`         |
| `--views V ...`        | `views`                   | 不指定 = 不渲染视角；`auto` 单独使用              |
| `--backend bpy         | pyrender`                 | `backend`                            |
| `--assets DIR`         | `asset_dir`               | 省略时尝试 ssl 旁 `assets/` 等              |
| `--hole_assets DIR`    | `hole_asset_dir`          | 门窗 fallback：`--assets` 找不到 asset_id 时使用，再回退 config `model_hole_path` |
| `--texture DIR`        | `texture_dir`             | 含 `floor/wall/ceiling_texture.png`   |
| `--glb`                | `export_glb=True`         | 几何开关（须配合 `--visible_geometry` 或 `--holo_geometry`） |
| `--ply`                | `export_point_cloud=True` | 几何开关 + 各视角 `planar_faces`（后者与 `--visible_geometry` 无关） |
| `--visible_geometry`   | `visible_geometry=True`   | **各视角**视锥内 GLB/PLY/体素（需 `--glb` / `--ply` / `--voxel` 至少其一） |
| `--holo_geometry`      | `holo_geometry=True`      | **根目录**全场景 GLB/点云/体素（需 `--glb` / `--ply` / `--voxel` 至少其一） |
| `--voxel`              | `export_voxel=True`       | 256³ 彩色占用场：各视角需 `--visible_geometry`，根目录需 `--holo_geometry` |
| `--semantic`           | `semantic=True`           | 各视角语义图 + JSON + bbox_2d              |
| `--depth`              | `depth=True`              | 各视角 depth + normal（bpy）              |
| `--pano`               | `pano=True`               | 各 `render_view` 视角额外全景（topdown 跳过）   |
| `--pano_resolution N`  | `pano_resolution`         | 默认 4096                              |
| `--width W`            | `width`                   | 各视角渲染宽，默认 1000（§4.0）                 |
| `--height H`           | `height`                  | 各视角渲染高，默认 1000（§4.0）                 |
| `--manual_fov DEG`     | `manual_fov`              | 长轴 FOV（度），覆盖 auto_fov（§4.0）           |
| `--no_auto_fov`        | `auto_fov=False`          | 关闭自动 FOV；透视视角建议配合 `--manual_fov`     |
| `--normalized_topdown` | `normalized_topdown=True` | 输出 Y=`{output}_normalized`           |
| `--no_floor_path`      | `floor_path=False`        | 跳过 `topdown_normalized/` 路径采样        |
| `--samples N`          | `samples`                 | Blender 采样数                          |
| `--no-resume`          | `resume=False`            | 禁用断点续跑，强制重渲染已完成视角                 |
| `--video`              | `video=True`              | 沿地板路径渲染切线 video（§5.1.1）；可单独使用或与 `--views auto` 叠加 |
| `--video_spacing M`    | `video_frame_spacing`     | video 全景帧弧长间距（米），默认 0.2；帧数 N≈路径周长/M |


**预设视角名**（`views` 列表元素）：`topdown`, `front`, `behind`, `left`, `right`, `leftfront`, `rightfront`, `leftbehind`, `rightbehind`, `left_seq`。相机位置由场景 `meta` 自动推算（与 §3.2 类似规则），`look_at` 为 `[center_x, center_y, z_max/2]`。


| 参数                                  | 说明                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------ |
| `views=None`                        | 只写 `data.json` / 可选全场景几何（需 `holo_geometry`），**不渲染任何视角**                                         |
| `views="auto"`                      | 强制 `normalized_topdown=True` + `floor_path=True`；渲染 `topdown` + 路径自动生成 sparse auto 视角（§5.1） |
| `video=True` / `--video`            | 强制 `normalized_topdown=True` + `floor_path=True`；渲染 video 轨迹（§5.1.1）。单独使用时跳过 sparse auto 与 `topdown/`；与 `views="auto"` 叠加时在 auto 基础上追加 video |
| `video_frame_spacing` / `--video_spacing` | video 全景帧弧长间距（米），默认 0.2；帧数由闭环路径长度自动计算 |
| `normalized_topdown`                | 唯一输出根 `Y={output_root}_normalized`；先 pixel-align SSL，再 `Y/topdown_normalized/` |
| `export_glb` / `export_point_cloud` / `export_voxel` | 几何类型开关；**不单独写盘**，须配合 `visible_geometry`（各视角）或 `holo_geometry`（根目录） |
| `visible_geometry`                  | 各视角视锥内 GLB/PLY/体素（§7.1 / §7.5）                     |
| `holo_geometry`                     | 根目录全场景 GLB/点云/体素（§7.1.1）                           |
| `semantic` / `depth`                | 映射到各视角 `topdown_view` / `render_view` 的 `render_semantic` / `render_depth`     |
| `asset_mode`                        | `"none"` 仅用 SSL 已有 `asset_id`；`retrieve`/`generate` 需配合 `image` 等              |
| `gen_texture`                       | 无外部贴图时内部生成 floor/wall/ceiling 纹理                                               |


**bpy + depth 行为：** 每个视角在独立子进程（`worker_render_view`）渲染，避免 Cycles 内存泄漏；根目录全场景几何由 `worker_render_post` 导出（仅 `--holo_geometry` 时）。

**返回值：** `(y_dir, standard_ssl_text, floor_result)`。`floor_result` 在 `normalized_topdown` 且 `floor_path=True` 时含 `path_points_ssl` 等。


| 导出开关                 | 说明                                |
| -------------------- | --------------------------------- |
| `export_glb` / `export_point_cloud` / `export_voxel` | 几何类型开关；须配合 `visible_geometry` 或 `holo_geometry` |
| `visible_geometry`   | 各视角视锥裁剪 GLB/PLY/体素（§7.1）             |
| `holo_geometry`      | 根目录全场景 GLB/点云/体素（§7.1.1，`worker_render_post`） |
| `export_point_cloud` | 另：各视角 `planar_faces`（§7.2，与 `visible_geometry` 无关） |
| `semantic` / `depth` | §7.3 / §7.4                       |




### 4.4 像素对齐俯视图（`normalized_topdown`）

三种等价层级（由低到高）：


| 层级      | 用法                                                                      |
| ------- | ----------------------------------------------------------------------- |
| Context | `ctx.normalized_topdown_view(output_dir, ...)`                          |
| 封装 API  | `render_normalized_topdown(input_text, output_dir, ...)`                |
| 通用渲染    | `render_ssl(..., normalized_topdown=True)` 或 CLI `--normalized_topdown` |


**与** `topdown_view` **的区别：**


|                    | `topdown_view`（§4.1）     | `normalized_topdown_view` / `--normalized_topdown` |
| ------------------ | ------------------------ | -------------------------------------------------- |
| SSL 坐标             | 不变                       | **平移** XY，图像左上 ↔ 地面 `(0,0)`                        |
| 默认分辨率              | 1024²                    | **固定 1000²**（不可改）                       |
| 输出目录               | `{Y}/{view}/topdown.png` | `{Y}/topdown_normalized/topdown.png`               |
| `camera_para.json` | SSL 世界系                  | 含 `pixel2real_ratio`、图像坐标系（SpatialFactory）         |
| 地板路径               | 无                        | `floor_path=True` 时自动 nav_mask 采样（§5.2）            |




#### `render_normalized_topdown`

```python
def render_normalized_topdown(
    input_text: str,
    output_dir: str,              # 实际 Y = {output_dir}_normalized
    backend: str = "bpy",
    asset_dir: Optional[str] = None,
    texture_dir: Optional[str] = None,
    show_ceiling: bool = False,
    floor_path: bool = True,      # False → 只渲染俯视图，不采样路径
    export_glb: bool = False,
    export_point_cloud: bool = False,
    ...
) -> Union[str, Dict[str, Any]]
```

等价于 `render_ssl(..., normalized_topdown=True, views=None)`。`floor_path=True` 且成功时返回 dict，含 `path_points_ssl`、`topdown_normalized_dir`、`floor_path` 等；否则返回 `y_dir` 字符串。

#### `render_ssl` 的 `normalized_topdown` 相关参数


| 参数 / CLI                                      | 说明                                                                                    |
| --------------------------------------------- | ------------------------------------------------------------------------------------- |
| `normalized_topdown` / `--normalized_topdown` | 输出根 `Y={output_root}_normalized`；先 pixel-align，后续 views/GLB/点云均用规范化 SSL               |
| `floor_path` / `--no_floor_path`              | 是否在 `topdown_normalized/` 跑 nav_mask 路径（默认 True）                                      |
| `topdown_normalized/` 分辨率                     | **固定 1000×1000**，不可配置（SpatialFactory Stage 1）                                        |
| `normalized_topdown_show_ceiling`             | 规范化 pass 是否渲染天花                                                                       |
| `views="auto"`                                | **隐式** `normalized_topdown=True` + `floor_path=True`，并额外渲染 `topdown/` + sparse auto 视角（§5.1） |
| `--video`                                     | **隐式** `normalized_topdown=True` + `floor_path=True`；仅 video 时只渲染 video 轨迹（§5.1.1）；与 `--views auto` 叠加时追加 video |


可与 `--views topdown front left_seq` **叠加**：先规范化，再在 `Y/` 下渲染常规多视角（坐标系已是规范化 SSL）。

#### `BpySceneCtx.normalized_topdown_view`

```python
ctx.normalized_topdown_view(
    output_dir: str,               # 目录内写 topdown.png / ssl.txt / camera_para.json
    show_ceiling: bool = False,
    render_depth: bool = False,    # floor_path 依赖
    render_semantic: bool = False,
    align: Optional[dict] = None,  # 已 pixel-align 时传入，跳过二次平移
    write_ssl: bool = True,
    round_decimals: int = 2,
    ...
)
```

分辨率**固定 1000×1000**，无 `width` / `height` 参数。

由 `prepare_pixel_aligned_topdown_context` 计算平移；`align` 非空时场景已在规范化 SSL 下（`render_ssl` 内部路径）。**不支持** `visible_geometry` / `export_glb`（会从 kwargs 丢弃）。

像素对齐公式、nav_mask pipeline、config 键见 **§5.2**。

---



## 5. 高级工作流



### 5.1 自动视角（`--views auto`）

等价于 `**--normalized_topdown` + 地板路径 + 常规 topdown + 按路径自动生成视角**（目前仅 `backend=bpy`）。

```bash
python scenebuilder/render_ssl.py --ssl scene.txt \
  --views auto --output out \
  --glb --ply --visible_geometry --voxel --semantic --depth --pano \
  --assets path/to/assets
# → Y = out_normalized/
```



#### 流程

1. **Step 1–2**：与 §5.2 相同——pixel-aligned SSL 规范化 + `topdown_normalized/` 地板路径规划
  路径输出 `path_points_ssl`：地面闭环路点 `0…n-1`（首尾相连）
2. **Step 3**：
  - 常规 `**Y/topdown/**` 俯视图（1024²，与 `--views topdown` 相同；**不是** Step 2 的像素对齐俯视图）
  - `core/auto_views.py` 根据路径点生成相机并子进程渲染，manifest 写入 `Y/auto_views.json`



#### 两种自动渲染

**类型 A — 单帧（多次** `render_view`**）**

- 路径点采样（`sample_path_indices`）：
  - **n < 40**：索引 `0, 4, 8, …`（stride=4）
  - **40 ≤ n ≤ 100**：在闭环上**均匀取 15 个**路点（如 n=100 约每隔 7 点）
  - **n > 100**：**均匀取 20 个**路点
- 相机位置：`(x, y, z)`，`x,y` 为路点；`z ∈ [0.5, 2.5]` 随机，且 `<` 墙高最大值
- `world_up = (0, 0, 1)`；初始 `look_at` = 场景 3D bbox 中心
- 俯仰：在射线 AB 上调整，随机 **−40° ~ 5°**
- FOV：**60° ~ 90°** 随机；分辨率 **1000×1000**
- 每个路点一个输出目录 `auto_path_{索引}/`（如 `auto_path_0000/`）

**类型 B — 三帧序列（一次** `render_view`**）**

- 从 `n` 个路点中选 **XY 离场景 bbox 中心最近** 的一点，记索引为 `k`
- 相机位置固定 `(x, y, 1.5)`；`look_at` 初始为 bbox 中心
- 三帧 `look_at`（同相机位置）：
  - 帧 1：向左偏航 **10° ~ 40°**
  - 帧 2：中心（不偏航）
  - 帧 3：向右偏航 **0° ~ 40°**
- FOV：**60° ~ 90°** 随机；分辨率 **1000×1000**
- 输出目录 `**auto_path_{k:04d}_seq/**`



#### 附属产物

auto 模式下 Step 3 的 `**topdown/**`、`**auto_path_***`、`**auto_path_*_seq**` 均走同一套 `worker_render_view`，CLI 开关**全部生效**：


| CLI                            | auto 单帧        | auto 序列 | auto 的 `topdown/` |
| ------------------------------ | -------------- | ------- | ----------------- |
| `--depth`                      | ✅              | ✅ 每帧    | ✅                 |
| `--semantic`                   | ✅              | ✅ 每帧    | ✅                 |
| `--pano`                       | ✅              | ✅ 每帧    | ❌                 |
| `--glb` + `--visible_geometry` | ✅              | ✅ 多帧并集 + `viewvis_*_tri.npz`（§7.1.2） | ✅                 |
| `--ply` + `--visible_geometry` | ✅              | ✅ 多帧并集 + `viewvis_*.npz`（§7.1.2） | ✅                 |
| `--voxel` + `--visible_geometry` | ✅            | ✅ 多帧并集  | ✅                 |
| `--ply`（无 visible_geometry）    | ✅ planar_faces | ✅ 每帧    | ✅                 |


根目录 `**scene.glb` / `pointcloud/scene_all.ply` / `voxel/**` 由 `worker_render_post` 导出（全场景，需 `--holo_geometry`）。

#### 输出示例

```
out_normalized/
├── topdown/                     # Step 3 常规俯视图（1024²）
├── auto_views.json
├── auto_path_0000/ …
└── auto_path_0012_seq/          # 三帧序列
```

> 手动序列（如 `left_seq`）目录名仍为 `{timestamp}_seq/`。



### 5.1.1 Video 轨迹（`--video`）

沿 `topdown_normalized/` 的地板闭环 `path_points_ssl` 生成**一条**切线方向平滑序列，实现见 `core/video_views.py`。与 §5.1 sparse auto **独立**。

#### 两种使用模式

| 模式 | CLI | Step 3 渲染内容 |
| ---- | --- | --------------- |
| **仅 video** | `--video`（不传 `--views`） | 只渲染 `video_{N}` / `video_{N}_pano/`；**不**渲染 sparse auto、**不**渲染常规 `topdown/` |
| **auto + video** | `--views auto --video` | 常规 `topdown/` + sparse auto + 追加 video |

两种模式均隐式开启 `normalized_topdown=True`、`floor_path=True`。

#### 帧数与轨迹

- 帧数 **N = round(闭环路径弧长 / spacing)**，默认 `spacing=0.2` m（`--video_spacing`），最小 3 帧
- 目录名 `video_{N}`；全景帧在 **`video_{N}_pano/`**
- 看向：沿路径前进**切线**方向
- 高度 `z ∈ [1.0, 1.5]` m、FOV `60°~90°` 各随机一次，全程恒定

#### RGB vs 几何

| 类型 | video 行为 |
| ---- | ---------- |
| **RGB** | **仅全景图**（equirectangular）；忽略 `--pano` / `--depth` / `--semantic` |
| **几何** | 仍受 `--glb` / `--ply` / `--visible_geometry` / `--voxel` 控制，写入 `video_{N}_pano/` |

轨迹 manifest → `Y/auto_views.json`（`camera_positions`、`look_at_targets`、…）；resume 复用，不重新随机。

```bash
python scenebuilder/render_ssl.py --ssl scene.txt --output out --video \
  --glb --ply --visible_geometry --video_spacing 0.2
```

#### 输出示例（仅 video）

```
out_normalized/
├── topdown_normalized/
├── auto_views.json
├── video_40/
└── video_40_pano/               # 40 帧全景（示例：8m 周长 / 0.2m）
```



### 5.2 像素对齐与地板路径（实现细节）

API 与快速示例见 **§3.4**、**§4.4**。本节说明对齐规则与 nav_mask pipeline。

```python
def render_normalized_topdown(
    input_text: str,
    output_dir: str,
    backend: str = "bpy",
    asset_dir: Optional[str] = None,
    texture_dir: Optional[str] = None,
    show_ceiling: bool = False,
    floor_path: bool = True,   # True 时渲染 depth+semantic 并调用 nav_mask_path
    ...
) -> Union[str, Dict[str, Any]]
```

`topdown_normalized/` 输出分辨率**固定 1000×1000**，不可通过参数修改。

底层渲染：`BpySceneCtx.normalized_topdown_view()` / `SceneCtx.normalized_topdown_view()`（`core/util_data.prepare_pixel_aligned_topdown_context` 做 XY 平移）。

### 像素对齐规则

1. 计算 topdown 相机与 FOV（与常规俯视图相同；默认 1000×1000 时长轴 FOV，见 §4.0）
2. `pixel2real_ratio = camera_z × tan(fov/2) / 500`（正方形时 fov 为水平 = 垂直 FOV）
3. **平移整场景 SSL**，使相机地面投影落在图像像素 `(ratio×500, ratio×500)`（图像坐标系）
4. 渲染 1000×1000 俯视图 → 图像左上角 `(0,0)` 对应地面 `(0,0)`
5. `camera_para.json` 使用**图像坐标系** + `pixel2real_ratio`（SpatialFactory 兼容）



### 地板路径 pipeline（`core/nav_mask_path.py`）

`floor_path=True`（默认）时在渲染完成后自动调用 `run_nav_mask_floor_path(output_dir, config)`：

```
topdown.png + topdown_depth.png + topdown_semantic.*
        ↓
① depth_mask     — 深度在「相机到地面距离 ± tol」内的像素
② structure_mask — 语义图中墙/门/窗
③ floor_mask     — 语义图中 floor
        ↓
nav_mask = (depth_mask − structure_mask) ∪ floor_mask
        ↓
沿最大连通域内边界采样闭环路径（默认 8–20 点）
        ↓
floor_path_points.json / floor_path_ssl.txt / topdown_floor_path.png
```

**nav mask 合并公式**（`combine_nav_mask`）：

```
nav_mask = (depth_mask \ structure_mask) ∪ floor_mask
```

即从深度可行走区域中去掉墙门窗，再并入语义地板区域。

输出目录树见 **§6**。

### 配置项（`config.yaml`）


| 键                            | 默认     | 说明                         |
| ---------------------------- | ------ | -------------------------- |
| `nav_mask_depth_tolerance_m` | `0.05` | 深度 mask：相机到地面距离 ± 此值（米）    |
| `nav_mask_inset_m`           | `0.15` | 路径采样：沿 nav mask 内边界向内偏移（米） |
| `nav_mask_point_spacing_m`   | `0.3`  | 闭环路径点间距（米）                 |


---



## 6. 输出目录与坐标系



### 6.1 目录树

**唯一输出根目录 Y**（所有产物都在 Y 下）：


| `--normalized_topdown` | Y                      |
| ---------------------- | ---------------------- |
| 否                      | `{output}/`            |
| 是                      | `{output}_normalized/` |


`--normalized_topdown` 是**全局配置**：先规范化 SSL，后续所有渲染（各视角、GLB、点云）均基于该坐标系。

**流程（启用** `--normalized_topdown`**）：**

1. 确定 Y = `{output}_normalized`，写入规范化 `ssl.txt` + `data.json`
2. `Y/topdown_normalized/`：1000² 像素对齐俯视图 + 地板路径（自动）
3. `Y/topdown/`、`Y/{时间戳}/`、`Y/{时间戳}_seq/` …：按 `--views` 或 `auto` 渲染
4. `Y/scene.glb`、`Y/pointcloud/`、`Y/voxel/`：全场景导出（需 `--holo_geometry`）

```
Y/                                    # {output} 或 {output}_normalized
├── ssl.txt
├── data.json
├── scene.glb                         # [--holo_geometry + --glb]
├── pointcloud/                       # [--holo_geometry + --ply]
├── voxel/                            # [--holo_geometry + --voxel]
├── topdown_normalized/               # [--normalized_topdown] 像素对齐 + 路径规划
│   ├── topdown.png                   # 1000×1000
│   ├── camera_para.json              # 图像坐标 + pixel2real_ratio
│   ├── topdown_depth.png             # [floor_path]
│   ├── topdown_semantic.*
│   ├── nav_mask*.png                 # [floor_path]
│   ├── floor_path_ssl.txt            # [floor_path]
│   └── topdown_floor_path.png        # [floor_path]
├── auto_views.json                   # [--views auto] 或 [--video]
├── auto_path_0000/ …                 # [--views auto] 单帧
├── auto_path_0008_seq/               # [--views auto] 三帧序列
├── video_{N}/                        # [--video]
├── video_{N}_pano/                   # [--video] 全景序列
├── {timestamp}_seq/                  # left_seq 等手动序列
├── topdown/                          # [--views topdown] 1024² 常规俯视图
│   ├── topdown.png
│   ├── topdown_depth.png             # [--depth]
│   ├── topdown_semantic.*            # [--semantic]
│   ├── planar_faces.json             # [--ply]
│   ├── pointcloud/                   # [--visible_geometry + --ply]
│   └── voxel/                        # [--visible_geometry + --voxel]
└── {毫秒时间戳}/                      # 单相机视角
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
# 规范化 + auto 路径视角
python scenebuilder/render_ssl.py --ssl path/to/ssl.txt \
  --views auto --output path/to/out --glb --assets path/to/assets

# 规范化 + 多视角
python scenebuilder/render_ssl.py --ssl path/to/ssl.txt \
  --normalized_topdown --views topdown left_seq \
  --output path/to/out --glb --assets path/to/assets
# → 唯一输出: path/to/out_normalized/
# left_seq → {timestamp}_seq/
```

---



### 6.2 几何导出坐标系（世界 SSL 与 OpenCV）

各视角在世界（或规范化）SSL 下 construct / 渲染。开启 `--visible_geometry` 且 `--glb` / `--ply` / `--voxel` 时，**每个视角目录**下的可见 GLB/PLY/体素 **主文件为世界 SSL**（体素为 `occupancy_world.npz`）；同时写出 `*_opencv.*` 副本（OpenCV 相机系，见 `core/geometry_opencv.py`）。根目录全场景几何（`scene.glb`、`pointcloud/scene_all.ply`、`voxel/`）须 `--holo_geometry`，不做视锥裁剪。


| 坐标系            | 适用文件                                                                                                                                                                 | 原点与轴向                                   |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| **世界 SSL**     | 根目录 `ssl.txt`、`data.json`；`worker_render_post`（`--holo_geometry`）的 `scene.glb`、`pointcloud/scene_all.ply`、`voxel/occupancy_world.npz`；`{视角}/pointcloud/*.ply`（无 `_opencv`）、`scene_visible.ply`、`scene_visible.glb` | 与 §2 一致：+X 右、+Y 上、+Z 高                  |
| **OpenCV 相机系** | 上述每个几何文件的 `*_opencv.ply` / `*_opencv.glb`                                                                                                                            | 原点：相机光心；+X：图像右；+Y：图像下；+Z：沿视线向场景深处（深度为正） |


**约定：**

- 主 **PLY** 顶点与法线均为 Blender 世界坐标（= SSL Z-up，或 `--normalized_topdown` 后的规范化世界坐标），与 §2 轴向一致；属性为 `x y z nx ny nz red green blue`。
- 主 **GLB** 与 `**_opencv.glb`** 均经 Blender `export_scene.gltf` 写盘，自动做 **Z-up → glTF Y-up**：`(gx, gy, gz) = (bx, bz, -by)`。主 GLB 的 mesh 为世界 SSL；`_opencv.glb` 的 mesh 为 OpenCV 变换后的 `(ox, oy, oz)`，磁盘上为 `(ox, oz, -oy)`，读回 OpenCV 坐标用 `gltf_yup_to_blender_zup` 即可。
- `**_opencv.glb**` 在主 GLB 导出后，于 Blender 内对 mesh 顶点调用与 PLY 相同的 `transform_points_to_opencv`，再二次 glTF 导出（**保留材质/UV**），不做额外轴补偿。
- `**_opencv.ply**`：对世界系点与法线分别做 OpenCV 变换后写出：
  - 点：`p_cam = R @ (p_world - eye)`
  - 法线：`n_cam = normalize(R @ n_world)`（只旋转、不平移）
  - ASCII 顶点即为 OpenCV `(ox, oy, oz, nx, ny, nz)`（无 glTF 轴变换）
- OpenCV 轴向：+X 右、+Y 下、+Z 向场景深处。
- `**left_seq` 等多帧序列**：OpenCV 副本以**序列首帧**相机为参考系；可见几何为各帧视锥并集。若 `--visible_geometry`，序列目录还会在 `pointcloud/`（`--ply`）和/或 `glb/`（`--glb`）下写 `viewvis` sidecar（§7.1.2）。
- `metadata_visible.json` / `metadata.json` 中可为合并点云记录 `"path_opencv": "scene_visible_opencv.ply"` 等字段。
- `planar_faces.json`、深度/语义 PNG 仍用世界 SSL 或与渲染图对齐的像素坐标。
- 各视角目录**不再**写出 `{视角}/ssl.txt`（场景 SSL 仅在输出根目录 Y）；每个视角目录自动写出 `**ssl_opencv.txt`**（OpenCV 相机系，见 §6.4）。

`**camera_para.json`（每视角）** 记录相机位姿、深度/法线元数据，以及 **ScanNet / OpenSpatial 兼容的 c2w 外参与 pinhole 内参**（见 §6.3）。示例：

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

`camera_position` / `look_at_target` 为 SSL 世界坐标（米），便于人类阅读；**几何投影请优先使用** `c2w` **+** `intrinsic`。

---



### 6.3 相机外参 c2w 与内参 intrinsic（ScanNet / OpenSpatial 兼容）

每个 `{basename}_camera_para.json`（以及 `topdown_normalized/camera_para.json` 的 SSL 分支）写入 **4×4 c2w** 与 **4×4 pinhole K**，可直接 `np.array(json["c2w"])` 使用，等价于 ScanNet 的 `{frame}_pose.txt` / `{frame}_intrinsic.txt`。

#### 6.3.1 坐标系约定


| 坐标系                | 轴向                         | 用途                                       |
| ------------------ | -------------------------- | ---------------------------------------- |
| **World（SSL）**     | +X 右、+Y 上（俯视）、+Z 高；地面 ∥ XY | 与 `ssl.txt`、3D bbox、主 PLY 点云同系 |
| **Camera（OpenCV）** | +X 右、+Y 下、+Z 前（深度为正）       | 深度图、法线图 `{view}_normal.png`、`*_opencv.ply` |


`world_convention: "ssl_z_up"`，`camera_convention: "opencv"`。

#### 6.3.2 外参 `c2w`（camera → world）

```
c2w = | R11  R12  R13  tx |
      | R21  R22  R23  ty |
      | R31  R32  R33  tz |
      |  0    0    0    1 |
```

- **R（3×3）**：OpenCV 相机坐标轴在 **SSL 世界系**下的方向；列为相机 +X(右)、+Y(下)、+Z(前)。
- **t = [tx, ty, tz]ᵀ**：相机光心在 SSL 世界系下的位置（**米**），与 `camera_position` 一致。
- **含义**：将 OpenCV 相机系点变到世界系：`P_world = c2w @ [X_cam, Y_cam, Z_cam, 1]ᵀ`。
- **逆变换**：`w2c = inv(c2w)`，`P_cam = w2c @ P_world`（齐次 4 维）。

构造方式：由 `camera_position`、`look_at_target`、`world_up` 与 Blender 渲染相机一致地导出（`geometry_opencv.build_opencv_c2w_matrix`）。**无** ScanNet 式 `axis_align_matrix`——若输入 SSL 已规范化，则 c2w 已在规范化 SSL 世界系下。

#### 6.3.3 内参 `intrinsic`（pinhole K，无畸变）

```
intrinsic = | fx   0   cx   0 |
            |  0  fy   cy   0 |
            |  0   0    1   0 |
            |  0   0    0   1 |
```


| 元素                     | 含义                |
| ---------------------- | ----------------- |
| `fx = intrinsic[0][0]` | x 方向焦距（像素）        |
| `fy = intrinsic[1][1]` | y 方向焦距（像素）        |
| `cx = intrinsic[0][2]` | 主点 u（像素，图像左上角为原点） |
| `cy = intrinsic[1][2]` | 主点 v（像素）          |


与 Blender `sensor_fit=AUTO` 及当前 `image_size` 一致（§4.0）：`manual_fov` / `auto_fov` 给出长轴 FOV 角 `angle`，按宽高比推出 `tan_x`、`tan_y`；`fx = width / (2·tan_x)`，`fy = height / (2·tan_y)`，**`fx = fy`**；`cx = width/2`，`cy = height/2`（规则同 `util_bpy.camera_frustum_tangents`）。

**深度反投影（相机系，米）：**

```
X_cam = (u - cx) * depth_m / fx
Y_cam = (v - cy) * depth_m / fy
Z_cam = depth_m
```

其中 `depth_m = depth_pixel / depth_scale`（`depth_scale` 见 §7.4.1）。

**世界点投影到像素：**

```
P_cam = w2c @ P_world                         # 齐次 4 维
uv_h  = intrinsic @ P_cam
u, v  = uv_h[0] / uv_h[2], uv_h[1] / uv_h[2]
```

完整链路：`pixel = K @ w2c @ P_world`（3D box 投影校验等同 ScanNet `3dbox_filter`）。

#### 6.3.4 配套字段


| 字段                | 典型值                       | 说明                         |
| ----------------- | ------------------------- | -------------------------- |
| `depth_scale`     | 动态 float                  | 深度 PNG ÷ `depth_scale` = 米 |
| `depth_unit`      | `"meter"`                 | 深度单位                       |
| `is_metric_depth` | `true`                    | 米制深度（有 `--depth` 时）        |
| `image_size`      | `[W, H]`                  | 与 RGB/深度/法线图同分辨率           |
| `intrinsic_model` | `"pinhole_no_distortion"` | 不考虑畸变                      |


全景 `{view}_pano.png` 的 `camera_para` 含 `c2w`，但 `**intrinsic` 省略**（`projection: equirectangular`），pinhole K 不适用。

#### 6.3.5 Python 读取示例

```python
import json
import numpy as np

with open("topdown_camera_para.json") as f:
    cam = json.load(f)

c2w = np.array(cam["c2w"], dtype=float)          # (4, 4)
K = np.array(cam["intrinsic"], dtype=float)       # (4, 4)
w2c = np.linalg.inv(c2w)

depth_m = imageio.imread("topdown_depth.png").astype(float) / cam["depth_scale"]
# 像素 (u, v) + depth_m[v, u] → P_cam → P_world = (c2w @ [X,Y,Z,1])[:3]
```

`topdown_normalized/camera_para.json` 中 `camera_position` 可能为**图像坐标**（`coordinate_system: "image"`），但 `**c2w` / `intrinsic` 仍基于 SSL 世界系的 `camera_position_ssl`**（与场景几何同系）。

---



### 6.4 视角 OpenCV SSL（`ssl_opencv.txt`）

每个视角目录（含 `topdown/`、`auto_path_0012/`、`auto_path_0023_seq/`、`auto_path_0012_pano/` 等）在 `topdown_view` / `render_view` 渲染完成后自动写出 `**ssl_opencv.txt**`：将根目录 `ssl.txt` 的几何变换到 **OpenCV 相机坐标系**（与 §6.3 `c2w` / 深度 / 3D Grounding 一致）。

#### 6.4.1 参考相机


| 渲染模式                                | OpenCV 参考系                          |
| ----------------------------------- | ----------------------------------- |
| 单帧 `render_view` / `topdown_view`   | 该次渲染的相机位姿                           |
| 相机序列 `left_seq` / `auto_path_*_seq` | **序列首帧**相机（所有帧与子目录 `_pano` 共用同一参考系） |
| `topdown_normalized/`               | 像素对齐后的 SSL 俯视图相机                    |


参考相机与 `{basename}_camera_para.json` 中的 `c2w` 描述的是同一相机；`c2w` 为 **OpenCV 相机系 → SSL 世界系**，`w2c = inv(c2w)` 为 **SSL 世界系 → OpenCV 相机系**。

#### 6.4.2 坐标系

- **OpenCV 相机系**：原点 = 光心；+X 右、+Y 下、+Z 前（深度为正）。
- 所有点坐标单位为**米**。
- `world_up`：原 SSL 世界竖直向上 `[0, 0, 1]` 变换到当前 OpenCV 相机系下的方向（通常不为 `[0,0,1]`）。



#### 6.4.3 相对根目录 `ssl.txt` 的字段变化


| 实体                | 世界 SSL (`ssl.txt`)                        | OpenCV SSL (`ssl_opencv.txt`)                        |
| ----------------- | ----------------------------------------- | ---------------------------------------------------- |
| **Room**          | `Room(room_type="...")`                   | 增加 `world_up=[a,b,c]`                                |
| **Wall**          | `p`, `q` 为 SSL 世界 XY（第三维常为 0）             | `p`, `q` 为 OpenCV 相机系 **3D** 端点                      |
| **Door / Window** | `center` 为 SSL 世界坐标                       | `center` 为 OpenCV 相机坐标                               |
| **Bbox**          | `center` + `angle_z`（度，绕 SSL +Z）+ `scale` | `center`（相机系）+ `pose=[roll,pitch,yaw]` + `scale`（不变） |


`width` / `height`（门窗）、墙 `height`、bbox `scale` 等**标量尺寸不变**（局部几何长度，米）。

#### 6.4.4 Bbox `pose` 约定（标准面对面位姿 + 内旋 XYZ）

```ssl
Bbox(label="dining table0", center=[0.5, -0.1, 2.3], pose=[0.3, -0.05, 0.1], scale=[3.05, 0.93, 1.21], asset_id="17116819")
```



##### 字段含义


| 字段                          | 含义                                     |
| --------------------------- | -------------------------------------- |
| `pose = [roll, pitch, yaw]` | 从标准面对面位姿到实际位姿的 **内旋 XYZ** 欧拉角，单位**弧度** |
| `scale = [xl, yl, zl]`      | 沿盒体**自身局部轴**的全长（米），不是相机轴上的投影宽度         |
| `roll`                      | 第一步，绕当前 OA/local **X(front)** 轴内旋      |
| `pitch`                     | 第二步，绕已转动后的 OA/local **Y(left)** 轴内旋    |
| `yaw`                       | 第三步，绕已转动后的 OA/local **Z(top)** 轴内旋     |


这里的 `pose` 是有物理语义的欧拉角：先定义一个与观察者面对面的 OA/local 标准坐标系，再从这个标准位姿出发，按 `X -> Y -> Z` 的动轴顺序内旋到物体在当前 OpenCV 相机系下的实际位姿。

SSL bbox 的三根语义轴为：


| 语义轴     | bbox 局部轴   |
| ------- | ---------- |
| `front` | `local -Y` |
| `left`  | `local +X` |
| `back`  | `local +Y` |
| `top`   | `local +Z` |


OpenCV 相机系为 `+X` 右、`+Y` 下、`+Z` 前。因此：

OA 的标准位姿为：

- `front -> camera -Z`：物体正面对着我们。
- `left -> camera +X`：物体左侧投到图像右边。
- `top -> camera -Y`：物体上方投到图像上方。

这三根轴构成右手系：`front x left = top`，即 `camera -Z x camera +X = camera -Y`。

设标准位姿矩阵：

```python
R0 = [front0, left0, top0]
   = [camera -Z, camera +X, camera -Y]
```

实际位姿矩阵：

```python
R_actual = [front_cam, left_cam, top_cam]
```

相对旋转为：

```python
R_delta = R0.T @ R_actual
pose = as_euler("XYZ", R_delta) = [roll, pitch, yaw]
```

还原时：

```python
R_actual = R0 @ Rotation.from_euler("XYZ", pose).as_matrix()
```

SciPy 大写 `"XYZ"` 表示内旋，即从标准 local 坐标系出发，依次绕当前 `X(front)`、转后的 `Y(left)`、再转后的 `Z(top)` 旋转。

##### 欧拉角主值与万向节锁

本仓库采用 SciPy `"XYZ"` 的主值规范导出唯一文本表示：


| 分量      | 主值范围          |
| ------- | ------------- |
| `roll`  | `[-π, π]`     |
| `pitch` | `[-π/2, π/2]` |
| `yaw`   | `[-π, π]`     |


当 `pitch` 接近 `+π/2` 或 `-π/2` 时发生万向节锁：最终 3D 姿态仍然唯一，但 `roll` 与 `yaw` 的欧拉角分配不唯一。此时本仓库采用规范解：

- 固定 `yaw = 0`
- 将剩余自由度合并到 `roll`

例如 topdown 视角下，很多落地物体会出现 `pitch=π/2`；打印机示例可规范表示为 `pose=[π/2, π/2, 0]`。若用于训练/评估，建议用由 `pose` 还原出的旋转矩阵或 `front/left/top` 三轴方向计算误差，而不是直接对欧拉角做 L1/L2。

##### 从世界 SSL 如何算出 `pose`

1. 世界系 OBB：`angle_z` → 仅绕 SSL +Z 的 `R_world_box`，平移 `t_world = center`，尺寸 `scale`。
2. `T_world_box = [R_world_box | t_world]` 表示 **bbox 局部系 → SSL 世界系**；`T_cam_box = inv(c2w) @ T_world_box` 表示 **bbox 局部系 → OpenCV 相机系**。
3. 从 `R_cam_box = T_cam_box[:3,:3]` 提取语义轴：

```python
front_cam = R_cam_box @ [0, -1, 0]
left_cam  = R_cam_box @ [1,  0, 0]
top_cam   = R_cam_box @ [0,  0, 1]
```

1. 由 `R_delta = R0.T @ [front_cam,left_cam,top_cam]` 分解 `pose=[roll,pitch,yaw]`；`scale` 不变。

这里 `R_world_box = Rz(angle_z)` 是局部轴到 SSL 世界轴的旋转。它与“`angle_z=0°` 时物体面向 −Y”并不矛盾：前向向量不是局部 +X，而是 `R_world_box @ [0, -1, 0]`。

这个表示既保留欧拉角的完整 3D 姿态表达，又有明确的物理零姿态。例如物体正面对着相机且 `left/top` 与图像右/上对齐时 `pose=[0,0,0]`；topdown 打印机这类极端俯视姿态也可以解释为从面对面标准位姿经过内旋到实际 `front/left/top` 三轴。

#### 6.4.5 示例片段

世界 SSL（根目录）：

```ssl
Room(room_type="dining room")
Wall(label="wall0", p=[0.0, 0.0, 0], q=[5.0, 0.0, 0], height=2.7)
Bbox(label="dining table0", center=[5.1, 2.77, 0.6], angle_z=0, scale=[3.05, 0.93, 1.21], asset_id="17116819")
```

同一视角 `ssl_opencv.txt`（示意数值）：

```ssl
Room(room_type="dining room", world_up=[0.12, -0.98, 0.05])
Wall(label="wall0", p=[1.2, 0.3, 4.5], q=[-0.8, 0.3, 2.1], height=2.7)
Bbox(label="dining table0", center=[0.5, -0.1, 2.3], pose=[0.42, -0.15, 0.02], scale=[3.05, 0.93, 1.21], asset_id="17116819")
```



#### 6.4.6 输出路径


| 目录                       | `ssl_opencv.txt` |
| ------------------------ | ---------------- |
| `Y/topdown/`             | ✅                |
| `Y/topdown_normalized/`  | ✅                |
| `Y/auto_path_0012/`      | ✅                |
| `Y/auto_path_0023_seq/`  | ✅（首帧相机系）         |
| `Y/auto_path_0012_pano/` | ✅（与对应透视视角同一参考相机） |
| `Y/ssl.txt`（根）           | ❌ 仍为世界 SSL       |


实现：`core/ssl_opencv.py`；由 `BpySceneCtx.write_opencv_ssl_for_view()` / `SceneCtx.write_opencv_ssl_for_view()` 在渲染保存阶段调用。

**三类点云的区别：**


| 路径                             | 坐标系              | 内容                                  |
| ------------------------------ | ---------------- | ----------------------------------- |
| `output_root/pointcloud/`      | 世界 SSL           | 场景内**所有**物体（`--holo_geometry --ply`，post 阶段）          |
| `{视角}/pointcloud/*.ply`        | 世界 SSL           | 该视角**可见**物体（需 `--visible_geometry --ply`） |
| `{视角}/pointcloud/*_opencv.ply` | OpenCV（该视角/首帧相机） | 与上一行相同几何，换到 OpenCV 相机系              |
| `{seq}/pointcloud/scene_visible_viewvis_*.npz` | 无（0/1 标记） | **仅多相机序列**（`*_seq/`）：每点相对各帧的可见性；第 *i* 行与 PLY 第 *i*+1 行对齐（§7.1.2） |
| `output_root/voxel/`           | 世界 SSL           | 全场景 256³ 占用场（`--holo_geometry --voxel`） |
| `{视角}/voxel/occupancy_*.npz`   | 世界 SSL / OpenCV 相机系 | 该视角可见 256³ 占用场（`--visible_geometry --voxel`） |
| `{视角}/planar_faces.json`       | 世界 SSL（3D 顶点）    | 墙/门/窗/地板/天花内表面（需 `--ply`）           |


**命名与 context 规范（**`ssl.txt`**、context 键、点云文件名一致）：**

- 地板 / 天花：`floor/floor.ply`、`ceiling/ceiling.ply`（可见几何时为 `floor/floor_visible.ply`、`floor/floor_visible_cutted.ply` 等，规则同墙/家具）
- 墙：`wall0`, `wall1`, … → `walls/wall0.ply`（无 `asset_id`）
- 门/窗：`door0`, `window0`, … → `doors/door0_{asset_id}.ply`；若 `asset_id` 在资产目录不存在则**保留空洞**、去掉 `asset_id`，文件名与 SSL 均不含 `asset_id`
- 家具：`sidetable0`, `armchair0`, … → `boxes/sidetable0_{asset_id}.ply`；若 `asset_id` 不存在则**删除该 bbox**

可见几何后缀：`_visible`（整体在视锥内）、`_visible_cutted`（被视锥裁切，见下）。

#### `_cutted` 后缀含义

`**_cutted` 表示该物体被当前相机视锥「切过一刀」**——物体仍被导出，但 mesh / 点云里只保留视锥内的部分；文件名用 `_cutted` 标记「原始物体并未完整落在画面内」。


| 后缀                | 含义                                                  | 典型场景                    |
| ----------------- | --------------------------------------------------- | ----------------------- |
| `_visible`        | 物体 **≥ 95%** 的三角形三顶点都在渲染视口 `[0,1]×[0,1]` 内，视为整体在画面里 | 房间中央的桌子、完全入镜的墙段         |
| `_visible_cutted` | 上述「完全在视口内」的三角形占比 **< 95%**，视锥边界切到了物体                | 画面边缘的窗、被裁掉一半的墙、只露出一角的家具 |


**判定与导出是两步：**

1. **遮挡剔除（物体级）**：被其它物体完全挡住 → **整物体不导出**（不会出现 `_visible` 或 `_cutted`）。
2. **视锥裁剪（三角形级）**：保留的物体按 Blender 渲染视锥裁掉视锥外三角形；裁完若还有几何则导出。
3. **命名**：根据裁剪**前** mesh 的「完全在视口内」面片占比是否 ≥ 95%，决定文件名用 `_visible` 还是 `_visible_cutted`。

示例：

```
floor/floor_visible_cutted.ply       # 地板被视锥裁切
floor/floor_visible_cutted_opencv.ply
ceiling/ceiling_visible.ply          # 天花可见时（如相机朝上）
walls/wall0_visible.ply          # 整面墙基本都在画面内
walls/wall2_visible_cutted.ply   # 墙的一部分在画面外，被视锥切掉
walls/wall2_visible_cutted_opencv.ply  # 同上，OpenCV 坐标副本
```

`metadata_visible.json` 里每个 object 有 `"frustum_cutted": true/false` 与 `"frustum_in_view_ratio"`（0–1），与文件名一致。

**注意：** `_cutted` 只描述**视锥裁剪**，与是否被遮挡无关；被挡死的物体直接不出现在导出结果中。

---



## 7. 导出产物详解



### 7.1 可见几何 (`visible_geometry`)

开启 `--visible_geometry` 且同时开启 `--glb`、`--ply` 或 `--voxel` 之一时，每个视角目录下导出可见 GLB / 点云 / 体素；几何文件另有 `_opencv` 副本（见 §6.2）。体素详见 **§7.5**。

**注意：** 体素**不是**从 GLB 读回，而是与 GLB 共用同一套内存中的可见三角形；`--voxel` 可单独使用（不必 `--glb`）。

### 处理流程（按顺序）

1. **遮挡可见性（物体级，基于完整 mesh）**
  - 对物体表面采样点做射线检测（透明墙/天花/地板可穿透）
  - **完全被挡** → 丢弃该物体
  - **部分可见或全部可见** → 保留，进入下一步
2. **视锥裁剪（三角形级）**
  - 使用 Blender `calc_matrix_camera` 投影矩阵，在 clip space 做几何裁剪
  - 切掉视锥外的三角形或三角形部分
  - 裁剪后无几何 → 丢弃该物体
3. **命名规则（**`_visible` **/** `_cutted`**）**
  - 详见上文 **「**`_cutted` **后缀含义」**
  - 统计时仅 **三顶点都在** 视口 `[0,1]×[0,1]` 的面片算「在视锥内」
  - 占比 **≥ 95%** → `_visible`；**< 95%** → `_visible_cutted`
  - `metadata_visible.json` 含 `"frustum_cutted"` 与 `"frustum_in_view_ratio"`



### 适用类别

floor、ceiling、walls、doors、windows、boxes 均参与可见性判定与视锥裁剪。

---

### 7.1.1 全场景几何 (`holo_geometry`)

开启 `--holo_geometry` 且同时开启 `--glb`、`--ply` 或 `--voxel` 之一时，在输出根目录 `Y/` 导出**完整场景**几何（不做遮挡/视锥裁剪），由 `worker_render_post` 在全部视角渲染完成后统一写出：

| 开关组合 | 根目录产物 |
| -------- | ---------- |
| `--holo_geometry --glb` | `scene.glb` |
| `--holo_geometry --ply` | `pointcloud/scene_all.ply` + 分物体 PLY |
| `--holo_geometry --voxel` | `voxel/occupancy_world.npz` 等（仅世界 SSL，无 OpenCV 副本） |

与 `--visible_geometry` **独立**：可只开其一，也可同时开启（各视角可见 + 根目录全场景）。

```bash
# 仅各视角可见几何
--glb --ply --visible_geometry --voxel

# 仅根目录全场景
--glb --ply --holo_geometry --voxel

# 两者都要
--glb --ply --visible_geometry --holo_geometry --voxel
```

---

### 7.1.2 多相机序列可见性 sidecar（`viewvis`）

当 `render_view` 传入**相机序列**（嵌套列表，目录 `*_seq/`）且帧数 **>1**，并开启 `--visible_geometry` 时，合并几何仍是各帧视锥**并集**（§7.1）。会额外写出 sidecar，对每个**采样元素**相对每一帧相机打 0/1 标注。

**单帧视角不写出。**

#### 点云（`--visible_geometry --ply`）

| 文件 | 可见性类型 | `visibility[i, k] == 1` |
| ---- | ---------- | ------------------------ |
| `pointcloud/scene_visible_viewvis_point.npz` | **点级（遮挡）** | 采样点 *i* 在帧 *k* 视锥内且未被遮挡 |
| `pointcloud/scene_visible_viewvis_object.npz` | **物体级（导出规则）** | 点 *i* 所属物体在相机 *k* 通过 §7.1，且点在视锥内 |

第 *i* 行与 `scene_visible.ply` / `scene_visible_opencv.ply` 第 *i*+1 顶点对齐。元数据：`pointcloud/scene_visible_viewvis.json`。

#### Mesh / GLB（`--visible_geometry --glb`）

| 文件 | 可见性类型 | `visibility[i, k] == 1` |
| ---- | ---------- | ------------------------ |
| `glb/scene_visible_viewvis_point_tri.npz` | **点级（遮挡）** | 三角形 *i* 重心在视锥内且未被遮挡 |
| `glb/scene_visible_viewvis_object_tri.npz` | **物体级（导出规则）** | 三角形 *i* 所属物体在相机 *k* 通过 §7.1，且重心在视锥内 |

第 *i* 行对应 **`scene_visible.glb` 导出顺序**中的第 *i* 个三角形（`export_visible_glb` 的 entry → record → triangle 循环）。采样位置为**三角形重心**。一份 sidecar 同时适用于 `scene_visible.glb` 与 `scene_visible_opencv.glb`。元数据：`glb/scene_visible_viewvis_tri.json`。

#### NPZ 结构（点云或 mesh 通用）

```python
import numpy as np

data = np.load("glb/scene_visible_viewvis_point_tri.npz")  # 或 pointcloud/..._point.npz
vis = data["visibility"]       # uint8, shape (N, K)
frames = data["camera_frames"]
# vis[i, k] → 元素 i 在 frames[k] 相机下是否可见
```

#### 依赖与性能

| 项 | 说明 |
| --- | --- |
| 触发 | 序列 **>1** 帧 + `--visible_geometry` +（`--ply` 和/或 `--glb`） |
| 建议 | 开启 `--depth` 做点级 sidecar（§7.4.1） |
| 无 depth | 点级 sidecar 射线 fallback，较慢 |
| 物体级 sidecar | 每相机物体级射线 + 视锥，开销小 |
| 实现 | `core/viewvis_export.py`；PLY 由 `export_visible_point_cloud()`，mesh 由 `export_visible_glb()` 调用 |

序列目录示例：

```
auto_path_0023_seq/
├── 1735123456789.png
├── 1735123456789_depth.png
├── 1735123456789_camera_para.json
├── …
├── glb/scene_visible.glb
├── glb/scene_visible_opencv.glb
├── glb/scene_visible_viewvis_point_tri.npz
├── glb/scene_visible_viewvis_object_tri.npz
├── glb/scene_visible_viewvis_tri.json
├── pointcloud/scene_visible.ply              # 若 --ply
├── pointcloud/scene_visible_viewvis_*.npz
└── metadata_visible.json
```

#### 7.1.3 按视角分解并集几何（`split_viewvis_geometry.py`）

在 §7.1.2 sidecar 就绪后，运行独立脚本，从合并 PLY/GLB 中导出**每帧一个子集**（与 `camera_frames[k]` 一一对应）：

```bash
python split_viewvis_geometry.py left_seq/pointcloud/scene_visible_opencv.ply view
python split_viewvis_geometry.py left_seq/glb/scene_visible_opencv.glb object
```

| CLI 模式 | Sidecar | 保留规则 |
| -------- | ------- | -------- |
| `view` | `*_viewvis_point*.npz` | 当 `visibility[i,k]==1` 时保留第 *i* 行（点级遮挡） |
| `object` | `*_viewvis_object*.npz` | 物体级导出规则 + 视锥通过时保留第 *i* 行 |

默认输出目录：输入文件旁的 `{文件名}_by_{模式}/`；文件名为 `{相机时间戳}.ply` / `.glb`；`manifest.json` 记录保留/总数。库：`core/viewvis_split.py`。另见 [README § 按视角分解](../README.zh-CN.md#按视角分解可见几何split_viewvis_geometrypy)。

---

### 7.2 平面内表面顶点 (`export_point_cloud` / `--ply`)

开启 `--ply` 时，**每个视角**在渲染完成后自动额外导出（无需单独 flag）：


| 文件                  | 说明                   |
| ------------------- | -------------------- |
| `planar_faces.json` | 墙/门/窗/地板/天花内表面多边形顶点  |
| `{view}_lines.png`  | 在渲染图副本上绘制顶点连线（每对象一色） |




### 几何定义

与 scenebuilder 建 mesh 逻辑一致：

- **墙**：SSL 中 `p`/`q` 即内墙底两点，高度为 `align_height ? z_max : wall.height`；带门/窗洞时 JSON 含 `outer` 环与 `hole` 环
- **门/窗**：按 `center`、`width`、`height` 及所属墙计算内面四顶点（同 `create_door_or_window_mesh`）
- **地板**：房间 `vertices` 多边形 @ z=0
- **天花**：房间 `vertices` 多边形 @ z=z_max（仅 `show_ceiling=True` 且已构建时）



### 视锥裁剪与顶点顺序

- 多边形在 **Blender** `calc_matrix_camera` **齐次 clip space** 裁剪（与可见几何相同），沿边插值 world 坐标以保持共面
- **保持边界拓扑顺序**（地板/天花沿用房间 `vertices` 环路；墙/门/窗沿用 mesh 定义顺序；视锥裁剪不重新按角度排序）
- 若从相机看为 CW 则整体反转；再旋转起点为 z 最小 → y 最小 → x 最小的顶点
- 同时记录 3D 坐标、投影像素坐标 `vertices_2d_px` 与遮挡标记 `occluded`（1=被遮挡未呈现在渲染图，0=可见）



### JSON 结构示例

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

**说明：** 平面顶点导出与 `visible_geometry` 独立——仅 `--ply` 即可；可见点云 `{视角}/pointcloud/` 仍须 `--visible_geometry`。

---



### 7.3 语义图 (`semantic`)

开启 `--semantic` 时，每个视角（透视 / 俯视 / 全景，规则相同）额外输出：


| 文件                     | 说明                                         |
| ---------------------- | ------------------------------------------ |
| `{view}_semantic.png`  | 按实体着色的语义分割图                                |
| `{view}_semantic.json` | 实体元数据、`color` ↔ mask、`pixel_num`、`bbox_2d` |
| `{view}_bbox_2d.png`   | 在**原彩色渲染图**上绘制检测框与 `label` 的可视化            |




#### 7.3.1 渲染机制

- 每个实体独立颜色：内部键 `entity:{category}:{label}` 经 HSV 哈希得到 RGB；同场景内保证颜色唯一（便于精确索引）
- 颜色保证不与背景 `(0,0,0)` 相同
- bpy：为每个实体创建 proxy mesh + **不透明 Emission** 材质，`view_transform=Raw`
- **语义专用抗锯齿关闭**：`cycles.samples=1`、`filter_width=0.01`、关闭 denoising（EEVEE 则 `taa_render_samples=1`），使 PNG 像素与 JSON `color` **逐通道精确一致**
- 语义通道**不保留**玻璃等透明材质：透明物体在语义图中按实心几何遮挡后方物体
- 索引方式：`pixel == color` 精确匹配（不再用最近邻）；旧语义图若有抗锯齿偏差，可调用 `attach_semantic_bbox_2d(..., max_dist_sq=12)` 兼容



#### 7.3.2 `label` 与 SSL 的对应

`semantic.json` 中的 `**label` 等同于规范化后 SSL 的 `label**`（不是原始输入里的 `id` / `room_id`）。


| 类型              | 规范化规则                       | 示例                            |
| --------------- | --------------------------- | ----------------------------- |
| Wall            | `wall{index}`               | `wall0`                       |
| Door            | 全局递增 `door{n}`              | `door0`                       |
| Window          | 全局递增 `window{n}`            | `window0`                     |
| Bbox            | `slug(label/class)+ordinal` | `sidetable0`                  |
| floor / ceiling | 固定字符串                       | `floor` / `ceiling`（SSL 无对应行） |


进不进 JSON **看的是是否成功进入** `mesh_nodes`，不是单纯看字段有没有 `asset_id`：


| 情况（默认 `geometry_mode=gltf`）             | 是否进 `semantic.json` |
| --------------------------------------- | ------------------- |
| 门/窗/家具：`asset_id` 缺失或无效，未进 `mesh_nodes` | **否**               |
| 门/窗/家具：`asset_id` 有效并加载进 `mesh_nodes`   | **是**（与当前视角是否被遮挡无关） |
| 墙 / 地板 / 天花板（无 `asset_id`，但已建网格）        | **是**               |


补充：

- 家具 `asset_id` 有值但资源不存在 → 规范化阶段**整条删除**，不会进场景也不会进 JSON
- 门/窗 `asset_id` 无效 → 保留墙上空洞，不加载模型 → **不进** JSON
- `mixed` / `bbox` 模式下，无 `asset_id` 的家具可用方块回退进 `mesh_nodes`，从而进 JSON



#### 7.3.3 遮挡与 mask

- 语义图按**不透明几何深度**写色；前方网格挡住的像素只属于前方实体
- **被遮挡的部分一定没有该物体的 mask**，即使遮挡物在彩色图中是透明的（语义通道按实心处理）
- 因此：彩色图里透过玻璃可见的椅子，在语义图里仍可能被玻璃门完全挡住



#### 7.3.4 `{view}_semantic.json` 字段

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


| 字段          | 说明                                                                                               |
| ----------- | ------------------------------------------------------------------------------------------------ |
| `category`  | `floor` / `ceiling` / `walls` / `doors` / `windows` / `boxes`                                    |
| `label`     | 规范化实体名，与 SSL `label` 一致                                                                          |
| `color`     | 该实体在 `*_semantic.png` 中的 RGB，用于反查 mask                                                           |
| `pixel_num` | 当前视角语义 mask 的像素个数。默认按 JSON `color` **精确 RGB 匹配**（配合语义渲染关闭抗锯齿）；`max_dist_sq>0` 时可回退带阈值最近邻以兼容旧图    |
| `bbox_2d`   | `[x1, y1, x2, y2]`，左上角为 `(0,0)`；由该实体 mask **全部前景像素**（含多个连通域）的外接矩形得到；**仅当** `pixel_num > 0` **时存在**，否则字段缺省 |


说明：

- JSON 的 `objects` 列表包含场景中所有已加载实体，**不表示当前视角可见**
- `pixel_num == 0`（完全不可见 / 被挡完）→ **不写** `bbox_2d`
- 已废弃字段：`id`（改用 `label`）、`color_key`（仅内部渲染用）



#### 7.3.5 `{view}_bbox_2d.png`

- 在原渲染图（如 `topdown.png`、`{stamp}.png`）上叠加检测框与 `label`
- 框线颜色使用该实体的 `color`
- **只绘制有** `bbox_2d` **的物体**（`pixel_num == 0` 或缺省 `bbox_2d` 的跳过）

---



### 7.4 深度图与法线图 (`depth`)

开启 `--depth` 时，**bpy + Cycles** 在**同一次渲染**中通过合成器同时导出深度与法线（无额外渲染时间）。`pyrender` 后端仅导出深度，不含法线。

#### 7.4.1 深度图 `{view}_depth.png`


| 项目   | 说明                                                               |
| ---- | ---------------------------------------------------------------- |
| 格式   | **uint16 单通道 PNG**（非 EXR）                                        |
| 含义   | **OpenCV 相机系**下沿 +Z（视线向前）的**米制距离**（与 Cycles Z pass / 渲染 clip 一致） |
| 无效像素 | `pixel == 0`（背景、透明、超出 clip 等）；解码后 `depth_m == 0`                 |


**bpy 实现：** Cycles 合成器 **Z pass** → 临时 EXR → 转 uint16 PNG。失败时回退逐像素 `ray_cast`（此时**不**导出法线）。

#### 7.4.1.1 文件配对（单帧与序列）

深度 PNG 与同目录 `**{basename}_camera_para.json**` 一一对应（`basename` = 彩色图主文件名，不含扩展名）：


| 彩色图           | 深度图                 | 相机参数                       |
| ------------- | ------------------- | -------------------------- |
| `topdown.png` | `topdown_depth.png` | `topdown_camera_para.json` |
| `{stamp}.png` | `{stamp}_depth.png` | `{stamp}_camera_para.json` |


`render_view` 传入**相机轨迹序列**时，所有帧落在同一 `{timestamp}_seq/`（或 `auto_path_XXXX_seq/`）目录，例如：

```
auto_path_0023_seq/
├── 1735123456789.png
├── 1735123456789_depth.png
├── 1735123456789_camera_para.json    # depth_scale 仅适用于本帧深度
├── 1735123456790.png
├── 1735123456790_depth.png
├── 1735123456790_camera_para.json
└── ssl_opencv.txt
```

**重要：** 序列内**没有**全局统一的 `depth_scale`。每一帧在渲染时根据**该帧**有效深度的最大值单独计算编码倍率，并写入**该帧**的 `*_camera_para.json`。解码第 *i* 帧深度时，必须读取第 *i* 帧配对的 JSON，**不能**用首帧或其它帧的 `depth_scale` 代替。

#### 7.4.1.2 编码机制（导出时）

1. Cycles Z pass 得到米制深度图 `depth_m`（相机系，与 OpenCV +Z 深度一致）。
2. 取该帧有效像素（`depth_m > 0`）的最大值 `d_max`；若无有效像素则 `d_max` 退化为 1 m。
3. 设定映射上界 `n = d_max × 1.5`（留 50% 余量，避免远点贴边饱和）。
4. 编码倍率：`depth_scale = 65535 / n`（实现：`util.compute_depth_encode_scale`）。
5. 存盘：`pixel = round(depth_m × depth_scale)`，clamp 到 `[0, 65535]`；无效处写 `0`。

因此 `depth_scale` 是**每张深度图独有**的动态参数，随该帧场景最远可见距离变化；不同帧数值可以相同，但**不保证相同**。

#### 7.4.1.3 解码机制（读取时）

从配对 JSON 读取 `depth_scale` 与 `depth_unit`（恒为 `"meter"`）：

```
depth_m = pixel / depth_scale     # pixel 为 uint16 深度 PNG 单通道值
valid   = pixel > 0               # 0 表示无效，不应参与几何计算
```

**Python（单帧）：**

```python
import json
import imageio
import numpy as np
from scenebuilder.core import util

base = "1735123456789"  # 与 {base}.png 同名
with open(f"{base}_camera_para.json") as f:
    cam = json.load(f)

depth_u16 = imageio.imread(f"{base}_depth.png")
depth_m = util.decode_depth_uint16(depth_u16, cam["depth_scale"])  # (H, W), 米, OpenCV +Z
valid = depth_u16 > 0
```

**Python（序列批量，每帧各用各的 scale）：**

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
    # depth_m 为该帧相机系下的米制深度
```



#### 7.4.1.4 与反投影 / `ssl_opencv.txt` 的关系

解码得到的 `depth_m(u, v)` 是 OpenCV 相机系下该像素的 **+Z 深度（米）**。配合同帧 `{basename}_camera_para.json` 中的 `intrinsic`（§6.3）可反投影到相机系 3D 点：

```
X_cam = (u - cx) * depth_m / fx
Y_cam = (v - cy) * depth_m / fy
Z_cam = depth_m
```

再经 `c2w` 变换到 SSL 世界系：`P_world = c2w @ [X_cam, Y_cam, Z_cam, 1]ᵀ`。同目录 `ssl_opencv.txt`（§6.4）中的几何已在**序列首帧**（或单帧）OpenCV 相机系下表达，与深度反投影坐标系一致。

#### 7.4.1.5 `camera_para.json` 中的深度字段

```json
{
  "depth_unit": "meter",
  "depth_scale": 5811.535562,
  "is_metric_depth": true
}
```


| 字段                | 说明                                                  |
| ----------------- | --------------------------------------------------- |
| `depth_unit`      | 恒为 `"meter"`                                        |
| `depth_scale`     | **仅适用于与本 JSON 同 basename 的** `{basename}_depth.png` |
| `is_metric_depth` | 有深度导出时为 `true`                                      |




#### 7.4.2 法线图 `{view}_normal.png`（随 `--depth` 自动导出）


| 项目   | 说明                                                                        |
| ---- | ------------------------------------------------------------------------- |
| 格式   | **uint8 RGB PNG**（3 通道，与彩色图同分辨率、像素对齐）                                     |
| 来源   | Blender Cycles **Normal pass**（合成器写临时 EXR，再转 PNG）                         |
| 坐标系  | **OpenCV 相机系**（与 `{view}_depth.png` 一致；+X 右、+Y 下、+Z 沿视线向场景深处） |
| 无效像素 | 与深度对齐：对应 `{view}_depth.png` 像素为 `0` 时视为无效                                 |


同目录 `{basename}_camera_para.json` 在导出深度时会写入法线相关字段（`normal_space`、`normal_encode`、`normal_decode` 等），解码应以 JSON 为准。

#### 7.4.2.1 OpenCV 相机系具体指什么

法线三分量 `(nx, ny, nz)` 是在 **该视角 OpenCV 相机坐标系**下表达的单位向量（导出前对 Cycles Normal pass 做 `n_cam = normalize(R @ n_world)`，与 `*_opencv.ply` 法线、深度图同系）：


| 分量   | 正方向含义        |
| ---- | ------------ |
| `nx` | +X，图像**右**   |
| `ny` | +Y，图像**下**   |
| `nz` | +Z，沿视线**进场景**（与深度正方向一致） |


**与深度一致：** 深度 `{view}_depth.png` 与法线 `{view}_normal.png` **均在 OpenCV 相机系**；可直接用于单目法线监督/预测，无需再手动乘 `R`。

**与主 PLY 的区别：** 根目录或 `{view}/pointcloud/*.ply`（无 `_opencv`）的点坐标与法线仍为 **SSL 世界系**（§6.2）；仅 `_normal.png` 与 `*_opencv.ply` 为相机系。

#### 7.4.2.2 法线方向的具体定义

每个有效像素存的是：**该像素射线命中的、彩色图同一次渲染中可见的最前表面**上，Cycles 用于着色的**几何法线**（Normal pass，非 bump/normal map 贴图扰动），再变换到 OpenCV 相机系。

- **语义：** 单位向量 `n = (nx, ny, nz)`，指向该可见三角面在 mesh 上的外法线方向（Blender 按顶点 winding 计算），**相对当前相机轴**表达。
- **随视角而变：** 同一面墙在不同视角下，相机系法线分量不同；同一视角下看内侧面与外侧面时方向可相差约 180°。
- **常见直觉（随相机姿态变化，数值非固定）：**
  - 俯视图看地板上表面：法线大致指向**远离相机**（OpenCV 下 `nz` 常为负，即与世界 +Z 相反）。
  - 立面墙正对相机：法线主要在 `nx`/`ny` 平面内，`|nz|` 较小。
- **家具 GLB：** 随可见三角面朝向；与同视角 `scene_visible_opencv.ply` 中法线一致（若导出可见点云）。

#### 7.4.2.3 编码与解码（完整公式）

Cycles Normal pass 的 EXR 为世界系浮点向量（每分量 **[-1, 1]**）；写 PNG 前先变换到 OpenCV 相机系，再线性映射到 uint8：

**编码（导出时，**`util_bpy._encode_opencv_normal_png_from_cycles_exr`**）：**

```
n_cam = normalize(R @ n_world)     # R 与 depth / *_opencv.ply 相同
c = (n_cam + 1) / 2              # 每分量 ∈ [-1, 1] → c ∈ [0, 1]
pixel_channel = round(c * 255)
```

**解码（读取时，**`util.decode_normal_opencv_uint8` **或按 JSON 公式）：**

```
c = pixel_rgb / 255.0            # c ∈ [0, 1]
normal_opencv = c * 2.0 - 1.0  # 回到 [-1, 1]
# 可选：对 valid 像素再 normalize，修正量化误差
```

**示例（相机系，随视角变化；以下为示意）：**


| OpenCV 法线 `normal_opencv` | 编码后 RGB (约)       | 含义（相对当前相机）   |
| ------------------------- | ----------------- | ------------ |
| `(0, 0, +1)`              | `(128, 128, 255)` | 沿 +Z，指向场景深处 |
| `(0, 0, -1)`              | `(128, 128, 0)`   | 沿 −Z，指向相机   |
| `(+1, 0, 0)`              | `(255, 128, 128)` | 图像右侧         |
| `(0, +1, 0)`              | `(128, 255, 128)` | 图像下方         |


**无效 / 背景：** 无几何或 `depth==0` 的像素，Normal pass 可能为任意值。**必须**用深度掩码过滤，不要对无效像素做法线推理。

#### 7.4.2.4 转为 SSL 世界系（可选）

若需与 `ssl.txt`、主 PLY、`planar_faces.json` 对齐，可从 `{basename}_camera_para.json` 读取 `camera_position`、`look_at_target`、`world_up`，用 `core/geometry_opencv.py` 的 `opencv_rotation_from_pose` 得 `R`，再：

```
normal_world = (R.T @ normal_opencv.T).T   # 再 normalize
```

#### 7.4.2.5 `camera_para.json` 中的法线与标定字段

导出 `--depth` 时，除 `depth_unit` / `depth_scale` / `**c2w` / `intrinsic**`（§6.3）外还会写入法线字段：

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

`normal_decode` 与 `normal_encode` 互逆；程序内优先读 JSON，避免硬编码与文档不一致。

#### 7.4.3 Python 解码示例（深度 + 法线）

深度解算见 §7.4.1.3；此处演示同帧配对读取：

```python
import json
import imageio
import numpy as np
from scenebuilder.core import util

with open("topdown_camera_para.json") as f:
    cam = json.load(f)

depth_u16 = imageio.imread("topdown_depth.png")
depth_m = util.decode_depth_uint16(depth_u16, cam["depth_scale"])  # OpenCV +Z, 米

normal_u8 = imageio.imread("topdown_normal.png")
normal_cam = util.decode_normal_opencv_uint8(normal_u8)  # (H, W, 3), OpenCV 相机系

valid = depth_u16 > 0
nx = normal_cam[..., 0]  # +X 图像右
ny = normal_cam[..., 1]  # +Y 图像下
nz = normal_cam[..., 2]  # +Z 进场景（与 depth 一致）
normal_cam_valid = normal_cam.copy()
normal_cam_valid[~valid] = np.nan
```



#### 7.4.4 其他说明

- 透明遮挡物（自动透明的墙/天花/地板）在深度射线回退路径中跳过；Cycles pass 路径则与彩色图一致（透明处可见后方）
- **bpy + depth** 仍走每视角独立子进程（`render_ssl.worker_render_view`），避免合成器内存泄漏

---

### 7.5 体素 (`export_voxel` / `--voxel`)

体素有两条路径，共用 256³ 彩色占用场格式（详见下文），但数据源不同：

| 路径 | 触发条件 | 输出目录 | 数据源 |
| ---- | -------- | -------- | ------ |
| **各视角可见** | `--visible_geometry --voxel` | `{view}/voxel/` | 可见 + 视锥裁剪后的三角形；含 `occupancy_opencv.npz` |
| **根目录全场景** | `--holo_geometry --voxel` | `Y/voxel/` | 完整场景三角形；仅 `occupancy_world.npz` |

各视角体素**不是**从 GLB 读回，而与 `scene_visible.glb` 共用同一套内存中的可见三角形。根目录体素与 `scene.glb` 同级，由 `worker_render_post` 导出。

#### 各视角可见体素（触发条件）

```text
--visible_geometry 且 --voxel
```

可只开体素：`--visible_geometry --voxel`（不必 `--glb` / `--ply`）。

#### 根目录全场景体素（触发条件）

```text
--holo_geometry 且 --voxel
```

#### 输出文件


| 文件 | 说明 |
| ---- | ---- |
| `occupancy_world.npz` | 世界 SSL 系：`occupancy` uint8 `[256,256,256]` + `rgb` uint8 `[256,256,256,3]` |
| `occupancy_world_meta.json` | 网格几何与索引约定（见下） |
| `occupancy_world_preview.ply` | 占用格中心彩色点云，便于 MeshLab / CloudCompare / Blender 预览 |
| `occupancy_opencv.npz` | OpenCV 相机系占用场（同上结构） |
| `occupancy_opencv_meta.json` | 相机系 meta（含变换后的 `world_up`） |
| `occupancy_opencv_preview.ply` | 相机系预览点云 |
| `metadata_voxel.json` | 汇总索引 |


#### 网格定义

- **分辨率**：固定 `256³`
- **立方体跨度**：`span = max(x_extent, y_extent, z_extent)`（该坐标系下可见几何 AABB 最长轴）
- **格子边长**：`voxel_size = span / 256`
- **中心**：AABB 几何中心；meta 同时给出 `corner_min`
- **索引 → 3D 点**（体素中心）：

```text
p(i,j,k) = corner_min + (i + 0.5, j + 0.5, k + 0.5) * voxel_size    # i,j,k ∈ [0,255]
```

- **每格数据**：`occupancy` 0=空 / 1=占用；占用格 `rgb` 来自三角形材质/贴图采样色；空格 RGB 可忽略

#### meta 主要字段

`format`, `version`, `coordinate_space`, `center`, `voxel_size`, `span`, `corner_min`, `world_up`, `world_up_space`, `index_origin`, `index_to_point`, `occupied_count`, `voxelization`

#### 读取示例

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

#### 可视化

| 方式 | 说明 |
| ---- | ---- |
| `*_preview.ply` | 导出时自动生成，直接打开即可 |
| Python | `np.load` + Open3D / matplotlib 3D scatter |
| 体素方块 | 按 meta 在 Blender 中实例化 unit cube |

**体积提示**：单套 npz 未压缩约 **50–65 MB**（256³ dense）；世界 + 相机两套约翻倍。默认 **不 gzip 压缩**（meta 里 `compressed: false`），避免每视角额外等待；仍会导出 `*_preview.ply`。

**耗时提示**（`--voxel` 每视角额外）：体素化约 5–30s + 写盘约 3–10s；日志会打印 `体素化 Xs, 写盘 Ys`。

---



## 8. 配置与后端



### 8.1 配置文件 (`config.yaml`)

统一管理：画布尺寸、墙厚、默认 FOV、HDRI 路径、家具模型库路径、光照强度、nav_mask 参数（§5.2）等。可通过 `BpySceneCtx` / `SceneCtx` 的 `config` 属性读取。

### 8.2 版本对比


| 特性       | scenebuilder (Pyrender) | scenebuilder_bpy (Blender)             |
| -------- | --------------------- | ------------------------------------ |
| 渲染品质     | 基础 OpenGL             | PBR (Eevee/Cycles)                   |
| 转角处理     | 简单重叠                  | 斜接修正 (Miter Joint)                   |
| 遮挡处理     | 简单裁切                  | 智能半透明材质                              |
| 可见几何     | 支持（手动 FOV 平面）         | 支持（`calc_matrix_camera` 对齐渲染）        |
| 可见体素     | -                     | 支持（256³ 彩色占用场，`--voxel`）              |
| 平面内表面顶点  | -                     | 支持（`--ply` 自动导出 JSON + 连线图）          |
| 语义图      | 支持                    | 支持（proxy + Raw）                      |
| 深度图      | Pyrender 深度缓冲         | Cycles Z pass + ray_cast 回退          |
| 法线图      | -                     | Cycles Normal pass（随 `--depth` 同次渲染） |
| 深度 + 多视角 | 单进程                   | 每视角独立子进程（防内存泄漏）                      |


---



## 9. 资产处理（`get_mesh`）

`render_ssl` 会先将 SSL 解析为 JSON，再调用 `get_mesh` 处理资产（检索或生成），最后交给渲染引擎。

- **默认**：`asset_mode="none"`，不修改 JSON，直接使用 SSL 中的 `asset_id` 占位
- **检索**：需先构建 LanceDB（见下）



### 9.1 检索分支



#### 数据库构建

```bash
python build_lancedb.py
```

- 存储位置：当前工作目录下 `manycore/`
- `furniture` 表首次构建约需 16 小时（全量 Embedding）
- 需配置 `Qwen3VLEmbedder` 路径（`util_data.py`）



#### 检索逻辑

**门/窗：** 按 `width`×`height` L2 距离匹配最近模型。

**家具 (Bbox)：**

1. **有图像**（`retrieve` + `image_path`）：按 `mesh_id` 分组，bbox_2d 裁剪 + label/caption 多模态 Embedding
2. **无图像**：label + caption 文本 Embedding，Batch 并行



### 9.2 生成分支（`asset_mode="generate"`）

- 需提供 `image_path`
- 按 `mesh_id` 分组 → 裁剪 → 扩图/超分 → 3D 生成（Hunyuan 等）
- 模型保存至 `asset_dir`（默认 `/data-nas/data/dataset/qunhe/Manycore-Future/generate/`）

---



## 附录 A. 更多调用示例

与 **§3.3**、**§5.2** 等价的一行命令：

```bash
# 全量 benchmark（同 §3.3）
python scenebuilder/render_ssl.py --ssl path/to/ssl.txt --views auto --output out \
  --glb --assets path/to/assets --ply --visible_geometry --voxel --semantic --depth --pano

# 仅像素对齐 + 地板路径（§5.2）
python scenebuilder/render_ssl.py --ssl path/to/ssl.txt \
  --normalized_topdown --output out_normalized --assets path/to/assets

# 静态多视角
python scenebuilder/render_ssl.py --ssl path/to/ssl.txt \
  --views topdown left_seq --output out --glb --semantic --depth
```

---



## 附录 B. 视锥裁剪与 Blender 原生 API（实现备忘）

bpy 路径下，**可见几何**（`visible_geometry`）与**平面内表面顶点**（`planar_faces.json` / `*_lines.png`）都依赖「当前渲染相机能看到什么、裁切后顶点在哪」。这块多次出问题的共同根因是：**没有用与 Cycles 实际渲染一致的投影矩阵**。

### 正确做法（当前实现）

与 Blender 渲染对齐，使用：

1. `**camera.calc_matrix_camera(depsgraph, x, y, ...)**` 获取投影矩阵
2. **相机** `matrix_world.inverted()` 作为 modelview
3. 在 **齐次 clip space** 中对三角形/多边形环做 Sutherland-Hodgman 裁剪
4. 沿边插值时 **同时插值** `clip` **与** `world`，保证裁切后的交点仍在原平面（墙/门/窗/地板/天花）上

可见几何的三角形裁剪（`scenebuilder_bpy._clip_triangle_to_render_frustum`）与平面顶点的多边形裁剪（`util_bpy.clip_polygon_to_render_frustum`）均遵循上述流程。

**遮挡标记** `occluded` 与视锥无关，使用 `scene.ray_cast`（与可见几何物体级遮挡判定相同），透明墙/天花/地板可穿透。

### 错误做法（已废弃，勿再使用）


| 做法                                                      | 问题                                     |
| ------------------------------------------------------- | -------------------------------------- |
| 手写视锥平面 / 手动 FOV 切平面                                     | 与 Blender 渲染视锥不对齐，俯视图/边缘物体易误判 cutted   |
| `world_to_camera_view` 得 `(u,v,z)` 后在屏幕空间裁切，交点用 3D 线性插值 | 透视下交点**脱离原平面**；部分在视锥内时墙/天花顶点坐标错乱、连线图自交 |
| 按质心角度对多边形顶点重排                                           | 破坏 L 形/凹多边形边界拓扑（地板顺序错误）                |




### 其他独立问题（非视锥 API）

- **环起点规则**：规范起点为 z 最小 → y 最小 → x 最小；仅旋转起点，不重排边界顺序  
- **cutted 统计**：仅当三角形**三顶点都在**视口 `[0,1]×[0,1]` 时才算「在视锥内」；面片占比 < 95% 才标 `_cutted`



### 经验结论

- **判断点是否在视锥内（统计/命名）**：可用 `world_to_camera_view`，阈值与渲染视口 `[0,1]×[0,1]`、`z>0` 一致  
- **裁切几何并求 3D 交点**：必须用 `**calc_matrix_camera` + clip space**，不能用屏幕 `(u,v)` 反推 3D  
- **导出多边形顶点顺序**：保留 SSL/房间 `vertices` 或 mesh 定义的边界顺序，只做 CCW 整体反转 + 起点旋转

俯视图因物体全在视锥内，旧错误裁剪不易暴露；**透视视角（如 left）部分裁切**时，非原生 API 裁剪的问题会非常明显。