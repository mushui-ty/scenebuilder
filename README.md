# Fast Scene 场景渲染工具库

从 SSL / JSON 描述（墙体、门窗、家具）构建 3D 室内场景，支持 **Blender (bpy)** 与 **Pyrender** 双后端渲染、多视角导出、语义图、深度图、可见几何与平面内表面顶点。

> SSL 世界坐标约定（+Y 为俯视图上方、各实体字段含义）见 **§2**。

---

# 一. 渲染模块

## 1. 安装与依赖

### Python 版本要求: 3.11

不是 3.11 无法安装最新版本的 `bpy`。

### 基础依赖

```bash
pip install numpy shapely scipy imageio pyyaml
```

### 渲染后端

| 后端 | 安装 | 适用场景 |
|------|------|----------|
| **bpy (推荐)** | `pip install bpy` | PBR 渲染、语义图、Cycles 深度、可见几何 |
| **pyrender** | `pip install pyrender trimesh` | 轻量 OpenGL 渲染、无 bpy 环境 |

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
cd fast-scene
pip install -e . --config-settings editable_mode=strict
```

---

## 2. SSL 坐标系与实体约定

库内所有 SSL 解析、场景 `context`、渲染与导出（含 `ssl.txt`、点云命名）共用同一套**世界坐标系**，单位为**米 (m)**。

### 2.1 坐标轴

俯视房间（topdown）时：

| 轴 | 正方向 | 俯视图中的方向 |
|----|--------|----------------|
| **X** | +X | 右 |
| **Y** | +Y | **上** |
| **Z** | +Z | 竖直向上（高度） |

- 地面在 **XY 平面**，一般取 **Z = 0** 为地面高度。
- 家具朝向 **`angle_z`**：`0°` 时物体面向 **−Y**（俯视图**下方**）；绕 **Z 轴逆时针**旋转角度增大（右手定则）。

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

| 字段 | 含义 |
|------|------|
| `p`, `q` | 墙段起点、终点；第三维写 `0` 即可，仅 XY 有效 |
| `height` | 墙高，从 `z=0` 到 `z=height` |

**Door / Window** — 门洞 / 窗洞，挂在某面墙上：

```ssl
Door(label="door0", center=[x, y, z], width=2.32, height=2.3, wall="wall2", asset_id="61064792")
Window(label="window0", center=[x, y, z], width=2.37, height=2.2, wall="wall0")
```

| 字段 | 含义 |
|------|------|
| `center` | 洞口的**几何中心**（3D）。例如 `height=2.3, z=1.15` 表示中心在高度中间，底边约在 `z ≈ 0` |
| `width` | 沿**墙走向**的宽度 |
| `height` | 沿 **Z 轴**的高度 |
| `wall` | 所属墙的 `label`（如 `wall0`） |
| `asset_id` | 可选；缺失时仍挖洞，但不加载门/窗模型 |

加载时若 `center` 的 XY 不在墙线上，会吸附到最近墙的垂足。

**Bbox** — 家具或有向包围盒：

```ssl
Bbox(label="sidetable0", center=[1.96, 0.21, 0.31], angle_z=180, scale=[0.42, 0.42, 0.63], asset_id="56056912")
```

| 字段 | 含义 |
|------|------|
| `center` | 旋转后 OBB 的**几何中心** |
| `angle_z` | 绕 Z 轴旋转角（度），逆时针为正 |
| `scale` | 局部 XYZ 三轴上的**全长**（非半长）。落地物体通常 `center.z ≈ scale.z / 2` |
| `asset_id` | 对应 `{asset_id}.glb`；不存在时该 bbox 会被剔除 |

GLB 加载流程：模型居中 → 按 `scale` 缩放 → 绕 Z 转 `angle_z` → 平移到 `center`。

### 2.3 与渲染视角的关系

`render_ssl` 默认透视相机（如 `front`）放在 **−Y 一侧**（`y = center_y − span_y/3`），朝向房间中心，即朝 **+Y** 看进房间。这与「+Y 在俯视图上方、`angle_z=0°` 朝 −Y」一致。

`topdown` 相机从 **+Z** 向下俯视 XY 平面。

### 2.4 标准 SSL 输出

经 `normalize_scene_data()` 整理后，输出的 `ssl.txt` 使用规范 `label`（`wall0`, `door0`, `sidetable0` …），去掉原始 `id` / `room_id`；字段语义与上述约定相同。

---

## 3. Quickstart

### 3.1 渲染 SSL（推荐入口）

```python
from fast_scene.render_ssl import render_ssl

with open("ssl.txt", "r", encoding="utf-8") as f:
    ssl_text = f.read()

output_dir, updated_ssl = render_ssl(
    ssl_text,
    backend="bpy",
    output_root="output",
    views=["topdown", "front"],   # None 或省略 = 全部视角
    export_glb=True,
    export_point_cloud=True,
    visible_geometry=True,        # 每个视角额外导出可见几何
    semantic=True,
    depth=True,
    asset_dir="/path/to/assets",
    texture_dir="/path/to/texture",  # 含 floor/wall/ceiling_texture.png
)
```

**命令行（推荐，直接调用 `render_ssl.py`）：**

```bash
python fast_scene/render_ssl.py --ssl path/to/ssl.txt \
  --views topdown left_seq \
  --output path/to/output \
  --glb --ply --visible_geometry --semantic --depth \
  --texture path/to/texture --assets path/to/assets
```

| CLI 参数 | 对应 `render_ssl()` 参数 | 说明 |
|----------|--------------------------|------|
| `--ssl` | `input_text`（从文件读取） | SSL 或 JSON 场景文件 |
| `--output` | `output_root` | 输出目录；省略时为 ssl 同目录下 `render_output` |
| `--views` | `views` | 视角列表，或 `all`（默认） |
| `--backend` | `backend` | `bpy`（默认）或 `pyrender` |
| `--assets` | `asset_dir` | 3D 资产目录；省略时尝试 ssl 旁的 `assets/` 等 |
| `--texture` | `texture_dir` | 贴图目录 |
| `--glb` | `export_glb=True` | 导出全场景 `scene.glb` |
| `--ply` | `export_point_cloud=True` | 导出全场景点云 + 各视角平面顶点 |
| `--visible_geometry` | `visible_geometry=True` | 各视角可见 GLB/PLY（需配合 `--glb` 或 `--ply`） |
| `--semantic` | `semantic=True` | 语义分割图 |
| `--depth` | `depth=True` | 深度图 + 法线图（bpy） |
| `--view_transform` | `view_transform=True` | 渲染前变换到各视角 SSL 坐标系 |
| `--normalized_topdown` | — | 像素对齐俯视图模式（调用 `render_normalized_topdown`） |
| `--no_floor_path` | — | 配合 `--normalized_topdown`：跳过地板路径采样 |
| `--samples` | `samples` | Blender 采样数 |

可用视角：`topdown`, `front`, `behind`, `left`, `right`, `leftfront`, `rightfront`, `leftbehind`, `rightbehind`, `left_seq`, `all`

> 历史包装脚本 `SpatialFactory/scripts/render_scene.py` 的静态视角渲染与上述 CLI 等价；新用法请直接调用本库 `fast_scene/render_ssl.py`。

### 3.2 直接调用 Context API

```python
from fast_scene.core.fast_scene_bpy import BpySceneCtx

ctx = BpySceneCtx(room_type="living_room", gen_asset_dir="/path/to/assets")
ctx.add_walls(walls_data)
ctx.add_doors(doors_data)
ctx.add_windows(windows_data)
ctx.add_boxes(furniture_data)

ctx.topdown_view(
    "topdown/topdown.png",
    show_ceiling=False,
    rebuild=True,
    render_depth=True,
    render_semantic=True,
    visible_geometry=True,
    export_glb=True,
    export_point_cloud=True,
)

ctx.render_view(
    "view.png",
    camera_position=[6, 4, 2],
    look_at_target=[0, 0, 1],
    visible_geometry=True,
    export_glb=True,
)
```

---

## 4. `render_ssl` API

```python
def render_ssl(
    input_text: str,                # SSL 文本或 JSON 字符串
    backend: str = "bpy",           # "bpy" | "pyrender"
    output_root: str = "output_ssl",
    image: Optional[str] = None,    # retrieve/generate 模式用
    retrieve_hole: bool = True,
    asset_mode: Literal["none", "retrieve", "generate"] = "none",
    outpaint_image_dir: Optional[str] = None,
    asset_dir: Optional[str] = None,
    gen_3d_model: Literal["hunyuan-3d-rapid", "hunyuan-3d-pro"] = "hunyuan-3d-pro",
    gen_texture: bool = False,
    texture_dir: Optional[str] = None,
    correct_tilt: bool = True,
    correct_yaw: bool = True,
    views: Optional[list] = None,   # None = 全部视角
    export_glb: bool = False,       # 导出全场景 scene.glb
    export_point_cloud: bool = False,  # 导出全场景点云到 output_root/pointcloud/
    visible_geometry: bool = False, # 每个视角额外导出可见 GLB/点云
    semantic: bool = False,         # 每个视角导出语义图
    depth: bool = False,            # 每个视角导出深度图 + 法线图（bpy）
    view_transform: bool = False,   # 渲染前变换到各视角 SSL 坐标系
    samples: Optional[int] = None,  # Blender 采样数
)
```

命令行入口：`python fast_scene/render_ssl.py --help`（参数见 §3.1 表格）。

| 参数 | 说明 |
|------|------|
| `export_glb` | 在 `output_root/scene.glb` 导出**完整场景**（与视角无关） |
| `export_point_cloud` | 在 `output_root/pointcloud/` 导出**完整场景**点云；并在**每个视角目录**自动导出平面内表面顶点 JSON 与连线图（见 §7） |
| `visible_geometry` | 需配合 `--glb` 或 `--ply`：在**每个视角目录**下额外导出该视角可见几何（GLB/可见点云） |
| `semantic` | 每个视角输出 `{view}_semantic.png` + `{view}_semantic.json` |
| `depth` | 每个视角输出 `{view}_depth.png` + `{view}_normal.png`（bpy：Cycles Z/Normal pass，同一次渲染） |
| `texture_dir` | 优先使用外部 `floor/wall/ceiling_texture.png`；否则 `gen_texture=True` 时内部生成 |

**bpy + depth 特殊行为：** 为避免 Cycles 合成器内存泄漏，每个视角在 `render_ssl.py` 内通过独立子进程（`worker_render_view`）渲染，全部视角完成后由 `worker_render_post` 导出全场景 GLB/点云。

---

## 5. 输出目录结构

以 `output_root/` 为例：

```
output_root/
├── data.json                 # 场景 JSON（世界 SSL）
├── ssl.txt                   # 场景级标准 SSL（世界坐标，未做视角变换）
├── scene.glb                 # [--glb] 完整场景（世界 SSL）
├── pointcloud/               # [--ply] 完整场景点云（世界 SSL，post 阶段导出）
│   ├── metadata.json
│   ├── scene_all.ply
│   ├── scene_all_opencv.ply  # 若 post 传入相机位姿则生成（少见）
│   ├── floor.ply
│   ├── walls/wall0.ply
│   ├── doors/door0_{asset_id}.ply
│   ├── windows/window0_{asset_id}.ply
│   └── boxes/sidetable0_{asset_id}.ply
├── topdown/                  # 俯视图（固定目录名）
│   ├── ssl.txt               # [--view_transform] 或默认视角 SSL 文本
│   ├── topdown.png
│   ├── topdown_depth.png     # [--depth]
│   ├── topdown_normal.png    # [--depth] 世界空间法线（与深度同次渲染）
│   ├── topdown_semantic.png  # [--semantic]
│   ├── topdown_semantic.json
│   ├── topdown_lines.png     # [--ply] 平面内表面顶点连线 overlay
│   ├── planar_faces.json     # [--ply] 内表面顶点（世界 SSL 3D 坐标）
│   ├── camera_para.json      # 含 world + *_view 字段（见 §5.1）
│   ├── scene_visible.glb     # [--visible_geometry + --glb] 视角 SSL
│   ├── scene_visible_opencv.glb  # 同上，OpenCV 相机系副本
│   └── pointcloud/           # [--visible_geometry + --ply]
│       ├── metadata_visible.json   # objects[].path / path_opencv
│       ├── scene_visible.ply       # 视角 SSL（主文件）
│       ├── scene_visible_opencv.ply  # OpenCV 相机系副本
│       ├── floor_visible[_cutted].ply
│       ├── floor_visible[_cutted]_opencv.ply
│       ├── walls/wall0_visible[_cutted].ply
│       ├── walls/wall0_visible[_cutted]_opencv.ply
│       └── boxes/sidetable0_{asset_id}_visible[_cutted].ply
│           └── …_opencv.ply    # 每个 PLY 均有对应 _opencv 副本
└── {毫秒时间戳}/              # 单视角或 left_seq 等多帧序列
    ├── ssl.txt               # 视角 SSL（序列以首帧定义坐标系）
    ├── {stamp}.png           # 单视角一张；序列多帧
    ├── {stamp}_depth.png
    ├── {stamp}_normal.png    # [--depth]
    ├── {stamp}_semantic.png
    ├── {stamp}_lines.png     # [--ply]
    ├── planar_faces.json     # [--ply]
    ├── camera_para.json
    ├── scene_visible.glb
    ├── scene_visible_opencv.glb
    └── pointcloud/           # 可见点云：主文件 + _opencv 副本（结构同 topdown/）
```

### 5.1 几何导出坐标系（视角 SSL 与 OpenCV）

开启 `--glb` / `--ply` 且（对可见几何）`--visible_geometry` 时，**每个视角目录**下的 GLB/PLY 会成对导出：**主文件（视角 SSL）** + **`_opencv` 副本（OpenCV 相机系）**。实现见 `core/geometry_opencv.py`。

| 坐标系 | 适用文件 | 原点与轴向 |
|--------|----------|------------|
| **世界 SSL** | 根目录 `ssl.txt`、`data.json`；`worker_render_post` 的 `scene.glb`、`pointcloud/scene_all.ply` | 与 §2 一致：+X 右、+Y 上、+Z 高 |
| **视角 SSL** | `{视角}/pointcloud/*.ply`（无 `_opencv`）、`scene_visible.ply`、`scene_visible.glb` | 原点：相机位置到地面的垂足；+Y：视线在地面上的投影；+Z：竖直向上 |
| **OpenCV 相机系** | 上述每个几何文件的 `*_opencv.ply` / `*_opencv.glb` | 原点：相机光心；+X：图像右；+Y：图像下；+Z：沿视线向场景深处（深度为正） |

**约定：**

- 主 PLY/GLB 先从 Blender 世界坐标变换到**该视角的视角 SSL**，再写盘。
- `_opencv` 副本在**视角 SSL 坐标**下，再变换到以**首帧参考相机**为原点的 OpenCV 系（`camera_position_view = (0,0,h)`，`look_at_target_view = (0,x,e)`）。
- **`left_seq` 等多帧序列**：OpenCV 副本统一以**序列首帧**相机为参考系；可见几何为各帧视锥并集。
- `metadata_visible.json` / `metadata.json` 中可为合并点云记录 `"path_opencv": "scene_visible_opencv.ply"` 等字段。
- `planar_faces.json`、深度/语义 PNG 仍用**世界 SSL** 或与渲染图对齐的像素坐标，不生成 `_opencv` 副本。

**`camera_para.json`（每视角）** 同时记录世界与视角 SSL 相机位姿，例如：

```json
{
  "camera_position": [wx, wy, wz],
  "look_at_target": [wx, wy, wz],
  "world_up": [0, 0, 1],
  "camera_position_view": [0, 0, h],
  "look_at_target_view": [0, x, e],
  "world_up_view": [...],
  "view_origin_xy": [a, b],
  "view_theta": 0.42,
  "fov_y": 1.2,
  "aspectRatio": 1.0,
  "depth_scale": 1000.0
}
```

- 单视角 / topdown / 序列首帧：`camera_position_view = (0,0,h)`，`look_at_target_view = (0,x,e)`。
- 序列后续帧：在同一视角 SSL 基底下写入该帧完整 `*_view` 坐标。
- 加 `--view_transform` 时还会在 `{视角}/ssl.txt` 写出变换后的场景 SSL，且在该视角 SSL 下 construct / 渲染 / 导出几何。

**三类点云的区别：**

| 路径 | 坐标系 | 内容 |
|------|--------|------|
| `output_root/pointcloud/` | 世界 SSL | 场景内**所有**物体（post 阶段，与视角无关） |
| `{视角}/pointcloud/*.ply` | 视角 SSL | 该视角**可见**物体（需 `--visible_geometry`） |
| `{视角}/pointcloud/*_opencv.ply` | OpenCV（首帧相机） | 与上一行相同几何，换到 OpenCV 相机系 |
| `{视角}/planar_faces.json` | 世界 SSL（3D 顶点） | 墙/门/窗/地板/天花内表面（需 `--ply`） |

**命名与 context 规范（`ssl.txt`、context 键、点云文件名一致）：**

- 墙：`wall0`, `wall1`, … → `walls/wall0.ply`（无 `asset_id`）
- 门/窗：`door0`, `window0`, … → `doors/door0_{asset_id}.ply`；若 `asset_id` 在资产目录不存在则**保留空洞**、去掉 `asset_id`，文件名与 SSL 均不含 `asset_id`
- 家具：`sidetable0`, `armchair0`, … → `boxes/sidetable0_{asset_id}.ply`；若 `asset_id` 不存在则**删除该 bbox**

可见几何后缀：`_visible`（整体在视锥内）、`_visible_cutted`（被视锥裁切，见下）。

### `_cutted` 后缀含义

**`_cutted` 表示该物体被当前相机视锥「切过一刀」**——物体仍被导出，但 mesh / 点云里只保留视锥内的部分；文件名用 `_cutted` 标记「原始物体并未完整落在画面内」。

| 后缀 | 含义 | 典型场景 |
|------|------|----------|
| `_visible` | 物体 **≥ 95%** 的三角形三顶点都在渲染视口 `[0,1]×[0,1]` 内，视为整体在画面里 | 房间中央的桌子、完全入镜的墙段 |
| `_visible_cutted` | 上述「完全在视口内」的三角形占比 **< 95%**，视锥边界切到了物体 | 画面边缘的窗、被裁掉一半的墙、只露出一角的家具 |

**判定与导出是两步：**

1. **遮挡剔除（物体级）**：被其它物体完全挡住 → **整物体不导出**（不会出现 `_visible` 或 `_cutted`）。
2. **视锥裁剪（三角形级）**：保留的物体按 Blender 渲染视锥裁掉视锥外三角形；裁完若还有几何则导出。
3. **命名**：根据裁剪**前** mesh 的「完全在视口内」面片占比是否 ≥ 95%，决定文件名用 `_visible` 还是 `_visible_cutted`。

示例：

```
walls/wall0_visible.ply          # 整面墙基本都在画面内
walls/wall2_visible_cutted.ply   # 墙的一部分在画面外，被视锥切掉
walls/wall2_visible_cutted_opencv.ply  # 同上，OpenCV 坐标副本
```

`metadata_visible.json` 里每个 object 有 `"frustum_cutted": true/false` 与 `"frustum_in_view_ratio"`（0–1），与文件名一致。

**注意：** `_cutted` 只描述**视锥裁剪**，与是否被遮挡无关；被挡死的物体直接不出现在导出结果中。

---

## 6. 可见几何 (`visible_geometry`)

开启 `--visible_geometry` 且同时开启 `--glb` 或 `--ply` 时，每个视角目录下导出 `scene_visible.glb` / `pointcloud/`；每个几何文件另有 `_opencv` 副本（见 §5.1）。

### 处理流程（按顺序）

1. **遮挡可见性（物体级，基于完整 mesh）**
   - 对物体表面采样点做射线检测（透明墙/天花/地板可穿透）
   - **完全被挡** → 丢弃该物体
   - **部分可见或全部可见** → 保留，进入下一步

2. **视锥裁剪（三角形级）**
   - 使用 Blender `calc_matrix_camera` 投影矩阵，在 clip space 做几何裁剪
   - 切掉视锥外的三角形或三角形部分
   - 裁剪后无几何 → 丢弃该物体

3. **命名规则（`_visible` / `_cutted`）**
   - 详见上文 **「`_cutted` 后缀含义」**
   - 统计时仅 **三顶点都在** 视口 `[0,1]×[0,1]` 的面片算「在视锥内」
   - 占比 **≥ 95%** → `_visible`；**< 95%** → `_visible_cutted`
   - `metadata_visible.json` 含 `"frustum_cutted"` 与 `"frustum_in_view_ratio"`

### 适用类别

floor、ceiling、walls、doors、windows、boxes 均参与可见性判定与视锥裁剪。

---

## 7. 平面内表面顶点 (`export_point_cloud` / `--ply`)

开启 `--ply` 时，**每个视角**在渲染完成后自动额外导出（无需单独 flag）：

| 文件 | 说明 |
|------|------|
| `planar_faces.json` | 墙/门/窗/地板/天花内表面多边形顶点 |
| `{view}_lines.png` | 在渲染图副本上绘制顶点连线（每对象一色） |

### 几何定义

与 fast_scene 建 mesh 逻辑一致：

- **墙**：SSL 中 `p`/`q` 即内墙底两点，高度为 `align_height ? z_max : wall.height`；带门/窗洞时 JSON 含 `outer` 环与 `hole` 环
- **门/窗**：按 `center`、`width`、`height` 及所属墙计算内面四顶点（同 `create_door_or_window_mesh`）
- **地板**：房间 `vertices` 多边形 @ z=0
- **天花**：房间 `vertices` 多边形 @ z=z_max（仅 `show_ceiling=True` 且已构建时）

### 视锥裁剪与顶点顺序

- 多边形在 **Blender `calc_matrix_camera` 齐次 clip space** 裁剪（与可见几何相同），沿边插值 world 坐标以保持共面
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

## 8. 语义图 (`semantic`)

- 每个实体独立颜色（`entity:{category}:{id}` HSV 哈希），每面墙不同色
- 颜色不与背景 `(0,0,0)` 相同
- bpy：创建 proxy mesh + Emission 材质，`view_transform=Raw`，关闭 HDRI/灯光后单独渲染
- 输出：`{view}_semantic.png` + `{view}_semantic.json`（含 entity 与 RGB 映射）

---

## 9. 深度图与法线图 (`depth`)

开启 `--depth` 时，**bpy + Cycles** 在**同一次渲染**中通过合成器同时导出深度与法线（无额外渲染时间）。`pyrender` 后端仅导出深度，不含法线。

### 9.1 深度图 `{view}_depth.png`

| 项目 | 说明 |
|------|------|
| 格式 | **uint16 单通道 PNG**（非 EXR） |
| 含义 | 相机坐标系下沿视线方向的**米制距离**（与 Cycles Z pass 一致，近大远小） |
| 编码 | `pixel = round(depth_m * depth_scale)`，clamp 到 `[0, 65535]` |
| 解码 | `depth_m = pixel / depth_scale` |
| `depth_scale` | 写入同目录 `camera_para.json` 的 `depth_scale` 字段；由场景有效最大深度动态计算（`max(depth)*1.5` 映射到 65535） |
| 无效像素 | `depth_m == 0`（背景、透明区域、超出 clip 等） |

**bpy 实现：** Cycles 合成器 **Z pass** → 临时 EXR → 转 uint16 PNG。失败时回退逐像素 `ray_cast`（此时**不**导出法线）。

### 9.2 法线图 `{view}_normal.png`（随 `--depth` 自动导出）

| 项目 | 说明 |
|------|------|
| 格式 | **uint8 RGB PNG**（3 通道） |
| 坐标系 | **世界空间**（World Space），与 Blender Cycles **Normal pass** 一致 |
| 含义 | 每个像素处**可见最前表面**的单位法向量 `n = (nx, ny, nz)`，方向指向房间外侧/表面外法线（随 mesh 朝向，与光照计算用法线一致） |
| 编码 | Cycles 输出每通道 `c in [0, 1]`，存盘为 `pixel_channel = round(c * 255)` |
| 解码 | `normal_world = (pixel_rgb / 255.0) * 2.0 - 1.0`，得到 `[-1, 1]` 三分量；可按需再 `normalize` |
| 无效像素 | 与深度对齐：当对应 `depth` 像素为 `0` 时视为无效，不应依赖法线值 |
| 背景 | 无几何处 Normal pass 可能为 `(0.5, 0.5, 1.0)`（解码为 `(0,0,1)`），**务必用 depth 掩码过滤** |

**为何是世界空间而非相机空间：** Cycles Normal pass 原生即为世界空间；便于与 SSL/场景坐标、平面顶点 JSON 的 3D 坐标直接对比。若需要相机空间法线，可在解码后左乘相机旋转的逆：`normal_cam = R_cw @ normal_world`。

**`camera_para.json` 元数据（有深度时一并写入）：**

```json
{
  "depth_unit": "meter",
  "depth_scale": 5811.535562,
  "normal_space": "world",
  "normal_encoding": "uint8_rgb",
  "normal_decode": "normal_world = (pixel_rgb / 255.0) * 2.0 - 1.0",
  "normal_invalid_mask": "depth_pixel == 0"
}
```

### 9.3 Python 解码示例

```python
import imageio
import numpy as np
from fast_scene.core import util

depth_u16 = imageio.imread("topdown_depth.png")
depth_scale = 5811.535562  # from camera_para.json
depth_m = depth_u16.astype(np.float64) / depth_scale

normal_u8 = imageio.imread("topdown_normal.png")
normal_world = util.decode_normal_world_uint8(normal_u8)  # (H, W, 3)

valid = depth_m > 0
nx, ny, nz = normal_world[..., 0], normal_world[..., 1], normal_world[..., 2]
# 仅对 valid 像素使用法线
```

### 9.4 其他说明

- 透明遮挡物（自动透明的墙/天花/地板）在深度射线回退路径中跳过；Cycles pass 路径则与彩色图一致（透明处可见后方）
- **bpy + depth** 仍走每视角独立子进程（`render_ssl.worker_render_view`），避免合成器内存泄漏

---

## 10. 核心渲染参数（`topdown_view` / `render_view`）

两函数共享大部分参数。

### 共有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_path` | `str` | - | 输出图像路径 |
| `width` / `height` | `int` | `1024` | 分辨率 |
| `geometry_mode` | `str` | `"gltf"` | `"gltf"` / `"mixed"` / `"bbox"` |
| `lighting_type` | `str` | `"array"` | `"array"` / `"area"` / `"none"` |
| `visible_shadow` | `bool` | `True` | 墙/天花是否投射阴影 |
| `align_height` | `bool` | `True` | 墙体高度对齐到 `z_max` |
| `rebuild` | `bool` | `False` | 强制重建场景（改材质/光照后建议 `True`） |
| `show_wall/window/door/ceiling` | `bool` | `True` | 各元素可见性 |
| `auto_fov` | `bool` | `True` | 自动计算 FOV |
| `manual_fov` | `float` | `None` | 手动 FOV（角度制），覆盖 `auto_fov` |
| `auto_transparent` | `bool` | `True` | 自动透明化挡在相机前的墙体 |
| `transparent_alpha` | `float` | `0.0` | 透明墙 Alpha |
| `use_HDRI` | `bool` | `True` | 启用 config 中 HDRI |
| `hdri_transparent_background` | `bool` | 俯视图 `True`，其他 `False` | HDRI 光照保留、背景透明 |
| `render_depth` | `bool` | `False` | 导出 `{view}_depth.png` + `{view}_normal.png`（bpy Cycles pass） |
| `render_semantic` | `bool` | `False` | 导出语义图 |
| `export_glb` | `bool` | `False` | 在该视角目录导出 `scene_visible.glb`（需 `visible_geometry=True`） |
| `export_point_cloud` | `bool` | `False` | 在该视角目录导出可见点云（需 `visible_geometry=True`）；同时自动导出 `planar_faces.json` 与 `{view}_lines.png` |
| `visible_geometry` | `bool` | `False` | 启用可见几何导出流程（GLB / 可见点云） |

### `topdown_view` 建议

- `show_ceiling=False`
- `hdri_transparent_background=True`

### `render_view` 独有参数

| 参数 | 说明 |
|------|------|
| `camera_position` | 相机坐标 `[x, y, z]`（必填） |
| `look_at_target` | 观察目标点（默认房间中心） |

---

## 11. 版本对比

| 特性 | fast_scene (Pyrender) | fast_scene_bpy (Blender) |
|------|----------------------|--------------------------|
| 渲染品质 | 基础 OpenGL | PBR (Eevee/Cycles) |
| 转角处理 | 简单重叠 | 斜接修正 (Miter Joint) |
| 遮挡处理 | 简单裁切 | 智能半透明材质 |
| 可见几何 | 支持（手动 FOV 平面） | 支持（`calc_matrix_camera` 对齐渲染） |
| 平面内表面顶点 | - | 支持（`--ply` 自动导出 JSON + 连线图） |
| 语义图 | 支持 | 支持（proxy + Raw） |
| 深度图 | Pyrender 深度缓冲 | Cycles Z pass + ray_cast 回退 |
| 法线图 | - | Cycles Normal pass（随 `--depth` 同次渲染） |
| 深度 + 多视角 | 单进程 | 每视角独立子进程（防内存泄漏） |

---

## 12. 配置文件 (`config.yaml`)

统一管理：画布尺寸、墙厚、默认 FOV、HDRI 路径、家具模型库路径、光照强度等。可通过 `BpySceneCtx` / `SceneCtx` 的 `config` 属性读取。

---

# 二. 资产处理模块 (`get_mesh`)

`render_ssl` 会先将 SSL 解析为 JSON，再调用 `get_mesh` 处理资产（检索或生成），最后交给渲染引擎。

- **默认**：`asset_mode="none"`，不修改 JSON，直接使用 SSL 中的 `asset_id` 占位
- **检索**：需先构建 LanceDB（见下）

## 检索分支

### 1. 数据库构建

```bash
python build_lancedb.py
```

- 存储位置：当前工作目录下 `manycore/`
- `furniture` 表首次构建约需 16 小时（全量 Embedding）
- 需配置 `Qwen3VLEmbedder` 路径（`util_data.py`）

### 2. 检索逻辑

**门/窗：** 按 `width`×`height` L2 距离匹配最近模型。

**家具 (Bbox)：**

1. **有图像**（`retrieve` + `image_path`）：按 `mesh_id` 分组，bbox_2d 裁剪 + label/caption 多模态 Embedding
2. **无图像**：label + caption 文本 Embedding，Batch 并行

## 生成分支 (`asset_mode="generate"`)

- 需提供 `image_path`
- 按 `mesh_id` 分组 → 裁剪 → 扩图/超分 → 3D 生成（Hunyuan 等）
- 模型保存至 `gen_asset_dir`（默认 `/data-nas/data/dataset/qunhe/Manycore-Future/generate/`）

---

## 附录：常用 Python 调用示例

```python
# 仅俯视图 + 可见几何
render_ssl(ssl_text, backend="bpy", views=["topdown"],
           export_glb=True, export_point_cloud=True, visible_geometry=True)

# 全视角 + 语义 + 深度（bpy 自动走子进程 depth）
render_ssl(ssl_text, backend="bpy", views=None,
           semantic=True, depth=True, samples=64)

# 低内存：大场景 (>25 物体) 自动切换 simplified 模型路径
render_ssl(ssl_text, backend="bpy", asset_dir="/path/to/assets")
```

---

## 附录：视锥裁剪与 Blender 原生 API（实现备忘）

bpy 路径下，**可见几何**（`visible_geometry`）与**平面内表面顶点**（`planar_faces.json` / `*_lines.png`）都依赖「当前渲染相机能看到什么、裁切后顶点在哪」。这块多次出问题的共同根因是：**没有用与 Cycles 实际渲染一致的投影矩阵**。

### 正确做法（当前实现）

与 Blender 渲染对齐，使用：

1. **`camera.calc_matrix_camera(depsgraph, x, y, ...)`** 获取投影矩阵  
2. **相机 `matrix_world.inverted()`** 作为 modelview  
3. 在 **齐次 clip space** 中对三角形/多边形环做 Sutherland-Hodgman 裁剪  
4. 沿边插值时 **同时插值 `clip` 与 `world`**，保证裁切后的交点仍在原平面（墙/门/窗/地板/天花）上  

可见几何的三角形裁剪（`fast_scene_bpy._clip_triangle_to_render_frustum`）与平面顶点的多边形裁剪（`util_bpy.clip_polygon_to_render_frustum`）均遵循上述流程。

**遮挡标记 `occluded`** 与视锥无关，使用 `scene.ray_cast`（与可见几何物体级遮挡判定相同），透明墙/天花/地板可穿透。

### 错误做法（已废弃，勿再使用）

| 做法 | 问题 |
|------|------|
| 手写视锥平面 / 手动 FOV 切平面 | 与 Blender 渲染视锥不对齐，俯视图/边缘物体易误判 cutted |
| `world_to_camera_view` 得 `(u,v,z)` 后在屏幕空间裁切，交点用 3D 线性插值 | 透视下交点**脱离原平面**；部分在视锥内时墙/天花顶点坐标错乱、连线图自交 |
| 按质心角度对多边形顶点重排 | 破坏 L 形/凹多边形边界拓扑（地板顺序错误） |

### 其他独立问题（非视锥 API）

- **环起点规则**：规范起点为 z 最小 → y 最小 → x 最小；仅旋转起点，不重排边界顺序  
- **cutted 统计**：仅当三角形**三顶点都在**视口 `[0,1]×[0,1]` 时才算「在视锥内」；面片占比 < 95% 才标 `_cutted`  

### 经验结论

- **判断点是否在视锥内（统计/命名）**：可用 `world_to_camera_view`，阈值与渲染视口 `[0,1]×[0,1]`、`z>0` 一致  
- **裁切几何并求 3D 交点**：必须用 **`calc_matrix_camera` + clip space**，不能用屏幕 `(u,v)` 反推 3D  
- **导出多边形顶点顺序**：保留 SSL/房间 `vertices` 或 mesh 定义的边界顺序，只做 CCW 整体反转 + 起点旋转  

俯视图因物体全在视锥内，旧错误裁剪不易暴露；**透视视角（如 left）部分裁切**时，非原生 API 裁剪的问题会非常明显。

