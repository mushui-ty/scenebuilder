# SceneBuilder

[English](README.md) | **中文**

基于 **bpy / pyrender** 的 **3D 场景构建**、**渲染**、**路径规划**（覆盖式视频渲染）与**几何导出**工具库，输入为 **SSL** 结构化场景格式。

SceneBuilder 渲染示例

从 SSL 文本（墙、门、窗、家具）构建室内 3D 场景，支持多视角渲染、语义/深度图、全景图，导出 GLB/PLY/体素，并基于地板路径自动生成相机轨迹。

## 研发路线


| 阶段    | 状态     | 目标                                                                     |
| ----- | ------ | ---------------------------------------------------------------------- |
| **1** | ✅ 已完成  | 渲染与几何导出工具链（`render_ssl`、`topdown_view`、`render_view`、GLB/PLY/体素/语义/深度） |
| **2** | 🚧 进行中 | 发布 **场景 example 数据集**                                                  |
| **3** | ⏳ 未开始  | 构建 **canonical 资产检索（retrieve）** 系统                                     |
| **4** | ⏳ 未开始  | 接入 **资产生成（generation）** 能力                                             |


## 功能概览

- **双后端**：Blender（`bpy`，推荐）或 Pyrender（轻量）
- **多视角**：俯视、预设相机、自定义相机、`--views auto` 路径驱动渲染
- **导出**：RGB、深度、法线、语义 mask、全景、可见 GLB/PLY、256³ 体素
- **路径规划**：像素对齐俯视 + nav-mask 地板闭环，用于覆盖式视频
- **坐标系**：SSL 世界系 + OpenCV 相机副本（`ssl_opencv.txt`、`c2w`、`intrinsic`）



## 安装

需要 **Python 3.11**（最新版 `bpy` 依赖此版本）。

```bash
pip install numpy shapely scipy imageio pyyaml
pip install bpy          # 推荐
# 或: pip install pyrender trimesh

cd scenebuilder
pip install -e .
```

Linux 无头 Pyrender 需设置 `PYOPENGL_PLATFORM=egl` 并安装 Mesa/EGL 依赖，详见 [完整文档（中文）](docs/doc.zh-CN.md#1-安装与依赖)。

## 快速开始



### CLI — 批量渲染

```bash
python render_ssl.py \
  --ssl path/to/ssl.txt \
  --views auto \
  --output out \
  --assets path/to/assets \
  --glb --ply --visible_geometry --semantic --depth --pano
```



### Python — 单张俯视图

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



### Python — 自定义相机

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

CLI 也支持 `--camera_position`、`--look_at`、可选 `--up_vector`。未指定 `up_vector` 时，若视线与默认 up `[0,0,1]` 共线（如俯视），`render_view` 会自动回退为 `[0,1,0]`。

## 文档


| 主题                                 | 链接                                                    |
| ---------------------------------- | ----------------------------------------------------- |
| **完整参考**（API、SSL 坐标、输出目录、深度/语义/体素） | [docs/doc.zh-CN.md](docs/doc.zh-CN.md)                |
| SSL 坐标系                            | [§2](docs/doc.zh-CN.md#2-ssl-坐标系与实体约定)                |
| `render_ssl` / CLI 参数              | [§4.3](docs/doc.zh-CN.md#43-render_ssl--render_sslpy) |
| 输出目录与 `c2w`                        | [§6](docs/doc.zh-CN.md#6-输出目录与坐标系)                    |
| `--views auto` 与地板路径               | [§5](docs/doc.zh-CN.md#5-高级工作流)                       |
| English documentation              | [docs/doc.en.md](docs/doc.en.md)                      |




## 目录结构

```
scenebuilder/
├── render_ssl.py          # CLI 与高层批量 API
├── core/
│   ├── scenebuilder_bpy.py
│   ├── scenebuilder.py
│   ├── auto_views.py
│   └── nav_mask_path.py
├── assets/                # 示例图与家具 GLB
├── config.yaml
└── docs/
    ├── doc.zh-CN.md       # 完整文档（中文）
    └── doc.en.md          # Full documentation (English)
```



## 许可证

见 [LICENSE](LICENSE)。