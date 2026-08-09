# SceneBuilder

[English](README.md) | **中文**

基于 **bpy / pyrender** 的 **3D 场景构建**、**空间规划**（覆盖式视频渲染）、**渲染**与**几何导出**工具库，输入为 **SSL** 结构化场景格式。

<p align="center">
  <img src="assets/image.png" alt="SceneBuilder 渲染示例" width="800">
</p>

从 SSL 文本（墙、门、窗、家具）构建室内 3D 场景，支持多视角渲染、语义/深度图、全景图，导出 GLB/PLY/体素，并基于地板路径自动生成相机轨迹。

## 研发路线


| 阶段    | 状态     | 目标                                                                     |
| ----- | ------ | ---------------------------------------------------------------------- |
| **1** | ✅ 已完成  | 渲染与几何导出工具链（`render_ssl`、`topdown_view`、`render_view`、GLB/PLY/体素/语义/深度） |
| **2** | 🚧 进行中 | 发布 **structured scene data sets**（结构化场景数据集）                                                  |
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

## 高级功能 — `render_ssl` 批量管线

面向数据生产与 benchmark，`render_ssl.py` 是高层入口。一次调用可以：

- **像素对齐**规范化俯视坐标，并 **规划地板覆盖路径**
- 沿路径 **自动生成相机视角**（`--views auto`），也支持预设 / 自定义相机
- **导出几何**：各视角可见 GLB / PLY / 256³ 体素，以及输出根目录的全场景 holo 导出
- **导出监督信号**：语义 mask、米制深度 + 法线、等距圆柱全景图
- **断点续跑**（用 `--no-resume` 强制全量重跑）

示例 — 打开主要导出开关：

```bash
python render_ssl.py \
  --ssl path/to/ssl.txt \
  --output out \
  --assets path/to/assets \
  --visible_geometry --holo_geometry \
  --glb --ply --voxel \
  --semantic --depth --pano \
  --views auto
```

**输出目录结构**（`--views auto` 时输出到 `out_normalized/`）：

```
out_normalized/
├── ssl.txt, data.json
├── scene.glb                         # --holo_geometry --glb
├── pointcloud/, voxel/               # --holo_geometry --ply --voxel
├── topdown_normalized/               # 像素对齐俯视 + 地板路径
├── topdown/                          # 常规 1024² 俯视 + 导出
├── auto_views.json
├── auto_path_0000/ …                 # auto 单帧视角
├── auto_path_0008_seq/               # auto 三帧序列
└── {view}/                           # RGB、depth、semantic、pano、可见 glb/ply/voxel …
```

完整目录说明见 [§6 输出目录与坐标系](docs/doc.zh-CN.md#6-输出目录与坐标系)。

也支持 `--camera_position`、`--look_at`、可选 `--up_vector` 自定义相机；视线与默认 up `[0,0,1]` 共线时会自动回退为 `[0,1,0]`。

## 文档

**完整手册：** [中文](docs/doc.zh-CN.md) · [English](docs/doc.en.md)

建议按以下顺序阅读：

| 步骤 | 主题 | 中文 | English |
| ---- | ---- | ---- | ------- |
| 1 | 安装与依赖 | [§1](docs/doc.zh-CN.md#1-安装与依赖) | [§1](docs/doc.en.md#1-installation-and-dependencies) |
| 2 | SSL 格式与坐标系 | [§2](docs/doc.zh-CN.md#2-ssl-坐标系与实体约定) | [§2](docs/doc.en.md#2-ssl-coordinate-system-and-entities) |
| 3 | 快速开始与示例 | [§3](docs/doc.zh-CN.md#3-快速开始) | [§3](docs/doc.en.md#3-quick-start) |
| 4 | API 参考（`topdown_view` / `render_view` / `render_ssl`） | [§4](docs/doc.zh-CN.md#4-api-参考) | [§4](docs/doc.en.md#4-api-reference) |
| 5 | 自动视角与地板路径（`--views auto`） | [§5](docs/doc.zh-CN.md#5-高级工作流) | [§5](docs/doc.en.md#5-advanced-workflows) |
| 6 | 输出目录、`c2w` 与导出产物 | [§6](docs/doc.zh-CN.md#6-输出目录与坐标系) | [§6](docs/doc.en.md#6-output-layout-and-coordinate-systems) |
| 7 | 语义 / 深度 / 体素详解 | [§7](docs/doc.zh-CN.md#7-导出产物详解) | [§7](docs/doc.en.md#7-export-artifacts) |

## 许可证

本项目采用 [MIT License](LICENSE) 开源。