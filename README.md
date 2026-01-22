# Fast Scene 场景渲染工具库

这是一个室内场景渲染库，支持从简单的几何描述（墙体、家具、门窗的 layout json 或者 ssl）生成 3D 场景并进行任意视角渲染。

---
# 一. 渲染模块

## 1. 安装与依赖

### Python 版本要求: 3.11  !!!!

不是 3.11 无法安装最新版本的 bpy

### 基础依赖 (Python 环境)

适用于 `fast_scene` (Pyrender版) 和 `fast_scene_bpy` (Blender版) 的通用工具逻辑：

```bash
pip install numpy shapely scipy imageio yaml
```

### 渲染后端依赖

- **Pyrender 版**: `pip install pyrender trimesh`
  ```bash
  conda install -c conda-forge libstdcxx-ng=12 -y  # 使用 conda 安装 c++
  apt update
  apt install -y libegl1-mesa-dev libgles2-mesa-dev mesa-utils xvfb libgl1-mesa-glx libglu1-mesa libxrender1 libxext6
  ```
- **Blender 版**: `pip install bpy`
  ```bash
  apt update
  apt install -y libxi6 libxrender1 libxrandr2 libxfixes3 libxcursor1 libxinerama1 libxxf86vm1 libgl1-mesa-glx libglu1-mesa libxkbcommon0 libxkbcommon-dev libgl1-mesa-glx libgl1-mesa-dev libxi6 libxrender1 libxrandr2 libxfixes3 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxss1 libgtk-3-0 libgtk-3-dev libgconf-2-4 libasound2 libpulse0
  ```

---

### 安装到python 环境
  ```bash
  pip install -e .
  ```
  这样安装导入会报 Linting 错误, 但是可以正常运行

## 2. Quickstart

渲染 ssl 使用 render_ssl.py

渲染指定房间 id 使用 render_room_id.py

渲染 json 文件使用 fast_scene_bpy.py 中的例子

```python
# 以 Blender 版本为例, 可选版本 pyrender
from fast_scene.fast_scene_bpy import BpySceneCtx  # 使用 blender 渲染
# from fast_scene import SceneCtx       # 使用 pyrender 渲染

ctx = BpySceneCtx(scene_type="living_room")

# 1. 添加元素
ctx.add_walls(walls_data)
ctx.add_door(center=[2.0, 0, 1.0], width=0.9, height=2.0)
ctx.add_window(center=[0, 3.5, 1.5], width=1.2, height=1.0)
ctx.add_boxes(furniture_data)

# 2. 渲染
ctx.topdown_view("topdown.png")
ctx.render_view("side_view.png", camera_position=[6, 4, 2], look_at_target=[0, 0, 1])
```

---

## 3. 核心渲染函数参数详解

## 渲染 API 详述

`topdown_view` 和 `render_view` 共享大部分控制参数。

### 核心共有参数

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `output_path` | `str` | - | 输出图像路径（支持 .png, .jpg） |
| `width` / `height` | `int` | `1024` | 渲染分辨率 |
| `geometry_mode` | `str` | `"gltf"` | 几何体模式: `"gltf"`(仅模型), `"mixed"`(模型优先), `"bbox"`(纯方块) |
| `lighting_type` | `str` | `"array"` | 灯光模式: `"array"`(阵列筒灯), `"area"`(单盏大柔光灯), `"none"`(无灯) |
| `visible_shadow` | `bool` | `True` | 墙体和天花板是否投射阴影。设为 `False` 可让 HDRI 光线穿透进入室内 |
| `align_height` | `bool` | `True` | 是否将所有墙体的高度统一对齐到场景的最大高度 `z_max` |
| `rebuild` | `bool` | `False` | 是否强制清空当前 Blender 场景并重新构建（包括灯光和 HDRI） |
| `show_wall` / `window` / `door` / `ceiling` | `bool` | `True` | 分别控制墙体、窗户、门、天花板的可见性 |
| `auto_fov` | `bool` | `True` | 是否根据场景内容自动计算最佳视角 (FOV) |
| `manual_fov` | `float` | `None` | 手动指定视角范围（角度制）。若提供，则覆盖 `auto_fov` |
| `auto_transparent` | `bool` | `True` | **核心功能**：自动检测并透明化挡在相机前的墙体 |
| `transparent_alpha` | `float` | `0.3` | 自动透明化时，被遮挡墙体的 Alpha 透明度 (0-1) |
| `use_HDRI` | `bool` | `True` | 是否启用配置文件中指定的 HDRI 环境光 |
| `hdri_transparent_background` | `bool` | `False` | 启用 HDRI 时是否隐藏背景贴图（保留光照）。俯视图默认 `True` |
| `render_depth` | `bool` | `False` | 是否同时导出深度图 (同名 `.exr` 格式) |

---
**!!!注意在改变了模型光照等属性重新渲染时要将`rebuild`设为`True`**
**!!!一般情况不用修改后面的一堆默认参数, 以下情况除外:**

### `topdown_view` (俯视图渲染)

用于生成从上往下的场景视图。通常建议将 `show_ceiling` 设为 `False`, `hdri_transparent_background`设置为 `True`以隐藏 HDRI 贴图

### `render_view` (任意视角渲染)

用于生成指定相机位置和观察点的自由视角视图。

*   **独有参数**：
    *   `camera_position`: `list` (必填)。相机 3D 坐标 `[x, y, z]`。
    *   `look_at_target`: `list`。相机看向的目标点（默认指向房间中心）。

如果相机在房间中且希望房间是封闭的, 建议设置参数`show_ceiling` 为默认 `True` 


---

## 4. 版本对比

| 特性               | fast_scene (Pyrender) | fast_scene_bpy (Blender)                 |
| :----------------- | :-------------------- | :--------------------------------------- |
| **渲染品质** | 基础 OpenGL 效果      | PBR 物理渲染 (Eevee/Cycles)              |
| **转角处理** | 简单重叠              | **斜接修正 (Miter Joint)**，无黑影 |
| **遮挡处理** | 简单裁切              | 智能半透明材质                           |
| **光照支持** | 点光源                | 点光源 + HDRI 贴图                       |

---

## 5. 配置文件 (config.yaml)

用户可以通过 `config.yaml` 统一管理：墙体厚度、地板纹理路径、家具模型库路径、默认光照强度等全局参数。

---

# 二. 检索模块

在执行 `render_ssl.py` 进行资产检索之前，必须先完成 LanceDB 向量数据库的构建。

## 1. 数据库构建

运行以下脚本以构建检索所需的三个表（`door`, `window`, `furniture`）：

```bash
python build_lancedb.py
```

**注意事项：**
- **存储位置**：数据库将构建在当前工作目录下的 `manycore` 文件夹中。
- **构建耗时**：由于 `furniture` 表需要对全量资产计算 Embedding 向量，首次构建大约需要 **16 小时**（取决于 GPU 性能）。
- **环境依赖**：确保已安装 `lancedb` 并在 `fast_scene/util_data.py` 中正确配置了 `Qwen3VLEmbedder` 的路径。

## 2. 检索逻辑说明

系统会自动根据 `scene_json` 中的信息匹配最接近的 3D 资产：

### A. 门与窗 (Holes)
- **匹配特征**：使用物体的 `width` 和 `height` 构造 2D 向量。
- **检索度量**：使用 **L2 距离** 查找尺寸最接近的模型。

### B. 家具 (Bboxes)
检索逻辑支持两种模式，均采用 **余弦相似度 (Cosine Similarity)** 进行匹配：

1. **有图像输入 (retrieve 模式下提供 image_path)**:
   - **资产分组**：根据 `asset_id` 属性进行分组，确保同一资产在场景中多次出现时只计算一次并共享结果。
   - **多模态 Embedding**：对图像进行 `bbox_2d` 裁剪，结合 `label` 和 `caption` 构造多模态输入，调用 Qwen3-VL-Embedding 模型进行 Batch 计算。
2. **无图像输入**:
   - **文本检索**：仅根据 `label`（必须提供）和 `caption`（可选）构造文本 Prompt 进行检索。
   - **并行处理**：对场景内所有待处理物体进行 Batch Embedding 计算以提高效率。

---

# 三. 生成模块 (generate)

当 `asset_mode="generate"` 时，系统将进入 3D 资产生成流程（目前为占位实现）：
- **前置条件**：必须提供 `image_path`。
- **处理流程**：
  1. 按 `asset_id` 对物体分组。
  2. 对每组首个物体进行图像裁剪。
  3. 执行图像补全与超分辨率处理。
  4. 调用 3D 生成工具生成 `.glb` 模型。
- **存储路径**：生成的模型将以 8 位 UUID 命名，存储在 `/data-nas/data/dataset/qunhe/Manycore-Future/generate/` 目录下。
