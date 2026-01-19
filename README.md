# Fast Scene 场景渲染工具库

这是一个室内场景渲染库，支持从简单的几何描述（墙体、家具、门窗的 layout json 或者 ssl）生成 3D 场景并进行任意视角渲染。

---

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
  apt install -y libxi6 libxrender1 libxrandr2 libxfixes3 libxcursor1 libxinerama1 libxxf86vm1 libgl1-mesa-glx libglu1-mesa libxkbcommon0 libxkbcommon-dev libgl1-mesa-glx libgl1-mesa-dev libxi6 libxrender1 libxrandr2 libxfixes3 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxss1 libgtk-3-0 libgtk-3-dev libgconf-2-4 libasound2 libpulse0
  ```

---

## 2. Quickstart

```python
# 以 Blender 版本为例, 可选版本 pyrender
from fast_scene_bpy import BpySceneCtx  # 使用 blender 渲染
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

### `topdown_view` (俯视图渲染)

用于生成从上往下的正交或透视投影视图。

| 参数                            | 类型      | 默认值     | 说明                                                                 |
| :------------------------------ | :-------- | :--------- | :------------------------------------------------------------------- |
| `output_path`                 | `str`   | -          | 输出图像路径（支持 .png, .jpg）                                      |
| `width` / `height`          | `int`   | `1024`   | 渲染分辨率                                                           |
| `geometry_mode`               | `str`   | `"gltf"` | 几何体模式:`"gltf"`(仅模型), `"mixed"`(混合), `"bbox"`(纯方块) |
| `show_wall`                   | `bool`  | `True`   | 是否显示墙体                                                         |
| `show_window`                 | `bool`  | `True`   | 是否显示窗户                                                         |
| `show_door`                   | `bool`  | `True`   | 是否显示门                                                           |
| `show_ceiling`                | `bool`  | `True`   | 是否显示天花板（俯视建议设为 `False`）                             |
| `up_vector`                   | `list`  | `None`   | 相机上方向（默认根据视角自动计算）                                   |
| `auto_fov`                    | `bool`  | `True`   | 是否根据场景范围自动计算最佳 FOV                                     |
| `manual_fov`                  | `float` | `None`   | 手动指定视角范围（角度制）                                           |
| `auto_transparent`            | `bool`  | `True`   | 是否自动将遮挡视野的物体设为透明                                     |
| `transparent_alpha`           | `float` | `0.3`    | 透明物体的 Alpha 值 (0-1)                                            |
| `render_depth`                | `bool`  | `False`  | 是否同时导出深度图 (.exr 格式)                                       |
| `use_HDRI`                    | `bool`  | `True`   | 是否启用 HDRI 环境光渲染                                             |
| `hdri_transparent_background` | `bool`  | `False`  | 启用 HDRI 时是否隐藏背景贴图（保留光照）                             |

### `render_view` (任意视角渲染)

用于生成指定相机位置和观察点的自由视角视图。

| 参数                            | 类型      | 默认值     | 说明                                                 |
| :------------------------------ | :-------- | :--------- | :--------------------------------------------------- |
| `output_path`                 | `str`   | -          | 输出图像路径                                         |
| `camera_position`             | `list`  | -          | 相机 3D 坐标 `[x, y, z]`                           |
| `look_at_target`              | `list`  | `None`   | 相机看向的目标点（默认指向房间中心）                 |
| `width` / `height`          | `int`   | `1024`   | 渲染分辨率                                           |
| `up_vector`                   | `list`  | `None`   | 相机的上方向向量（默认 `[0,0,1]`）                 |
| `auto_fov`                    | `bool`  | `True`   | 是否自动计算 FOV 以包含整个房间                      |
| `manual_fov`                  | `float` | `None`   | 手动指定视角范围（角度制）                           |
| `auto_transparent`            | `bool`  | `True`   | **核心功能**：自动检测并透明化挡在相机前的墙体 |
| `transparent_alpha`           | `float` | `0.3`    | 透明物体的 Alpha 值 (0-1)                            |
| `geometry_mode`               | `str`   | `"gltf"` | 几何体模式:`"gltf"`, `"mixed"`, `"bbox"`       |
| `render_depth`                | `bool`  | `False`  | 是否渲染深度信息                                     |
| `show_wall`                   | `bool`  | `True`   | 是否渲染墙体                                         |
| `show_window`                 | `bool`  | `True`   | 是否渲染窗户                                         |
| `show_door`                   | `bool`  | `True`   | 是否渲染门                                           |
| `show_ceiling`                | `bool`  | `True`   | 是否渲染天花板                                       |
| `use_HDRI`                    | `bool`  | `True`   | 是否启用高级 HDRI 环境光                             |
| `hdri_transparent_background` | `bool`  | `False`  | 启用 HDRI 时是否隐藏背景贴图                         |

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
