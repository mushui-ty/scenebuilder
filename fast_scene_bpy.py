"""
Fast Scene - 纯 Blender 渲染版本（无 trimesh 依赖）

用 bpy 替代 trimesh/pyrender

使用方式：
   $BLENDER_PATH --background --python fast_scene_bpy.py 或者在 pip install bpy 后直接使用 python fast_scene_bpy.py
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import json
import yaml
import numpy as np
import time
from typing import List, Dict, Any, Optional

# 导入 util 的数据处理函数（不使用其 mesh 创建函数）
try:
    from . import util, util_bpy
except ImportError:
    import util
    import util_bpy

try:
    import bpy
    import mathutils  # type: ignore[import]
    from mathutils import Matrix  # type: ignore[import]
except ImportError as e:
    print(f"❌ 错误：无法导入 bpy 模块")
    print(f"   请在 Blender Python 环境中运行此脚本")
    raise e


class BpySceneCtx:
    """场景上下文管理器 - 纯 Blender 版本"""
    
    
    def __init__(self, scene_type: str):
        # 与 fast_scene.py 完全一致的数据结构
        self.context = {
            "meta": {"scene_type": scene_type},
            "walls": {},
            "boxes": {}
        }
        self.scene = None
        self.if_set_lights = False
        
        # 保存所有对象引用（与 fast_scene.py 的 mesh_nodes 对应）
        self.mesh_nodes = {
            "walls": {},      # wall_id -> node
            "doors": {},      # door_id -> node
            "windows": {},    # window_id -> node
            "boxes": {},      # box_id -> node
            "floor": None,    # floor node
            "ceiling": None,  # ceiling node
        }

        # 加载配置
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 初始化 Blender 场景
        self._init_blender_scene()

    def _init_blender_scene(self):
        """初始化 Blender 场景"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        if "Scene" in bpy.data.scenes:
            self.scene = bpy.data.scenes["Scene"]
        else:
            self.scene = bpy.data.scenes.new("Scene")
        bpy.context.window.scene = self.scene
        
        # 设置渲染引擎
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.device = 'GPU'
        
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        for device in prefs.get_devices_for_type('CUDA'):
            device.use = True
        
        self.scene.cycles.samples = 32
        self.scene.cycles.use_denoising = True
        self.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        
        self.scene_collection = self.scene.collection
        
        print(f"✅ Blender 场景初始化完成 (Cycles + CUDA)")

    # ==================== 数据管理函数（与 fast_scene.py 完全一致）====================
    def add_walls(self, walls: List[Dict[str, Any]]):
        """添加墙体并计算场景元数据"""
        walls_converted = [{"s": w["p"][:2], "e": w["q"][:2], "height": w["height"]} for w in walls]
        all_points = [p for w in walls_converted for p in [w["s"], w["e"]]]
        x_coords, y_coords = zip(*all_points)
        vertices = util.calculate_minimum_area_polygon(walls_converted)
        
        self.context["meta"].update({
            "bounds": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
            "center": [(min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2],
            "span": [max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)],
            "z_max": max(w["height"] for w in walls_converted),
            "vertices": vertices
        })
        
        print(f"📐 输入墙体数量: {len(walls_converted)}")
        print(f"📐 计算得到的顶点数量: {len(vertices)}")
        
        for wall in walls_converted:
            wall_id = util.generate_unique_id()
            self.context["walls"][wall_id] = {
                **wall,
                "orientation": util.calculate_wall_orientation(wall["s"], wall["e"], self.context["meta"]["vertices"]),
                "doors": {},
                "windows": {}
            }

    def add_doors(self, doors: List[Dict[str, Any]]):
        """批量添加门"""
        for door in doors:
            self.add_door(door["center"], door["width"], door["height"], door.get("mesh_id"))

    def add_door(self, center: List[float], width: float, height: float, mesh_id: Optional[int] = None):
        """添加单个门"""
        wall_id = util.find_closest_wall(center, self.context["walls"])
        if wall_id:
            wall = self.context["walls"][wall_id]
            door_id = util.generate_unique_id()
            door_data = {
                "center": util.snap_to_wall(center, wall),
                "width": width,
                "height": height
            }
            if mesh_id:
                door_data["mesh_id"] = mesh_id
            self.context["walls"][wall_id]["doors"][door_id] = door_data

    def add_windows(self, windows: List[Dict[str, Any]]):
        """批量添加窗"""
        for window in windows:
            self.add_window(window["center"], window["width"], window["height"], window.get("mesh_id"))

    def add_window(self, center: List[float], width: float, height: float, mesh_id: Optional[int] = None):
        """添加单个窗"""
        wall_id = util.find_closest_wall(center, self.context["walls"])
        if wall_id:
            wall = self.context["walls"][wall_id]
            window_id = util.generate_unique_id()
            window_data = {
                "center": util.snap_to_wall(center, wall),
                "width": width,
                "height": height
            }
            if mesh_id:
                window_data["mesh_id"] = mesh_id
            self.context["walls"][wall_id]["windows"][window_id] = window_data

    def add_boxes(self, boxes: List[Dict[str, Any]]):
        """批量添加家具"""
        for box in boxes:
            self.add_box(box["center"], box["angle_z"], box["scale"],
                        box.get("label"), box.get("caption"), box.get("mesh_id"))

    def add_box(self, center: List[float], angle_z: float, scale: List[float],
                label: Optional[str] = None, caption: Optional[str] = None, 
                mesh_id: Optional[int] = None) -> str:
        """添加单个家具，返回ID"""
        box_id = util.generate_unique_id()
        box_data = {"center": center, "angle_z": angle_z, "scale": scale}

        if label: box_data["label"] = label
        if caption: box_data["caption"] = caption
        if mesh_id: box_data["mesh_id"] = mesh_id

        self.context["boxes"][box_id] = box_data

        box_top = center[2] + scale[2] / 2
        if box_top > self.context["meta"].get("z_max", 0):
            self.context["meta"]["z_max"] = box_top

        return box_id

    def get_context(self) -> Dict:
        """获取场景上下文"""
        return self.context
    
    def get_boxes(self) -> Dict:
        """获取所有家具boxes"""
        return self.context["boxes"]

    # ==================== 纯 bpy 几何体创建（替代 util 中的 trimesh 函数）====================
    # helper functions now live in util_bpy

    # ==================== 场景构建（与 fast_scene.py 逻辑完全一致）====================
    
    def construct_floor(self, show_wall=True, show_window=True, show_door=True, show_ceiling=True):
        """构建地板和墙体（与 fast_scene.py 完全一致的逻辑）"""
        # 注意：这里不创建 self.scene = pyrender.Scene()，而是使用已有的 bpy scene
        vertices = self.context["meta"]["vertices"]

        if len(vertices) < 3:
            return

        # 创建地板
        bounds = self.context["meta"]["bounds"]
        wall_thickness = self.config.get("wall_thickness", 0.1)
        floor_obj = util_bpy.create_floor_mesh_bpy(self.scene_collection, vertices)
        if floor_obj is None:
            print("⚠️ 地板几何体创建失败，跳过后续处理")
            return

        # 加载纹理
        texture_path = self.config["floor_blender_texture_path"]
        if texture_path and os.path.exists(texture_path):
            mat = util_bpy.create_material_with_texture("Floor_Material", texture_path)
        else:
            mat = util_bpy.create_material_with_color("Floor_Material", [0.8, 0.7, 0.5, 1.0])
        if mat:
            floor_obj.data.materials.append(mat)
        
        self.mesh_nodes["floor"] = {"node": floor_obj, "mesh": floor_obj}  # 保存引用

        # 创建墙体
        if show_wall:
            wall_color = self.config["wall_color"]
            wall_texture_path = self.config["wall_blender_texture_path"]

            # --- 预计算 Miter Joint (斜接) 偏移 ---
            wall_outer_points = util_bpy.calculate_miter_joints(self.context["walls"], wall_thickness)

            for wall_id, wall in self.context["walls"].items():
                # 收集门窗信息
                openings = []
                for door in wall.get("doors", {}).values():
                    openings.append(door)
                for window in wall.get("windows", {}).values():
                    openings.append(window)
                
                # 获取预计算的偏移点
                outer_s, outer_e = wall_outer_points.get(wall_id, (None, None))

                # 创建墙体mesh（带门窗挖洞）
                wall_obj = util_bpy.create_single_wall_mesh_bpy(
                    self.scene_collection,
                    wall["s"], wall["e"], wall["height"], wall["orientation"],
                    chip=False,
                    openings=openings if openings else None,
                    wall_thickness=wall_thickness,
                    name=f"Wall_{wall_id[:4]}",
                    outer_s=outer_s,
                    outer_e=outer_e
                )

                # 应用纹理或颜色
                if wall_texture_path and os.path.exists(wall_texture_path):
                    mat = util_bpy.create_material_with_texture(f"Wall_{wall_id}_Material", wall_texture_path)
                else:
                    mat = util_bpy.create_material_with_color(f"Wall_{wall_id}_Material", wall_color)
                
                if wall_obj:
                    wall_obj.data.materials.append(mat)
                    self.mesh_nodes["walls"][wall_id] = {
                        "node": wall_obj,
                        "mesh": wall_obj,
                        "wall_data": wall
                    }
                
                # 注意：bpy 版本暂不实现边缘线条（可选功能）
        else:
            print("🚫 跳过墙体创建 (show_wall=False)")
        
        # 添加门窗
        if show_door or show_window:
            print(f"🚪 开始处理门窗 (按 mesh_id 加载)...")
            wall_thickness = self.config.get("wall_thickness", 0.1)
            model_hole_path = self.config.get("model_hole_path")

            for wall_id, wall in self.context["walls"].items():
                wall_s = np.array(wall["s"])
                wall_e = np.array(wall["e"])
                wall_vec = wall_e - wall_s
                wall_len = np.linalg.norm(wall_vec)
                if wall_len < 1e-6:
                    continue

                wall_dir = wall_vec / wall_len
                # orientation 是指向内部的法线，向外扩的方向是 -orientation
                outward_normal = -np.array(wall["orientation"])
                # 计算墙的旋转角度 (角度制，供 apply_box_transform 使用)
                wall_angle_deg = np.degrees(np.arctan2(wall_dir[1], wall_dir[0]))

                # 定义要处理的类型列表
                types_to_process = []
                if show_door:
                    types_to_process.append(("door", "doors"))
                if show_window:
                    types_to_process.append(("window", "windows"))

                for item_type, key in types_to_process:
                    for item_id, item in wall.get(key, {}).items():
                        mesh_id = item.get("mesh_id")
                        if not mesh_id:
                            continue

                        # 计算真实中心：在原中心基础上向外偏移厚度的一半
                        inner_center = np.array(item["center"])
                        real_center = inner_center.copy()
                        real_center[0] += outward_normal[0] * (wall_thickness / 2)
                        real_center[1] += outward_normal[1] * (wall_thickness / 2)

                        # 构造 box 格式的数据
                        item_box_data = {
                            "center": real_center.tolist(),
                            "scale": [item["width"], wall_thickness, item["height"]],
                            "angle_z": wall_angle_deg,
                            "mesh_id": mesh_id
                        }

                        # 仿照 bbox 添加逻辑
                        mesh_root = util_bpy.load_mesh_to_origin(mesh_id, model_hole_path)
                        if mesh_root:
                            try:
                                success_transform = util_bpy.apply_box_transform(mesh_root, item_box_data)
                            except Exception as e:
                                success_transform = False
                                print(f"  ❌ {item_type} {item_id[:4]} 变换异常: {e}")

                            if success_transform:
                                self.mesh_nodes[f"{item_type}s"][item_id] = {
                                    "node": mesh_root,
                                    "mesh": mesh_root,
                                    f"{item_type}_data": {"wall_id": wall_id}
                                }
                                print(f"  ✅ {item_type} {item_id[:4]} (mesh_id={mesh_id}) 添加成功")
                            else:
                                print(f"  ❌ {item_type} {item_id[:4]} 变换失败")
                        else:
                            print(f"  ❌ {item_type} {item_id[:4]} (mesh_id={mesh_id}) 导入失败")
        else:
            print("🚫 跳过门窗创建 (show_door=False, show_window=False)")
        
        # 创建天花板
        if show_ceiling:
            z_max = self.context["meta"]["z_max"]
            ceiling_obj = util_bpy.create_ceiling_mesh_bpy(self.scene_collection, vertices, z_max)
            if ceiling_obj:
                ceiling_texture_path = self.config["ceiling_blender_texture_path"]
                if ceiling_texture_path and os.path.exists(ceiling_texture_path):
                    mat = util_bpy.create_material_with_texture("Ceiling_Material", ceiling_texture_path)
                else:
                    mat = util_bpy.create_material_with_color("Ceiling_Material", [0.9, 0.9, 0.9, 1.0])
                if mat:
                    ceiling_obj.data.materials.append(mat)

                self.mesh_nodes["ceiling"] = {"node": ceiling_obj, "mesh": ceiling_obj}
                print(f"✅ 已添加天花板 (高度: {z_max:.2f}m)")
            else:
                print("⚠️ 天花板几何体创建失败")
        else:
            print("🚫 跳过天花板 (show_ceiling=False)")

    def construct_scene(self, show_wall: bool = True, show_window: bool = True, 
                       show_door: bool = True, show_ceiling: bool = True,
                       geometry_mode: str = "gltf"):
        """
        构建完整场景。
        
        Args:
            show_wall, show_window, show_door, show_ceiling: 是否渲染对应的结构
            geometry_mode: 几何体模式，可选:
                - "gltf": 仅加载 GLTF 模型，若无 mesh_id 或文件不存在则跳过该物体
                - "mixed": 优先加载 GLTF 模型，若失败则回退到 bbox 几何体
                - "bbox": 所有物体均强制使用 bbox 几何体表示
        """
        total_start = time.perf_counter()

        self.construct_floor(show_wall=show_wall, show_window=show_window, 
                             show_door=show_door, show_ceiling=show_ceiling)

        print(f"📦 加载家具 ({len(self.context['boxes'])} 个物体, 模式: {geometry_mode})...")
        success_count = 0
        load_time = 0
        transform_time = 0
        
        model_path = self.config.get("model_path")

        for box_id, box in self.context["boxes"].items():
            mesh_id = box.get("mesh_id")
            
            # --- 模式判断逻辑 ---
            should_load_gltf = False
            should_fallback_bbox = False
            
            if geometry_mode == "bbox":
                should_fallback_bbox = True
            elif geometry_mode == "gltf":
                if mesh_id:
                    should_load_gltf = True
                # else: skip
            elif geometry_mode == "mixed":
                if mesh_id:
                    should_load_gltf = True
                else:
                    should_fallback_bbox = True
            
            # --- 执行加载 ---
            mesh_root = None
            if should_load_gltf:
                load_start = time.perf_counter()
                mesh_root = util_bpy.load_mesh_to_origin(mesh_id, model_path)
                load_time += time.perf_counter() - load_start
                
                if mesh_root:
                    transform_start = time.perf_counter()
                    try:
                        success_transform = util_bpy.apply_box_transform(mesh_root, box)
                    except Exception as e:
                        print(f"  ❌ {box.get('class', 'unknown')} (mesh_id={mesh_id}) 变换异常: {e}")
                        success_transform = False
                    transform_time += time.perf_counter() - transform_start
                    
                    if success_transform:
                        self.mesh_nodes["boxes"][box_id] = {
                            "node": mesh_root, "mesh": mesh_root, "box_data": box
                        }
                        success_count += 1
                        print(f"  ✅ {box.get('class', 'unknown')} 添加成功")
                        continue
                
                # 如果 gltf 加载失败，判断是否需要回退
                if geometry_mode == "mixed":
                    should_fallback_bbox = True
                else:
                    print(f"  ❌ {box.get('class', 'unknown')} (mesh_id={mesh_id}) 导入失败且未开启混合模式，跳过")
                    continue

            if should_fallback_bbox:
                try:
                    bbox_mesh = util_bpy.create_bbox_geometry(
                        box["center"], box["scale"], box["angle_z"],
                        color=util_bpy.get_class_color(box.get("class"))
                    )
                    self.mesh_nodes["boxes"][box_id] = {
                        "node": bbox_mesh, "mesh": bbox_mesh, "box_data": box
                    }
                    success_count += 1
                    print(f"  ✅ {box.get('class', 'unknown')} (bbox几何体) 添加成功")
                except Exception as e:
                    print(f"  ❌ {box.get('class', 'unknown')} (bbox几何体): {e}")

        total_time = time.perf_counter() - total_start
        print(f"✅ 场景构建完成! 成功添加 {success_count}/{len(self.context['boxes'])} 个物体")
        print(f"   构建耗时: {total_time:.2f}s (加载: {load_time:.2f}s, 变换: {transform_time:.2f}s)")

        total_time = time.perf_counter() - total_start
        print(f"✅ 场景构建完成! 成功添加 {success_count}/{len(self.context['boxes'])} 个物体")
        print(f"   构建耗时: {total_time:.2f}s (加载: {load_time:.2f}s, 变换: {transform_time:.2f}s)")

    def setup_lighting(self, point_light_intensity: float = 1000,
                      use_point_lights: bool = True, ambient_light_color: list = None,
                      ambient_strength: float = 2.0):
        """设置场景光照（与 fast_scene.py 逻辑一致，但使用 bpy 实现）"""
        if self.if_set_lights:
            print("⚠️  光照已设置，跳过重复设置")
            return
        
        center = self.context["meta"]["center"]
        z_max = self.context["meta"]["z_max"]
        span = self.context["meta"]["span"]
        
        # 设置环境光
        world = self.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            self.scene.world = world
        
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            if ambient_light_color is None:
                ambient_light_color = [1.0, 1.0, 1.0, 1.0]
            elif len(ambient_light_color) == 3:
                ambient_light_color = ambient_light_color + [1.0]
            bg_node.inputs['Color'].default_value = ambient_light_color
            bg_node.inputs['Strength'].default_value = ambient_strength
        
        # 添加点光源
        if use_point_lights:
            corner_positions = [
                [center[0] - span[0]/2, center[1] - span[1]/2, z_max+3],
                [center[0] + span[0]/2, center[1] - span[1]/2, z_max+3],
                [center[0] - span[0]/2, center[1] + span[1]/2, z_max+3],
                [center[0] + span[0]/2, center[1] + span[1]/2, z_max+3],
            ]
            for i, pos in enumerate(corner_positions):
                light_data = bpy.data.lights.new(name=f"PointLight_{i}", type='POINT')
                light_data.energy = point_light_intensity
                light_obj = bpy.data.objects.new(name=f"PointLight_{i}", object_data=light_data)
                self.scene_collection.objects.link(light_obj)
                light_obj.location = pos
            
            print(f"💡 已添加点光源x{len(corner_positions)}，强度={point_light_intensity}")
        
        self.if_set_lights = True

    def topdown_view(self, output_path: str, width: int = 1024, height: int = 1024,
                     geometry_mode: str = "gltf", show_wall: bool = True, 
                     show_window: bool = True, show_door: bool = True, show_ceiling: bool = True,
                     up_vector: list = None,
                     auto_fov: bool = True, manual_fov: float = None,
                     auto_transparent: bool = True, transparent_alpha: float = 0.3,
                     render_depth: bool = False, use_HDRI: bool = True,
                     hdri_transparent_background: bool = True):
        """俯视图渲染（与 fast_scene.py 逻辑完全一致）"""
        total_start = time.perf_counter()

        construct_time = 0
        if self.mesh_nodes["floor"] is None:
            construct_start = time.perf_counter()
            self.construct_scene(geometry_mode=geometry_mode, 
                               show_wall=show_wall, show_window=show_window, 
                               show_door=show_door, show_ceiling=show_ceiling)
            construct_time = time.perf_counter() - construct_start

        # Setup lighting
        setup_start = time.perf_counter()
        if not self.if_set_lights:
            # 顶视图使用纯环境光，避免大面积阴影
            self.setup_lighting(use_point_lights=False, ambient_light_color=[1.0, 1.0, 1.0], ambient_strength=3.0)
        if use_HDRI:
            hdri_path = self.config.get("hdri_path")
            if hdri_path and util_bpy.apply_hdri_to_world(self.scene, hdri_path, strength=1.0):
                print(f"🌇 使用 HDRI 环境光: {hdri_path}")
                if hdri_transparent_background:
                    self.scene.render.film_transparent = True
            else:
                print(f"⚠️ HDRI 文件不可用或未配置: {hdri_path}")
        # 禁用所有阴影，匹配 fast_scene.py 无阴影的效果
        # util_bpy.disable_shadows_for_all(self.scene_collection)

        # Setup camera
        center = self.context["meta"]["center"]
        span = self.context["meta"]["span"]
        z_max = self.context["meta"]["z_max"]
        bounds = self.context["meta"]["bounds"]

        camera_height = z_max + max(span) * 1.5
        camera_position = np.array([center[0], center[1], camera_height], dtype=float)
        look_at_target = np.array([center[0], center[1], 0.0], dtype=float)
        if up_vector is None:
            up_vector = [0.0, 1.0, 0.0]

        up_vector = np.array(up_vector, dtype=float)
        if np.linalg.norm(up_vector) < 1e-6:
            up_vector = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            up_vector = up_vector / np.linalg.norm(up_vector)

        if manual_fov is not None:
            fov_y = np.radians(manual_fov)
        elif auto_fov:
            indoor_fov = self.config.get('indoor_fov', 160)
            outdoor_fov_scale = self.config.get('outdoor_fov_scale', 1.05)
            fov_y = util.calculate_optimal_fov(
                camera_position, look_at_target,
                self.context["meta"]["vertices"],
                z_max, bounds,
                indoor_fov, outdoor_fov_scale
            )
        else:
            fov_y = np.radians(70.0)

        up_vector_np = np.array(up_vector, dtype=float)

        forward = look_at_target - camera_position
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up_vector_np)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            fallback_up = np.array([0.0, 0.0, 1.0], dtype=float)
            right = np.cross(forward, fallback_up)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                fallback_up = np.array([1.0, 0.0, 0.0], dtype=float)
                right = np.cross(forward, fallback_up)
                right_norm = np.linalg.norm(right)
        right = right / max(right_norm, 1e-6)
        up = np.cross(right, forward)

        camera_data = bpy.data.cameras.new(name="Camera")
        camera_obj = bpy.data.objects.new("Camera", camera_data)
        self.scene_collection.objects.link(camera_obj)
        self.scene.camera = camera_obj
        camera_matrix = Matrix((
            (float(right[0]), float(up[0]), float(-forward[0]), float(camera_position[0])),
            (float(right[1]), float(up[1]), float(-forward[1]), float(camera_position[1])),
            (float(right[2]), float(up[2]), float(-forward[2]), float(camera_position[2])),
            (0.0, 0.0, 0.0, 1.0),
        ))
        camera_obj.matrix_world = camera_matrix

        camera_data.type = 'PERSP'
        camera_data.lens_unit = 'FOV'
        camera_data.angle = fov_y

        setup_time = time.perf_counter() - setup_start

        wall_transparency_records = {}
        object_transparency_records = []
        view_layer = self.scene.view_layers[0] if len(self.scene.view_layers) > 0 else None
        prev_view_layer_pass = view_layer.use_pass_z if view_layer else False
        if view_layer:
            view_layer.use_pass_z = render_depth

        if auto_transparent:
            transparent_wall_ids = util.find_walls_to_make_transparent(
                camera_position.tolist()[:2],
                look_at_target.tolist()[:2],
                self.context["meta"]["vertices"],
                self.context["walls"]
            )
            if transparent_wall_ids:
                print(f"🔍 检测到 {len(transparent_wall_ids)} 面遮挡墙体，设为透明...")
                for wall_id in transparent_wall_ids:
                    wall_obj = self.mesh_nodes["walls"].get(wall_id, {}).get("node")
                    replacements = util_bpy.apply_wall_transparency(wall_obj, transparent_alpha)
                    if replacements:
                        wall_transparency_records[wall_id] = replacements

            camera_z = camera_position[2]
            if camera_z > z_max:
                ceiling_info = self.mesh_nodes.get("ceiling")
                if ceiling_info:
                    record = util_bpy.apply_object_transparency(ceiling_info.get("node"), transparent_alpha)
                    if record:
                        object_transparency_records.append(record)
            elif camera_z < 0:
                floor_info = self.mesh_nodes.get("floor")
                if floor_info:
                    record = util_bpy.apply_object_transparency(floor_info.get("node"), transparent_alpha)
                    if record:
                        object_transparency_records.append(record)

        # 设置渲染输出
        self.scene.render.resolution_x = width
        self.scene.render.resolution_y = height
        self.scene.render.filepath = output_path

        print(f"🎬 渲染中 ({width}x{height})...")
        try:
            render_start = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            render_time = time.perf_counter() - render_start
            if render_depth:
                depth_path = os.path.join(
                    os.path.dirname(output_path),
                    f"{os.path.splitext(os.path.basename(output_path))[0]}_depth.exr",
                )
                util_bpy.render_depth_exr(self.scene, depth_path)
        finally:
            for wall_id, slots in wall_transparency_records.items():
                wall_obj = self.mesh_nodes["walls"].get(wall_id, {}).get("node")
                util_bpy.restore_wall_transparency(wall_obj, slots)
            for record in object_transparency_records:
                util_bpy.restore_object_transparency(record)
            if view_layer:
                view_layer.use_pass_z = prev_view_layer_pass
            if camera_obj:
                bpy.data.objects.remove(camera_obj, do_unlink=True)
            if camera_data:
                bpy.data.cameras.remove(camera_data, do_unlink=True)

        # Save camera parameters（与 fast_scene.py 一致）
        save_start = time.perf_counter()
        para_path = os.path.join(os.path.dirname(output_path), 'camera_para.json')
        with open(para_path, 'w') as f:
            json.dump({
                "camera_position": camera_position.tolist(),
                "aspectRatio": 1.0,
                "fov_y": fov_y
            }, f, indent=4)
        save_time = time.perf_counter() - save_start

        total_time = time.perf_counter() - total_start
        print(f"✅ 渲染完成! 保存至: {output_path}")
        print(f"   构建: {construct_time:.2f}s, 设置: {setup_time:.2f}s, 渲染: {render_time:.2f}s, 保存: {save_time:.2f}s")
        print(f"   总耗时: {total_time:.2f}s")

    def render_view(self, output_path: str, camera_position: list, look_at_target: list = None, 
                    width: int = 1024, height: int = 1024, up_vector: list = None, 
                    auto_fov: bool = True, manual_fov: float = None,
                    auto_transparent: bool = True, transparent_alpha: float = 0.3,
                    geometry_mode: str = "gltf", render_depth: bool = False,
                    show_wall: bool = True, show_window: bool = True, show_door: bool = True, show_ceiling: bool = True,
                    use_HDRI: bool = True, hdri_transparent_background: bool = False):
        total_start = time.perf_counter()

        construct_time = 0.0
        if self.mesh_nodes["floor"] is None:
            construct_start = time.perf_counter()
            self.construct_scene(geometry_mode=geometry_mode,
                                 show_wall=show_wall, show_window=show_window,
                                 show_door=show_door, show_ceiling=show_ceiling)
            construct_time = time.perf_counter() - construct_start

        setup_start = time.perf_counter()
        if not self.if_set_lights:
            self.setup_lighting()
        if use_HDRI:
            hdri_path = self.config.get("hdri_path")
            if hdri_path and util_bpy.apply_hdri_to_world(self.scene, hdri_path, strength=1.0):
                print(f"🌇 使用 HDRI 环境光: {hdri_path}")
                if hdri_transparent_background:
                    self.scene.render.film_transparent = True
            else:
                print(f"⚠️ HDRI 文件不可用或未配置: {hdri_path}")
        setup_time = time.perf_counter() - setup_start

        center = self.context["meta"]["center"]
        z_max = self.context["meta"]["z_max"]
        bounds = self.context["meta"]["bounds"]

        if look_at_target is None:
            look_at_target = [center[0], center[1], z_max / 2]
        if up_vector is None:
            up_vector = [0.0, 0.0, 1.0]

        camera_position = np.array(camera_position, dtype=float)
        look_at_target = np.array(look_at_target, dtype=float)
        up_vector = np.array(up_vector, dtype=float)
        if np.linalg.norm(up_vector) < 1e-6:
            up_vector = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            up_vector = up_vector / np.linalg.norm(up_vector)

        forward = look_at_target - camera_position
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up_vector)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            fallback_up = np.array([0.0, 1.0, 0.0], dtype=float)
            right = np.cross(forward, fallback_up)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                fallback_up = np.array([1.0, 0.0, 0.0], dtype=float)
                right = np.cross(forward, fallback_up)
                right_norm = np.linalg.norm(right)
        right = right / max(right_norm, 1e-6)
        up = np.cross(right, forward)

        camera_matrix = Matrix((
            (float(right[0]), float(up[0]), float(-forward[0]), float(camera_position[0])),
            (float(right[1]), float(up[1]), float(-forward[1]), float(camera_position[1])),
            (float(right[2]), float(up[2]), float(-forward[2]), float(camera_position[2])),
            (0.0, 0.0, 0.0, 1.0),
        ))

        if manual_fov is not None:
            fov_y = np.radians(manual_fov)
        elif auto_fov:
            indoor_fov = self.config.get('indoor_fov', 120)
            outdoor_fov_scale = self.config.get('outdoor_fov_scale', 1.05)
            fov_y = util.calculate_optimal_fov(
                camera_position, look_at_target,
                self.context["meta"]["vertices"],
                z_max, bounds,
                indoor_fov, outdoor_fov_scale
            )
        else:
            fov_y = np.radians(70.0)

        wall_transparency_records = {}
        object_transparency_records = []
        if auto_transparent:
            transparent_wall_ids = util.find_walls_to_make_transparent(
                camera_position.tolist()[:2],
                look_at_target.tolist()[:2],
                self.context["meta"]["vertices"],
                self.context["walls"]
            )
            if transparent_wall_ids:
                print(f"🔍 检测到 {len(transparent_wall_ids)} 面遮挡墙体，设为透明...")
                for wall_id in transparent_wall_ids:
                    wall_obj = self.mesh_nodes["walls"].get(wall_id, {}).get("node")
                    replacements = util_bpy.apply_wall_transparency(wall_obj, transparent_alpha)
                    if replacements:
                        wall_transparency_records[wall_id] = replacements

            camera_z = camera_position[2]
            if camera_z > z_max:
                ceiling_info = self.mesh_nodes.get("ceiling")
                if ceiling_info:
                    ceiling_record = util_bpy.apply_object_transparency(ceiling_info.get("node"), transparent_alpha)
                    if ceiling_record:
                        object_transparency_records.append(ceiling_record)
            elif camera_z < 0:
                floor_info = self.mesh_nodes.get("floor")
                if floor_info:
                    floor_record = util_bpy.apply_object_transparency(floor_info.get("node"), transparent_alpha)
                    if floor_record:
                        object_transparency_records.append(floor_record)

        camera_data = bpy.data.cameras.new(name="Camera")
        camera_obj = bpy.data.objects.new("Camera", camera_data)
        self.scene_collection.objects.link(camera_obj)
        prev_camera = self.scene.camera
        self.scene.camera = camera_obj
        camera_obj.matrix_world = camera_matrix

        camera_data.type = 'PERSP'
        camera_data.lens_unit = 'FOV'
        camera_data.angle = float(fov_y)

        view_layer = self.scene.view_layers[0] if len(self.scene.view_layers) > 0 else None
        prev_view_layer_pass = view_layer.use_pass_z if view_layer else False
        if view_layer:
            view_layer.use_pass_z = render_depth

        self.scene.render.resolution_x = width
        self.scene.render.resolution_y = height
        self.scene.render.filepath = output_path

        print(f"🎬 渲染中 ({width}x{height})...")
        try:
            render_start = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            render_time = time.perf_counter() - render_start
            if render_depth:
                depth_path = os.path.join(
                    os.path.dirname(output_path),
                    f"{os.path.splitext(os.path.basename(output_path))[0]}_depth.exr",
                )
                util_bpy.render_depth_exr(self.scene, depth_path)
            total_time = time.perf_counter() - total_start
            print(f"✅ 渲染完成! 保存至: {output_path}")
            print(f"   构建: {construct_time:.2f}s, 设置: {setup_time:.2f}s, 渲染: {render_time:.2f}s")
            if render_depth:
                print("   深度图已生成")
            print(f"   总耗时: {total_time:.2f}s")
        except Exception as e:
            print(f"❌ 渲染失败: {e}")
            raise
        finally:
            for wall_id, slots in wall_transparency_records.items():
                wall_obj = self.mesh_nodes["walls"].get(wall_id, {}).get("node")
                util_bpy.restore_wall_transparency(wall_obj, slots)
            for record in object_transparency_records:
                util_bpy.restore_object_transparency(record)
            if view_layer:
                view_layer.use_pass_z = prev_view_layer_pass
            if prev_camera:
                self.scene.camera = prev_camera
            else:
                self.scene.camera = None
            if camera_obj:
                bpy.data.objects.remove(camera_obj, do_unlink=True)
            if camera_data:
                bpy.data.cameras.remove(camera_data, do_unlink=True)
    
# ==================== 测试代码 ====================




if __name__ == "__main__":
    print("="*60)
    print("🔍 检查 Blender Python 环境")
    print("="*60)
    print(f"✅ bpy 版本: {bpy.app.version_string}")
    print(f"✅ Blender 版本: {'.'.join(map(str, bpy.app.version))}")
    print()
    
    # 显示渲染设备
    print("🖥️  可用的渲染设备:")
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for device in prefs.get_devices_for_type('CUDA'):
        status = "✅ 已启用" if device.use else "⚪ 未启用"
        print(f"   - {device.name} (type: {device.type}) {status}")
    print()
    
    # 测试场景渲染
    line = 15
    jsonl_path = '/root/datasets/manycore/spatialllm_raw.jsonl'
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    test_dir = os.path.join(base_dir, 'test_bpy')
    os.makedirs(test_dir, exist_ok=True)
    topdown_path = os.path.join(test_dir, 'scene_topdown.png')
    
    try:
        data = util.read_jsonl_line(jsonl_path, line)
        print(f"📁 加载场景: {data['room']['room_type']} ({len(data['bbox'])} 个物体)")
        
        ctx = BpySceneCtx(data['room']['room_type'])
        ctx.add_walls(data['wall'])
        ctx.add_doors(data.get('door', []))
        ctx.add_windows(data.get('window', []))
        ctx.add_boxes(data['bbox'])
        
        print("\n" + "="*60)
        print("渲染俯视图")
        print("="*60)
        ctx.topdown_view(topdown_path, use_bbox_geometry=False,
                        show_wall=True, show_window=True, show_door=True, show_ceiling=False, auto_fov=True, auto_transparent=False, use_HDRI=True, render_depth=True)

        look_at = [
            (ctx.context["meta"]["center"][0], ctx.context["meta"]["center"][1], ctx.context["meta"]["z_max"] / 2)
        ][0]
        views = [
            ("right", [look_at[0] + ctx.context["meta"]["span"][0]/2 - 0.1, look_at[1], ctx.context["meta"]["z_max"] * 2/3]),
            ("left", [look_at[0] - ctx.context["meta"]["span"][0]/2 + 0.1, look_at[1], ctx.context["meta"]["z_max"] * 2/3]),
            ("front", [look_at[0], look_at[1] + ctx.context["meta"]["span"][1]/2 - 0.1, ctx.context["meta"]["z_max"] * 2/3]),
            ("back", [look_at[0], look_at[1] - ctx.context["meta"]["span"][1]/2 + 0.1, ctx.context["meta"]["z_max"] * 2/3]),
        ]
        for view_name, camera_pos in views:
            output_file = os.path.join(test_dir, f'scene_view_{view_name}.png')
            print(f"\n📷 渲染 {view_name} 视角...")
            print(f"   相机位置: {camera_pos}")
            print(f"   看向目标: {look_at}")
            ctx.render_view(
                output_path=output_file,
                camera_position=camera_pos,
                look_at_target=look_at,
                width=1024,
                height=1024,
                transparent_alpha=0.3,
                use_bbox_geometry=False,
                show_wall=True,
                show_window=False,
                show_door=False, 
                show_ceiling=False, 
                auto_fov=True, 
                auto_transparent=False, 
                use_HDRI=True, 
                render_depth=False
            )
        
        print("\n✅ 全部完成!")
        print(ctx.get_context())
        
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
