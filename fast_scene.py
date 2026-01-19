import os
# Set PyOpenGL platform to EGL BEFORE importing pyrender
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import json
import yaml
import numpy as np
import pyrender
import trimesh
import time
import colorsys
import hashlib
import imageio
from typing import List, Dict, Any, Optional

try:
    from . import util
except ImportError:
    import util

class SceneCtx:
    """场景上下文管理器"""
    # 立方体贴图每个面的坐标轴（right/up/forward），与标准Cubemap约定一致
    CUBEMAP_FACE_AXES = {
        "right": {
            "forward": np.array([1.0, 0.0, 0.0]),
            "up": np.array([0.0, 0.0, 1.0])
        },
        "left": {
            "forward": np.array([-1.0, 0.0, 0.0]),
            "up": np.array([0.0, 0.0, 1.0])
        },
        "front": {
            "forward": np.array([0.0, 1.0, 0.0]),
            "up": np.array([0.0, 0.0, 1.0])
        },
        "back": {
            "forward": np.array([0.0, -1.0, 0.0]),
            "up": np.array([0.0, 0.0, 1.0])
        },
        "top": {
            "forward": np.array([0.0, 0.0, 1.0]),
            "up": np.array([0.0, -1.0, 0.0])
        },
        "bottom": {
            "forward": np.array([0.0, 0.0, -1.0]),
            "up": np.array([0.0, 1.0, 0.0])
        }
    }

    def __init__(self, scene_type: str):
        self.context = {
            "meta": {"scene_type": scene_type},
            "walls": {},
            "boxes": {}
        }
        self.scene = None
        self.if_set_lights = False
        
        # 保存所有节点引用，key为唯一ID
        self.mesh_nodes = {
            "walls": {},      # wall_id -> node
            "doors": {},      # door_id -> node
            "windows": {},    # window_id -> node
            "boxes": {},      # box_id -> node
            "floor": None,    # floor node
        }

        # Load config
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def add_walls(self, walls: List[Dict[str, Any]]):
        """添加墙体并计算场景元数据"""
        # 转换格式: p/q -> s/e
        walls_converted = [{"s": w["p"][:2], "e": w["q"][:2], "height": w["height"]} for w in walls]

        # 计算bounds和vertices
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
        
        # 调试信息：检查墙体和顶点的对应关系
        print(f"📐 输入墙体数量: {len(walls_converted)}")
        print(f"📐 计算得到的顶点数量: {len(vertices)}")
        print(f"📐 Vertices顺序: {[f'({v[0]:.2f},{v[1]:.2f})' for v in vertices]}")
        
        # 检查每面墙是否在顶点列表中
        walls_not_in_vertices = []
        for i, wall in enumerate(walls_converted):
            s_in = tuple(wall["s"]) in vertices
            e_in = tuple(wall["e"]) in vertices
            if not (s_in and e_in):
                walls_not_in_vertices.append(i)
                print(f"   ⚠️  墙体{i} 的端点不在vertices中: s={wall['s']} ({s_in}), e={wall['e']} ({e_in})")
        
        if not walls_not_in_vertices:
            print(f"✅ 所有墙体的端点都在vertices中，没有墙体被分割")

        # 添加墙体
        for wall in walls_converted:
            wall_id = util.generate_unique_id()
            self.context["walls"][wall_id] = {
                **wall,
                "orientation": util.calculate_wall_orientation(wall["s"], wall["e"], self.context["meta"]["vertices"]),
                "doors": {},
                "windows": {}
            }

    def add_wall(self, s: List[float], e: List[float], height: float):
        """添加单面墙"""
        # 简单实现：重新调用add_walls
        current_walls = [{"p": w["s"] + [0], "q": w["e"] + [0], "height": w["height"]}
                        for w in self.context["walls"].values()]
        current_walls.append({"p": s[:2] + [0], "q": e[:2] + [0], "height": height})
        self.context["walls"] = {}  # 清空
        self.add_walls(current_walls)

    def add_doors(self, doors: List[Dict[str, Any]]):
        """批量添加门"""
        for door in doors:
            self.add_door(door["center"], door["width"], door["height"])

    def add_door(self, center: List[float], width: float, height: float):
        """添加单个门"""
        wall_id = util.find_closest_wall(center, self.context["walls"])
        if wall_id:
            wall = self.context["walls"][wall_id]
            door_id = util.generate_unique_id()
            self.context["walls"][wall_id]["doors"][door_id] = {
                "center": util.snap_to_wall(center, wall),
                "width": width,
                "height": height
            }

    def add_windows(self, windows: List[Dict[str, Any]]):
        """批量添加窗"""
        for window in windows:
            self.add_window(window["center"], window["width"], window["height"])

    def add_window(self, center: List[float], width: float, height: float):
        """添加单个窗"""
        wall_id = util.find_closest_wall(center, self.context["walls"])
        if wall_id:
            wall = self.context["walls"][wall_id]
            window_id = util.generate_unique_id()
            self.context["walls"][wall_id]["windows"][window_id] = {
                "center": util.snap_to_wall(center, wall),
                "width": width,
                "height": height
            }

    def add_boxes(self, boxes: List[Dict[str, Any]]):
        """批量添加家具"""
        for box in boxes:
            self.add_box(box["center"], box["angle_z"], box["scale"], box["class"],
                        box.get("label"), box.get("caption"), box.get("mesh_id"))

    def add_box(self, center: List[float], angle_z: float, scale: List[float],
                class_name: str, label: Optional[str] = None,
                caption: Optional[str] = None, mesh_id: Optional[int] = None) -> str:
        """添加单个家具，返回ID"""
        box_id = util.generate_unique_id()
        box_data = {"center": center, "angle_z": angle_z, "scale": scale, "class": class_name}

        if label: box_data["label"] = label
        if caption: box_data["caption"] = caption
        if mesh_id: box_data["mesh_id"] = mesh_id

        self.context["boxes"][box_id] = box_data

        # 更新z_max
        box_top = center[2] + scale[2] / 2
        if box_top > self.context["meta"].get("z_max", 0):
            self.context["meta"]["z_max"] = box_top

        return box_id

    def delete_box(self, box_id: str):
        """删除家具并重新计算z_max"""
        if box_id not in self.context["boxes"]:
            raise ValueError(f"Box ID {box_id} not found")

        del self.context["boxes"][box_id]
        self.scene = None  # 标记需要重建

        # 重新计算z_max: 取墙体和剩余boxes中的最大高度
        wall_max = max((w["height"] for w in self.context["walls"].values()), default=0)

        box_max = 0
        for box in self.context["boxes"].values():
            box_top = box["center"][2] + box["scale"][2] / 2
            box_max = max(box_max, box_top)

        self.context["meta"]["z_max"] = max(wall_max, box_max)

    def get_context(self) -> Dict:
        """获取场景上下文"""
        return self.context
    
    def get_boxes(self) -> Dict:
        """获取所有家具boxes"""
        return self.context["boxes"]

    def export_wall_ssl(self, output_dir: str):
        """
        导出墙体SSL格式文件 (只包含 Room 和 Wall)
        基于计算出的最小面积多边形的vertices
        
        Args:
            output_dir: 输出目录路径
        """
        output_path = os.path.join(output_dir, 'wall_ssl.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        
        # 生成 room_id
        room_id = util.generate_unique_id()
        room_type = self.context["meta"]["scene_type"]
        lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
        
        # 从计算出的vertices生成墙体
        vertices = self.context["meta"]["vertices"]
        height = self.context["meta"]["z_max"]  # 使用场景的最大高度
        
        # 将连续的顶点对转换为墙体
        for i in range(len(vertices)):
            wall_id = util.generate_unique_id()
            # 当前顶点和下一个顶点（循环）
            p = list(vertices[i]) + [0.0]  # 转换为3D坐标
            q = list(vertices[(i + 1) % len(vertices)]) + [0.0]
            lines.append(f'Wall(id="{wall_id}", room_id="{room_id}", p={p}, q={q}, height={height})')
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ 墙体SSL已导出: {output_path} ({len(vertices)} 面墙)")
        return output_path

    def export_wall_hole_ssl(self, output_dir: str):
        """
        导出墙体和门窗SSL格式文件 (包含 Room, Wall, Door, Window)
        基于计算出的最小面积多边形的vertices
        
        Args:
            output_dir: 输出目录路径
        """
        output_path = os.path.join(output_dir, 'wall_hole_ssl.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        
        # 生成 room_id
        room_id = util.generate_unique_id()
        room_type = self.context["meta"]["scene_type"]
        lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
        
        # 从计算出的vertices生成墙体，并记录wall_id和墙体几何信息的映射
        vertices = self.context["meta"]["vertices"]
        height = self.context["meta"]["z_max"]
        
        # 存储新墙体的ID和几何信息 [(wall_id, p, q), ...]
        new_walls = []
        for i in range(len(vertices)):
            wall_id = util.generate_unique_id()
            p = list(vertices[i]) + [0.0]
            q = list(vertices[(i + 1) % len(vertices)]) + [0.0]
            new_walls.append((wall_id, p, q))
            lines.append(f'Wall(id="{wall_id}", room_id="{room_id}", p={p}, q={q}, height={height})')
        
        # 收集所有门窗，并找到它们最接近的新墙体
        doors = []
        windows = []
        
        for wall_id, wall in self.context["walls"].items():
            for door_id, door in wall.get("doors", {}).items():
                doors.append((door_id, door))
            for window_id, window in wall.get("windows", {}).items():
                windows.append((window_id, window))
        
        # 为每个门找到最接近的新墙体并导出
        for door_id, door in doors:
            center = door["center"]
            width = door["width"]
            door_height = door["height"]
            
            # 找到最接近的新墙体
            closest_wall_id = util.find_closest_wall_from_list(center, new_walls)
            lines.append(f'Door(id="{door_id}", wall_id="{closest_wall_id}", center={center}, width={width}, height={door_height})')
        
        # 为每个窗找到最接近的新墙体并导出
        for window_id, window in windows:
            center = window["center"]
            width = window["width"]
            window_height = window["height"]
            
            # 找到最接近的新墙体
            closest_wall_id = util.find_closest_wall_from_list(center, new_walls)
            lines.append(f'Window(id="{window_id}", wall_id="{closest_wall_id}", center={center}, width={width}, height={window_height})')
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ 墙体和门窗SSL已导出: {output_path} ({len(vertices)} 面墙, {len(doors)} 个门, {len(windows)} 个窗)")
        return output_path

    def export_ssl(self, output_dir: str):
        """
        导出完整SSL格式文件 (包含 Room, Wall, Door, Window, Bbox)
        基于计算出的最小面积多边形的vertices
        
        Args:
            output_dir: 输出目录路径
        """
        output_path = os.path.join(output_dir, 'ssl.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        
        # 生成 room_id
        room_id = util.generate_unique_id()
        room_type = self.context["meta"]["scene_type"]
        lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
        
        # 从计算出的vertices生成墙体
        vertices = self.context["meta"]["vertices"]
        height = self.context["meta"]["z_max"]
        
        new_walls = []
        for i in range(len(vertices)):
            wall_id = util.generate_unique_id()
            p = list(vertices[i]) + [0.0]
            q = list(vertices[(i + 1) % len(vertices)]) + [0.0]
            new_walls.append((wall_id, p, q))
            lines.append(f'Wall(id="{wall_id}", room_id="{room_id}", p={p}, q={q}, height={height})')
        
        # 收集所有门窗
        doors = []
        windows = []
        
        for wall_id, wall in self.context["walls"].items():
            for door_id, door in wall.get("doors", {}).items():
                doors.append((door_id, door))
            for window_id, window in wall.get("windows", {}).items():
                windows.append((window_id, window))
        
        # 导出门
        for door_id, door in doors:
            center = door["center"]
            width = door["width"]
            door_height = door["height"]
            closest_wall_id = util.find_closest_wall_from_list(center, new_walls)
            lines.append(f'Door(id="{door_id}", wall_id="{closest_wall_id}", center={center}, width={width}, height={door_height})')
        
        # 导出窗
        for window_id, window in windows:
            center = window["center"]
            width = window["width"]
            window_height = window["height"]
            closest_wall_id = util.find_closest_wall_from_list(center, new_walls)
            lines.append(f'Window(id="{window_id}", wall_id="{closest_wall_id}", center={center}, width={width}, height={window_height})')
        
        # 导出所有家具
        for box_id, box in self.context["boxes"].items():
            label = box.get("label", box.get("class", "unknown"))
            center = box["center"]
            angle_z = box["angle_z"]
            scale = box["scale"]
            
            # 构建 Bbox 行，包含 mesh_id（如果有的话）
            bbox_str = f'Bbox(id="{box_id}", room_id="{room_id}", label="{label}", center={center}, angle_z={angle_z}, scale={scale}'
            
            if box.get("mesh_id") is not None:
                bbox_str += f', mesh_id={box["mesh_id"]}'
            
            bbox_str += ')'
            lines.append(bbox_str)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ 完整SSL已导出: {output_path} ({len(vertices)} 面墙, {len(doors)} 个门, {len(windows)} 个窗, {len(self.context['boxes'])} 个家具)")
        return output_path
    
    def construct_floor(self, show_wall=True, show_window=True, show_door=True):
        """构建地板和墙体
        
        Args:
            show_wall: 是否显示墙体 (默认True)
            show_window: 是否显示窗户 (默认True)
            show_door: 是否显示门 (默认True)
        """
        self.scene = pyrender.Scene()
        vertices = self.context["meta"]["vertices"]

        if len(vertices) < 3:
            return

        # 创建地板
        bounds = self.context["meta"]["bounds"]
        texture_scale = self.config.get("texture_scale", None)
        wall_thickness = self.config.get("wall_thickness", 0.1)
        floor_mesh = util.create_floor_mesh(vertices, bounds, texture_scale=texture_scale)

        # 加载纹理
        texture_path = self.config.get("floor_texture_path")
        if texture_path and os.path.exists(texture_path):
            try:
                import PIL.Image
                texture = PIL.Image.open(texture_path).convert('RGB')
                floor_mesh.visual = trimesh.visual.TextureVisuals(uv=floor_mesh.visual.uv, image=texture)
            except:
                floor_mesh.visual.material = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[0.8, 0.7, 0.5, 1.0])

        self.scene.add(pyrender.Mesh.from_trimesh(floor_mesh))

        # 创建墙体 - 为每面墙单独创建mesh
        if show_wall:
            wall_color = self.config["wall_color"]
            wall_texture_path = self.config.get("wall_texture_path")

            # 加载墙体纹理
            wall_texture = None
            if wall_texture_path and os.path.exists(wall_texture_path):
                try:
                    import PIL.Image
                    import PIL.ImageEnhance
                    wall_texture = PIL.Image.open(wall_texture_path).convert('RGB')
                    
                    # 提升贴图亮度，让白色贴图渲染时更亮
                    enhancer = PIL.ImageEnhance.Brightness(wall_texture)
                    wall_texture = enhancer.enhance(1.5)  # 提升150%亮度
                    
                    print(f"✅ 成功加载墙体纹理: {wall_texture_path} ({wall_texture.size}), 亮度提升180%")
                except Exception as e:
                    print(f"⚠️ 加载墙体纹理失败: {e}")
            
            # 为每面墙单独创建mesh（类似floor的方式）
            for wall_id, wall in self.context["walls"].items():
                # 收集该墙的门窗信息
                openings = []
                for door in wall.get("doors", {}).values():
                    openings.append(door)
                for window in wall.get("windows", {}).values():
                    openings.append(window)
                
                # 创建单面墙的mesh（带门窗挖洞）
                wall_mesh = util.create_single_wall_mesh(
                    wall["s"],
                    wall["e"],
                    wall["height"],
                    wall["orientation"],
                    chip=False,  # 使用带厚度的实体墙模式
                    openings=openings if openings else None,
                    wall_thickness=wall_thickness,
                    texture_scale=texture_scale
                )

                # 应用纹理或颜色（完全照抄floor的方式）
                if wall_texture:
                    # 使用TextureVisuals，贴图亮度已在加载时提升
                    wall_mesh.visual = trimesh.visual.TextureVisuals(
                        uv=wall_mesh.visual.uv,
                        image=wall_texture
                    )
                else:
                    # 使用PBR材质设置颜色
                    wall_mesh.visual.material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=wall_color
                    )

                # 添加节点并保存引用（启用双面渲染）
                pyrender_mesh = pyrender.Mesh.from_trimesh(wall_mesh)
                
                
                node = self.scene.add(pyrender_mesh)
                self.mesh_nodes["walls"][wall_id] = {
                    "node": node,
                    "mesh": wall_mesh,  # 保存trimesh对象用于相交检测
                    "wall_data": wall
                }
                
                # 为墙体添加边缘线条，使分界更明显（可通过配置启用/禁用）
                if self.config.get("wall_edge_enabled", True):
                    edge_color = self.config.get("wall_edge_color", [0.1, 0.1, 0.1, 1.0])
                    edge_offset = self.config.get("wall_edge_offset", 0.01)  # 边缘线向内偏移距离
                    edge_lines = util.create_wall_edge_lines(
                        wall["s"], wall["e"], wall["height"], 
                        wall["orientation"], edge_color, edge_offset
                    )
                    if edge_lines:
                        edge_node = self.scene.add(edge_lines)
                        # 将边缘线条节点也保存到墙体引用中
                        self.mesh_nodes["walls"][wall_id]["edge_node"] = edge_node
        else:
            print("🚫 跳过墙体创建 (show_wall=False)")
        
        # 添加门窗
        if show_door or show_window:
            door_texture_path = self.config.get("door_texture_path", "/root/projects/utils/fast-scene/gltf/door.png")
            window_texture_path = self.config.get("window_texture_path", "/root/projects/utils/fast-scene/gltf/window.png")
            
            doors_and_windows = util.create_windows_and_doors(
                self.context["walls"],
                door_texture_path if show_door else None,
                window_texture_path if show_window else None
            )
            
            print(f"🚪 添加门窗 ({len(doors_and_windows)} 个)...")
            for item in doors_and_windows:
                item_type = item["type"]
                item_id = item["id"]
                item_mesh = item["mesh"]
                
                # 根据参数决定是否添加
                should_add = False
                if item_type == "door" and show_door:
                    should_add = True
                elif item_type == "window" and show_window:
                    should_add = True
                
                if should_add and item_mesh and len(item_mesh.vertices) > 0:
                    node = self.scene.add(pyrender.Mesh.from_trimesh(item_mesh))
                    self.mesh_nodes[f"{item_type}s"][item_id] = {
                        "node": node,
                        "mesh": item_mesh,
                        f"{item_type}_data": {
                            "wall_id": item["wall_id"]
                        }
                    }
                    print(f"  ✅ {item_type} 添加成功")
        else:
            print("🚫 跳过门窗创建 (show_door=False, show_window=False)")
        
        # 创建天花板
        z_max = self.context["meta"]["z_max"]
        ceiling_mesh = util.create_ceiling_mesh(vertices, bounds, z_max)
        
        # 加载天花板纹理（完全照抄墙体逻辑）
        ceiling_color = [0.9, 0.9, 0.9, 1.0]
        ceiling_texture_path = self.config.get("ceiling_texture_path")
        
        # 加载天花板纹理
        ceiling_texture = None
        if ceiling_texture_path and os.path.exists(ceiling_texture_path):
            try:
                import PIL.Image
                import PIL.ImageEnhance
                ceiling_texture = PIL.Image.open(ceiling_texture_path).convert('RGB')
                
                # 提升贴图亮度，让白色贴图渲染时更亮
                enhancer = PIL.ImageEnhance.Brightness(ceiling_texture)
                ceiling_texture = enhancer.enhance(1.5)  # 提升150%亮度
                
                print(f"✅ 成功加载天花板纹理: {ceiling_texture_path} ({ceiling_texture.size}), 亮度提升150%")
            except Exception as e:
                print(f"⚠️ 加载天花板纹理失败: {e}")
        
        # 应用纹理或颜色（完全照抄墙体的方式）
        if ceiling_texture:
            # 使用TextureVisuals，贴图亮度已在加载时提升
            ceiling_mesh.visual = trimesh.visual.TextureVisuals(
                uv=ceiling_mesh.visual.uv,
                image=ceiling_texture
            )
        else:
            # 使用PBR材质设置颜色
            ceiling_mesh.visual.material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=ceiling_color
            )
        
        self.scene.add(pyrender.Mesh.from_trimesh(ceiling_mesh))
        print(f"✅ 已添加天花板 (高度: {z_max:.2f}m)")

    def construct_scene(
        self,
        show_wall: bool = True,
        show_window: bool = True,
        show_door: bool = True,
        fix_coordinate: bool = False,
        use_bbox_geometry: bool = False,
    ):
        """构建完整场景
        
        Args:
            show_wall: 是否渲染墙体
            show_window: 是否渲染窗
            show_door: 是否渲染门
            fix_coordinate: 是否对模型进行坐标系修正
            use_bbox_geometry: 是否强制全部家具使用bbox几何体
        """
        total_start = time.perf_counter()
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

        self.construct_floor(show_wall=show_wall, show_window=show_window, show_door=show_door)

        print(f"📦 加载家具 ({len(self.context['boxes'])} 个物体)...")
        success_count = 0
        load_time = 0
        transform_time = 0

        # 根据参数统一设置/清除bbox标记
        for box in self.context["boxes"].values():
            if use_bbox_geometry:
                box["use_bbox_geometry"] = True
            else:
                box.pop("use_bbox_geometry", None)

        for box in self.context["boxes"].values():
            # 检查是否使用bbox几何体
            if box.get("use_bbox_geometry"):
                try:
                    # 创建bbox几何体
                    bbox_mesh = self._create_bbox_geometry(
                        box["center"],
                        box["scale"],
                        box["angle_z"],
                        color=self._get_class_color(box.get("class"))
                    )
                    
                    # 添加到场景
                    box_id = box.get("id", util.generate_unique_id())
                    pyrender_mesh = pyrender.Mesh.from_trimesh(bbox_mesh)
                    node = self.scene.add(pyrender_mesh)
                    self.mesh_nodes["boxes"][box_id] = {
                        "node": node,
                        "mesh": bbox_mesh,
                        "box_data": box
                    }
                    success_count += 1
                    print(f"  ✅ {box.get('class', 'unknown')} (bbox几何体) 添加成功")
                except Exception as e:
                    print(f"  ❌ {box.get('class', 'unknown')} (bbox几何体): {e}")
                continue
            
            if not box.get("mesh_id"):
                continue

            # 统计加载时间
            load_start = time.perf_counter()
            loaded = util.load_mesh(box["mesh_id"], self.config)
            load_time += time.perf_counter() - load_start

            if not loaded:
                continue

            try:
                # 统计变换时间
                transform_start = time.perf_counter()

                # 阶段1: 坐标系修正 (GLTF Y-up -> Scene Z-up)
                if fix_coordinate:
                    if isinstance(loaded, trimesh.Scene):
                        bounds = loaded.bounds
                        current_center = (bounds[0] + bounds[1]) / 2
                    else:
                        current_center = loaded.centroid

                    # 直接操作vertices: 移到原点 + 坐标变换 [x,y,z] -> [x,-z,y]
                    if isinstance(loaded, trimesh.Scene):
                        for geom in loaded.geometry.values():
                            if hasattr(geom, 'vertices'):
                                geom.vertices -= current_center
                                v = geom.vertices.copy()
                                geom.vertices[:, 0] = v[:, 0]   # x' = x
                                geom.vertices[:, 1] = -v[:, 2]  # y' = -z
                                geom.vertices[:, 2] = v[:, 1]   # z' = y
                    else:
                        if hasattr(loaded, 'vertices'):
                            loaded.vertices -= current_center
                            v = loaded.vertices.copy()
                            loaded.vertices[:, 0] = v[:, 0]
                            loaded.vertices[:, 1] = -v[:, 2]
                            loaded.vertices[:, 2] = v[:, 1]

                # 阶段2: Bbox变换 (Scale -> Rotate -> Translate)
                if isinstance(loaded, trimesh.Scene):
                    new_bounds = loaded.bounds
                    new_size = new_bounds[1] - new_bounds[0]
                else:
                    new_size = loaded.bounding_box.extents

                target_scale = np.array(box["scale"])
                target_center = np.array(box["center"])
                target_angle = box["angle_z"]

                # Handle zero dimensions
                if np.any(new_size == 0):
                    new_size = np.where(new_size == 0, 1.0, new_size)

                scale_factors = target_scale / new_size

                # Build rotation matrix (around Z-axis)
                # 物体初始已面向-y方向，直接绕z轴逆时针旋转（从+z看向-z，右手系）
                # 0°=-y方向，90°=+x方向，180°=+y方向，270°=-x方向
                theta = np.radians(target_angle)
                R = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0,              0,             1]
                ])

                # Apply bbox transform - Scale, Rotate, Translate
                # 注意：vertices是行向量，需要用R的转置
                if isinstance(loaded, trimesh.Scene):
                    for geom in loaded.geometry.values():
                        if hasattr(geom, 'vertices'):
                            geom.vertices *= scale_factors
                            geom.vertices = geom.vertices.dot(R.T)  # 行向量用转置
                            geom.vertices += target_center
                else:
                    if hasattr(loaded, 'vertices'):
                        loaded.vertices *= scale_factors
                        loaded.vertices = loaded.vertices.dot(R.T)  # 行向量用转置
                        loaded.vertices += target_center

                # 添加到场景并保存节点引用
                box_id = box.get("id", util.generate_unique_id())
                if isinstance(loaded, trimesh.Scene):
                    mesh_scene = pyrender.Scene.from_trimesh_scene(loaded)
                    nodes_list = []
                    for node in mesh_scene.get_nodes():
                        if node.mesh is not None:
                            added_node = self.scene.add(node.mesh)
                            nodes_list.append(added_node)
                    self.mesh_nodes["boxes"][box_id] = {
                        "nodes": nodes_list,  # 多个节点
                        "mesh": loaded,
                        "box_data": box
                    }
                else:
                    pyrender_mesh = pyrender.Mesh.from_trimesh(loaded)
                    node = self.scene.add(pyrender_mesh)
                    self.mesh_nodes["boxes"][box_id] = {
                        "node": node,  # 单个节点
                        "mesh": loaded,
                        "box_data": box

                    }

                transform_time += time.perf_counter() - transform_start
                success_count += 1
                print(f"  ✅ {box['class']} 添加成功")

            except Exception as e:
                print(f"  ❌ {box.get('class', 'unknown')}: {e}")

        total_time = time.perf_counter() - total_start
        print(f"✅ 场景构建完成! 成功添加 {success_count}/{len(self.context['boxes'])} 个物体")
        print(f"   构建耗时: {total_time:.2f}s (加载: {load_time:.2f}s, 变换: {transform_time:.2f}s)")

    def _get_class_color(self, class_name: Optional[str]):
        """
        根据类别名称生成稳定且可区分的颜色（RGBA，alpha固定0.9）。
        """
        label = (class_name or "default").lower()
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        h = int.from_bytes(digest[0:2], "big") / 65535.0
        s = 0.55 + (digest[2] / 255.0) * (0.85 - 0.55)
        v = 0.65 + (digest[3] / 255.0) * (0.85 - 0.65)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return [float(r), float(g), float(b), 0.9]

    def _create_bbox_geometry(self, center, scale, angle_z, color=None):
        """
        创建 bbox 几何体（立方体 + 箭头）
        
        Args:
            center: bbox中心位置 [x, y, z]
            scale: bbox尺寸 [x, y, z]
            angle_z: 绕z轴旋转角度（度），0°面向-y方向
            color: 立方体颜色，默认为半透明蓝色
        
        Returns:
            trimesh.Trimesh: 包含立方体和箭头的mesh
        """
        center = np.array(center)
        scale = np.array(scale)
        angle_z = float(angle_z)
        
        # 默认颜色：基于类别生成
        if color is None:
            color = list(self._get_class_color(None))
        else:
            color = list(color)
        
        # 1. 创建立方体（在原点，大小为scale）
        box_mesh = trimesh.creation.box(extents=scale)
        
        # 2. 创建箭头（在立方体顶部中心，指向物体朝向）
        # 箭头长度：取scale的x和y中较小值的30%
        arrow_length = min(scale[0], scale[1]) * 0.3
        arrow_radius = max(arrow_length * 0.05, 0.02)
        
        # 箭头在局部坐标系中指向-y方向（0°时的朝向）
        # 箭头会和立方体一起旋转，所以只需要指向局部-y方向即可
        arrow_direction_local = np.array([0, -1, 0])  # 局部坐标系中的-y方向
        
        # 箭头起点：立方体顶部中心
        arrow_start = np.array([0, 0, scale[2] / 2])
        
        # 箭头头部长度
        head_length = arrow_length * 0.25
        body_length = arrow_length * 0.75
        
        # 创建箭头身体（圆柱），默认沿z轴向上
        arrow_body = trimesh.creation.cylinder(
            radius=arrow_radius,
            height=body_length,
            sections=20
        )
        
        # 创建箭头头部（圆锥），默认沿z轴向上
        arrow_head = trimesh.creation.cone(
            radius=arrow_radius * 3,
            height=head_length,
            sections=20
        )
        
        # 旋转箭头到局部-y方向（从z轴向上旋转到-y方向）
        z_axis = np.array([0, 0, 1])
        arrow_transform = trimesh.geometry.align_vectors(z_axis, arrow_direction_local)
        
        # 应用旋转
        arrow_body.apply_transform(arrow_transform)
        arrow_head.apply_transform(arrow_transform)
        
        # 移动箭头到正确位置（在立方体顶部中心）
        # 箭头身体中心在起点沿箭头方向移动body_length/2
        arrow_body.apply_translation(arrow_start + arrow_direction_local * body_length / 2)
        # 箭头头部中心在起点沿箭头方向移动body_length + head_length/2
        arrow_head.apply_translation(arrow_start + arrow_direction_local * (body_length + head_length / 2))
        
        # 设置颜色
        box_mesh.visual.vertex_colors = color
        arrow_body.visual.vertex_colors = [0.0, 0.8, 0.0, 1.0]  # 绿色箭头
        arrow_head.visual.vertex_colors = [0.0, 0.8, 0.0, 1.0]  # 绿色箭头
        
        # 合并所有mesh
        combined_mesh = trimesh.util.concatenate([box_mesh, arrow_body, arrow_head])
        
        # 3. 应用旋转和位移变换（与fast-scene中bbox的变换逻辑一致）
        # 构建旋转矩阵（绕z轴逆时针旋转）
        theta = np.radians(angle_z)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        
        # 应用变换：先旋转，再平移
        # vertices是行向量，需要用R的转置
        combined_mesh.vertices = combined_mesh.vertices.dot(R.T)
        combined_mesh.vertices += center
        
        return combined_mesh

    def setup_lighting(self, point_light_intensity: float = 20,
                      use_point_lights: bool = True, ambient_light_color: list = None):
        """
        设置场景光照
        
        Args:
            point_light_intensity: 点光源强度
            use_point_lights: 是否使用点光源
            ambient_light_color: 环境光颜色 [R, G, B]，默认为白色 [1.0, 1.0, 1.0]
        """
        if self.if_set_lights:
            print("⚠️  光照已设置，跳过重复设置")
            return
            
        center = self.context["meta"]["center"]
        z_max = self.context["meta"]["z_max"]
        span = self.context["meta"]["span"]
        
        # 0. 设置全局环境光颜色（简单的环境光，不是IBL）
        if ambient_light_color is None:
            # 适中的环境光，配合贴图亮度增强使用
            ambient_light_color = [0.6, 0.6, 0.6]  # 60%亮度，平衡所有物体表现
        
        # pyrender的ambient_light是全局环境光，影响所有材质
        # 注意：这不是真正的环境贴图IBL，pyrender不支持HDR环境贴图
        self.scene.ambient_light = np.array(ambient_light_color, dtype=np.float32)        
        
        # 2. 在四个角落添加辅助点光源，确保局部细节被照亮
        if use_point_lights:
            corner_positions = [
                [center[0] - span[0]/2, center[1] - span[1]/2, z_max+3],  # 左下
                [center[0] + span[0]/2, center[1] - span[1]/2, z_max+3],  # 右下
                [center[0] - span[0]/2, center[1] + span[1]/2, z_max+3],  # 左上
                [center[0] + span[0]/2, center[1] + span[1]/2, z_max+3],  # 右上
                # [center[0], center[1], z_max-0.5],  # 天花板下方
                # [center[0] - span[0]/2, center[1] - span[1]/2, -1],  # 地板下方 
                # [center[0] + span[0]/2, center[1] - span[1]/2, -1],  # 地板下方
                # [center[0] - span[0]/2, center[1] + span[1]/2, -1],  # 地板下方
                # [center[0] + span[0]/2, center[1] + span[1]/2, -1],  # 地板下方
            ]
            for pos in corner_positions:
                light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=point_light_intensity)
                light_pose = trimesh.transformations.translation_matrix(pos)
                self.scene.add(light, pose=light_pose)
            
            print(f"💡 已添加点光源x{len(corner_positions)}，强度={point_light_intensity}")
        
        self.if_set_lights = True


    def topdown_view(self, output_path: str, width: int = 1024, height: int = 1024,
                     use_bbox_geometry: bool = False, show_wall: bool = True, 
                     show_window: bool = True, show_door: bool = True,
                     up_vector: list = None,
                     auto_fov: bool = True, manual_fov: float = None,
                     auto_transparent: bool = True, transparent_alpha: float = 0.3,
                     render_depth: bool = False):
        """俯视图渲染

        Args:
            output_path: 输出图片路径
            width: 图片宽度
            height: 图片高度
            use_bbox_geometry: 是否强制家具使用bbox几何体
            show_wall/show_window/show_door: 是否渲染对应组件
            up_vector: 相机上方向，默认 [0, 1, 0]
            auto_fov/manual_fov: FOV 计算逻辑（与 render_view 保持一致）
            auto_transparent: 是否透明遮挡墙体
            transparent_alpha: 透明墙体 alpha
            render_depth: 是否输出深度图
        """
        total_start = time.perf_counter()

        os.environ['PYOPENGL_PLATFORM'] = 'egl'

        construct_time = 0
        if self.scene is None:
            construct_start = time.perf_counter()
            self.construct_scene(use_bbox_geometry=use_bbox_geometry, 
                                 show_wall=show_wall, show_window=show_window, show_door=show_door)
            construct_time = time.perf_counter() - construct_start

        setup_start = time.perf_counter()
        if not self.if_set_lights:
            self.setup_lighting()

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

        forward = look_at_target - camera_position
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up_vector)
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

        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward
        camera_pose[:3, 3] = camera_position

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

        wall_transparency_ids = []
        if auto_transparent:
            wall_transparency_ids = util.find_walls_to_make_transparent(
                camera_position.tolist()[:2],
                look_at_target.tolist()[:2],
                self.context["meta"]["vertices"],
                self.context["walls"]
            )
            if wall_transparency_ids:
                print(f"🔍 检测到 {len(wall_transparency_ids)} 面遮挡墙体，设为透明...")
                for wall_id in wall_transparency_ids:
                    util.set_mesh_alpha(self.mesh_nodes, "walls", wall_id, transparent_alpha)

        camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=width/height)
        camera_node = self.scene.add(camera, pose=camera_pose)
        setup_time = time.perf_counter() - setup_start

        renderer = pyrender.OffscreenRenderer(width, height)
        try:
            print(f"🎬 渲染中 ({width}x{height})...")
            render_start = time.perf_counter()
            color, depth = renderer.render(self.scene)
            render_time = time.perf_counter() - render_start

            imageio.imwrite(output_path, color)
            if render_depth:
                depth_mm = np.clip(depth * 4000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
                base_name = os.path.splitext(os.path.basename(output_path))[0]
                depth_path = os.path.join(os.path.dirname(output_path), f"{base_name}_depth.png")
                imageio.imwrite(depth_path, depth_mm)

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
            if render_depth:
                print("   深度图已生成")
            print(f"   总耗时: {total_time:.2f}s")

        except Exception as e:
            print(f"❌ Render failed: {e}")
            raise
        finally:
            if auto_transparent and wall_transparency_ids:
                for wall_id in wall_transparency_ids:
                    util.reset_mesh_alpha(self.mesh_nodes, "walls", wall_id)
            self.scene.remove_node(camera_node)
            renderer.delete()
    
    def render_view(self, output_path: str, camera_position: list, look_at_target: list = None, 
                    width: int = 2048, height: int = 2048, up_vector: list = None, 
                    auto_fov: bool = False, manual_fov: float = None,
                    auto_transparent: bool = True, transparent_alpha: float = 0.3,
                    use_bbox_geometry: bool = False, render_depth: bool = False,
                    show_wall: bool = True, show_window: bool = True, show_door: bool = True):
        """
        通用渲染方法：从任意角度渲染场景
        
        Args:
            output_path: 输出图片路径
            camera_position: 相机位置 [x, y, z]
            look_at_target: 相机看向的目标点 [x, y, z]，默认为场景中心
            width: 图片宽度
            height: 图片高度
            up_vector: 相机的上方向 [x, y, z]，默认为[0, 0, 1]（Z轴向上）
            auto_fov: 是否自动计算FOV使场景刚好填满画面
            manual_fov: 手动指定FOV（角度），仅当auto_fov=False时生效
            auto_transparent: 是否自动将遮挡视线的墙体设为透明
            transparent_alpha: 透明墙体的alpha值 (0.0-1.0)
            show_wall: 是否显示墙体
            show_window: 是否显示窗户
            show_door: 是否显示门
        """
        total_start = time.perf_counter()
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

        if self.scene is None:
            self.construct_scene(use_bbox_geometry=use_bbox_geometry,
                               show_wall=show_wall, show_window=show_window, show_door=show_door)

        # Setup lighting if needed
        if not self.if_set_lights:
            self.setup_lighting()

        # 默认参数
        center = self.context["meta"]["center"]
        z_max = self.context["meta"]["z_max"]
        bounds = self.context["meta"]["bounds"]
        
        if look_at_target is None:
            look_at_target = [center[0], center[1], z_max/2]
        
        if up_vector is None:
            up_vector = [0, 0, 1]
        
        # 转换为numpy数组
        camera_position = np.array(camera_position, dtype=float)
        look_at_target = np.array(look_at_target, dtype=float)
        up_vector = np.array(up_vector, dtype=float)
        
        # 构建look-at矩阵
        forward = look_at_target - camera_position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # 构建相机位姿矩阵（OpenGL相机朝向-Z）
        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward
        camera_pose[:3, 3] = camera_position
        
        # 计算FOV
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
            fov_y = np.radians(70.0)  # 默认70度
        
        # 自动检测并设置遮挡墙体为透明
        transparent_wall_ids = []
        if auto_transparent:
            transparent_wall_ids = util.find_walls_to_make_transparent(
                camera_position.tolist()[:2],  # 只用x, y坐标
                look_at_target.tolist()[:2],   # 只用x, y坐标
                self.context["meta"]["vertices"],  # 凹多边形顶点
                self.context["walls"]  # 墙体数据
            )
            if transparent_wall_ids:
                print(f"🔍 检测到 {len(transparent_wall_ids)} 面遮挡墙体，设为透明...")
                for wall_id in transparent_wall_ids:
                    util.set_mesh_alpha(self.mesh_nodes, "walls", wall_id, transparent_alpha)
        
        # 创建相机并渲染
        camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=width/height)
        camera_node = self.scene.add(camera, pose=camera_pose)
        
        print(f"🎬 渲染中 ({width}x{height})...")
        renderer = pyrender.OffscreenRenderer(width, height)
        
        try:
            render_start = time.perf_counter()
            color, depth = renderer.render(self.scene)
            render_time = time.perf_counter() - render_start
            
            import imageio
            imageio.imwrite(output_path, color)
            if render_depth:
                depth_mm = np.clip(depth * 4000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
                image_dir = os.path.dirname(output_path)
                image_basename = os.path.basename(output_path)
                image_name = image_basename.split(".")[0]
                depth_path = os.path.join(image_dir, f"{image_name}_depth.png")
                imageio.imwrite(depth_path, depth_mm)
            total_time = time.perf_counter() - total_start
            print(f"✅ 渲染完成! 保存至: {output_path}")
            print(f"   渲染耗时: {render_time:.2f}s, 总耗时: {total_time:.2f}s")
            
        except Exception as e:
            print(f"❌ 渲染失败: {e}")
            raise
        finally:
            # 恢复透明墙体的原始材质
            if auto_transparent and transparent_wall_ids:
                for wall_id in transparent_wall_ids:
                    util.reset_mesh_alpha(self.mesh_nodes, "walls", wall_id)
            
            self.scene.remove_node(camera_node)
            renderer.delete()
    def _cubemap_to_equirectangular_map(self, cubemap_faces, width, height):
        """
        将立方体贴图转换为等距柱状投影（equirectangular）图
        
        Args:
            cubemap_faces: 字典，包含6个面的数据（可以是RGB或深度） {'right', ...}
            width: 输出全景图宽度
            height: 输出全景图高度
            
        Returns:
            equirectangular图 (numpy array)
        """
        face_size = cubemap_faces['right'].shape[0]
        sample_face = cubemap_faces['right']
        if sample_face.ndim == 3:
            panorama_shape = (height, width, sample_face.shape[2])
        else:
            panorama_shape = (height, width)
        panorama = np.zeros(panorama_shape, dtype=sample_face.dtype)
        
        # 创建坐标网格
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # 转换为球面坐标
        # equirectangular约定：theta是经度（水平角），phi是纬度（垂直角）
        # theta=0对应正前方（+Y），theta=π/2对应右侧（+X）
        # phi=0对应水平面，phi=π/2对应上方（+Z）
        theta = 2.0 * np.pi * x_coords / width - np.pi  # 经度 [-π, π]，从左侧开始
        # 注意：图像坐标系y=0在顶部，y=height在底部，需要翻转
        phi = np.pi / 2.0 - np.pi * y_coords / height  # 纬度 [π/2, -π/2]，从上方开始
        
        # 将球面坐标转换为3D方向向量
        # 标准约定：theta=0时dir_y=1（正前方），theta=π/2时dir_x=1（右侧）
        cos_phi = np.cos(phi)
        dir_x = cos_phi * np.sin(theta)  # 右侧为正
        dir_y = cos_phi * np.cos(theta)  # 前方为正
        dir_z = np.sin(phi)              # 上方为正
        
        abs_x = np.abs(dir_x)
        abs_y = np.abs(dir_y)
        abs_z = np.abs(dir_z)
        
        mask_right = (abs_x >= abs_y) & (abs_x >= abs_z) & (dir_x > 0)
        mask_left = (abs_x >= abs_y) & (abs_x >= abs_z) & (dir_x <= 0)
        mask_front = (abs_y >= abs_x) & (abs_y >= abs_z) & (dir_y > 0)
        mask_back = (abs_y >= abs_x) & (abs_y >= abs_z) & (dir_y <= 0)
        mask_top = (abs_z >= abs_x) & (abs_z >= abs_y) & (dir_z > 0)
        mask_bottom = (abs_z >= abs_x) & (abs_z >= abs_y) & (dir_z <= 0)
        
        masks = {
            "right": mask_right,
            "left": mask_left,
            "front": mask_front,
            "back": mask_back,
            "top": mask_top,
            "bottom": mask_bottom
        }
        
        # 为每个面计算UV坐标并采样，统一使用面坐标轴
        for face_name, mask in masks.items():
            if not np.any(mask):
                continue
            
            axes = self.CUBEMAP_FACE_AXES[face_name]
            forward = axes["forward"]
            up = axes["up"]
            right = np.cross(forward, up)
            
            dir_forward = dir_x * forward[0] + dir_y * forward[1] + dir_z * forward[2]
            dir_right = dir_x * right[0] + dir_y * right[1] + dir_z * right[2]
            dir_up = dir_x * up[0] + dir_y * up[1] + dir_z * up[2]
            
            denom = np.abs(dir_forward) + 1e-10
            u = 0.5 * (1.0 + (dir_right / denom))
            v = 0.5 * (1.0 - (dir_up / denom))  # 图像坐标y轴向下，因此取负号
            
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
            px = (u * (face_size - 1)).astype(int)
            py = (v * (face_size - 1)).astype(int)
            
            panorama[mask] = cubemap_faces[face_name][py[mask], px[mask]]
        
        return panorama

    def render_panorama(self, output_path: str, width: int = 4096, height: int = 2048,
                        face_size: int = 2048, depth_output_path: Optional[str] = None,
                        use_bbox_geometry: bool = False, show_wall: bool = True,
                        show_window: bool = True, show_door: bool = True):
        """
        渲染全景图（等距柱状投影，equirectangular）
        使用立方体贴图方法：先渲染6个面，再转换为equirectangular格式

        Args:
            output_path: 输出全景图路径
            width: 全景图宽度（建议为2的倍数，如4096）
            height: 全景图高度（应为宽度的1/2，满足2:1比例）
            face_size: 立方体贴图每个面的分辨率
            depth_output_path: 若提供，将额外输出全景深度图
            show_wall: 是否显示墙体
            show_window: 是否显示窗户
            show_door: 是否显示门
        """
        total_start = time.perf_counter()
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

        if self.scene is None:
            self.construct_scene(use_bbox_geometry=use_bbox_geometry,
                               show_wall=show_wall, show_window=show_window, show_door=show_door)

        # Setup lighting if needed
        if not self.if_set_lights:
            self.setup_lighting()

        # 计算所有bbox的中心作为相机位置
        if not self.context["boxes"]:
            # 如果没有bbox，使用场景中心
            camera_position = np.array(self.context["meta"]["center"] + [0, 0, 2.2], dtype=float)  # 默认高度1.5m
        else:
            # 计算所有bbox的平均中心
            all_centers = []
            for box_id, box_data in self.context["boxes"].items():
                all_centers.append(box_data["center"])
            all_centers = np.array(all_centers)
            camera_position = np.mean(all_centers, axis=0).astype(float)

        print(f"📍 相机位置: {camera_position}")

        cubemap_faces = {}
        cubemap_depths = {}
        renderer = pyrender.OffscreenRenderer(face_size, face_size)

        print(f"🎬 开始渲染立方体贴图 ({face_size}x{face_size} x 6面)...")

        for face_name, axes in self.CUBEMAP_FACE_AXES.items():
            print(f"📷 渲染 {face_name} 面...")

            # 使用预定义坐标轴，确保与转换阶段一致
            forward = axes["forward"]
            up = axes["up"]
            right = np.cross(forward, up)

            camera_pose = np.eye(4)
            camera_pose[:3, 0] = right
            camera_pose[:3, 1] = up
            camera_pose[:3, 2] = -forward  # OpenGL相机朝向-Z
            camera_pose[:3, 3] = camera_position

            # 创建90度FOV的立方体相机
            camera = pyrender.PerspectiveCamera(yfov=np.radians(90), aspectRatio=1.0)
            camera_node = self.scene.add(camera, pose=camera_pose)

            try:
                color, depth = renderer.render(self.scene)
                cubemap_faces[face_name] = color
                cubemap_depths[face_name] = depth
            finally:
                self.scene.remove_node(camera_node)

        renderer.delete()

        # 将立方体贴图转换为equirectangular全景图
        print(f"🔄 转换立方体贴图为equirectangular格式 ({width}x{height})...")
        panorama = self._cubemap_to_equirectangular_map(cubemap_faces, width, height)

        # 保存全景图
        import imageio
        imageio.imwrite(output_path, panorama)

        total_time = time.perf_counter() - total_start
        print(f"✅ 全景图渲染完成! 保存至: {output_path}")
        print(f"   总耗时: {total_time:.2f}s")
        print(f"   分辨率: {width}x{height}")

        if depth_output_path:
            panorama_depth = self._cubemap_to_equirectangular_map(cubemap_depths, width, height).astype(np.float32)
            raw_path = depth_output_path + ".npy"
            np.save(raw_path, panorama_depth)

            tiff_path = depth_output_path + ".tiff"
            imageio.imwrite(tiff_path, panorama_depth)

            depth_mm = np.clip(panorama_depth * 8000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            png_path = depth_output_path + ".png"
            imageio.imwrite(png_path, depth_mm)
            print(f"   深度图已保存: {raw_path} (npy), {tiff_path} (32-bit TIFF), {png_path} (16-bit PNG, scale x4000)")

def read_jsonl_line(file_path, line_number):
    """
    读取JSONL文件的第n行（0-based）
    
    Args:
        file_path: JSONL文件路径
        line_number: 行号（从0开始）
    
    Returns:
        dict: 该行的JSON数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == line_number:
                return json.loads(line.strip())
    
    raise ValueError(f"行号 {line_number} 超出文件范围")
if __name__ == "__main__":
    """测试场景渲染"""
    line = 16
    jsonl_path = '/root/projects/utils/fast-scene/data-agent/spatial-layout.jsonl'
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    # example_path = os.path.join(base_dir, 'example.json')
    test_dir = os.path.join(base_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    topdown_path = os.path.join(test_dir, 'scene_topdown.png')

    try:
        data = read_jsonl_line(jsonl_path, line)

        print(f"📁 加载场景: {data['room']['label']} ({len(data['bbox'])} 个物体)")

        ctx = SceneCtx(data['room']['label'])
        ctx.add_walls(data['wall'])
        ctx.add_doors(data.get('door', []))
        ctx.add_windows(data.get('window', []))
        ctx.add_boxes(data['bbox'])
        ctx.topdown_view(topdown_path, use_bbox_geometry=False, show_wall=True, show_window=False, show_door=False)

        # 从房间外四个方向看向室内
        print("\n" + "="*60)
        print("📸 从房间外四个方向渲染")
        print("="*60)
        
        center = ctx.context["meta"]["center"]
        span = ctx.context["meta"]["span"]
        z_max = ctx.context["meta"]["z_max"]
        
        # 看向的目标点（地面中心）
        look_at = [center[0], center[1], 0]
        
        # 4个方向的相机位置
        views = [
            ("right", [center[0] + span[0]/2 + 0.5, center[1], z_max * 2/3]),
            ("left", [center[0] - span[0]/2 - 0.5, center[1], z_max * 2/3]),
            ("front", [center[0], center[1] + span[1]/2 + 0.5, z_max * 2/3]),
            ("back", [center[0], center[1] - span[1]/2 - 0.5, z_max * 2/3]),
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
                auto_transparent=True,
                transparent_alpha=0.3, 
                use_bbox_geometry=False,
                show_wall=True,
                show_window=False,
                show_door=False
            )

        # 渲染全景图（立方体贴图 + equirectangular）
        print("\n" + "="*60)
        print("🌀 渲染全景图")
        print("="*60)
        panorama_path = os.path.join(test_dir, 'scene_panorama_equirectangular.png')
        ctx.render_panorama(
            output_path=panorama_path,
            width=2048,
            height=1024,
            face_size=1024,
            depth_output_path=os.path.join(test_dir, 'scene_panorama_depth'),
            use_bbox_geometry=True,
            show_wall=True,
            show_window=False,
            show_door=False
        )

        # # 导出调试用的六个立方体面，便于逐面排查
        # print("\n" + "="*60)
        # print("🧪 导出立方体六个面的调试图")
        # print("="*60)
        # if ctx.scene is None:
        #     ctx.construct_scene()
        # if not ctx.if_set_lights:
        #     ctx.setup_lighting()

        # # 与render_panorama保持一致的相机位置
        # if ctx.context["boxes"]:
        #     centers = np.array([box["center"] for box in ctx.context["boxes"].values()])
        #     cube_camera_pos = centers.mean(axis=0)
        # else:
        #     cube_camera_pos = np.array(ctx.context["meta"]["center"] + [2.2])

        # face_renderer = pyrender.OffscreenRenderer(512, 512)
        # cube_face_dir = test_dir
        # for face_name, axes in SceneCtx.CUBEMAP_FACE_AXES.items():
        #     forward = axes["forward"]
        #     up = axes["up"]
        #     right = np.cross(forward, up)

        #     pose = np.eye(4)
        #     pose[:3, 0] = right
        #     pose[:3, 1] = up
        #     pose[:3, 2] = -forward
        #     pose[:3, 3] = cube_camera_pos

        #     camera = pyrender.PerspectiveCamera(yfov=np.pi/2, aspectRatio=1.0)
        #     cam_node = ctx.scene.add(camera, pose=pose)
        #     try:
        #         color, depth = face_renderer.render(ctx.scene)
        #         face_path = os.path.join(cube_face_dir, f'scene_panorama_face_{face_name}.png')
        #         imageio.imwrite(face_path, color)

        #         depth_float = depth.astype(np.float32)
        #         depth_raw_path = os.path.join(cube_face_dir, f'scene_panorama_face_{face_name}_depth.npy')
        #         np.save(depth_raw_path, depth_float)

        #         depth_tiff_path = os.path.join(cube_face_dir, f'scene_panorama_face_{face_name}_depth.tiff')
        #         imageio.imwrite(depth_tiff_path, depth_float)

        #         depth_mm = np.clip(depth_float * 8000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        #         depth_png_path = os.path.join(cube_face_dir, f'scene_panorama_face_{face_name}_depth.png')
        #         imageio.imwrite(depth_png_path, depth_mm)
        #         print(f"   ✅ {face_name} 面: {face_path}")
        #         print(f"      ↳ 深度: {depth_raw_path} (npy), {depth_tiff_path} (32-bit TIFF), {depth_png_path} (16-bit PNG, scale x4000)")
        #     finally:
        #         ctx.scene.remove_node(cam_node)
        # face_renderer.delete()

        # print("\n" + "="*60)
        # print("📋 场景上下文:")
        # print("="*60)
        # print(json.dumps(ctx.get_context(), indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()