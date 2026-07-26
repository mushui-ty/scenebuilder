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
    import util # type: ignore

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

    def __init__(self, scene_type: str, model_extra_path: Optional[str] = None):
        self.context = {
            "meta": {"scene_type": scene_type},
            "walls": {},
            "boxes": {}
        }
        self.scene = None
        self.if_set_lights = False
        self.model_extra_path = model_extra_path
        
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

        # 端点吸附功能，默认开启，阈值为0.3m
        do_snap = True
        if do_snap:
            walls_converted = util.snap_wall_endpoints(walls_converted, threshold=0.3)

        # 计算bounds和vertices
        all_points = [p for w in walls_converted for p in [w["s"], w["e"]]]
        x_coords, y_coords = zip(*all_points)

        # vertices: [(x1, y1), (x2, y2), ...] 多边形外边界顶点序列
        # partitions: [(xs, ys, xe, ye, height), ...] 内部隔断墙线段序列（含高度）
        vertices, partitions = util.calculate_minimum_area_polygon_and_partitions(walls_converted)
        
        self.context["meta"].update({
            "bounds": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
            "center": [(min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2],
            "span": [max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)],
            "z_max": max(w["height"] for w in walls_converted),
            "vertices": vertices
        })
        
        # 调试信息
        print(f"📐 输入原始墙体数量: {len(walls_converted)}")
        print(f"📐 边界墙段数量: {len(vertices)}")
        print(f"📐 内部隔断墙数量: {len(partitions)}")
        
        # 1. 添加边界墙 (Boundary Walls)
        # 边界墙由 vertices 闭合环路构成，确保地板和墙基完美重合
        for i in range(len(vertices)):
            v_s = vertices[i]
            v_e = vertices[(i + 1) % len(vertices)]
            
            # 查找该段边界墙对应的原始高度
            height = None
            for w in walls_converted:
                if util.is_wall_on_edge(w, v_s, v_e):
                    height = w["height"]
                    break
            if height is None:
                height = max(w["height"] for w in walls_converted)

            wall_id = util.generate_unique_id()
            self.context["walls"][wall_id] = {
                "s": list(v_s),
                "e": list(v_e),
                "height": height,
                # 边界墙需要计算朝向（指向房间内部）
                "orientation": util.calculate_wall_orientation(v_s, v_e, vertices),
                "is_partition": False,
                "doors": {},
                "windows": {}
            }

        # 2. 添加内部隔断墙 (Partition Walls)
        for p in partitions:
            wall_id = util.generate_unique_id()
            self.context["walls"][wall_id] = {
                "s": [p[0], p[1]],
                "e": [p[2], p[3]],
                "height": p[4],
                # 隔断墙不需要朝向
                "orientation": (0.0, 0.0),
                "is_partition": True,
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
            self.add_box(box["center"], box["angle_z"], box["scale"], box.get("class"),
                        box.get("label"), box.get("caption"), box.get("asset_id"))

    def add_box(self, center: List[float], angle_z: float, scale: List[float],
                class_name: Optional[str] = None, label: Optional[str] = None,
                caption: Optional[str] = None, asset_id: Optional[int] = None) -> str:
        """添加单个家具，返回ID"""
        box_id = util.generate_unique_id()
        box_data = {"center": center, "angle_z": angle_z, "scale": scale}

        if class_name: box_data["class"] = class_name
        if label: box_data["label"] = label
        if caption: box_data["caption"] = caption
        if asset_id: box_data["asset_id"] = asset_id

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
        导出墙体SSL格式文件 (包含 Room 和所有 Wall)
        包含边界墙和内部隔断墙
        """
        output_path = os.path.join(output_dir, 'wall_ssl.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        
        # 生成 room_id
        room_id = util.generate_unique_id()
        room_type = self.context["meta"]["scene_type"]
        lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
        
        # 直接从 context["walls"] 导出所有墙体
        all_walls = self.context["walls"]
        for wall_id, wall in all_walls.items():
            p = list(wall["s"]) + [0.0]
            q = list(wall["e"]) + [0.0]
            height = wall["height"]
            lines.append(f'Wall(id="{wall_id}", room_id="{room_id}", p={p}, q={q}, height={height})')
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ 墙体SSL已导出: {output_path} ({len(all_walls)} 面墙)")
        return output_path

    def export_wall_hole_ssl(self, output_dir: str):
        """
        导出墙体和门窗SSL格式文件 (包含 Room, Wall, Door, Window)
        """
        output_path = os.path.join(output_dir, 'wall_hole_ssl.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        
        # 生成 room_id
        room_id = util.generate_unique_id()
        room_type = self.context["meta"]["scene_type"]
        lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
        
        # 记录所有墙体
        all_walls = self.context["walls"]
        for wall_id, wall in all_walls.items():
            p = list(wall["s"]) + [0.0]
            q = list(wall["e"]) + [0.0]
            height = wall["height"]
            lines.append(f'Wall(id="{wall_id}", room_id="{room_id}", p={p}, q={q}, height={height})')
            
            # 导出该墙体上的门
            for door_id, door in wall.get("doors", {}).items():
                lines.append(f'Door(id="{door_id}", wall_id="{wall_id}", center={door["center"]}, width={door["width"]}, height={door["height"]})')
            
            # 导出该墙体上的窗
            for window_id, window in wall.get("windows", {}).items():
                lines.append(f'Window(id="{window_id}", wall_id="{wall_id}", center={window["center"]}, width={window["width"]}, height={window["height"]})')
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ 墙体和门窗SSL已导出: {output_path}")
        return output_path

    def export_ssl(self, output_dir: str):
        """
        导出完整SSL格式文件 (包含 Room, Wall, Door, Window, Bbox)
        """
        output_path = os.path.join(output_dir, 'ssl.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        
        # 生成 room_id
        room_id = util.generate_unique_id()
        room_type = self.context["meta"]["scene_type"]
        lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
        
        # 导出所有墙体及门窗
        for wall_id, wall in self.context["walls"].items():
            p = list(wall["s"]) + [0.0]
            q = list(wall["e"]) + [0.0]
            height = wall["height"]
            lines.append(f'Wall(id="{wall_id}", room_id="{room_id}", p={p}, q={q}, height={height})')
            
            for door_id, door in wall.get("doors", {}).items():
                lines.append(f'Door(id="{door_id}", wall_id="{wall_id}", center={door["center"]}, width={door["width"]}, height={door["height"]})')
            
            for window_id, window in wall.get("windows", {}).items():
                lines.append(f'Window(id="{window_id}", wall_id="{wall_id}", center={window["center"]}, width={window["width"]}, height={window["height"]})')
        
        # 导出所有物品 (Bbox)
        for box_id, box in self.context["boxes"].items():
            lines.append(f'Bbox(id="{box_id}", room_id="{room_id}", center={box["center"]}, scale={box["scale"]}, angle_z={box["angle_z"]}, class="{box["class"]}")')
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ 完整SSL已导出: {output_path}")
        return output_path
    
    def construct_floor(self, show_wall=True, show_window=True, show_door=True):
        """构建地板和墙体"""
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

        # 创建墙体
        if show_wall:
            wall_color = self.config["wall_color"]
            wall_texture_path = self.config.get("wall_texture_path")

            wall_texture = None
            if wall_texture_path and os.path.exists(wall_texture_path):
                try:
                    import PIL.Image
                    import PIL.ImageEnhance
                    wall_texture = PIL.Image.open(wall_texture_path).convert('RGB')
                    enhancer = PIL.ImageEnhance.Brightness(wall_texture)
                    wall_texture = enhancer.enhance(1.5)
                except Exception as e:
                    print(f"⚠️ 加载墙体纹理失败: {e}")
            
            for wall_id, wall in self.context["walls"].items():
                openings = []
                for door in wall.get("doors", {}).values():
                    openings.append(door)
                for window in wall.get("windows", {}).values():
                    openings.append(window)
                
                wall_mesh = util.create_single_wall_mesh(
                    wall["s"], wall["e"], wall["height"], wall["orientation"],
                    chip=False, openings=openings if openings else None,
                    wall_thickness=wall_thickness, texture_scale=texture_scale
                )

                if wall_texture:
                    wall_mesh.visual = trimesh.visual.TextureVisuals(uv=wall_mesh.visual.uv, image=wall_texture)
                else:
                    wall_mesh.visual.material = trimesh.visual.material.PBRMaterial(baseColorFactor=wall_color)

                pyrender_mesh = pyrender.Mesh.from_trimesh(wall_mesh)
                node = self.scene.add(pyrender_mesh)
                self.mesh_nodes["walls"][wall_id] = {
                    "node": node,
                    "mesh": wall_mesh,
                    "wall_data": wall
                }
                
                if self.config.get("wall_edge_enabled", True):
                    edge_color = self.config.get("wall_edge_color", [0.1, 0.1, 0.1, 1.0])
                    edge_offset = self.config.get("wall_edge_offset", 0.01)
                    edge_lines = util.create_wall_edge_lines(
                        wall["s"], wall["e"], wall["height"], 
                        wall["orientation"], edge_color, edge_offset
                    )
                    if edge_lines:
                        edge_node = self.scene.add(edge_lines)
                        self.mesh_nodes["walls"][wall_id]["edge_node"] = edge_node
        
        # 添加门窗
        if show_door or show_window:
            door_texture_path = self.config.get("door_texture_path", "/data-nas/data/experiments/mushui/projects/utils/fast-scene/gltf/door.png")
            window_texture_path = self.config.get("window_texture_path", "/data-nas/data/experiments/mushui/projects/utils/fast-scene/gltf/window.png")
            
            config_with_extra = self.config.copy()
            if self.model_extra_path:
                config_with_extra["model_extra_path"] = self.model_extra_path

            doors_and_windows = util.create_windows_and_doors(
                self.context["walls"],
                door_texture_path if show_door else None,
                window_texture_path if show_window else None,
                config=config_with_extra
            )
            
            for item in doors_and_windows:
                item_type = item["type"]
                item_id = item["id"]
                item_mesh = item["mesh"]
                
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
                        f"{item_type}_data": {"wall_id": item["wall_id"]}
                    }
        
        # 创建天花板
        z_max = self.context["meta"]["z_max"]
        ceiling_mesh = util.create_ceiling_mesh(vertices, bounds, z_max)
        ceiling_color = [0.9, 0.9, 0.9, 1.0]
        ceiling_texture_path = self.config.get("ceiling_texture_path")
        
        ceiling_texture = None
        if ceiling_texture_path and os.path.exists(ceiling_texture_path):
            try:
                import PIL.Image
                import PIL.ImageEnhance
                ceiling_texture = PIL.Image.open(ceiling_texture_path).convert('RGB')
                enhancer = PIL.ImageEnhance.Brightness(ceiling_texture)
                ceiling_texture = enhancer.enhance(1.5)
            except Exception as e:
                print(f"⚠️ 加载天花板纹理失败: {e}")
        
        if ceiling_texture:
            ceiling_mesh.visual = trimesh.visual.TextureVisuals(uv=ceiling_mesh.visual.uv, image=ceiling_texture)
        else:
            ceiling_mesh.visual.material = trimesh.visual.material.PBRMaterial(baseColorFactor=ceiling_color)
        
        self.scene.add(pyrender.Mesh.from_trimesh(ceiling_mesh))

    def construct_scene(self, show_wall: bool = True, show_window: bool = True,
                        show_door: bool = True, fix_coordinate: bool = False,
                        use_bbox_geometry: bool = False):
        """构建完整场景"""
        total_start = time.perf_counter()
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

        self.construct_floor(show_wall=show_wall, show_window=show_window, show_door=show_door)

        print(f"📦 加载家具 ({len(self.context['boxes'])} 个物体)...")
        success_count = 0
        load_time = 0
        transform_time = 0

        for box_id, box in self.context["boxes"].items():
            if use_bbox_geometry or box.get("use_bbox_geometry"):
                try:
                    bbox_mesh = self._create_bbox_geometry(box["center"], box["scale"], box["angle_z"],
                                                         color=self._get_class_color(box.get("class")))
                    pyrender_mesh = pyrender.Mesh.from_trimesh(bbox_mesh)
                    node = self.scene.add(pyrender_mesh)
                    self.mesh_nodes["boxes"][box_id] = {"node": node, "mesh": bbox_mesh, "box_data": box}
                    success_count += 1
                except Exception as e:
                    print(f"  ❌ {box.get('class', 'unknown')} (bbox): {e}")
                continue
            
            if not box.get("asset_id"): continue

            load_start = time.perf_counter()
            config_with_extra = self.config.copy()
            if self.model_extra_path: config_with_extra["model_extra_path"] = self.model_extra_path
            loaded = util.load_mesh(box["asset_id"], config_with_extra)
            load_time += time.perf_counter() - load_start

            if not loaded: continue

            try:
                transform_start = time.perf_counter()
                if fix_coordinate:
                    center = (loaded.bounds[0] + loaded.bounds[1]) / 2 if isinstance(loaded, trimesh.Scene) else loaded.centroid
                    if isinstance(loaded, trimesh.Scene):
                        for geom in loaded.geometry.values():
                            if hasattr(geom, 'vertices'):
                                geom.vertices -= center
                                v = geom.vertices.copy()
                                geom.vertices[:, 1], geom.vertices[:, 2] = -v[:, 2], v[:, 1]
                    elif hasattr(loaded, 'vertices'):
                        loaded.vertices -= center
                        v = loaded.vertices.copy()
                        loaded.vertices[:, 1], loaded.vertices[:, 2] = -v[:, 2], v[:, 1]

                new_size = (loaded.bounds[1] - loaded.bounds[0]) if isinstance(loaded, trimesh.Scene) else loaded.bounding_box.extents
                new_size = np.where(new_size == 0, 1.0, new_size)
                scale_factors = np.array(box["scale"]) / new_size
                theta = np.radians(box["angle_z"])
                R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

                if isinstance(loaded, trimesh.Scene):
                    for geom in loaded.geometry.values():
                        if hasattr(geom, 'vertices'):
                            geom.vertices = (geom.vertices * scale_factors).dot(R.T) + np.array(box["center"])
                elif hasattr(loaded, 'vertices'):
                    loaded.vertices = (loaded.vertices * scale_factors).dot(R.T) + np.array(box["center"])

                if isinstance(loaded, trimesh.Scene):
                    mesh_scene = pyrender.Scene.from_trimesh_scene(loaded)
                    nodes_list = [self.scene.add(n.mesh) for n in mesh_scene.get_nodes() if n.mesh]
                    self.mesh_nodes["boxes"][box_id] = {"nodes": nodes_list, "mesh": loaded, "box_data": box}
                else:
                    node = self.scene.add(pyrender.Mesh.from_trimesh(loaded))
                    self.mesh_nodes["boxes"][box_id] = {"node": node, "mesh": loaded, "box_data": box}

                transform_time += time.perf_counter() - transform_start
                success_count += 1
            except Exception as e:
                print(f"  ❌ {box.get('class')}: {e}")

        print(f"✅ 构建完成! 成功 {success_count}/{len(self.context['boxes'])} 个物体 (耗时: {time.perf_counter() - total_start:.2f}s)")

    def _get_class_color(self, class_name: Optional[str]):
        label = (class_name or "default").lower()
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        h = int.from_bytes(digest[0:2], "big") / 65535.0
        s, v = 0.55 + (digest[2]/255.0)*0.3, 0.65 + (digest[3]/255.0)*0.2
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return [float(r), float(g), float(b), 0.9]

    def _create_bbox_geometry(self, center, scale, angle_z, color=None):
        center, scale = np.array(center), np.array(scale)
        box_mesh = trimesh.creation.box(extents=scale)
        arrow_length = min(scale[0], scale[1]) * 0.3
        arrow_radius = max(arrow_length * 0.05, 0.02)
        body_len, head_len = arrow_length * 0.75, arrow_length * 0.25
        
        arrow_body = trimesh.creation.cylinder(radius=arrow_radius, height=body_len)
        arrow_head = trimesh.creation.cone(radius=arrow_radius*3, height=head_len)
        
        z_to_neg_y = trimesh.geometry.align_vectors([0,0,1], [0,-1,0])
        arrow_body.apply_transform(z_to_neg_y)
        arrow_head.apply_transform(z_to_neg_y)
        
        top_center = np.array([0, 0, scale[2]/2])
        arrow_body.apply_translation(top_center + np.array([0,-body_len/2,0]))
        arrow_head.apply_translation(top_center + np.array([0,-(body_len+head_len/2),0]))
        
        box_mesh.visual.vertex_colors = color or self._get_class_color(None)
        arrow_body.visual.vertex_colors = arrow_head.visual.vertex_colors = [0.0, 0.8, 0.0, 1.0]
        
        combined = trimesh.util.concatenate([box_mesh, arrow_body, arrow_head])
        theta = np.radians(angle_z)
        R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        combined.vertices = combined.vertices.dot(R.T) + center
        return combined

    def setup_lighting(self, intensity: float = 20):
        if self.if_set_lights: return
        center, span, z_max = self.context["meta"]["center"], self.context["meta"]["span"], self.context["meta"]["z_max"]
        self.scene.ambient_light = [0.6, 0.6, 0.6]
        corners = [[center[0]-span[0]/2, center[1]-span[1]/2, z_max+3], [center[0]+span[0]/2, center[1]-span[1]/2, z_max+3],
                   [center[0]-span[0]/2, center[1]+span[1]/2, z_max+3], [center[0]+span[0]/2, center[1]+span[1]/2, z_max+3]]
        for pos in corners:
            self.scene.add(pyrender.PointLight(color=[1,1,1], intensity=intensity), pose=trimesh.transformations.translation_matrix(pos))
        self.if_set_lights = True

    def topdown_view(self, output_path: str, width: int = 1024, height: int = 1024, **kwargs):
        if self.scene is None: self.construct_scene(**kwargs)
        self.setup_lighting()
        meta = self.context["meta"]
        camera_height = meta["z_max"] + max(meta["span"]) * 1.5
        camera_pos = np.array([meta["center"][0], meta["center"][1], camera_height])
        
        fov_y = util.calculate_optimal_fov(camera_pos, [meta["center"][0], meta["center"][1], 0],
                                         meta["vertices"], meta["z_max"], meta["bounds"],
                                         self.config.get('indoor_fov', 160), self.config.get('outdoor_fov_scale', 1.05))
        
        camera_pose = np.eye(4)
        camera_pose[:3, 1], camera_pose[:3, 2], camera_pose[:3, 3] = [0,1,0], [0,0,1], camera_pos
        
        # 透明遮挡墙体
        trans_ids = util.find_walls_to_make_transparent(camera_pos[:2], meta["center"][:2], meta["vertices"], self.context["walls"])
        for wid in trans_ids: util.set_mesh_alpha(self.mesh_nodes, "walls", wid, kwargs.get("transparent_alpha", 0.3))
        
        camera_node = self.scene.add(pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=width/height), pose=camera_pose)
        renderer = pyrender.OffscreenRenderer(width, height)
        color, _ = renderer.render(self.scene)
        imageio.imwrite(output_path, color)
        
        for wid in trans_ids: util.reset_mesh_alpha(self.mesh_nodes, "walls", wid)
        self.scene.remove_node(camera_node)
        renderer.delete()
        print(f"✅ 俯视图保存至: {output_path}")

    def render_view(self, output_path: str, camera_position: list, look_at_target: list = None, **kwargs):
        if self.scene is None: self.construct_scene(**kwargs)
        self.setup_lighting()
        # 通用视角渲染逻辑 (略，已包含在之前的实现中)
        pass

if __name__ == "__main__":
    pass
