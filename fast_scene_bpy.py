"""
Fast Scene - 纯 Blender 渲染版本（无 trimesh 依赖）

用 bpy 替代 trimesh/pyrender

使用方式：
   $BLENDER_PATH --background --python fast_scene_bpy.py 或者在 pip install bpy 后直接使用 python fast_scene_bpy.py
"""

import os
import sys
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_PLATFORM'] = 'surfaceless'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import json
import yaml
import numpy as np
import time
from typing import List, Dict, Any, Optional, Literal

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
    
    
    def __init__(self, scene_type: str, model_extra_path: Optional[str] = None, 
                 render_engine: Literal["CYCLES", "EEVEE"] = "CYCLES"):
        # 与 fast_scene.py 完全一致的数据结构
        self.context = {
            "meta": {"scene_type": scene_type},
            "walls": {},
            "boxes": {}
        }
        self.scene = None
        self.if_set_lights = False
        self.model_extra_path = model_extra_path
        self.render_engine = render_engine
        
        # 模型缓存：asset_id -> master_collection
        self.asset_cache = {}
        self.asset_bounds = {}
        
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

    def set_model_path(self, path: str):
        """更改模型查找路径"""
        print(f"🔄 更改模型路径为: {path}")
        self.config["model_path"] = path

    def _init_blender_scene(self):
        """初始化 Blender 场景"""
        # 使用工厂设置重置场景，这比手动删除物体更彻底，有助于在无头模式下初始化 EGL 上下文
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        if "Scene" in bpy.data.scenes:
            self.scene = bpy.data.scenes["Scene"]
        else:
            self.scene = bpy.data.scenes.new("Scene")
        bpy.context.window.scene = self.scene
        
        # 设置渲染引擎
        engine_map = {
            "CYCLES": "CYCLES",
            "EEVEE": "BLENDER_EEVEE"
        }
        self.scene.render.engine = engine_map.get(self.render_engine, "CYCLES")
        
        if self.render_engine == "CYCLES":
            self.scene.cycles.device = 'GPU'
            
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'CUDA'
            for device in prefs.get_devices_for_type('CUDA'):
                device.use = True
            
            self.scene.cycles.samples = self.config.get("blender_samples", 32)
            self.scene.cycles.use_denoising = True
            self.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
            print(f"✅ Blender 场景初始化完成 (Cycles + CUDA)")
        else:
            # EEVEE Next 相关设置 (Blender 4.2+)
            if hasattr(self.scene, "eevee"):
                # EEVEE Next 在 4.2 中有一些新参数，这里可以根据需要配置
                self.scene.eevee.taa_render_samples = self.config.get("blender_samples", 32)
            print(f"✅ Blender 场景初始化完成 (EEVEE Next)")
        
        self.scene_collection = self.scene.collection

    def clear_scene(self):
        """彻底清空所有物体、灯光和相机，并重置状态"""
        # 切换到 Object 模式以防万一
        if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
            
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # --- 深度清理内存 ---
        # 1. 物理删除所有图像块（这是最占内存的残留之一）
        for img in bpy.data.images:
            if img.users == 0 or not img.filepath: # 只删除未使用的或占位图
                try:
                    bpy.data.images.remove(img, do_unlink=True)
                except:
                    pass
        
        # 2. 清空所有孤立的数据块（材质、网格等）
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        # -------------------
        
        # 3. 清空 Collection 缓存
        for coll in self.asset_cache.values():
            if coll:
                try:
                    if coll.name in bpy.data.collections:
                        bpy.data.collections.remove(coll, do_unlink=True)
                except (ReferenceError, AttributeError):
                    pass
        self.asset_cache = {}
        # -------------------
        
        # 重置引用状态
        self.mesh_nodes = {
            "walls": {},      
            "doors": {},      
            "windows": {},    
            "boxes": {},      
            "floor": None,    
            "ceiling": None,  
        }
        self.if_set_lights = False
        print("🧹 场景已完全清空并重置状态")

    # ==================== 数据管理函数（与 fast_scene.py 完全一致）====================
    def add_walls(self, walls: List[Dict[str, Any]]):
        """添加墙体并计算场景元数据"""
        # walls_converted: [{"s": [x1, y1], "e": [x2, y2], "height": h}, ...] 统一后的墙体列表格式
        walls_converted = [{"s": w["p"][:2], "e": w["q"][:2], "height": w["height"]} for w in walls]
        
        # 端点吸附功能，默认开启，阈值为0.3m
        do_snap = True
        if do_snap:
            walls_converted = util.snap_wall_endpoints(walls_converted, threshold=0.3)

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
        
        print(f"📐 输入原始墙体数量: {len(walls_converted)}")
        print(f"📐 边界墙段数量: {len(vertices)}")
        print(f"📐 内部隔断墙数量: {len(partitions)}")

        # 1. 添加边界墙 (Boundary Walls)
        # 边界墙由 vertices 闭合环路构成，确保地板和墙基完美重合
        for i in range(len(vertices)):
            v_s = vertices[i]
            v_e = vertices[(i + 1) % len(vertices)]
            
            # 查找该段边界墙对应的原始高度
            # 优先匹配覆盖该线段的原始墙体，若无匹配（如桥接线）则取邻近墙体高度
            height = None
            for w in walls_converted:
                if util.is_wall_on_edge(w, v_s, v_e):
                    height = w["height"]
                    break
            if height is None:
                # 容错：取所有原始墙体中的最大高度作为默认值
                height = max(w["height"] for w in walls_converted)

            wall_id = util.generate_unique_id()
            self.context["walls"][wall_id] = {
                "s": list(v_s),
                "e": list(v_e),
                "height": height,
                # 边界墙需要计算朝向（指向房间内部），用于后续单向厚度偏移
                "orientation": util.calculate_wall_orientation(v_s, v_e, vertices),
                "is_partition": False,
                "doors": {},
                "windows": {}
            }

        # 2. 添加内部隔断墙 (Partition Walls)
        # 隔断墙不属于外部轮廓，通常两侧都在房间内
        for p in partitions:
            wall_id = util.generate_unique_id()
            self.context["walls"][wall_id] = {
                "s": [p[0], p[1]],
                "e": [p[2], p[3]],
                "height": p[4],
                # 隔断墙不需要朝向，后续渲染时应以中心线为基准向两侧平分厚度
                "orientation": (0.0, 0.0),
                "is_partition": True,
                "doors": {},
                "windows": {}
            }

    def add_wall(self, s: List[float], e: List[float], height: float):
        """添加单面墙体"""
        # 转换现有墙体为输入格式，并加入新墙体
        current_walls = [{"p": w["s"] + [0], "q": w["e"] + [0], "height": w["height"]}
                        for w in self.context["walls"].values()]
        current_walls.append({"p": s[:2] + [0], "q": e[:2] + [0], "height": height})
        
        # 清空当前状态并重新批量添加（以触发重新分类和吸附逻辑）
        self.context["walls"] = {}
        self.add_walls(current_walls)

    def add_doors(self, doors: List[Dict[str, Any]]):
        """批量添加门"""
        for door in doors:
            self.add_door(door["center"], door["width"], door["height"], door.get("asset_id"))

    def add_door(self, center: List[float], width: float, height: float, asset_id: Optional[int] = None):
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
            if asset_id:
                door_data["asset_id"] = asset_id
            self.context["walls"][wall_id]["doors"][door_id] = door_data

    def add_windows(self, windows: List[Dict[str, Any]]):
        """批量添加窗"""
        for window in windows:
            self.add_window(window["center"], window["width"], window["height"], window.get("asset_id"))

    def add_window(self, center: List[float], width: float, height: float, asset_id: Optional[int] = None):
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
            if asset_id:
                window_data["asset_id"] = asset_id
            self.context["walls"][wall_id]["windows"][window_id] = window_data

    def add_boxes(self, boxes: List[Dict[str, Any]]):
        """批量添加家具"""
        for box in boxes:
            self.add_box(box["center"], box["angle_z"], box["scale"],
                        box.get("label"), box.get("caption"), box.get("asset_id"))

    def add_box(self, center: List[float], angle_z: float, scale: List[float],
                label: Optional[str] = None, caption: Optional[str] = None, 
                asset_id: Optional[int] = None) -> str:
        """添加单个家具，返回ID"""
        box_id = util.generate_unique_id()
        box_data = {"center": center, "angle_z": angle_z, "scale": scale}

        if label: box_data["label"] = label
        if caption: box_data["caption"] = caption
        if asset_id: box_data["asset_id"] = asset_id

        self.context["boxes"][box_id] = box_data

        box_top = center[2] + scale[2] / 2
        if box_top > self.context["meta"].get("z_max", 0):
            self.context["meta"]["z_max"] = box_top

        return box_id

    def delete_box(self, box_id: str):
        """删除家具，并从 Blender 场景中移除对应物体"""
        if box_id not in self.context["boxes"]:
            print(f"⚠️  未找到 Box ID: {box_id}，无法删除")
            return

        # 1. 如果该物体已经渲染，则在 Blender 中物理删除
        if box_id in self.mesh_nodes["boxes"]:
            box_info = self.mesh_nodes["boxes"][box_id]
            node = box_info.get("node")
            if node:
                # 递归删除对象及其所有子对象（对于 import 的 gltf 根节点）
                objs_to_remove = [node] + list(node.children_recursive)
                for obj in objs_to_remove:
                    try:
                        bpy.data.objects.remove(obj, do_unlink=True)
                    except (ReferenceError, AttributeError):
                        pass
            del self.mesh_nodes["boxes"][box_id]

        # 2. 从 context 中移除
        del self.context["boxes"][box_id]

        # 3. 重新计算 z_max
        wall_max = max((w["height"] for w in self.context["walls"].values()), default=0)
        box_max = 0
        for box in self.context["boxes"].values():
            box_top = box["center"][2] + box["scale"][2] / 2
            box_max = max(box_max, box_top)
        self.context["meta"]["z_max"] = max(wall_max, box_max)
        
        print(f"🗑️  家具 {box_id[:4]} 已删除")

    def get_context(self) -> Dict:
        """获取场景上下文"""
        return self.context
    
    def get_boxes(self) -> Dict:
        """获取所有家具boxes"""
        return self.context["boxes"]

    def set_wall_blender_texture_path(self, path: str):
        """设置墙体纹理路径"""
        self.config["wall_blender_texture_path"] = path
        print(f"📝 墙体纹理路径已更新: {path}")

    def set_floor_blender_texture_path(self, path: str):
        """设置地板纹理路径"""
        self.config["floor_blender_texture_path"] = path
        print(f"📝 地板纹理路径已更新: {path}")

    def set_ceiling_blender_texture_path(self, path: str):
        """设置天花板纹理路径"""
        self.config["ceiling_blender_texture_path"] = path
        print(f"📝 天花板纹理路径已更新: {path}")

    def set_hdri_path(self, path: str):
        """设置 HDRI 环境贴图路径"""
        self.config["hdri_path"] = path
        print(f"📝 HDRI 路径已更新: {path}")

    def set_blender_samples(self, samples: int):
        """设置 Blender 渲染采样数"""
        self.config["blender_samples"] = samples
        if self.scene and hasattr(self.scene, "cycles"):
            self.scene.cycles.samples = samples
        print(f"📝 Blender 渲染采样数已更新: {samples}")

    def _get_or_create_asset_collection(self, asset_id: int, model_paths: List[str]):
        """获取或创建资产的母版 Collection (用于实例化)"""
        if asset_id in self.asset_cache:
            return self.asset_cache[asset_id]
        
        # 1. 创建一个新的独立 Collection
        coll_name = f"AssetCollection_{asset_id}"
        if coll_name in bpy.data.collections:
            self.asset_cache[asset_id] = bpy.data.collections[coll_name]
            return self.asset_cache[asset_id]
            
        new_coll = bpy.data.collections.new(coll_name)
        # 暂时链接到场景以允许导入
        self.scene.collection.children.link(new_coll)
        
        # 2. 设置为活动集合并导入
        # 注意：Blender 的某些导入操作依赖于 active_collection
        orig_collection = bpy.context.view_layer.active_layer_collection
        
        # 递归寻找目标 layer_collection
        def find_layer_collection(layer_coll, name):
            if layer_coll.name == name: return layer_coll
            for child in layer_coll.children:
                res = find_layer_collection(child, name)
                if res: return res
            return None
            
        target_layer_coll = find_layer_collection(bpy.context.view_layer.layer_collection, coll_name)
        if target_layer_coll:
            bpy.context.view_layer.active_layer_collection = target_layer_coll
            
        # 3. 导入模型
        # 我们稍微修改逻辑，直接在 util_bpy.load_mesh_to_origin 中导入
        # 注意：load_mesh_to_origin 内部会把对象放到场景主集合中
        # 我们需要在导入后把它们移到我们的 new_coll 中
        mesh_root = util_bpy.load_mesh_to_origin(asset_id, model_paths)
        
        if mesh_root:
            # 将 mesh_root 及其所有子对象移入新集合
            objs_to_move = [mesh_root] + list(mesh_root.children_recursive)
            
            # --- [优化] 预先计算母版包围盒并存入缓存 ---
            master_bounds = util_bpy._get_combined_bounds(objs_to_move)
            self.asset_bounds[asset_id] = master_bounds
            
            for obj in objs_to_move:
                for coll in obj.users_collection:
                    coll.objects.unlink(obj)
                new_coll.objects.link(obj)
            
            # 导入成功后，从主场景 Collection 中移除这个资产母版 Collection (保持在数据块中即可)
            self.scene.collection.children.unlink(new_coll)
            self.asset_cache[asset_id] = new_coll
            print(f"📦 [母版加载] {asset_id} 加载成功并存入缓存 (Size: {master_bounds[1]-master_bounds[0] if master_bounds else 'None'})")
            return new_coll
        else:
            # 导入失败，清理
            self.scene.collection.children.unlink(new_coll)
            bpy.data.collections.remove(new_coll)
            self.asset_cache[asset_id] = None
            return None

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
        
        # 直接从 context["walls"] 导出所有墙体（包括边界和隔断）
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
        包含边界墙和内部隔断墙
        """
        output_path = os.path.join(output_dir, 'wall_hole_ssl.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        
        # 生成 room_id
        room_id = util.generate_unique_id()
        room_type = self.context["meta"]["scene_type"]
        lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
        
        # 遍历所有墙体及其携带的门窗
        for wall_id, wall in self.context["walls"].items():
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
        
        # 导出所有家具 (Bbox)
        for box_id, box in self.context["boxes"].items():
            label = box.get("label", box.get("class", "unknown"))
            center = box["center"]
            angle_z = box["angle_z"]
            scale = box["scale"]
            
            bbox_str = f'Bbox(id="{box_id}", room_id="{room_id}", label="{label}", center={center}, angle_z={angle_z}, scale={scale}'
            if box.get("asset_id") is not None:
                bbox_str += f', asset_id={box["asset_id"]}'
            bbox_str += ')'
            lines.append(bbox_str)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ 完整SSL已导出: {output_path}")
        return output_path

    # ==================== 纯 bpy 几何体创建（替代 util 中的 trimesh 函数）====================
    # helper functions now live in util_bpy

    # ==================== 场景构建（与 fast_scene.py 逻辑完全一致）====================
    
    def construct_floor(self, show_wall=True, show_window=True, show_door=True, show_ceiling=True, align_height: bool = True):
        """构建地板和墙体（与 fast_scene.py 完全一致的逻辑）"""
        # 注意：这里不创建 self.scene = pyrender.Scene()，而是使用已有的 bpy scene
        vertices = self.context["meta"]["vertices"]

        if len(vertices) < 3:
            return

        # 创建地板
        bounds = self.context["meta"]["bounds"]
        z_max = self.context["meta"]["z_max"]
        wall_thickness = self.config.get("wall_thickness", 0.1)
        
        f_scale = self.config.get("floor_texture_scale", 1.0)
        w_scale = self.config.get("wall_texture_scale", 1.0)
        c_scale = self.config.get("ceiling_texture_scale", 1.0)

        floor_obj = util_bpy.create_floor_mesh_bpy(self.scene_collection, vertices, texture_scale=f_scale)
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

            # --- 收集所有生成的墙体对象，用于后续布尔合并 ---
            all_wall_objs = []

            for wall_id, wall in self.context["walls"].items():
                is_partition = wall.get("is_partition", False)
                
                # 收集门窗信息
                openings = []
                for door in wall.get("doors", {}).values():
                    openings.append((door, "door"))
                for window in wall.get("windows", {}).values():
                    openings.append((window, "window"))
                
                # 获取预计算的偏移点
                outer_s, outer_e = wall_outer_points.get(wall_id, (None, None))

                # 墙体高度
                wall_h = z_max if align_height else wall["height"]

                # 对于隔断墙，检查端点是否连接在其他墙上，决定是否延长
                extend_s, extend_e = False, False
                if is_partition:
                    s_pt = tuple(wall["s"])
                    e_pt = tuple(wall["e"])
                    for other_id, other_wall in self.context["walls"].items():
                        if other_id == wall_id: continue
                        if util.point_on_segment(s_pt, tuple(other_wall["s"]), tuple(other_wall["e"])):
                            extend_s = True
                        if util.point_on_segment(e_pt, tuple(other_wall["s"]), tuple(other_wall["e"])):
                            extend_e = True

                # 创建墙体mesh
                wall_obj = util_bpy.create_single_wall_mesh_bpy(
                    self.scene_collection,
                    wall["s"], wall["e"], wall_h, wall["orientation"],
                    chip=False,
                    openings=openings if openings else None,
                    wall_thickness=wall_thickness,
                    name=f"Wall_{wall_id[:4]}",
                    outer_s=outer_s,
                    outer_e=outer_e,
                    texture_scale=w_scale,
                    is_partition=is_partition,
                    extend_s=extend_s,
                    extend_e=extend_e
                )

                if wall_obj:
                    # 应用纹理或颜色
                    if wall_texture_path and os.path.exists(wall_texture_path):
                        mat = util_bpy.create_material_with_texture(f"Wall_{wall_id}_Material", wall_texture_path)
                    else:
                        mat = util_bpy.create_material_with_color(f"Wall_{wall_id}_Material", wall_color)
                    
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
            print(f"🚪 开始处理门窗 (按 asset_id 加载)...")
            wall_thickness = self.config.get("wall_thickness", 0.1)
            
            hole_paths = []
            path_hole = self.config.get("model_hole_path")
            if path_hole: hole_paths.append(path_hole)
            if self.model_extra_path: hole_paths.append(self.model_extra_path)

            for wall_id, wall in self.context["walls"].items():
                wall_s = np.array(wall["s"])
                wall_e = np.array(wall["e"])
                wall_vec = wall_e - wall_s
                wall_len = np.linalg.norm(wall_vec)
                if wall_len < 1e-6:
                    continue

                wall_dir = wall_vec / wall_len
                # orientation 是指向内部的法线
                is_partition = wall.get("is_partition", False)
                orientation = np.array(wall["orientation"])
                
                # 策略 B：为隔断墙构造一个稳定的虚拟法线，用于保证模型正反面一致性
                if is_partition and np.linalg.norm(orientation) < 1e-4:
                    # 取墙体走向的垂直向量 [-dy, dx]
                    outward_normal = np.array([-wall_dir[1], wall_dir[0]])
                else:
                    # 外墙：指向内部的法线取取反即为向外偏移的方向
                    outward_normal = -orientation

                # 计算墙的旋转角度 (角度制)
                wall_angle_deg = np.degrees(np.arctan2(wall_dir[1], wall_dir[0]))

                # 定义要处理的类型列表
                types_to_process = []
                if show_door:
                    types_to_process.append(("door", "doors"))
                if show_window:
                    types_to_process.append(("window", "windows"))

                for item_type, key in types_to_process:
                    for item_id, item in wall.get(key, {}).items():
                        asset_id = item.get("asset_id")
                        if not asset_id:
                            continue

                        # 计算真实中心
                        inner_center = np.array(item["center"])
                        real_center = inner_center.copy()
                        
                        # 只有外墙需要向外侧偏移一半厚度，隔断墙模型直接放在中心线上
                        if not is_partition:
                            real_center[0] += outward_normal[0] * (wall_thickness / 2)
                            real_center[1] += outward_normal[1] * (wall_thickness / 2)

                        # 构造 box 格式的数据
                        item_box_data = {
                            "center": real_center.tolist(),
                            # 门窗模型的厚度通常由模型自身决定，但这里传入 wall_thickness 作为参考
                            "scale": [item["width"], wall_thickness, item["height"]],
                            "angle_z": wall_angle_deg,
                            "asset_id": asset_id
                        }

                        # 仿照 bbox 添加逻辑
                        # --- [优化] 门窗实例化 ---
                        asset_coll = self._get_or_create_asset_collection(asset_id, hole_paths)
                        if asset_coll:
                            # 创建实例
                            instance_name = f"Instance_{item_type}_{item_id[:4]}"
                            instance_obj = bpy.data.objects.new(instance_name, None)
                            instance_obj.instance_type = 'COLLECTION'
                            instance_obj.instance_collection = asset_coll
                            self.scene_collection.objects.link(instance_obj)
                            
                            try:
                                # 应用变换 (与家具逻辑一致)
                                # [优化] 传入预计算的 master_bounds
                                master_b = self.asset_bounds.get(asset_id)
                                success_transform = util_bpy.apply_box_transform(instance_obj, item_box_data, master_bounds=master_b)
                            except Exception as e:
                                print(f"  ❌ {item_type} {item_id[:4]} 实例化变换异常: {e}")
                                success_transform = False

                            if success_transform:
                                self.mesh_nodes[f"{item_type}s"][item_id] = {
                                    "node": instance_obj,
                                    "mesh": instance_obj,
                                    f"{item_type}_data": {"wall_id": wall_id}
                                }
                                # print(f"  ✅ {item_type} {item_id[:4]} (asset_id={asset_id}) 实例化成功")
                            else:
                                print(f"  ❌ {item_type} {item_id[:4]} 变换失败")
                                # 如果变换失败，清理
                                bpy.data.objects.remove(instance_obj, do_unlink=True)
                        else:
                            print(f"  ❌ {item_type} {item_id[:4]} (asset_id={asset_id}) 母版加载失败")
        else:
            print("🚫 跳过门窗创建 (show_door=False, show_window=False)")
        
        # 创建天花板
        if show_ceiling:
            z_max = self.context["meta"]["z_max"]
            ceiling_obj = util_bpy.create_ceiling_mesh_bpy(self.scene_collection, vertices, z_max, texture_scale=c_scale)
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
                       geometry_mode: str = "gltf", align_height: bool = True,
                       rebuild: bool = False):
        """
        构建完整场景。
        
        Args:
            show_wall, show_window, show_door, show_ceiling: 是否渲染对应的结构
            geometry_mode: 几何体模式，可选:
                - "gltf": 仅加载 GLTF 模型，若无 asset_id 或文件不存在则跳过该物体
                - "mixed": 优先加载 GLTF 模型，若失败则回退到 bbox 几何体
                - "bbox": 所有物体均强制使用 bbox 几何体表示
            align_height: 是否对齐墙体高度到 z_max
            rebuild: 是否重新构建场景（清空当前所有物体）
        """
        if rebuild:
            self.clear_scene()

        total_start = time.perf_counter()

        self.construct_floor(show_wall=show_wall, show_window=show_window, 
                             show_door=show_door, show_ceiling=show_ceiling, align_height=align_height)

        print(f"📦 加载家具 ({len(self.context['boxes'])} 个物体, 模式: {geometry_mode})...")
        success_count = 0
        load_time = 0
        transform_time = 0
        
        model_paths = []
        path1 = self.config.get("model_path")
        path2 = self.config.get("model_generate_path")
        if path1: model_paths.append(path1)
        if path2: model_paths.append(path2)
        if self.model_extra_path: model_paths.append(self.model_extra_path)

        for box_id, box in self.context["boxes"].items():
            asset_id = box.get("asset_id")
            
            # --- 模式判断逻辑 ---
            should_load_gltf = False
            should_fallback_bbox = False
            
            if geometry_mode == "bbox":
                should_fallback_bbox = True
            elif geometry_mode == "gltf":
                if asset_id:
                    should_load_gltf = True
                # else: skip
            elif geometry_mode == "mixed":
                if asset_id:
                    should_load_gltf = True
                else:
                    should_fallback_bbox = True
            
            # --- 执行加载 ---
            mesh_root = None
            if should_load_gltf:
                load_start = time.perf_counter()
                
                # --- [优化] 使用实例化 (Collection Instance) ---
                asset_coll = self._get_or_create_asset_collection(asset_id, model_paths)
                load_time += time.perf_counter() - load_start
                
                if asset_coll:
                    transform_start = time.perf_counter()
                    # 创建实例 (Empty 对象)
                    instance_name = f"Instance_{asset_id}_{box_id[:4]}"
                    instance_obj = bpy.data.objects.new(instance_name, None)
                    instance_obj.instance_type = 'COLLECTION'
                    instance_obj.instance_collection = asset_coll
                    self.scene_collection.objects.link(instance_obj)
                    
                    try:
                        # [优化] 传入预计算的 master_bounds
                        master_b = self.asset_bounds.get(asset_id)
                        success_transform = util_bpy.apply_box_transform(instance_obj, box, master_bounds=master_b)
                    except Exception as e:
                        print(f"  ❌ {box.get('class', 'unknown')} (asset_id={asset_id}) 实例化变换异常: {e}")
                        success_transform = False
                    transform_time += time.perf_counter() - transform_start
                    
                    if success_transform:
                        self.mesh_nodes["boxes"][box_id] = {
                            "node": instance_obj, "mesh": instance_obj, "box_data": box
                        }
                        success_count += 1
                        name = box.get('label', box.get('class', 'unknown'))
                        # print(f"  ✅ {name} (asset_id={asset_id}) 实例化成功")
                        continue
                
                # 如果 gltf 加载失败，判断是否需要回退
                if geometry_mode == "mixed":
                    should_fallback_bbox = True
                else:
                    print(f"  ❌ {box.get('class', 'unknown')} (asset_id={asset_id}) 导入失败且未开启混合模式，跳过")
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
                    name = box.get('label', box.get('class', 'unknown'))
                    print(f"  ✅ {name} (bbox几何体) 添加成功")
                except Exception as e:
                    name = box.get('label', box.get('class', 'unknown'))
                    print(f"  ❌ {name} (bbox几何体): {e}")

        total_time = time.perf_counter() - total_start
        print(f"✅ 场景构建完成! 成功添加 {success_count}/{len(self.context['boxes'])} 个物体")
        print(f"   构建耗时: {total_time:.2f}s (加载: {load_time:.2f}s, 变换: {transform_time:.2f}s)")

    def setup_lighting(self, intensity: float = 250,
                      lighting_type: Literal["area", "array", "none"] = "array",
                      ambient_light_color: list = None,
                      ambient_strength: float = 2.0):
        """设置场景光照，支持单盏大面积区域光、阵列光或不设置"""
        if self.if_set_lights:
            print("⚠️  光照已设置，跳过重复设置")
            return
        
        if lighting_type == "none":
            print("🌑 已选择 'none' 模式，不添加人造灯光")
            self.if_set_lights = True
            return

        center = self.context["meta"]["center"]
        z_max = self.context["meta"]["z_max"]
        span = self.context["meta"]["span"]
        
        # 柔和黄白色 (Soft Warm White)
        warm_white = [1.0, 0.95, 0.8]

        # 1. 设置环境光
        world = self.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            self.scene.world = world
        
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            if ambient_light_color is None:
                # 真实的自然环境光通常带一点点蓝色 (Daylight Blue)
                ambient_light_color = [0.95, 0.97, 1.0, 1.0]
            elif len(ambient_light_color) == 3:
                ambient_light_color = ambient_light_color + [1.0]
            bg_node.inputs['Color'].default_value = ambient_light_color
            bg_node.inputs['Strength'].default_value = ambient_strength
        
        # 2. 添加室内人造灯 (高度在 z_max - 0.1)
        light_z = z_max - 0.1
        
        # 更加符合摄影感的室内暖白光 (Warm White ~3500K)
        # 这种颜色与环境光的淡蓝色对比会产生非常真实的室内氛围
        realistic_warm = [1.0, 0.88, 0.75]
        
        if lighting_type == "area":
            # 方案一：单盏大面积区域光
            light_data = bpy.data.lights.new(name="MainAreaLight", type='AREA')
            light_data.shape = 'RECTANGLE'
            # 覆盖房间 80% 的面积
            light_data.size = span[0] * 0.8
            light_data.size_y = span[1] * 0.8
            light_data.energy = intensity
            light_data.color = realistic_warm
            
            light_obj = bpy.data.objects.new(name="MainAreaLight", object_data=light_data)
            self.scene_collection.objects.link(light_obj)
            light_obj.location = (center[0], center[1], light_z)
            print(f"💡 已添加单盏大面积区域光 (Intensity={intensity})")
            
        elif lighting_type == "array":
            # 方案二：阵列区域光 (筒灯风格)
            # 每隔 1.5 米布置一盏，且避开边缘
            nx = int(span[0] / 1.5)
            ny = int(span[1] / 1.5)
            
            # 计算 X 轴坐标 (居中排布)
            if nx <= 1:
                x_coords = [center[0]]
                nx = 1
            else:
                total_w = (nx - 1) * 1.5
                x_coords = np.linspace(center[0] - total_w/2, center[0] + total_w/2, nx)
                
            # 计算 Y 轴坐标 (居中排布)
            if ny <= 1:
                y_coords = [center[1]]
                ny = 1
            else:
                total_h = (ny - 1) * 1.5
                y_coords = np.linspace(center[1] - total_h/2, center[1] + total_h/2, ny)
            
            # 每个光源使用恒定强度，确保大房间亮度自动增加
            each_intensity = self.config.get("array_light_intensity", 50.0)
            count = 0
            for x in x_coords:
                for y in y_coords:
                    count += 1
                    l_name = f"ArrayLight_{count}"
                    l_data = bpy.data.lights.new(name=l_name, type='AREA')
                    l_data.shape = 'DISK'
                    l_data.size = 0.3  # 直径 0.3 米的圆盘
                    l_data.energy = each_intensity
                    l_data.color = realistic_warm
                    
                    l_obj = bpy.data.objects.new(name=l_name, object_data=l_data)
                    self.scene_collection.objects.link(l_obj)
                    l_obj.location = (x, y, light_z)
            print(f"💡 已添加加密阵列区域光 x{count} (Each Intensity={each_intensity}, Total={each_intensity*count})")
        
        self.if_set_lights = True

    def topdown_view(self, output_path: str, width: int = 1024, height: int = 1024,
                     geometry_mode: str = "gltf", show_wall: bool = True, 
                     show_window: bool = True, show_door: bool = True, show_ceiling: bool = True,
                     up_vector: list = None,
                     auto_fov: bool = True, manual_fov: float = None,
                     auto_transparent: bool = True, transparent_alpha: float = 0.3,
                     render_depth: bool = False, use_HDRI: bool = True,
                     hdri_transparent_background: bool = True,
                     visible_shadow: bool = True,
                     lighting_type: Literal["area", "array", "none"] = "array",
                     align_height: bool = True,
                     rebuild: bool = False):
        """俯视图渲染（与 fast_scene.py 逻辑完全一致）"""
        total_start = time.perf_counter()

        construct_time = 0
        if rebuild or self.mesh_nodes["floor"] is None:
            construct_start = time.perf_counter()
            self.construct_scene(geometry_mode=geometry_mode, 
                               show_wall=show_wall, show_window=show_window, 
                               show_door=show_door, show_ceiling=show_ceiling, 
                               align_height=align_height, rebuild=rebuild)
            construct_time = time.perf_counter() - construct_start

        # 设置阴影可见性
        for wall_info in self.mesh_nodes["walls"].values():
            wall_obj = wall_info.get("node")
            if wall_obj:
                wall_obj.visible_shadow = visible_shadow

        ceiling_info = self.mesh_nodes.get("ceiling")
        if ceiling_info:
            ceiling_obj = ceiling_info.get("node")
            if ceiling_obj:
                ceiling_obj.visible_shadow = visible_shadow

        # Setup lighting
        setup_start = time.perf_counter()
        if not self.if_set_lights:
            # 顶视图合并环境光与指定的人造灯
            self.setup_lighting(lighting_type=lighting_type, ambient_light_color=[1.0, 1.0, 1.0], ambient_strength=3.0)

        # 显式控制背景透明度
        self.scene.render.film_transparent = hdri_transparent_background

        if use_HDRI:
            hdri_path = self.config.get("hdri_path")
            if hdri_path and util_bpy.apply_hdri_to_world(self.scene, hdri_path, strength=1.0):
                print(f"🌇 使用 HDRI 环境光: {hdri_path}")
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

        up_vector = np.array(up_vector if up_vector else [0.0, 1.0, 0.0], dtype=float)
        up_norm = np.linalg.norm(up_vector)
        up_vector = up_vector / up_norm if up_norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=float)

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

        forward = look_at_target - camera_position
        forward = forward / np.linalg.norm(forward) if np.linalg.norm(forward) > 1e-6 else np.array([0.0, 0.0, -1.0], dtype=float)

        right = np.cross(forward, up_vector)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            fallback_up = np.array([0.0, 0.0, 1.0], dtype=float)
            right = np.cross(forward, fallback_up)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                right = np.array([1.0, 0.0, 0.0], dtype=float)
                right_norm = 1.0
        right = right / right_norm
        up = np.cross(right, forward)

        camera_data = bpy.data.cameras.new(name=f"Camera_{id(self)}")
        camera_obj = bpy.data.objects.new(f"Camera_{id(self)}", camera_data)
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
        self.scene.render.image_settings.color_mode = 'RGBA'

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
        print(f"   构建: {construct_time:.2f}s, 设置: {setup_time:.2f}s, 渲染: {render_time:.2f}s, 保存: {save_time:.2f}s, 总耗时: {total_time:.2f}s")

    def render_view(self, output_path: str, camera_position: list, look_at_target: list = None, 
                    width: int = 1024, height: int = 1024, up_vector: list = None, 
                    auto_fov: bool = True, manual_fov: float = None,
                    auto_transparent: bool = True, transparent_alpha: float = 0.3,
                    geometry_mode: str = "gltf", render_depth: bool = False,
                    show_wall: bool = True, show_window: bool = True, show_door: bool = True, show_ceiling: bool = True,
                    use_HDRI: bool = True, hdri_transparent_background: bool = False,
                    visible_shadow: bool = True,
                    lighting_type: Literal["area", "array", "none"] = "array",
                    align_height: bool = True,
                    rebuild: bool = False):
        total_start = time.perf_counter()

        construct_time = 0.0
        if rebuild or self.mesh_nodes["floor"] is None:
            construct_start = time.perf_counter()
            self.construct_scene(geometry_mode=geometry_mode,
                                 show_wall=show_wall, show_window=show_window,
                                 show_door=show_door, show_ceiling=show_ceiling, 
                                 align_height=align_height, rebuild=rebuild)
            construct_time = time.perf_counter() - construct_start

        # 设置阴影可见性
        for wall_info in self.mesh_nodes["walls"].values():
            wall_obj = wall_info.get("node")
            if wall_obj:
                wall_obj.visible_shadow = visible_shadow
        ceiling_info = self.mesh_nodes.get("ceiling")
        if ceiling_info:
            ceiling_obj = ceiling_info.get("node")
            if ceiling_obj:
                ceiling_obj.visible_shadow = visible_shadow

        setup_start = time.perf_counter()
        if not self.if_set_lights:
            self.setup_lighting(lighting_type=lighting_type)

        if use_HDRI:
            hdri_path = self.config.get("hdri_path")
            if hdri_path and util_bpy.apply_hdri_to_world(self.scene, hdri_path, strength=1.0):
                print(f"🌇 使用 HDRI 环境光: {hdri_path}")
                # 显式控制：True 则透明，False 则显示 HDRI 贴图
                self.scene.render.film_transparent = hdri_transparent_background
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

        render = self.scene.render
        render.image_settings.color_mode = 'RGBA'
        render.film_transparent = hdri_transparent_background

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
            print(f"   构建: {construct_time:.2f}s, 设置: {setup_time:.2f}s, 渲染: {render_time:.2f}s, 总耗时: {total_time:.2f}s")
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
    jsonl_path = '/data-nas/data/experiments/mushui/datasets/manycore/spatiallm_raw.jsonl'
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
        ctx.topdown_view(topdown_path, geometry_mode="gltf",
                        show_wall=True, show_window=True, show_door=True, show_ceiling=False, auto_fov=True, auto_transparent=False, use_HDRI=True, render_depth=True)

        look_at = [
            (ctx.context["meta"]["center"][0], ctx.context["meta"]["center"][1], ctx.context["meta"]["z_max"] / 2)
        ][0]
        views = [
            # ("right", [look_at[0] + ctx.context["meta"]["span"][0]/2 - 0.1, look_at[1], ctx.context["meta"]["z_max"] * 2/3]),
            # ("left", [look_at[0] - ctx.context["meta"]["span"][0]/2 + 0.1, look_at[1], ctx.context["meta"]["z_max"] * 2/3]),
            # ("front", [look_at[0], look_at[1] + ctx.context["meta"]["span"][1]/2 - 0.1, ctx.context["meta"]["z_max"] * 2/3]),
            # ("back", [look_at[0], look_at[1] - ctx.context["meta"]["span"][1]/2 + 0.1, ctx.context["meta"]["z_max"] * 2/3]),
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
                geometry_mode="gltf",
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
