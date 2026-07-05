"""
Fast Scene - 纯 Blender 渲染版本（无 trimesh 依赖）

用 bpy 替代 trimesh/pyrender

使用方式：
   $BLENDER_PATH --background --python -m fast_scene.core.fast_scene_bpy
   或者在 pip install bpy 后: python -m fast_scene.core.fast_scene_bpy
"""

import os
import sys
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_PLATFORM'] = 'surfaceless'
current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

import json
import numpy as np
import time
import hashlib
from typing import List, Dict, Any, Optional, Literal, Tuple, Union

# 导入 util 的数据处理函数（不使用其 mesh 创建函数）
try:
    from . import util, util_bpy, util_data
    from .config_utils import CONFIG_PATH, load_config
except ImportError:
    import util
    import util_bpy
    import util_data  # type: ignore
    from config_utils import CONFIG_PATH, load_config

try:
    import bpy
    import mathutils  # type: ignore[import]
    from mathutils import Matrix  # type: ignore[import]
except ImportError:
    print("❌ 错误：无法导入 bpy 模块")
    print("   请在 Blender Python 环境中运行此脚本")
    raise


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
        self._point_cloud_material_cache = {}
        self._point_cloud_image_cache = {}
        
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
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        self.config = load_config(config_path)

        # 初始化 Blender 场景
        self._init_blender_scene()

    def set_model_path(self, path: str):
        """更改模型查找路径"""
        print(f"🔄 更改模型路径为: {path}")
        self.config["model_path"] = path

    def _asset_search_paths(self, default_key: str = "model_path") -> List[str]:
        """extra_path → 默认路径 → generate 路径（去重保序）。"""
        candidates = []
        if self.model_extra_path:
            candidates.append(self.model_extra_path)
        default = self.config.get(default_key)
        if default:
            candidates.append(default)
        generate = self.config.get("model_generate_path")
        if generate:
            candidates.append(generate)
        seen = set()
        ordered = []
        for path in candidates:
            if path and path not in seen:
                seen.add(path)
                ordered.append(path)
        return ordered

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
        # 2b. 释放 Cycles 渲染缓冲
        try:
            from . import util_bpy
            util_bpy.cleanup_bpy_render_memory(self.scene)
        except Exception:
            pass
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
        self._point_cloud_material_cache = {}
        self._point_cloud_image_cache = {}
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
        """添加墙体并计算场景元数据（自动跳过 height<=0 的墙，支持室外场景）"""
        # walls_converted: 用全部墙体来计算多边形和元数据（包含 height=0 的室外边界）
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

        # z_max: 取所有墙体的最大高度，室外场景全为0时 z_max=0
        z_max = max(w["height"] for w in walls_converted)

        self.context["meta"].update({
            "bounds": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
            "center": [(min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2],
            "span": [max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)],
            "z_max": z_max,
            "vertices": vertices
        })
        self.context["meta"].pop("_scene_normalized", None)

        # 过滤 height<=0 的墙，室外场景只保留地板多边形，不构建墙体对象
        valid_walls = [w for w in walls_converted if w["height"] > 0]
        if len(valid_walls) < len(walls_converted):
            print(f"🏞️ 室外场景：跳过 {len(walls_converted) - len(valid_walls)} 面高度为0的墙")
        if not valid_walls:
            print(f"🏞️ 纯室外场景：仅计算地板多边形({len(vertices)}顶点)，不构建墙体")
            return

        print(f"📐 输入原始墙体数量: {len(valid_walls)}")
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
            for w in valid_walls:
                if util.is_wall_on_edge(w, v_s, v_e):
                    height = w["height"]
                    break
            if height is None:
                # 容错：取所有有效墙体中的最大高度作为默认值
                height = max(w["height"] for w in valid_walls)

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
            if p[4] <= 0:
                continue  # 跳过高度为0的隔断
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
        self.context["meta"].pop("_scene_normalized", None)
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
        self.context["meta"].pop("_scene_normalized", None)
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
        self.context["meta"].pop("_scene_normalized", None)
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

    def normalize_scene_data(self) -> None:
        """统一 context label / 校验 asset_id（与输出 ssl 一致）。"""
        try:
            from . import util_data
        except ImportError:
            import util_data  # type: ignore
        util_data.normalize_scene_context(
            self.context,
            model_paths=self._asset_search_paths("model_path"),
            hole_paths=self._asset_search_paths("model_hole_path"),
        )

    def _object_export_identity(self, category: str, object_id: str):
        """点云/可见几何命名：(label, asset_id|None)。"""
        if category == "walls":
            wall = self.context["walls"].get(object_id, {})
            return wall.get("label", object_id), None
        if category == "boxes":
            box = self.context["boxes"].get(object_id, {})
            return box.get("label", object_id), box.get("asset_id")
        if category in ("doors", "windows"):
            key = "doors" if category == "doors" else "windows"
            for wall in self.context["walls"].values():
                item = wall.get(key, {}).get(object_id)
                if item is not None:
                    return item.get("label", object_id), item.get("asset_id")
            return object_id, None
        return object_id, None

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
            
            hole_paths = self._asset_search_paths("model_hole_path")

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
        # 室外场景（所有墙高度为0）不构建天花板，即使 show_ceiling=True
        wall_max_height = max((w["height"] for w in self.context["walls"].values()), default=0)
        if show_ceiling and wall_max_height > 0:
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
            if not show_ceiling:
                print("🚫 跳过天花板 (show_ceiling=False)")
            else:
                print("🏞️ 跳过天花板 (室外场景，墙体最大高度为0)")

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

        self.normalize_scene_data()
        total_start = time.perf_counter()

        self.construct_floor(show_wall=show_wall, show_window=show_window, 
                             show_door=show_door, show_ceiling=show_ceiling, align_height=align_height)

        print(f"📦 加载家具 ({len(self.context['boxes'])} 个物体, 模式: {geometry_mode})...")
        success_count = 0
        load_time = 0
        transform_time = 0
        
        model_paths = self._asset_search_paths("model_path")

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

    def export_glb(self, output_path: str, rebuild: bool = False, **construct_kwargs):
        """导出当前 Blender 场景为 GLB 文件。"""
        if rebuild or self.mesh_nodes["floor"] is None:
            self.construct_scene(rebuild=rebuild, **construct_kwargs)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        bpy.context.view_layer.update()

        try:
            bpy.ops.export_scene.gltf(
                filepath=output_path,
                export_format='GLB',
            )
        except TypeError:
            # 兼容少数 Blender 版本的参数差异，.glb 后缀仍会导出二进制 glTF。
            bpy.ops.export_scene.gltf(filepath=output_path)

        print(f"✅ GLB 已导出: {output_path}")
        return output_path

    def export_point_cloud(self, output_dir: str, rebuild: bool = False, **construct_kwargs):
        """按场景对象分别采样表面点云并导出为彩色 PLY。"""
        wall_max_height = max((w["height"] for w in self.context["walls"].values()), default=0)
        need_ceiling = wall_max_height > 0
        if rebuild or self.mesh_nodes["floor"] is None or (need_ceiling and self.mesh_nodes["ceiling"] is None):
            kwargs = {"show_ceiling": True}
            kwargs.update(construct_kwargs)
            self.construct_scene(rebuild=True, **kwargs)

        os.makedirs(output_dir, exist_ok=True)
        default_samples = 5000
        box_samples = 5000
        metadata = {
            "samples_per_object": default_samples,
            "box_samples_per_object": box_samples,
            "box_sampling_strategy": "surface_area_50_ses_50",
            "door_window_sampling_strategy": "surface_area_50_ses_50",
            "ses_angle_threshold_degrees": 30.0,
            "objects": [],
        }
        all_points = []
        all_colors = []

        def export_one(category: str, object_id: str, node, filename: str, sample_count: int,
                       extra: Optional[Dict[str, Any]] = None):
            if node is None:
                return
            use_ses = category in ("boxes", "doors", "windows")
            points, colors = self._sample_bpy_node_surface(
                node,
                sample_count,
                use_ses=use_ses,
            )
            if len(points) == 0:
                print(f"⚠️ 点云采样跳过空对象: {category}/{object_id}")
                return

            path = os.path.join(output_dir, filename)
            self._write_ply(path, points, colors)
            all_points.append(points)
            all_colors.append(colors)

            record = {
                "category": category,
                "id": object_id,
                "path": os.path.relpath(path, output_dir),
                "points": int(len(points)),
                "requested_samples": int(sample_count),
            }
            if extra:
                record.update(extra)
            metadata["objects"].append(record)

        floor_info = self.mesh_nodes.get("floor")
        if floor_info:
            export_one("floor", "floor", floor_info.get("node"), "floor.ply", default_samples)

        ceiling_info = self.mesh_nodes.get("ceiling")
        if ceiling_info:
            export_one("ceiling", "ceiling", ceiling_info.get("node"), "ceiling.ply", default_samples)

        for category in ["walls", "doors", "windows", "boxes"]:
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            for object_id, info in self.mesh_nodes[category].items():
                label, asset_id = self._object_export_identity(category, object_id)
                filename = util_data.build_pointcloud_ply_relpath(category, label, asset_id)
                extra = {"label": label}
                if asset_id is not None:
                    extra["asset_id"] = asset_id
                sample_count = box_samples if category == "boxes" else default_samples
                export_one(category, object_id, info.get("node"), filename, sample_count, extra)

        if all_points:
            merged_points = np.vstack(all_points)
            merged_colors = np.vstack(all_colors)
            scene_path = os.path.join(output_dir, "scene_all.ply")
            self._write_ply(scene_path, merged_points, merged_colors)
            metadata["scene_all"] = {
                "path": "scene_all.ply",
                "points": int(len(merged_points)),
            }

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ 点云已导出: {output_dir}")
        return output_dir

    def export_visible_geometry(
        self,
        output_dir: str,
        camera_obj,
        export_glb: bool = False,
        export_point_cloud: bool = False,
        transparent_objects: Optional[set] = None,
    ):
        """导出可见几何：先按遮挡剔除（部分可见则保留），再按渲染视锥裁剪视锥外部分。"""
        bpy.context.view_layer.update()
        visible_entries = self._visible_geometry_entries(
            camera_obj, transparent_objects=transparent_objects or set()
        )
        if not visible_entries:
            print("⚠️ 当前视角没有检测到可见几何")
            return

        if export_glb:
            self.export_visible_glb(os.path.join(output_dir, "scene_visible.glb"), visible_entries)
        if export_point_cloud:
            self.export_visible_point_cloud(os.path.join(output_dir, "pointcloud"), visible_entries)

    def export_visible_geometry_multi(
        self,
        output_dir: str,
        camera_states: List[Tuple[Any, set]],
        export_glb: bool = False,
        export_point_cloud: bool = False,
    ):
        """多相机合并导出：任意视角可见则保留，视锥裁剪取各视角并集。"""
        bpy.context.view_layer.update()
        visible_entries = self._visible_geometry_entries_multi(camera_states)
        if not visible_entries:
            print("⚠️ 多视角序列没有检测到可见几何")
            return
        if export_glb:
            self.export_visible_glb(os.path.join(output_dir, "scene_visible.glb"), visible_entries)
        if export_point_cloud:
            self.export_visible_point_cloud(os.path.join(output_dir, "pointcloud"), visible_entries)

    def export_planar_faces_and_lines(
        self,
        output_path: str,
        camera_obj,
        width: int,
        height: int,
        *,
        align_height: bool = True,
        show_wall: bool = True,
        show_door: bool = True,
        show_window: bool = True,
        show_ceiling: bool = True,
        transparent_objects: Optional[set] = None,
    ):
        """导出平面内表面顶点 JSON 与连线 overlay（随 --ply 默认启用）。"""
        try:
            from . import planar_faces
        except ImportError:
            import planar_faces  # type: ignore

        view_ctx = planar_faces.make_bpy_view_context(self.scene, camera_obj, width, height)
        include_floor = self.mesh_nodes.get("floor") is not None
        include_ceiling = show_ceiling and self.mesh_nodes.get("ceiling") is not None
        occlusion_fn = self.build_planar_vertex_occlusion_fn(
            camera_obj, transparent_objects=transparent_objects or set()
        )
        return planar_faces.export_planar_faces_for_view(
            self.context,
            output_path,
            view_ctx["clip_mats"],
            view_ctx["to_pixel"],
            view_ctx["camera_pos"],
            width,
            height,
            align_height=align_height,
            show_wall=show_wall,
            show_door=show_door,
            show_window=show_window,
            include_floor=include_floor,
            include_ceiling=include_ceiling,
            occlusion_fn=occlusion_fn,
        )

    def _resolve_planar_target_node(self, category: str, obj_id: str):
        if category == "floor":
            info = self.mesh_nodes.get("floor")
            return info.get("node") if info else None
        if category == "ceiling":
            info = self.mesh_nodes.get("ceiling")
            return info.get("node") if info else None
        if category in self.mesh_nodes:
            return self.mesh_nodes[category].get(obj_id, {}).get("node")
        return None

    def build_planar_vertex_occlusion_fn(self, camera_obj, transparent_objects: Optional[set] = None):
        """返回 (category, obj_id, point)->0/1，与可见几何相同的射线遮挡判定。"""
        transparent_objects = transparent_objects or set()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        target_cache: Dict[Tuple[str, str], set] = {}

        def _targets(category: str, obj_id: str) -> set:
            key = (category, obj_id)
            if key not in target_cache:
                node = self._resolve_planar_target_node(category, obj_id)
                target_cache[key] = self._visibility_target_objects(node, [])
            return target_cache[key]

        def occlusion_fn(category: str, obj_id: str, point) -> int:
            targets = _targets(category, obj_id)
            if not targets:
                return 0
            visible = self._point_visible_from_camera(
                point, camera_obj, targets, transparent_objects, [], depsgraph
            )
            return 0 if visible else 1

        return occlusion_fn

    def export_visible_glb(self, output_path: str, visible_entries: List[Dict[str, Any]]):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        temp_objects = []
        prev_selection = list(bpy.context.selected_objects)
        prev_active = bpy.context.view_layer.objects.active
        try:
            bpy.ops.object.select_all(action='DESELECT')
            for entry in visible_entries:
                name_suffix = "_visible_cutted" if entry.get("frustum_cutted") else "_visible"
                mesh = bpy.data.meshes.new(f"{entry['category']}_{entry['id']}{name_suffix}_mesh")
                vertices = []
                faces = []
                face_material_indices = []
                face_uvs = []
                materials = []
                material_map = {}

                for record in entry["records"]:
                    for mat in record.get("materials", []):
                        if mat is not None and mat.name not in material_map:
                            material_map[mat.name] = len(materials)
                            materials.append(mat)

                    material_indices = record.get("material_indices")
                    for tri_idx, tri in enumerate(record["triangles"]):
                        tri_uv = record["uvs"][tri_idx] if record.get("uvs") is not None else None
                        source_mat_index = int(material_indices[tri_idx]) if material_indices is not None else 0
                        target_mat_index = min(source_mat_index, max(len(materials) - 1, 0))
                        base = len(vertices)
                        vertices.extend([tuple(p) for p in tri])
                        faces.append((base, base + 1, base + 2))
                        face_material_indices.append(target_mat_index)
                        face_uvs.append(tri_uv)

                if not faces:
                    continue
                mesh.from_pydata(vertices, [], faces)
                mesh.update()
                for mat in materials:
                    mesh.materials.append(mat)
                for poly, mat_index in zip(mesh.polygons, face_material_indices):
                    poly.material_index = mat_index
                if any(uv is not None for uv in face_uvs):
                    uv_layer = mesh.uv_layers.new(name="UVMap")
                    for poly, tri_uv in zip(mesh.polygons, face_uvs):
                        if tri_uv is None:
                            continue
                        for loop_idx, uv in zip(poly.loop_indices, tri_uv):
                            uv_layer.data[loop_idx].uv = tuple(uv)

                obj = bpy.data.objects.new(f"{entry['category']}_{entry['id']}{name_suffix}", mesh)
                self.scene_collection.objects.link(obj)
                obj.select_set(True)
                temp_objects.append(obj)

            if not temp_objects:
                print("⚠️ scene_visible.glb 跳过：裁剪后没有几何")
                return
            bpy.context.view_layer.objects.active = temp_objects[0]
            bpy.context.view_layer.update()
            try:
                bpy.ops.export_scene.gltf(
                    filepath=output_path,
                    export_format='GLB',
                    use_selection=True,
                )
            except TypeError:
                bpy.ops.export_scene.gltf(filepath=output_path, use_selection=True)
            print(f"✅ 可见 GLB 已导出: {output_path}")
        finally:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in temp_objects:
                mesh = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if mesh:
                    bpy.data.meshes.remove(mesh, do_unlink=True)
            for obj in prev_selection:
                try:
                    obj.select_set(True)
                except ReferenceError:
                    pass
            bpy.context.view_layer.objects.active = prev_active

    def export_visible_point_cloud(self, output_dir: str, visible_entries: List[Dict[str, Any]]):
        os.makedirs(output_dir, exist_ok=True)
        default_samples = 5000
        box_samples = 5000
        all_points = []
        all_colors = []
        metadata = {
            "samples_per_object": default_samples,
            "box_samples_per_object": box_samples,
            "objects": [],
            "visible_geometry": True,
        }

        for entry in visible_entries:
            sample_count = box_samples if entry["category"] == "boxes" else default_samples
            points, colors = self._sample_visible_records(entry["records"], entry["camera_obj"], sample_count)
            if len(points) == 0:
                continue
            path = os.path.join(output_dir, entry["visible_ply"])
            self._write_ply(path, points, colors)
            all_points.append(points)
            all_colors.append(colors)
            metadata["objects"].append({
                "category": entry["category"],
                "id": entry["id"],
                "path": os.path.relpath(path, output_dir),
                "points": int(len(points)),
                "requested_samples": int(sample_count),
                "frustum_cutted": bool(entry.get("frustum_cutted", False)),
                "frustum_in_view_ratio": round(float(entry.get("frustum_in_view_ratio", 1.0)), 4),
            })

        if all_points:
            merged_points = np.vstack(all_points)
            merged_colors = np.vstack(all_colors)
            scene_path = os.path.join(output_dir, "scene_visible.ply")
            self._write_ply(scene_path, merged_points, merged_colors)
            metadata["scene_visible"] = {
                "path": "scene_visible.ply",
                "points": int(len(merged_points)),
            }

        with open(os.path.join(output_dir, "metadata_visible.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✅ 可见点云已导出: {output_dir}")

    def _visible_geometry_entries(self, camera_obj, transparent_objects: Optional[set] = None):
        transparent_objects = transparent_objects or set()
        entries = []
        clip_mats = self._render_frustum_clip_mats(camera_obj)
        for category, object_id, node, visible_ply in self._iter_point_cloud_nodes(visible_suffix=True):
            records = self._collect_bpy_mesh_records(node, use_ses=category in ("boxes", "doors", "windows"))
            if not records:
                continue
            target_objects = self._visibility_target_objects(node, records)
            if self._records_fully_occluded(records, camera_obj, target_objects, transparent_objects):
                continue
            clipped_records = self._clip_records_to_render_frustum(records, camera_obj, clip_mats=clip_mats)
            if not clipped_records:
                continue
            frustum_cutted = self._records_frustum_cutted(records, camera_obj)
            in_view_ratio_value = self._records_frustum_in_view_ratio(records, camera_obj)
            output_ply = self._append_output_suffix(visible_ply, "_cutted") if frustum_cutted else visible_ply
            entries.append({
                "category": category,
                "id": object_id,
                "node": node,
                "records": clipped_records,
                "visible_ply": output_ply,
                "frustum_cutted": frustum_cutted,
                "frustum_in_view_ratio": in_view_ratio_value,
                "camera_obj": camera_obj,
            })
        return entries

    def _visible_geometry_entries_multi(self, camera_states: List[Tuple[Any, set]]):
        """多相机：遮挡任意可见则保留；视锥裁剪合并各相机保留部分。"""
        entries = []
        for category, object_id, node, visible_ply in self._iter_point_cloud_nodes(visible_suffix=True):
            records = self._collect_bpy_mesh_records(node, use_ses=category in ("boxes", "doors", "windows"))
            if not records:
                continue
            target_objects = self._visibility_target_objects(node, records)

            visible_from_any = False
            for camera_obj, frame_transparent in camera_states:
                if self._records_visible_from_camera(
                    records, camera_obj, target_objects, frame_transparent
                ):
                    visible_from_any = True
                    break
            if not visible_from_any:
                continue

            all_clipped_records = []
            frustum_ratios = []
            for camera_obj, _frame_transparent in camera_states:
                clip_mats = self._render_frustum_clip_mats(camera_obj)
                clipped = self._clip_records_to_render_frustum(records, camera_obj, clip_mats=clip_mats)
                if clipped:
                    all_clipped_records.extend(clipped)
                frustum_ratios.append(self._records_frustum_in_view_ratio(records, camera_obj))
            if not all_clipped_records:
                continue

            max_ratio = max(frustum_ratios) if frustum_ratios else 0.0
            frustum_cutted = max_ratio < self.FRUSTUM_IN_VIEW_RATIO
            output_ply = self._append_output_suffix(visible_ply, "_cutted") if frustum_cutted else visible_ply
            primary_camera = camera_states[0][0] if camera_states else None
            entries.append({
                "category": category,
                "id": object_id,
                "node": node,
                "records": all_clipped_records,
                "visible_ply": output_ply,
                "frustum_cutted": frustum_cutted,
                "frustum_in_view_ratio": max_ratio,
                "camera_obj": primary_camera,
            })
        return entries

    @staticmethod
    def _append_output_suffix(path: str, suffix: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}{suffix}{ext}"

    def _iter_point_cloud_nodes(self, visible_suffix: bool = False):
        suffix = "_visible" if visible_suffix else ""
        floor_info = self.mesh_nodes.get("floor")
        if floor_info:
            yield "floor", "floor", floor_info.get("node"), f"floor{suffix}.ply"
        ceiling_info = self.mesh_nodes.get("ceiling")
        if ceiling_info:
            yield "ceiling", "ceiling", ceiling_info.get("node"), f"ceiling{suffix}.ply"
        for category in ["walls", "doors", "windows", "boxes"]:
            for object_id, info in self.mesh_nodes[category].items():
                label, asset_id = self._object_export_identity(category, object_id)
                suffixes = ["visible"] if visible_suffix else []
                filename = util_data.build_pointcloud_ply_relpath(
                    category, label, asset_id, suffixes,
                )
                yield category, object_id, info.get("node"), filename

    def _render_frustum_clip_mats(self, camera_obj):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        render = self.scene.render
        pct = float(render.resolution_percentage) / 100.0
        width = max(int(render.resolution_x * pct), 1)
        height = max(int(render.resolution_y * pct), 1)
        modelview = np.array(camera_obj.matrix_world.inverted(), dtype=float)
        calc_fn = getattr(camera_obj, "calc_matrix_camera", None)
        if calc_fn is None:
            calc_fn = camera_obj.data.calc_matrix_camera
        proj = np.array(
            calc_fn(
                depsgraph,
                x=width,
                y=height,
                scale_x=float(getattr(render, "pixel_aspect_x", 1.0)),
                scale_y=float(getattr(render, "pixel_aspect_y", 1.0)),
            ),
            dtype=float,
        )
        return proj, modelview

    @staticmethod
    def _render_frustum_clip_planes():
        eps = 1e-5
        return [
            (np.array([0.0, 0.0, 0.0, 1.0], dtype=float), eps),
            (np.array([1.0, 0.0, 0.0, 1.0], dtype=float), 0.0),
            (np.array([-1.0, 0.0, 0.0, 1.0], dtype=float), 0.0),
            (np.array([0.0, 1.0, 0.0, 1.0], dtype=float), 0.0),
            (np.array([0.0, -1.0, 0.0, 1.0], dtype=float), 0.0),
        ]

    @staticmethod
    def _clip_homogeneous_polygon_against_plane(polygon: List[Dict[str, Any]], plane_normal, plane_offset: float):
        normal = np.asarray(plane_normal, dtype=float)
        clipped = []
        prev = polygon[-1]
        prev_dist = float(np.dot(normal, prev["clip"]) - plane_offset)
        prev_inside = prev_dist >= -1e-8
        for curr in polygon:
            curr_dist = float(np.dot(normal, curr["clip"]) - plane_offset)
            curr_inside = curr_dist >= -1e-8
            if curr_inside != prev_inside:
                denom = prev_dist - curr_dist
                t = 0.0 if abs(denom) <= 1e-12 else prev_dist / denom
                t = float(np.clip(t, 0.0, 1.0))
                clipped.append({
                    "clip": prev["clip"] + t * (curr["clip"] - prev["clip"]),
                    "world": prev["world"] + t * (curr["world"] - prev["world"]),
                    "uv": (
                        None
                        if prev["uv"] is None or curr["uv"] is None
                        else prev["uv"] + t * (curr["uv"] - prev["uv"])
                    ),
                })
            if curr_inside:
                clipped.append(curr)
            prev = curr
            prev_dist = curr_dist
            prev_inside = curr_inside
        return clipped

    def _clip_triangle_to_render_frustum(
        self,
        triangle: np.ndarray,
        camera_obj,
        triangle_uv=None,
        clip_mats=None,
    ):
        if clip_mats is None:
            clip_mats = self._render_frustum_clip_mats(camera_obj)
        proj, modelview = clip_mats
        tri = np.asarray(triangle, dtype=float)
        payload = []
        for idx in range(3):
            ph = np.array([tri[idx, 0], tri[idx, 1], tri[idx, 2], 1.0], dtype=float)
            payload.append({
                "clip": proj @ (modelview @ ph),
                "world": tri[idx],
                "uv": None if triangle_uv is None else np.array(triangle_uv[idx], dtype=float),
            })
        polygon = payload
        for plane_normal, plane_offset in self._render_frustum_clip_planes():
            polygon = self._clip_homogeneous_polygon_against_plane(polygon, plane_normal, plane_offset)
            if len(polygon) < 3:
                return []
        clipped = []
        for i in range(1, len(polygon) - 1):
            items = [polygon[0], polygon[i], polygon[i + 1]]
            world_tri = np.array([item["world"] for item in items], dtype=float)
            if triangle_uv is None:
                clipped.append(world_tri)
            else:
                uv_tri = np.array([item["uv"] for item in items], dtype=float)
                clipped.append((world_tri, uv_tri))
        return clipped

    def _clip_records_to_render_frustum(
        self,
        records: List[Dict[str, Any]],
        camera_obj,
        clip_mats=None,
    ):
        """用 Blender 渲染投影矩阵裁剪视锥外的三角形部分。"""
        if clip_mats is None:
            clip_mats = self._render_frustum_clip_mats(camera_obj)
        clipped_records = []
        for record in records:
            clipped_triangles = []
            clipped_uvs = [] if record.get("uvs") is not None else None
            clipped_color_sources = []
            clipped_material_indices = []
            material_indices = record.get("material_indices")
            for tri_idx, tri in enumerate(record["triangles"]):
                tri_uv = record["uvs"][tri_idx] if record.get("uvs") is not None else None
                clipped = self._clip_triangle_to_render_frustum(
                    tri, camera_obj, tri_uv, clip_mats=clip_mats
                )
                for clipped_item in clipped:
                    if tri_uv is None:
                        clipped_tri = clipped_item
                        clipped_uv = None
                    else:
                        clipped_tri, clipped_uv = clipped_item
                    clipped_triangles.append(clipped_tri)
                    if clipped_uvs is not None and clipped_uv is not None:
                        clipped_uvs.append(clipped_uv)
                    clipped_color_sources.append(record["color_sources"][tri_idx])
                    if material_indices is not None:
                        clipped_material_indices.append(int(material_indices[tri_idx]))
            if not clipped_triangles:
                continue
            triangles_arr = np.array(clipped_triangles, dtype=float)
            area = float(np.sum(self._triangle_areas(triangles_arr)))
            if area <= 1e-12:
                continue
            new_record = dict(record)
            new_record["triangles"] = triangles_arr
            new_record["uvs"] = np.array(clipped_uvs, dtype=float) if clipped_uvs is not None else None
            new_record["color_sources"] = clipped_color_sources
            new_record["material_indices"] = (
                np.array(clipped_material_indices, dtype=np.int32) if material_indices is not None else None
            )
            new_record["area"] = area
            clipped_records.append(new_record)
        return clipped_records

    FRUSTUM_IN_VIEW_RATIO = 0.95

    def _vertex_in_render_view(self, co, camera_obj, margin: float = 0.002) -> bool:
        from bpy_extras.object_utils import world_to_camera_view

        u, v, z = world_to_camera_view(self.scene, camera_obj, mathutils.Vector(co))
        if z <= 0:
            return False
        return (-margin <= u <= 1.0 + margin) and (-margin <= v <= 1.0 + margin)

    def _triangle_fully_in_render_view(self, triangle: np.ndarray, camera_obj, margin: float = 0.002) -> bool:
        """三顶点都在渲染视口内才算该面片在视锥内（用于 cutted 统计）。"""
        tri = np.asarray(triangle, dtype=float)
        return all(self._vertex_in_render_view(p, camera_obj, margin=margin) for p in tri)

    def _records_frustum_in_view_ratio(self, records: List[Dict[str, Any]], camera_obj) -> float:
        total = 0
        in_view = 0
        for record in records:
            for tri in record["triangles"]:
                total += 1
                if self._triangle_fully_in_render_view(tri, camera_obj):
                    in_view += 1
        if total == 0:
            return 1.0
        return in_view / total

    def _records_frustum_cutted(
        self,
        records: List[Dict[str, Any]],
        camera_obj,
        in_view_ratio: float = FRUSTUM_IN_VIEW_RATIO,
    ) -> bool:
        """严格统计三顶点均在视口内的面片占比；低于阈值才为 cutted。"""
        return self._records_frustum_in_view_ratio(records, camera_obj) < in_view_ratio

    def _sample_visible_records(self, records: List[Dict[str, Any]], camera_obj, sample_count: int):
        areas = np.array([record["area"] for record in records], dtype=float)
        total_area = float(np.sum(areas))
        if total_area <= 1e-12:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
        counts = np.floor(areas / total_area * sample_count).astype(int)
        remainder = sample_count - int(np.sum(counts))
        if remainder > 0:
            order = np.argsort(-(areas / total_area * sample_count - counts))
            counts[order[:remainder]] += 1
        sampled_points = []
        sampled_colors = []
        for record, count in zip(records, counts):
            if count <= 0:
                continue
            pts, cols = self._sample_triangles(
                record["triangles"],
                record["color_sources"],
                count,
                uvs=record.get("uvs"),
            )
            if len(pts) > 0:
                sampled_points.append(pts)
                sampled_colors.append(cols)
        if not sampled_points:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
        return np.vstack(sampled_points), np.vstack(sampled_colors)

    def _visibility_target_objects(self, node, records: List[Dict[str, Any]]):
        targets = set()
        if node is not None:
            targets.add(node)
            instance_collection = getattr(node, "instance_collection", None)
            if getattr(node, "instance_type", None) == "COLLECTION" and instance_collection:
                for obj in instance_collection.all_objects:
                    targets.add(obj)
        for record in records:
            obj = record.get("object")
            if obj is not None:
                targets.add(obj)
        return targets

    def _ray_hit_matches_target(self, hit_obj, target_objects: set, records: List[Dict[str, Any]]):
        if hit_obj is None:
            return False
        if hit_obj in target_objects:
            return True
        hit_original = getattr(hit_obj, "original", None)
        if hit_original in target_objects:
            return True
        mesh_ids = {
            id(obj.data)
            for obj in target_objects
            if obj is not None and getattr(obj, "type", None) == "MESH" and obj.data is not None
        }
        mesh_ids.update(
            id(record["object"].data)
            for record in records
            if record.get("object") is not None and record["object"].data is not None
        )
        if hit_obj.data is not None and id(hit_obj.data) in mesh_ids:
            return True
        return False

    def _collect_visibility_sample_points(self, records, max_points: int = 128):
        points = []
        for record in records:
            tris = record["triangles"]
            if len(tris) == 0:
                continue
            points.extend(tris.reshape(-1, 3))
        if not points:
            return np.empty((0, 3), dtype=float)
        points = np.asarray(points, dtype=float)
        if len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = points[indices]
        return points

    def _point_visible_from_camera(
        self,
        point,
        camera_obj,
        target_objects: set,
        transparent_objects: set,
        records: List[Dict[str, Any]],
        depsgraph,
    ) -> bool:
        origin = np.array(camera_obj.matrix_world.translation, dtype=float)
        direction = np.asarray(point, dtype=float) - origin
        distance = float(np.linalg.norm(direction))
        if distance <= 1e-6:
            return True
        direction_unit = direction / distance
        direction_vec = mathutils.Vector(direction_unit)
        ray_origin = origin.copy()
        remaining = distance
        for _ in range(32):
            hit, hit_location, _, _, hit_obj, _ = self.scene.ray_cast(
                depsgraph,
                mathutils.Vector(ray_origin),
                direction_vec,
                distance=max(remaining - 1e-4, 0.0),
            )
            if not hit:
                return True
            if self._ray_hit_matches_target(hit_obj, target_objects, records):
                return True
            if hit_obj not in transparent_objects:
                return False
            hit_point = np.array(hit_location, dtype=float)
            advanced = float(np.linalg.norm(hit_point - ray_origin)) + 1e-4
            remaining -= advanced
            if remaining <= 1e-4:
                return True
            ray_origin = hit_point + direction_unit * 1e-4
        return False

    def _records_visible_from_camera(self, records, camera_obj, target_objects: set, transparent_objects: set):
        """至少有一个采样点无遮挡则视为可见（部分或全部可见）。"""
        points = self._collect_visibility_sample_points(records)
        if len(points) == 0:
            return False
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for point in points:
            if self._point_visible_from_camera(
                point, camera_obj, target_objects, transparent_objects, records, depsgraph
            ):
                return True
        return False

    def _records_fully_occluded(self, records, camera_obj, target_objects: set, transparent_objects: set):
        return not self._records_visible_from_camera(
            records, camera_obj, target_objects, transparent_objects
        )

    def _sample_bpy_node_surface(self, node, sample_count: int, use_ses: bool = False):
        mesh_records = self._collect_bpy_mesh_records(node, use_ses=use_ses)
        if not mesh_records:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)

        areas = np.array([record["area"] for record in mesh_records], dtype=float)
        total_area = float(np.sum(areas))
        if total_area <= 1e-12:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)

        sharp_lengths = np.array([record.get("sharp_edge_length", 0.0) for record in mesh_records], dtype=float)
        has_sharp_edges = use_ses and float(np.sum(sharp_lengths)) > 1e-12
        surface_sample_count = sample_count // 2 if has_sharp_edges else sample_count
        sharp_sample_count = sample_count - surface_sample_count if has_sharp_edges else 0

        counts = np.floor(areas / total_area * surface_sample_count).astype(int)
        remainder = surface_sample_count - int(np.sum(counts))
        if remainder > 0:
            order = np.argsort(-(areas / total_area * surface_sample_count - counts))
            counts[order[:remainder]] += 1

        sampled_points = []
        sampled_colors = []
        for record, count in zip(mesh_records, counts):
            if count <= 0:
                continue
            pts, cols = self._sample_triangles(
                record["triangles"],
                record["color_sources"],
                count,
                uvs=record.get("uvs"),
            )
            if len(pts) > 0:
                sampled_points.append(pts)
                sampled_colors.append(cols)

        if sharp_sample_count > 0:
            total_sharp_length = float(np.sum(sharp_lengths))
            sharp_counts = np.floor(sharp_lengths / total_sharp_length * sharp_sample_count).astype(int)
            sharp_remainder = sharp_sample_count - int(np.sum(sharp_counts))
            if sharp_remainder > 0:
                order = np.argsort(-(sharp_lengths / total_sharp_length * sharp_sample_count - sharp_counts))
                sharp_counts[order[:sharp_remainder]] += 1

            for record, count in zip(mesh_records, sharp_counts):
                if count <= 0:
                    continue
                pts, cols = self._sample_sharp_edges(record.get("sharp_edges", []), count)
                if len(pts) > 0:
                    sampled_points.append(pts)
                    sampled_colors.append(cols)

        if not sampled_points:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
        return np.vstack(sampled_points), np.vstack(sampled_colors)

    def _collect_bpy_mesh_records(self, node, use_ses: bool = False):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        records = []

        def collect_mesh(obj, matrix_world):
            if obj is None or obj.type != "MESH":
                return
            evaluated_obj = obj.evaluated_get(depsgraph)
            mesh = evaluated_obj.to_mesh()
            try:
                vertex_count = len(mesh.vertices)
                if vertex_count == 0:
                    return
                local_vertices = np.empty(vertex_count * 3, dtype=float)
                mesh.vertices.foreach_get("co", local_vertices)
                local_vertices = local_vertices.reshape((-1, 3))
                matrix = np.array(matrix_world, dtype=float)
                vertices = local_vertices @ matrix[:3, :3].T + matrix[:3, 3]

                mesh.calc_loop_triangles()
                loop_triangles = mesh.loop_triangles
                triangle_count = len(loop_triangles)
                if triangle_count == 0:
                    return

                vertex_indices = np.empty(triangle_count * 3, dtype=np.int32)
                loop_indices = np.empty(triangle_count * 3, dtype=np.int32)
                material_indices = np.empty(triangle_count, dtype=np.int32)
                loop_triangles.foreach_get("vertices", vertex_indices)
                loop_triangles.foreach_get("loops", loop_indices)
                loop_triangles.foreach_get("material_index", material_indices)
                vertex_indices = vertex_indices.reshape((-1, 3))
                loop_indices = loop_indices.reshape((-1, 3))
                triangles_arr = vertices[vertex_indices]

                uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None
                loop_uvs = None
                triangle_uvs = None
                if uv_layer is not None:
                    uv_values = np.empty(len(mesh.loops) * 2, dtype=float)
                    uv_layer.foreach_get("uv", uv_values)
                    loop_uvs = uv_values.reshape((-1, 2))
                    triangle_uvs = loop_uvs[loop_indices]

                material_sources = {
                    int(material_index): self._object_material_color_source(obj, int(material_index))
                    for material_index in np.unique(material_indices)
                }
                color_sources = [material_sources[int(material_index)] for material_index in material_indices]
                materials = [
                    slot.material for slot in getattr(obj, "material_slots", [])
                    if slot.material is not None
                ]

                tri_areas = self._triangle_areas(triangles_arr)
                area = float(np.sum(tri_areas))
                if area <= 1e-12:
                    return
                sharp_edges = (
                    self._extract_sharp_edges_from_triangles(
                        vertex_indices,
                        vertices,
                        triangle_uvs,
                        color_sources,
                        angle_threshold_degrees=30.0,
                    )
                    if use_ses else []
                )
                records.append({
                    "object": obj,
                    "triangles": triangles_arr,
                    "uvs": triangle_uvs,
                    "color_sources": color_sources,
                    "material_indices": material_indices,
                    "materials": materials,
                    "sharp_edges": sharp_edges,
                    "sharp_edge_length": float(sum(edge["length"] for edge in sharp_edges)),
                    "area": area,
                })
            finally:
                evaluated_obj.to_mesh_clear()

        if getattr(node, "type", None) == "MESH":
            collect_mesh(node, node.matrix_world.copy())

        instance_collection = getattr(node, "instance_collection", None)
        if getattr(node, "instance_type", None) == "COLLECTION" and instance_collection:
            for obj in instance_collection.all_objects:
                collect_mesh(obj, node.matrix_world @ obj.matrix_world)

        for obj in getattr(node, "children_recursive", []):
            collect_mesh(obj, obj.matrix_world.copy())

        return records

    def _extract_sharp_edges_from_triangles(
        self,
        vertex_indices: np.ndarray,
        vertices: np.ndarray,
        triangle_uvs,
        color_sources: List[Dict[str, Any]],
        angle_threshold_degrees: float,
    ):
        if len(vertex_indices) == 0:
            return []

        triangle_normals = self._triangle_normals(vertices[vertex_indices])
        edge_vertices = np.stack(
            [
                vertex_indices[:, [0, 1]],
                vertex_indices[:, [1, 2]],
                vertex_indices[:, [2, 0]],
            ],
            axis=1,
        ).reshape((-1, 2))
        sorted_edge_vertices = np.sort(edge_vertices, axis=1)
        order = np.lexsort((sorted_edge_vertices[:, 1], sorted_edge_vertices[:, 0]))
        sorted_keys = sorted_edge_vertices[order]
        group_starts = np.r_[0, np.nonzero(np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1))[0] + 1]
        group_ends = np.r_[group_starts[1:], len(order)]

        edge_triangle_indices = np.repeat(np.arange(len(vertex_indices)), 3)
        edge_normals = triangle_normals[edge_triangle_indices]
        edge_uvs = None
        if triangle_uvs is not None:
            edge_uvs = np.stack(
                [
                    triangle_uvs[:, [0, 1]],
                    triangle_uvs[:, [1, 2]],
                    triangle_uvs[:, [2, 0]],
                ],
                axis=1,
            ).reshape((-1, 2, 2))

        cos_threshold = float(np.cos(np.radians(angle_threshold_degrees)))
        group_lengths = group_ends - group_starts

        boundary_positions = order[group_starts[group_lengths == 1]]

        two_group_starts = group_starts[group_lengths == 2]
        two_group_positions = np.empty((0,), dtype=order.dtype)
        if len(two_group_starts) > 0:
            pos0 = order[two_group_starts]
            pos1 = order[two_group_starts + 1]
            n0 = edge_normals[pos0]
            n1 = edge_normals[pos1]
            valid = (np.linalg.norm(n0, axis=1) > 1e-12) & (np.linalg.norm(n1, axis=1) > 1e-12)
            cos_angles = np.einsum("ij,ij->i", n0, n1)
            is_sharp = valid & (cos_angles <= cos_threshold)
            two_group_positions = pos0[is_sharp]

        complex_positions = []
        complex_group_starts = group_starts[group_lengths > 2]
        complex_group_ends = group_ends[group_lengths > 2]
        for start, end in zip(complex_group_starts, complex_group_ends):
            positions = order[start:end]
            normals = edge_normals[positions]
            valid = np.linalg.norm(normals, axis=1) > 1e-12
            normals = normals[valid]
            if len(normals) < 2:
                continue
            cos_angles = np.clip(normals @ normals.T, -1.0, 1.0)
            if float(np.min(cos_angles)) <= cos_threshold:
                complex_positions.append(int(positions[0]))

        if complex_positions:
            sharp_positions = np.concatenate([
                boundary_positions,
                two_group_positions,
                np.array(complex_positions, dtype=order.dtype),
            ])
        else:
            sharp_positions = np.concatenate([boundary_positions, two_group_positions])

        sharp_edges = []
        for edge_pos in sharp_positions:
            edge_pos = int(edge_pos)
            v0, v1 = edge_vertices[edge_pos]
            p0 = vertices[v0]
            p1 = vertices[v1]
            length = float(np.linalg.norm(p1 - p0))
            if length <= 1e-12:
                continue
            uv0 = uv1 = None
            if edge_uvs is not None:
                uv0 = tuple(edge_uvs[edge_pos, 0])
                uv1 = tuple(edge_uvs[edge_pos, 1])
            tri_idx = int(edge_triangle_indices[edge_pos])
            sharp_edges.append({
                "p0": p0,
                "p1": p1,
                "uv0": uv0,
                "uv1": uv1,
                "color_source": color_sources[tri_idx],
                "length": length,
            })
        return sharp_edges

    def _extract_sharp_edges(self, edge_map: Dict[Any, List[Dict[str, Any]]], angle_threshold_degrees: float):
        sharp_edges = []
        threshold = np.radians(angle_threshold_degrees)
        for entries in edge_map.values():
            if not entries:
                continue
            is_sharp = len(entries) == 1
            if len(entries) >= 2:
                max_angle = 0.0
                for i in range(len(entries)):
                    for j in range(i + 1, len(entries)):
                        n0 = entries[i]["normal"]
                        n1 = entries[j]["normal"]
                        denom = np.linalg.norm(n0) * np.linalg.norm(n1)
                        if denom <= 1e-12:
                            continue
                        cos_angle = np.clip(float(np.dot(n0, n1) / denom), -1.0, 1.0)
                        max_angle = max(max_angle, float(np.arccos(cos_angle)))
                is_sharp = max_angle >= threshold
            if not is_sharp:
                continue
            entry = entries[0]
            p0 = np.array(entry["p0"], dtype=float)
            p1 = np.array(entry["p1"], dtype=float)
            length = float(np.linalg.norm(p1 - p0))
            if length <= 1e-12:
                continue
            sharp_edges.append({
                "p0": p0,
                "p1": p1,
                "uv0": entry.get("uv0"),
                "uv1": entry.get("uv1"),
                "color_source": entry.get("color_source", {"base_color": np.array([160, 160, 160], dtype=np.uint8)}),
                "length": length,
            })
        return sharp_edges

    @staticmethod
    def _polygon_normal(poly_vertices: np.ndarray):
        if len(poly_vertices) < 3:
            return np.array([0.0, 0.0, 0.0], dtype=float)
        origin = poly_vertices[0]
        for i in range(1, len(poly_vertices) - 1):
            normal = np.cross(poly_vertices[i] - origin, poly_vertices[i + 1] - origin)
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                return normal / norm
        return np.array([0.0, 0.0, 0.0], dtype=float)

    @staticmethod
    def _triangle_normals(triangles: np.ndarray):
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        return np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 1e-12)

    def _sample_sharp_edges(self, sharp_edges: List[Dict[str, Any]], sample_count: int):
        if not sharp_edges or sample_count <= 0:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
        lengths = np.array([edge["length"] for edge in sharp_edges], dtype=float)
        total_length = float(np.sum(lengths))
        if total_length <= 1e-12:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)

        edge_indices = np.random.choice(len(sharp_edges), size=sample_count, p=lengths / total_length)
        ts = np.random.random(sample_count)
        points = np.empty((sample_count, 3), dtype=float)
        colors = np.empty((sample_count, 3), dtype=np.uint8)
        for sample_idx, edge_idx in enumerate(edge_indices):
            edge = sharp_edges[edge_idx]
            t = ts[sample_idx]
            points[sample_idx] = (1.0 - t) * edge["p0"] + t * edge["p1"]
            colors[sample_idx] = self._edge_color(edge, t)
        return points, colors

    def _edge_color(self, edge: Dict[str, Any], t: float):
        source = edge.get("color_source", {})
        base_color = source.get("base_color", np.array([160, 160, 160], dtype=np.uint8))
        image = source.get("image")
        uv0 = edge.get("uv0")
        uv1 = edge.get("uv1")
        if image is None or uv0 is None or uv1 is None:
            return base_color
        uv = (1.0 - t) * np.array(uv0, dtype=float) + t * np.array(uv1, dtype=float)
        mapping = source.get("mapping")
        if mapping is not None:
            uv = self._apply_mapping_to_uv(uv, mapping)
        return self._sample_image_color(image, uv, base_color, source.get("base_factor"))

    def _object_material_color_source(self, obj, material_index: int):
        default = np.array([160, 160, 160], dtype=np.uint8)
        if obj is None or not getattr(obj, "material_slots", None):
            return {"base_color": default}
        if material_index >= len(obj.material_slots):
            material_index = 0
        mat = obj.material_slots[material_index].material
        if mat is None:
            return {"base_color": default}

        cache_key = mat.as_pointer()
        if cache_key in self._point_cloud_material_cache:
            return self._point_cloud_material_cache[cache_key]

        color = getattr(mat, "diffuse_color", None)
        image_source = None
        if mat.use_nodes and mat.node_tree:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf and "Base Color" in bsdf.inputs:
                color = bsdf.inputs["Base Color"].default_value
                image_source = self._find_linked_image_source(bsdf.inputs["Base Color"])
            if image_source is None:
                image_source = self._find_first_material_image_source(mat)
        if color is None:
            color = default

        rgb = np.clip(np.array(color[:3], dtype=float), 0.0, 1.0) * 255.0
        source = {
            "base_color": rgb.astype(np.uint8),
            "base_factor": np.clip(np.array(color[:3], dtype=float), 0.0, 1.0),
        }
        if image_source is not None:
            image_data = self._image_to_array(image_source["image"])
            if image_data is not None:
                source["image"] = image_data
                source["mapping"] = image_source.get("mapping")
        self._point_cloud_material_cache[cache_key] = source
        return source

    def _find_linked_image_source(self, socket):
        visited = set()

        def visit(node, mapping=None):
            if node is None or node.as_pointer() in visited:
                return None
            visited.add(node.as_pointer())
            if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
                node_mapping = mapping
                if "Vector" in node.inputs:
                    for link in node.inputs["Vector"].links:
                        if link.from_node.type == "MAPPING":
                            node_mapping = self._mapping_node_params(link.from_node)
                            break
                return {"image": node.image, "mapping": node_mapping}
            if node.type == "MAPPING":
                mapping = self._mapping_node_params(node)
            for input_socket in getattr(node, "inputs", []):
                for link in input_socket.links:
                    image_source = visit(link.from_node, mapping)
                    if image_source is not None:
                        return image_source
            return None

        for link in socket.links:
            image_source = visit(link.from_node)
            if image_source is not None:
                return image_source
        return None

    @staticmethod
    def _find_first_material_image_source(mat):
        if not mat.use_nodes or mat.node_tree is None:
            return None
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
                mapping = None
                if "Vector" in node.inputs:
                    for link in node.inputs["Vector"].links:
                        if link.from_node.type == "MAPPING":
                            mapping = BpySceneCtx._mapping_node_params(link.from_node)
                            break
                return {"image": node.image, "mapping": mapping}
        return None

    @staticmethod
    def _mapping_node_params(node):
        return {
            "location": np.array(node.inputs["Location"].default_value[:3], dtype=float),
            "rotation": np.array(node.inputs["Rotation"].default_value[:3], dtype=float),
            "scale": np.array(node.inputs["Scale"].default_value[:3], dtype=float),
        }

    def _image_to_array(self, image):
        if image is None:
            return None
        cache_key = image.as_pointer()
        if cache_key in self._point_cloud_image_cache:
            return self._point_cloud_image_cache[cache_key]
        width, height = image.size
        if width <= 0 or height <= 0:
            return None
        try:
            pixels = np.array(image.pixels[:], dtype=float).reshape((height, width, 4))
        except Exception:
            return None
        rgb_float = np.clip(pixels[:, :, :3], 0.0, 1.0)
        if getattr(image.colorspace_settings, "name", "") == "sRGB":
            rgb_float = BpySceneCtx._linear_to_srgb(rgb_float)
        rgb = rgb_float * 255.0
        image_data = rgb.astype(np.uint8)
        self._point_cloud_image_cache[cache_key] = image_data
        return image_data

    @staticmethod
    def _linear_to_srgb(values: np.ndarray):
        values = np.clip(values, 0.0, 1.0)
        return np.where(
            values <= 0.0031308,
            values * 12.92,
            1.055 * np.power(values, 1.0 / 2.4) - 0.055,
        )

    def _sample_triangles(self, triangles: np.ndarray, color_sources: List[Dict[str, Any]], sample_count: int, uvs=None):
        areas = self._triangle_areas(triangles)
        total_area = float(np.sum(areas))
        if total_area <= 1e-12:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)

        tri_indices = np.random.choice(len(triangles), size=sample_count, p=areas / total_area)
        chosen = triangles[tri_indices]
        r1 = np.sqrt(np.random.random(sample_count))[:, None]
        r2 = np.random.random(sample_count)[:, None]
        points = (1.0 - r1) * chosen[:, 0] + r1 * (1.0 - r2) * chosen[:, 1] + r1 * r2 * chosen[:, 2]
        colors = np.empty((sample_count, 3), dtype=np.uint8)
        for sample_idx, tri_idx in enumerate(tri_indices):
            source = color_sources[tri_idx]
            base_color = source.get("base_color", np.array([160, 160, 160], dtype=np.uint8))
            image = source.get("image")
            mapping = source.get("mapping")
            tri_uv = uvs[tri_idx] if uvs is not None else None
            if image is not None and tri_uv is not None:
                uv = (
                    (1.0 - r1[sample_idx, 0]) * np.array(tri_uv[0], dtype=float)
                    + r1[sample_idx, 0] * (1.0 - r2[sample_idx, 0]) * np.array(tri_uv[1], dtype=float)
                    + r1[sample_idx, 0] * r2[sample_idx, 0] * np.array(tri_uv[2], dtype=float)
                )
                if mapping is not None:
                    uv = self._apply_mapping_to_uv(uv, mapping)
                base_factor = source.get("base_factor")
                colors[sample_idx] = self._sample_image_color(image, uv, base_color, base_factor)
            else:
                colors[sample_idx] = base_color
        return points, colors

    @staticmethod
    def _apply_mapping_to_uv(uv: np.ndarray, mapping: Dict[str, np.ndarray]):
        vec = np.array([uv[0], uv[1], 0.0], dtype=float)
        scale = mapping.get("scale", np.ones(3))
        rotation = mapping.get("rotation", np.zeros(3))
        location = mapping.get("location", np.zeros(3))

        vec = vec * scale
        rx, ry, rz = rotation
        if abs(rx) > 1e-12:
            c, s = np.cos(rx), np.sin(rx)
            vec = np.array([vec[0], c * vec[1] - s * vec[2], s * vec[1] + c * vec[2]])
        if abs(ry) > 1e-12:
            c, s = np.cos(ry), np.sin(ry)
            vec = np.array([c * vec[0] + s * vec[2], vec[1], -s * vec[0] + c * vec[2]])
        if abs(rz) > 1e-12:
            c, s = np.cos(rz), np.sin(rz)
            vec = np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], vec[2]])
        vec = vec + location
        return vec[:2]

    @staticmethod
    def _sample_image_color(image: np.ndarray, uv: np.ndarray, fallback: np.ndarray, base_factor=None):
        if image is None or image.size == 0:
            return fallback
        height, width = image.shape[:2]
        u = float(uv[0]) % 1.0
        v = float(uv[1]) % 1.0
        x = min(width - 1, max(0, int(u * width)))
        # Blender image.pixels is stored from the bottom row, matching UV v=0.
        y = min(height - 1, max(0, int(v * height)))
        color = image[y, x].astype(float)
        if base_factor is not None:
            color = color * np.clip(base_factor, 0.0, 1.0)
        return np.clip(color, 0, 255).astype(np.uint8)

    @staticmethod
    def _triangle_areas(triangles: np.ndarray):
        return np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        ) * 0.5

    @staticmethod
    def _safe_filename(name: str):
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
        return safe or "object"

    @staticmethod
    def _write_ply(path: str, points: np.ndarray, colors: np.ndarray):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        colors = np.clip(colors, 0, 255).astype(np.uint8)
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for point, color in zip(points, colors):
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )

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

    def _semantic_outputs(self, output_path: str):
        base = os.path.splitext(os.path.basename(output_path))[0]
        out_dir = os.path.dirname(output_path)
        return (
            os.path.join(out_dir, f"{base}_semantic.png"),
            os.path.join(out_dir, f"{base}_semantic.json"),
        )

    def _semantic_color(self, color_key: str):
        return util.semantic_entity_color(color_key)

    def _semantic_material(self, color_key: str):
        mat_name = f"Semantic_{color_key.replace(':', '_').replace(' ', '_')}"
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()
            output = nodes.new(type="ShaderNodeOutputMaterial")
            emission = nodes.new(type="ShaderNodeEmission")
            mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        else:
            emission = mat.node_tree.nodes.get("Emission")
            if emission is None:
                for node in mat.node_tree.nodes:
                    if node.type == "EMISSION":
                        emission = node
                        break
        color = self._semantic_color(color_key)
        if emission is not None:
            emission.inputs["Color"].default_value = [c / 255.0 for c in color] + [1.0]
            emission.inputs["Strength"].default_value = 1.0
        return mat

    def _iter_semantic_scene_nodes(self):
        """Yield scene-level render nodes for semantic pass (never asset master meshes)."""

        def emit(category, color_key, node, object_meta):
            if node is not None:
                yield color_key, node, object_meta

        for category in ("floor", "ceiling"):
            info = self.mesh_nodes.get(category)
            if info:
                object_id = category
                color_key = f"entity:{category}:{object_id}"
                yield from emit(
                    category,
                    color_key,
                    info.get("node"),
                    {
                        "category": category,
                        "id": object_id,
                        "color_key": color_key,
                        "color": list(util.semantic_entity_color(color_key)),
                    },
                )
        for category in ("walls", "doors", "windows"):
            for object_id, info in self.mesh_nodes.get(category, {}).items():
                color_key = f"entity:{category}:{object_id}"
                yield from emit(
                    category,
                    color_key,
                    info.get("node"),
                    {
                        "category": category,
                        "id": object_id,
                        "color_key": color_key,
                        "color": list(util.semantic_entity_color(color_key)),
                    },
                )
        for object_id, info in self.mesh_nodes.get("boxes", {}).items():
            data = info.get("box_data", {})
            label = data.get("label", data.get("class", "object"))
            color_key = f"entity:boxes:{object_id}"
            object_meta = {
                "category": "boxes",
                "id": object_id,
                "label": label,
                "color_key": color_key,
                "color": list(util.semantic_entity_color(color_key)),
            }
            yield from emit("boxes", color_key, info.get("node"), object_meta)

    def _create_semantic_proxies(self, node, color_key: str):
        """Create render-only mesh copies with semantic materials; originals stay untouched."""
        semantic_mat = self._semantic_material(color_key)
        proxies = []

        def add_proxy(source_obj, matrix_world):
            if source_obj is None or getattr(source_obj, "type", None) != "MESH":
                return
            mesh_data = getattr(source_obj, "data", None)
            if mesh_data is None:
                return
            dup = source_obj.copy()
            dup.data = mesh_data.copy()
            slot_count = max(len(getattr(source_obj, "material_slots", [])), len(dup.data.materials), 1)
            dup.data.materials.clear()
            for _ in range(slot_count):
                dup.data.materials.append(semantic_mat)
            dup.hide_render = False
            dup.hide_viewport = False
            self.scene_collection.objects.link(dup)
            dup.matrix_world = matrix_world
            proxies.append(dup)

        if getattr(node, "type", None) == "MESH":
            add_proxy(node, node.matrix_world.copy())

        instance_collection = getattr(node, "instance_collection", None)
        if getattr(node, "instance_type", None) == "COLLECTION" and instance_collection:
            for obj in instance_collection.all_objects:
                add_proxy(obj, node.matrix_world @ obj.matrix_world)

        for obj in getattr(node, "children_recursive", []):
            if getattr(obj, "type", None) == "MESH":
                add_proxy(obj, obj.matrix_world.copy())

        return proxies

    def _semantic_metadata(self, objects: List[Dict[str, Any]]):
        deduped_objects = []
        seen = set()
        for obj in objects:
            obj_key = (obj.get("category"), obj.get("id"))
            if obj_key not in seen:
                deduped_objects.append(obj)
                seen.add(obj_key)
        return {
            "semantic": True,
            "format": "per_entity_color_png",
            "background": list(util.SEMANTIC_BACKGROUND),
            "objects": deduped_objects,
        }

    def _begin_semantic_render_env(self):
        backup = {"lights": [], "world_bg": None}
        for obj in bpy.data.objects:
            if getattr(obj, "type", None) == "LIGHT":
                backup["lights"].append((obj, obj.hide_render))
                obj.hide_render = True
        world = self.scene.world
        if world and world.use_nodes:
            bg = world.node_tree.nodes.get("Background")
            if bg:
                backup["world_bg"] = (
                    tuple(bg.inputs["Color"].default_value[:]),
                    float(bg.inputs["Strength"].default_value),
                )
                bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
                bg.inputs["Strength"].default_value = 0.0
        return backup

    def _restore_semantic_render_env(self, backup):
        if not backup:
            return
        for obj, hide in backup.get("lights", []):
            try:
                obj.hide_render = hide
            except ReferenceError:
                pass
        world_bg = backup.get("world_bg")
        if world_bg:
            world = self.scene.world
            bg = world.node_tree.nodes.get("Background") if world and world.use_nodes else None
            if bg:
                bg.inputs["Color"].default_value = world_bg[0]
                bg.inputs["Strength"].default_value = world_bg[1]

    def render_semantic_png(self, output_path: str):
        semantic_path, metadata_path = self._semantic_outputs(output_path)
        render = self.scene.render
        render_backup = (render.filepath, render.image_settings.file_format, render.film_transparent)
        objects = []
        hidden_nodes = []
        temp_proxies = []
        env_backup = None
        view_backup = None

        try:
            env_backup = self._begin_semantic_render_env()
            vs = self.scene.view_settings
            view_backup = (vs.view_transform, vs.look, float(vs.exposure), float(vs.gamma))
            vs.view_transform = "Raw"
            vs.look = "None"
            vs.exposure = 0.0
            vs.gamma = 1.0

            for color_key, node, object_meta in self._iter_semantic_scene_nodes():
                objects.append(object_meta)
                temp_proxies.extend(self._create_semantic_proxies(node, color_key))
                node.hide_render = True
                hidden_nodes.append(node)

            render.filepath = semantic_path
            render.image_settings.file_format = 'PNG'
            render.film_transparent = False
            bpy.ops.render.render(write_still=True)

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._semantic_metadata(objects), f, indent=2, ensure_ascii=False)
            print(f"✅ 语义图已导出: {semantic_path}")
        finally:
            for proxy in temp_proxies:
                mesh = proxy.data
                bpy.data.objects.remove(proxy, do_unlink=True)
                if mesh and mesh.users == 0:
                    bpy.data.meshes.remove(mesh, do_unlink=True)
            for node in hidden_nodes:
                node.hide_render = False
            if env_backup is not None:
                self._restore_semantic_render_env(env_backup)
            if view_backup is not None:
                vs = self.scene.view_settings
                vs.view_transform, vs.look, vs.exposure, vs.gamma = view_backup
            render.filepath, render.image_settings.file_format, render.film_transparent = render_backup

    def topdown_view(self, output_path: str, width: int = 1024, height: int = 1024,
                     geometry_mode: str = "gltf", show_wall: bool = True, 
                     show_window: bool = True, show_door: bool = True, show_ceiling: bool = True,
                     up_vector: list = None,
                     auto_fov: bool = True, manual_fov: float = None,
                     auto_transparent: bool = True, transparent_alpha: float = 0.0,
                     render_depth: bool = False, use_HDRI: bool = True,
                     hdri_transparent_background: bool = True,
                     visible_shadow: bool = True,
                     lighting_type: Literal["area", "array", "none"] = "array",
                     align_height: bool = True,
                     rebuild: bool = False,
                     export_glb: bool = False,
                     glb_path: Optional[str] = None,
                     export_point_cloud: bool = False,
                     visible_geometry: bool = False,
                     render_semantic: bool = False,
                     view_transform: bool = False):
        """俯视图渲染：output_path 为目录时在 {output}/topdown/topdown.png 输出单帧及附属产物。"""
        output_path = self._resolve_topdown_output_path(output_path)
        view_dir = os.path.dirname(output_path) or "."
        total_start = time.perf_counter()

        meta_w = self.context["meta"]
        world_cam_w = [
            meta_w["center"][0],
            meta_w["center"][1],
            meta_w["z_max"] + max(meta_w["span"]) * 1.5,
        ]
        world_look_w = [meta_w["center"][0], meta_w["center"][1], 0.0]
        world_up_raw = up_vector if up_vector else [0.0, 1.0, 0.0]

        with util_data.ViewSslSession(self, view_transform) as vss:
            vss.setup(world_cam_w, world_look_w, world_up_raw)
            vss.write_ssl(view_dir)

            construct_time = 0
            if view_transform or rebuild or self.mesh_nodes["floor"] is None:
                construct_start = time.perf_counter()
                self.construct_scene(
                    geometry_mode=geometry_mode,
                    show_wall=show_wall,
                    show_window=show_window,
                    show_door=show_door,
                    show_ceiling=show_ceiling,
                    align_height=align_height,
                    rebuild=view_transform or rebuild,
                )
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
                self.setup_lighting(lighting_type=lighting_type, ambient_light_color=[1.0, 1.0, 1.0], ambient_strength=3.0)

            self.scene.render.film_transparent = hdri_transparent_background

            if use_HDRI:
                hdri_path = self.config.get("hdri_path")
                if hdri_path and util_bpy.apply_hdri_to_world(self.scene, hdri_path, strength=1.0):
                    print(f"🌇 使用 HDRI 环境光: {hdri_path}")
                else:
                    print(f"⚠️ HDRI 文件不可用或未配置: {hdri_path}")

            center = self.context["meta"]["center"]
            span = self.context["meta"]["span"]
            z_max = self.context["meta"]["z_max"]
            bounds = self.context["meta"]["bounds"]

            if view_transform:
                render_cam, render_look = vss.render_camera_pose(
                    world_cam_w, world_look_w, reference_frame=True
                )
                camera_position = np.array(render_cam, dtype=float)
                look_at_target = np.array(render_look, dtype=float)
                up_vector = np.array(vss.render_world_up(world_up_raw), dtype=float)
            else:
                camera_height = z_max + max(span) * 1.5
                camera_position = np.array([center[0], center[1], camera_height], dtype=float)
                look_at_target = np.array([center[0], center[1], 0.0], dtype=float)
                up_vector = np.array(world_up_raw, dtype=float)

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

            self.scene.render.resolution_x = width
            self.scene.render.resolution_y = height
            self.scene.render.filepath = output_path
            self.scene.render.image_settings.color_mode = 'RGBA'

            print(f"🎬 渲染中 ({width}x{height})...")
            depth_scale = None
            transparent_objects = {
                self.mesh_nodes["walls"].get(wall_id, {}).get("node")
                for wall_id in wall_transparency_records
            }
            transparent_objects.update(record.get("object") for record in object_transparency_records)
            transparent_objects = {obj for obj in transparent_objects if obj is not None}
            try:
                render_start = time.perf_counter()
                if render_depth:
                    depth_path = os.path.join(
                        os.path.dirname(output_path),
                        f"{os.path.splitext(os.path.basename(output_path))[0]}_depth.png",
                    )
                    depth_scale = util_bpy.render_color_and_depth_png(
                        self.scene, output_path, depth_path, skip_objects=transparent_objects
                    )
                else:
                    bpy.ops.render.render(write_still=True)
                render_time = time.perf_counter() - render_start
                if visible_geometry and (export_glb or export_point_cloud):
                    self.export_visible_geometry(
                        os.path.dirname(output_path),
                        camera_obj,
                        export_glb=export_glb,
                        export_point_cloud=export_point_cloud,
                        transparent_objects=transparent_objects,
                    )
                if export_point_cloud:
                    self.export_planar_faces_and_lines(
                        output_path,
                        camera_obj,
                        width,
                        height,
                        align_height=align_height,
                        show_wall=show_wall,
                        show_door=show_door,
                        show_window=show_window,
                        show_ceiling=show_ceiling,
                        transparent_objects=transparent_objects,
                    )
                if render_semantic:
                    self.render_semantic_png(output_path)
            finally:
                for wall_id, slots in wall_transparency_records.items():
                    wall_obj = self.mesh_nodes["walls"].get(wall_id, {}).get("node")
                    util_bpy.restore_wall_transparency(wall_obj, slots)
                for record in object_transparency_records:
                    util_bpy.restore_object_transparency(record)
                if camera_obj and self.scene.camera == camera_obj:
                    self.scene.camera = None
                self._remove_camera_blocks(camera_obj, camera_data)
                util_bpy.cleanup_bpy_render_memory(self.scene)

            if export_glb and not visible_geometry and glb_path is not None:
                self.export_glb(glb_path or os.path.splitext(output_path)[0] + ".glb")

            save_start = time.perf_counter()
            para_path = self._camera_para_path(output_path)
            camera_para = vss.build_camera_para(
                world_cam_w,
                world_look_w,
                vss.world_up,
                fov_y,
                1.0,
                reference_frame=True,
                depth_scale=depth_scale,
                include_normal_fields=depth_scale is not None,
            )
            with open(para_path, 'w') as f:
                json.dump(camera_para, f, indent=4)
            save_time = time.perf_counter() - save_start

            total_time = time.perf_counter() - total_start
            print(f"✅ 渲染完成! 保存至: {output_path}")
            print(f"   构建: {construct_time:.2f}s, 设置: {setup_time:.2f}s, 渲染: {render_time:.2f}s, 保存: {save_time:.2f}s, 总耗时: {total_time:.2f}s")

    @staticmethod
    def _is_nested_point_list(values) -> bool:
        """是否为「列表套列表」：[[x,y,z], ...]；单个 [x,y,z] 返回 False。"""
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            return False
        first = values[0]
        if isinstance(first, np.ndarray):
            return first.ndim > 0 and len(first) >= 2
        return isinstance(first, (list, tuple)) and len(first) >= 2

    def _is_camera_sequence(self, values) -> bool:
        return self._is_nested_point_list(values)

    def _is_view_sequence(self, camera_position, look_at_target=None, up_vector=None) -> bool:
        """camera / look_at / up 任一侧为多层列表即走序列渲染。"""
        if self._is_nested_point_list(camera_position):
            return True
        if look_at_target is not None and self._is_nested_point_list(look_at_target):
            return True
        if up_vector is not None and self._is_nested_point_list(up_vector):
            return True
        return False

    @staticmethod
    def _is_image_output_path(path: str) -> bool:
        return path.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".webp"))

    def _resolve_view_output_path(self, output_path: str) -> str:
        """单相机：output_path 为目录时自动分配 {dir_stamp}/{image_stamp}.png。"""
        if self._is_image_output_path(output_path):
            return output_path
        return util.resolve_view_image_path(output_path)

    def _resolve_topdown_output_path(self, output_path: str) -> str:
        """俯视图：output_path 为目录时在 {output}/topdown/topdown.png 输出单帧。"""
        if self._is_image_output_path(output_path):
            return output_path
        return util.resolve_topdown_image_path(output_path)

    @staticmethod
    def _camera_para_path(output_path: str) -> str:
        """与主图同名前缀：{image_basename}_camera_para.json"""
        view_dir = os.path.dirname(output_path) or "."
        base = os.path.splitext(os.path.basename(output_path))[0]
        return os.path.join(view_dir, f"{base}_camera_para.json")

    def _expand_camera_sequence_args(
        self,
        camera_position,
        look_at_target=None,
        up_vector=None,
    ) -> Tuple[list, list, list]:
        """展开序列参数：多层列表侧定帧数；单层 [x,y,z] 向对侧广播（多对一 / 一对多 / 一对一）。"""
        center = self.context["meta"]["center"]
        z_max = self.context["meta"]["z_max"]
        default_look_at = [center[0], center[1], z_max / 2]
        default_up = [0.0, 0.0, 1.0]

        cam_is_seq = self._is_nested_point_list(camera_position)
        look_is_seq = look_at_target is not None and self._is_nested_point_list(look_at_target)
        up_is_seq = up_vector is not None and self._is_nested_point_list(up_vector)

        seq_lengths = []
        if cam_is_seq:
            seq_lengths.append(len(camera_position))
        if look_is_seq:
            seq_lengths.append(len(look_at_target))
        if up_is_seq:
            seq_lengths.append(len(up_vector))

        if not seq_lengths:
            raise ValueError("序列模式需要 camera_position / look_at_target / up_vector 至少一侧为多层列表")

        if len(set(seq_lengths)) != 1:
            raise ValueError(
                f"多层列表的 camera_position / look_at_target / up_vector 长度须一致: {seq_lengths}"
            )
        n = seq_lengths[0]

        if cam_is_seq:
            cam_positions = list(camera_position)
        else:
            cam_positions = [list(camera_position)] * n

        if look_at_target is None:
            look_ats = [default_look_at] * n
        elif look_is_seq:
            look_ats = [list(p) for p in look_at_target]
        else:
            look_ats = [list(look_at_target)] * n

        if up_vector is None:
            ups = [default_up] * n
        elif up_is_seq:
            ups = [list(p) for p in up_vector]
        else:
            ups = [list(up_vector)] * n

        return cam_positions, look_ats, ups

    def _build_view_camera_matrix_and_fov(
        self,
        camera_position,
        look_at_target=None,
        up_vector=None,
        auto_fov: bool = True,
        manual_fov: float = None,
    ):
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
        return camera_position, look_at_target, camera_matrix, float(fov_y)

    def _create_render_camera(self, camera_matrix, fov_y: float, width: int, height: int):
        camera_data = bpy.data.cameras.new(name="Camera")
        camera_obj = bpy.data.objects.new("Camera", camera_data)
        self.scene_collection.objects.link(camera_obj)
        prev_camera = self.scene.camera
        self.scene.camera = camera_obj
        camera_obj.matrix_world = camera_matrix
        camera_data.type = 'PERSP'
        camera_data.lens_unit = 'FOV'
        camera_data.angle = float(fov_y)
        self.scene.render.resolution_x = width
        self.scene.render.resolution_y = height
        return camera_obj, camera_data, prev_camera

    @staticmethod
    def _remove_camera_blocks(camera_obj, camera_data):
        """删除临时相机；object 移除后 data 可能已被 Blender 连带删除。"""
        data_name = camera_data.name if camera_data else None
        if camera_obj and camera_obj.name in bpy.data.objects:
            bpy.data.objects.remove(camera_obj, do_unlink=True)
        if data_name and data_name in bpy.data.cameras:
            bpy.data.cameras.remove(bpy.data.cameras[data_name], do_unlink=True)

    @staticmethod
    def _destroy_render_camera(camera_obj, camera_data, prev_camera, scene=None):
        scene = scene or bpy.context.scene
        if camera_obj and scene.camera == camera_obj:
            if prev_camera is not None and getattr(prev_camera, "name", None) in bpy.data.objects:
                scene.camera = prev_camera
            else:
                scene.camera = None
        BpySceneCtx._remove_camera_blocks(camera_obj, camera_data)

    def _apply_view_auto_transparency(self, camera_position, look_at_target, transparent_alpha, z_max):
        wall_transparency_records = {}
        object_transparency_records = []
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
        return wall_transparency_records, object_transparency_records

    def _collect_transparent_objects(self, wall_transparency_records, object_transparency_records):
        transparent_objects = set()
        for wall_id in wall_transparency_records:
            wall_obj = self.mesh_nodes["walls"].get(wall_id, {}).get("node")
            if wall_obj is not None:
                transparent_objects.add(wall_obj)
        for record in object_transparency_records:
            obj = record.get("object")
            if obj is not None:
                transparent_objects.add(obj)
        return transparent_objects

    def _restore_view_transparency(self, wall_transparency_records, object_transparency_records):
        for wall_id, slots in wall_transparency_records.items():
            wall_obj = self.mesh_nodes["walls"].get(wall_id, {}).get("node")
            util_bpy.restore_wall_transparency(wall_obj, slots)
        for record in object_transparency_records:
            util_bpy.restore_object_transparency(record)

    def _render_view_sequence(
        self,
        output_root: str,
        camera_positions: list,
        look_at_targets: list,
        up_vectors: list,
        width: int = 1024,
        height: int = 1024,
        auto_fov: bool = True,
        manual_fov: float = None,
        auto_transparent: bool = True,
        transparent_alpha: float = 0.0,
        geometry_mode: str = "gltf",
        render_depth: bool = False,
        show_wall: bool = True,
        show_window: bool = True,
        show_door: bool = True,
        show_ceiling: bool = True,
        use_HDRI: bool = True,
        hdri_transparent_background: bool = False,
        visible_shadow: bool = True,
        lighting_type: Literal["area", "array", "none"] = "array",
        align_height: bool = True,
        rebuild: bool = False,
        export_glb: bool = False,
        export_point_cloud: bool = False,
        visible_geometry: bool = False,
        render_semantic: bool = False,
        view_transform: bool = False,
    ):
        """多相机序列：所有帧写入同一序列目录；可见几何在该目录合并导出一份（多相机并集）。

        目录名时间戳 = 本次 render_view 调用；各帧 png/depth 等文件名时间戳 = 该帧渲染时刻（与目录名不同）。
        """
        total_start = time.perf_counter()
        seq_dir, dir_stamp = util.allocate_view_output_dir(output_root)
        used_frame_stamps = {dir_stamp}

        world_cam0 = list(camera_positions[0])
        world_look0 = list(look_at_targets[0])
        world_up0 = list(up_vectors[0])

        with util_data.ViewSslSession(self, view_transform) as vss:
            vss.setup(world_cam0, world_look0, world_up0)
            vss.write_ssl(seq_dir)

            if view_transform or rebuild or self.mesh_nodes["floor"] is None:
                self.construct_scene(
                    geometry_mode=geometry_mode,
                    show_wall=show_wall,
                    show_window=show_window,
                    show_door=show_door,
                    show_ceiling=show_ceiling,
                    align_height=align_height,
                    rebuild=view_transform or rebuild,
                )

            for wall_info in self.mesh_nodes["walls"].values():
                wall_obj = wall_info.get("node")
                if wall_obj:
                    wall_obj.visible_shadow = visible_shadow
            ceiling_info = self.mesh_nodes.get("ceiling")
            if ceiling_info:
                ceiling_obj = ceiling_info.get("node")
                if ceiling_obj:
                    ceiling_obj.visible_shadow = visible_shadow

            if not self.if_set_lights:
                self.setup_lighting(lighting_type=lighting_type)
            if use_HDRI:
                hdri_path = self.config.get("hdri_path")
                if hdri_path and util_bpy.apply_hdri_to_world(self.scene, hdri_path, strength=1.0):
                    print(f"🌇 使用 HDRI 环境光: {hdri_path}")
                    self.scene.render.film_transparent = hdri_transparent_background
                else:
                    print(f"⚠️ HDRI 文件不可用或未配置: {hdri_path}")

            z_max = self.context["meta"]["z_max"]
            frame_states = []
            n_frames = len(camera_positions)

            for frame_idx, (cam_pos, look_at, up_vec) in enumerate(
                zip(camera_positions, look_at_targets, up_vectors)
            ):
                world_cam = list(cam_pos)
                world_look = list(look_at)
                world_up = list(up_vec)
                render_cam, render_look, render_up = cam_pos, look_at, up_vec
                if view_transform:
                    render_cam, render_look = vss.render_camera_pose(
                        world_cam, world_look, reference_frame=(frame_idx == 0)
                    )
                    render_up = vss.render_world_up(world_up)

                image_stamp = util.allocate_millis_stamp(exclude=used_frame_stamps)
                used_frame_stamps.add(image_stamp)
                output_path = os.path.join(seq_dir, f"{image_stamp}.png")
                view_dir = seq_dir
                print(f"🎬 序列帧 {frame_idx + 1}/{n_frames} -> {output_path}")

                camera_position, look_at_target, camera_matrix, fov_y = self._build_view_camera_matrix_and_fov(
                    render_cam, render_look, render_up, auto_fov=auto_fov, manual_fov=manual_fov
                )

                wall_transparency_records = {}
                object_transparency_records = []
                if auto_transparent:
                    wall_transparency_records, object_transparency_records = self._apply_view_auto_transparency(
                        camera_position, look_at_target, transparent_alpha, z_max
                    )

                camera_obj, camera_data, prev_camera = self._create_render_camera(
                    camera_matrix, fov_y, width, height
                )
                self.scene.render.filepath = output_path
                render = self.scene.render
                render.image_settings.color_mode = 'RGBA'
                render.film_transparent = hdri_transparent_background

                transparent_objects = self._collect_transparent_objects(
                    wall_transparency_records, object_transparency_records
                )
                depth_scale = None
                try:
                    if render_depth:
                        depth_path = os.path.join(
                            view_dir,
                            f"{image_stamp}_depth.png",
                        )
                        depth_scale = util_bpy.render_color_and_depth_png(
                            self.scene, output_path, depth_path, skip_objects=transparent_objects
                        )
                    else:
                        bpy.ops.render.render(write_still=True)

                    if export_point_cloud:
                        self.export_planar_faces_and_lines(
                            output_path,
                            camera_obj,
                            width,
                            height,
                            align_height=align_height,
                            show_wall=show_wall,
                            show_door=show_door,
                            show_window=show_window,
                            show_ceiling=show_ceiling,
                            transparent_objects=transparent_objects,
                        )
                    if render_semantic:
                        self.render_semantic_png(output_path)

                    frame_states.append({
                        "camera_matrix": camera_matrix.copy(),
                        "fov_y": fov_y,
                        "transparent_objects": transparent_objects,
                    })

                    para_path = self._camera_para_path(output_path)
                    camera_para = vss.build_camera_para(
                        world_cam,
                        world_look,
                        world_up,
                        fov_y,
                        float(width) / float(height),
                        reference_frame=(frame_idx == 0),
                        depth_scale=depth_scale,
                        include_normal_fields=depth_scale is not None,
                    )
                    with open(para_path, 'w') as f:
                        json.dump(camera_para, f, indent=4)
                finally:
                    self._restore_view_transparency(wall_transparency_records, object_transparency_records)
                    self._destroy_render_camera(camera_obj, camera_data, prev_camera, self.scene)
                    util_bpy.cleanup_bpy_render_memory(self.scene)

            if visible_geometry and (export_glb or export_point_cloud) and frame_states:
                original_camera = self.scene.camera
                camera_states = []
                temp_cameras = []
                for state in frame_states:
                    cam_obj, cam_data, _prev = self._create_render_camera(
                        state["camera_matrix"], state["fov_y"], width, height
                    )
                    temp_cameras.append((cam_obj, cam_data))
                    camera_states.append((cam_obj, state["transparent_objects"]))
                try:
                    self.export_visible_geometry_multi(
                        seq_dir,
                        camera_states,
                        export_glb=export_glb,
                        export_point_cloud=export_point_cloud,
                    )
                finally:
                    for cam_obj, cam_data in reversed(temp_cameras):
                        if cam_obj and self.scene.camera == cam_obj:
                            if original_camera is not None and getattr(original_camera, "name", None) in bpy.data.objects:
                                self.scene.camera = original_camera
                            else:
                                self.scene.camera = None
                        self._remove_camera_blocks(cam_obj, cam_data)

        total_time = time.perf_counter() - total_start
        print(f"✅ 序列渲染完成 ({n_frames} 帧), 序列目录: {seq_dir}, 总耗时: {total_time:.2f}s")

    def render_view(self, output_path: str, camera_position: list, look_at_target: list = None, 
                    width: int = 1024, height: int = 1024, up_vector: list = None, 
                    auto_fov: bool = True, manual_fov: float = None,
                    auto_transparent: bool = True, transparent_alpha: float = 0.0,
                    geometry_mode: str = "gltf", render_depth: bool = False,
                    show_wall: bool = True, show_window: bool = True, show_door: bool = True, show_ceiling: bool = True,
                    use_HDRI: bool = True, hdri_transparent_background: bool = False,
                    visible_shadow: bool = True,
                    lighting_type: Literal["area", "array", "none"] = "array",
                    align_height: bool = True,
                    rebuild: bool = False,
                    export_glb: bool = False,
                    glb_path: Optional[str] = None,
                    export_point_cloud: bool = False,
                    visible_geometry: bool = False,
                    render_semantic: bool = False,
                    view_transform: bool = False):
        if self._is_view_sequence(camera_position, look_at_target, up_vector):
            cam_positions, look_ats, ups = self._expand_camera_sequence_args(
                camera_position, look_at_target, up_vector
            )
            return self._render_view_sequence(
                output_path,
                cam_positions,
                look_ats,
                ups,
                width=width,
                height=height,
                auto_fov=auto_fov,
                manual_fov=manual_fov,
                auto_transparent=auto_transparent,
                transparent_alpha=transparent_alpha,
                geometry_mode=geometry_mode,
                render_depth=render_depth,
                show_wall=show_wall,
                show_window=show_window,
                show_door=show_door,
                show_ceiling=show_ceiling,
                use_HDRI=use_HDRI,
                hdri_transparent_background=hdri_transparent_background,
                visible_shadow=visible_shadow,
                lighting_type=lighting_type,
                align_height=align_height,
                rebuild=rebuild,
                export_glb=export_glb,
                export_point_cloud=export_point_cloud,
                visible_geometry=visible_geometry,
                render_semantic=render_semantic,
                view_transform=view_transform,
            )

        output_path = self._resolve_view_output_path(output_path)
        view_dir = os.path.dirname(output_path) or "."
        center = self.context["meta"]["center"]
        z_max = self.context["meta"]["z_max"]
        if look_at_target is None:
            world_look_w = [center[0], center[1], z_max / 2]
        else:
            world_look_w = list(look_at_target)
        world_cam_w = list(camera_position)
        world_up_raw = up_vector if up_vector is not None else [0.0, 0.0, 1.0]

        total_start = time.perf_counter()

        with util_data.ViewSslSession(self, view_transform) as vss:
            vss.setup(world_cam_w, world_look_w, world_up_raw)
            vss.write_ssl(view_dir)

            construct_time = 0.0
            if view_transform or rebuild or self.mesh_nodes["floor"] is None:
                construct_start = time.perf_counter()
                self.construct_scene(
                    geometry_mode=geometry_mode,
                    show_wall=show_wall,
                    show_window=show_window,
                    show_door=show_door,
                    show_ceiling=show_ceiling,
                    align_height=align_height,
                    rebuild=view_transform or rebuild,
                )
                construct_time = time.perf_counter() - construct_start

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
                    self.scene.render.film_transparent = hdri_transparent_background
                else:
                    print(f"⚠️ HDRI 文件不可用或未配置: {hdri_path}")
            setup_time = time.perf_counter() - setup_start

            bounds = self.context["meta"]["bounds"]

            if view_transform:
                render_cam, render_look = vss.render_camera_pose(
                    world_cam_w, world_look_w, reference_frame=True
                )
                camera_position = np.array(render_cam, dtype=float)
                look_at_target = np.array(render_look, dtype=float)
                up_vector = np.array(vss.render_world_up(world_up_raw), dtype=float)
            else:
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
            object_transparency_records = {}
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

            self.scene.render.resolution_x = width
            self.scene.render.resolution_y = height
            self.scene.render.filepath = output_path

            render = self.scene.render
            render.image_settings.color_mode = 'RGBA'
            render.film_transparent = hdri_transparent_background

            print(f"🎬 渲染中 ({width}x{height})...")
            depth_scale = None
            transparent_objects = {
                self.mesh_nodes["walls"].get(wall_id, {}).get("node")
                for wall_id in wall_transparency_records
            }
            transparent_objects.update(record.get("object") for record in object_transparency_records)
            transparent_objects = {obj for obj in transparent_objects if obj is not None}
            try:
                render_start = time.perf_counter()
                if render_depth:
                    depth_path = os.path.join(
                        os.path.dirname(output_path),
                        f"{os.path.splitext(os.path.basename(output_path))[0]}_depth.png",
                    )
                    depth_scale = util_bpy.render_color_and_depth_png(
                        self.scene, output_path, depth_path, skip_objects=transparent_objects
                    )
                else:
                    bpy.ops.render.render(write_still=True)
                render_time = time.perf_counter() - render_start
                if visible_geometry and (export_glb or export_point_cloud):
                    self.export_visible_geometry(
                        os.path.dirname(output_path),
                        camera_obj,
                        export_glb=export_glb,
                        export_point_cloud=export_point_cloud,
                        transparent_objects=transparent_objects,
                    )
                if export_point_cloud:
                    self.export_planar_faces_and_lines(
                        output_path,
                        camera_obj,
                        width,
                        height,
                        align_height=align_height,
                        show_wall=show_wall,
                        show_door=show_door,
                        show_window=show_window,
                        show_ceiling=show_ceiling,
                        transparent_objects=transparent_objects,
                    )
                if render_semantic:
                    self.render_semantic_png(output_path)
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
                self._destroy_render_camera(camera_obj, camera_data, prev_camera, self.scene)
                util_bpy.cleanup_bpy_render_memory(self.scene)

            if export_glb and not visible_geometry and glb_path is not None:
                self.export_glb(glb_path or os.path.splitext(output_path)[0] + ".glb")

            para_path = self._camera_para_path(output_path)
            camera_para = vss.build_camera_para(
                world_cam_w,
                world_look_w,
                vss.world_up,
                float(fov_y),
                float(width) / float(height),
                reference_frame=True,
                depth_scale=depth_scale,
                include_normal_fields=depth_scale is not None,
            )
            with open(para_path, 'w') as f:
                json.dump(camera_para, f, indent=4)
    
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
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
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
                transparent_alpha=0.0,
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
