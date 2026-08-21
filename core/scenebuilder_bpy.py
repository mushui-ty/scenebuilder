"""
SceneBuilder - Pure Blender rendering version (no trimesh dependency)

Uses bpy instead of trimesh/pyrender

Usage:
   $BLENDER_PATH --background --python -m scenebuilder.core.scenebuilder_bpy
   Or after pip install bpy: python -m scenebuilder.core.scenebuilder_bpy
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
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Literal, Tuple, Union, Set, Callable

# Import util data processing functions (not its mesh creation functions)
try:
    from . import util, util_bpy, util_data, geometry_opencv as geo_cv
    from .config_utils import CONFIG_PATH, load_config
    from . import viewvis_export as vve
except ImportError:
    import util
    import util_bpy
    import util_data  # type: ignore
    import geometry_opencv as geo_cv  # type: ignore
    import viewvis_export as vve  # type: ignore
    from config_utils import CONFIG_PATH, load_config

try:
    import bpy
    import mathutils  # type: ignore[import]
    from mathutils import Matrix  # type: ignore[import]
except ImportError:
    print("❌ Error: Failed to import bpy module")
    print("   Please run this script in a Blender Python environment")
    raise


def configure_cycles_gpu_devices(prefs) -> str:
    """Configure Cycles GPU devices; prefer OptiX when available, else CUDA."""
    refresh = getattr(prefs, "refresh_devices", None)
    if callable(refresh):
        refresh()

    for backend in ("OPTIX", "CUDA"):
        try:
            devices = list(prefs.get_devices_for_type(backend) or [])
        except Exception:
            devices = []
        if not devices:
            continue
        if not any(getattr(device, "type", None) != "CPU" for device in devices):
            continue
        prefs.compute_device_type = backend
        for device in devices:
            device.use = True
        return backend

    return "NONE"


class BpySceneCtx:
    """Scene context manager - pure Blender version"""
    
    
    def __init__(self, scene_type: str, model_extra_path: Optional[str] = None,
                 hole_extra_path: Optional[str] = None,
                 render_engine: Literal["CYCLES", "EEVEE"] = "CYCLES"):
        # Data structure identical to scenebuilder.py
        self.context = {
            "meta": {"scene_type": scene_type},
            "walls": {},
            "boxes": {}
        }
        self.scene = None
        self.if_set_lights = False
        self.model_extra_path = model_extra_path
        self.hole_extra_path = hole_extra_path
        self.render_engine = render_engine
        
        # Model cache: asset_id -> master_collection
        self.asset_cache = {}
        self.asset_bounds = {}
        self._point_cloud_material_cache = {}
        self._point_cloud_image_cache = {}
        self._semantic_view_cache: Dict[str, Tuple[Any, List[Dict[str, Any]], Dict[Tuple[str, str], int]]] = {}
        
        # Keep all object references (corresponds to mesh_nodes in scenebuilder.py)
        self.mesh_nodes = {
            "walls": {},      # wall_id -> node
            "doors": {},      # door_id -> node
            "windows": {},    # window_id -> node
            "boxes": {},      # box_id -> node
            "floor": None,    # floor node
            "ceiling": None,  # ceiling node
        }

        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        self.config = load_config(config_path)

        # Initialize Blender scene
        self._init_blender_scene()

    def set_model_path(self, path: str):
        """Change model lookup path"""
        print(f"🔄 Changed model path to: {path}")
        self.config["model_path"] = path

    def _asset_search_paths(self, default_key: str = "model_path") -> List[str]:
        """extra_path → default path → generate path (deduplicated, order preserved)."""
        candidates = []
        if self.model_extra_path:
            candidates.append(self.model_extra_path)
        default = self.config.get(default_key)
        if default:
            candidates.append(default)
        generate = self.config.get("model_generate_path")
        if generate:
            candidates.append(generate)
        return self._dedupe_paths(candidates)

    def _hole_asset_search_paths(self) -> List[str]:
        """Door/window: assets dir → hole fallback → config model_hole_path → generate."""
        candidates = []
        if self.model_extra_path:
            candidates.append(self.model_extra_path)
        if self.hole_extra_path:
            candidates.append(self.hole_extra_path)
        default = self.config.get("model_hole_path")
        if default:
            candidates.append(default)
        generate = self.config.get("model_generate_path")
        if generate:
            candidates.append(generate)
        return self._dedupe_paths(candidates)

    @staticmethod
    def _dedupe_paths(candidates: List[str]) -> List[str]:
        seen = set()
        ordered = []
        for path in candidates:
            if path and path not in seen:
                seen.add(path)
                ordered.append(path)
        return ordered

    def _init_blender_scene(self):
        """Initialize Blender scene"""
        # Reset scene with factory settings; more thorough than manual deletion, helps init EGL context in headless mode
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        if "Scene" in bpy.data.scenes:
            self.scene = bpy.data.scenes["Scene"]
        else:
            self.scene = bpy.data.scenes.new("Scene")
        bpy.context.window.scene = self.scene
        
        # Set render engine
        engine_map = {
            "CYCLES": "CYCLES",
            "EEVEE": "BLENDER_EEVEE"
        }
        self.scene.render.engine = engine_map.get(self.render_engine, "CYCLES")
        
        if self.render_engine == "CYCLES":
            self.scene.cycles.device = 'GPU'
            
            prefs = bpy.context.preferences.addons['cycles'].preferences
            gpu_backend = configure_cycles_gpu_devices(prefs)
            
            self.scene.cycles.samples = self.config.get("blender_samples", 32)
            self.scene.cycles.use_denoising = True
            self.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
            if gpu_backend == "NONE":
                print("⚠️  Blender scene initialized (Cycles GPU requested, but no OptiX/CUDA device found)")
            else:
                print(f"✅ Blender scene initialized (Cycles + {gpu_backend})")
        else:
            # EEVEE Next settings (Blender 4.2+)
            if hasattr(self.scene, "eevee"):
                # EEVEE Next in 4.2 has new parameters; configure here as needed
                self.scene.eevee.taa_render_samples = self.config.get("blender_samples", 32)
            print(f"✅ Blender scene initialized (EEVEE Next)")
        
        self.scene_collection = self.scene.collection

    def clear_scene(self):
        """Completely clear all objects, lights, and cameras, and reset state"""
        # Switch to Object mode just in case
        if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
            
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # --- Deep memory cleanup ---
        # 1. Physically remove all image datablocks (one of the largest memory leftovers)
        for img in bpy.data.images:
            if img.users == 0 or not img.filepath: # Remove only unused or placeholder images
                try:
                    bpy.data.images.remove(img, do_unlink=True)
                except:
                    pass
        
        # 2. Purge all orphan datablocks (materials, meshes, etc.)
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        # 2b. Release Cycles render buffers
        try:
            from . import util_bpy
            util_bpy.cleanup_bpy_render_memory(self.scene)
        except Exception:
            pass
        # -------------------
        
        # 3. Clear Collection cache
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
        
        # Reset reference state
        self.mesh_nodes = {
            "walls": {},      
            "doors": {},      
            "windows": {},    
            "boxes": {},      
            "floor": None,    
            "ceiling": None,  
        }
        self.if_set_lights = False
        print("🧹 Scene fully cleared and state reset")

    # ==================== Data management (identical to scenebuilder.py) ====================
    def add_walls(self, walls: List[Dict[str, Any]]):
        """Add walls and compute scene metadata (auto-skip height<=0 walls; supports outdoor scenes)"""
        # walls_converted: use all walls to compute polygon and metadata (includes height=0 outdoor boundary)
        walls_converted = [{"s": w["p"][:2], "e": w["q"][:2], "height": w["height"]} for w in walls]

        # Endpoint snapping enabled by default, threshold 0.3m
        do_snap = True
        if do_snap:
            walls_converted = util.snap_wall_endpoints(walls_converted, threshold=0.3)

        all_points = [p for w in walls_converted for p in [w["s"], w["e"]]]
        x_coords, y_coords = zip(*all_points)
        # vertices: [(x1, y1), (x2, y2), ...] outer polygon boundary vertex sequence
        # partitions: [(xs, ys, xe, ye, height), ...] interior partition wall segments (with height)
        vertices, partitions = util.calculate_minimum_area_polygon_and_partitions(walls_converted)

        # z_max: max wall height; z_max=0 when all walls are height 0 (outdoor scene)
        z_max = max(w["height"] for w in walls_converted)

        self.context["meta"].update({
            "bounds": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
            "center": [(min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2],
            "span": [max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)],
            "z_max": z_max,
            "vertices": vertices
        })
        self.context["meta"].pop("_scene_normalized", None)

        # Filter height<=0 walls; outdoor scenes keep floor polygon only, no wall objects
        valid_walls = [w for w in walls_converted if w["height"] > 0]
        if len(valid_walls) < len(walls_converted):
            print(f"🏞️ Outdoor scene: skipping {len(walls_converted) - len(valid_walls)} walls with height 0")
        if not valid_walls:
            print(f"🏞️ Pure outdoor scene: computing floor polygon only ({len(vertices)} vertices), skipping wall construction")
            return

        print(f"📐 Input raw wall count: {len(valid_walls)}")
        print(f"📐 Boundary wall segment count: {len(vertices)}")
        print(f"📐 Interior partition wall count: {len(partitions)}")

        # 1. Add boundary walls
        # Boundary walls form a closed loop from vertices so floor and wall base align perfectly
        for i in range(len(vertices)):
            v_s = vertices[i]
            v_e = vertices[(i + 1) % len(vertices)]

            # Find original height for this boundary segment
            # Prefer walls covering this segment; if none (e.g. bridge edge), use neighbor height
            height = None
            for w in valid_walls:
                if util.is_wall_on_edge(w, v_s, v_e):
                    height = w["height"]
                    break
            if height is None:
                # Fallback: use max height among all valid walls
                height = max(w["height"] for w in valid_walls)

            wall_id = util.generate_unique_id()
            self.context["walls"][wall_id] = {
                "s": list(v_s),
                "e": list(v_e),
                "height": height,
                # Boundary walls need inward orientation for one-sided thickness offset
                "orientation": util.calculate_wall_orientation(v_s, v_e, vertices),
                "is_partition": False,
                "doors": {},
                "windows": {}
            }

        # 2. Add interior partition walls
        # Partitions are not on outer contour; both sides are typically inside the room
        for p in partitions:
            if p[4] <= 0:
                continue  # Skip zero-height partitions
            wall_id = util.generate_unique_id()
            self.context["walls"][wall_id] = {
                "s": [p[0], p[1]],
                "e": [p[2], p[3]],
                "height": p[4],
                # Partitions need no orientation; render thickness split evenly about centerline
                "orientation": (0.0, 0.0),
                "is_partition": True,
                "doors": {},
                "windows": {}
            }

    def add_wall(self, s: List[float], e: List[float], height: float):
        """Add a single wall"""
        # Convert existing walls to input format and append new wall
        current_walls = [{"p": w["s"] + [0], "q": w["e"] + [0], "height": w["height"]}
                        for w in self.context["walls"].values()]
        current_walls.append({"p": s[:2] + [0], "q": e[:2] + [0], "height": height})
        
        # Clear current state and re-add in batch (triggers reclassification and snapping)
        self.context["walls"] = {}
        self.add_walls(current_walls)

    def add_doors(self, doors: List[Dict[str, Any]]):
        """Add doors in batch"""
        self.context["meta"].pop("_scene_normalized", None)
        for door in doors:
            self.add_door(door["center"], door["width"], door["height"], door.get("asset_id"))

    def add_door(self, center: List[float], width: float, height: float, asset_id: Optional[int] = None):
        """Add a single door"""
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
        """Add windows in batch"""
        self.context["meta"].pop("_scene_normalized", None)
        for window in windows:
            self.add_window(window["center"], window["width"], window["height"], window.get("asset_id"))

    def add_window(self, center: List[float], width: float, height: float, asset_id: Optional[int] = None):
        """Add a single window"""
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
        """Add furniture boxes in batch"""
        self.context["meta"].pop("_scene_normalized", None)
        for box in boxes:
            self.add_box(box["center"], box["angle_z"], box["scale"],
                        box.get("label"), box.get("caption"), box.get("asset_id"))

    def add_box(self, center: List[float], angle_z: float, scale: List[float],
                label: Optional[str] = None, caption: Optional[str] = None, 
                asset_id: Optional[int] = None) -> str:
        """Add a single furniture box; returns ID"""
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
        """Delete furniture and remove corresponding objects from Blender scene"""
        if box_id not in self.context["boxes"]:
            print(f"⚠️  Box ID not found: {box_id}, cannot delete")
            return

        # 1. If already rendered, physically delete in Blender
        if box_id in self.mesh_nodes["boxes"]:
            box_info = self.mesh_nodes["boxes"][box_id]
            node = box_info.get("node")
            if node:
                # Recursively delete object and all children (for imported glTF root nodes)
                objs_to_remove = [node] + list(node.children_recursive)
                for obj in objs_to_remove:
                    try:
                        bpy.data.objects.remove(obj, do_unlink=True)
                    except (ReferenceError, AttributeError):
                        pass
            del self.mesh_nodes["boxes"][box_id]

        # 2. Remove from context
        del self.context["boxes"][box_id]

        # 3. Recompute z_max
        wall_max = max((w["height"] for w in self.context["walls"].values()), default=0)
        box_max = 0
        for box in self.context["boxes"].values():
            box_top = box["center"][2] + box["scale"][2] / 2
            box_max = max(box_max, box_top)
        self.context["meta"]["z_max"] = max(wall_max, box_max)
        
        print(f"🗑️  Furniture {box_id[:4]} deleted")

    def get_context(self) -> Dict:
        """Get scene context"""
        return self.context
    
    def get_boxes(self) -> Dict:
        """Get all furniture boxes"""
        return self.context["boxes"]

    def normalize_scene_data(self) -> None:
        """Normalize context labels / validate asset_id (consistent with exported SSL)."""
        try:
            from . import util_data
        except ImportError:
            import util_data  # type: ignore
        util_data.normalize_scene_context(
            self.context,
            model_paths=self._asset_search_paths("model_path"),
            hole_paths=self._hole_asset_search_paths(),
        )

    def _object_export_identity(self, category: str, object_id: str):
        """Point cloud / visible geometry naming: (label, asset_id|None)."""
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
        """Set wall texture path"""
        self.config["wall_blender_texture_path"] = path
        print(f"📝 Wall texture path updated: {path}")

    def set_floor_blender_texture_path(self, path: str):
        """Set floor texture path"""
        self.config["floor_blender_texture_path"] = path
        print(f"📝 Floor texture path updated: {path}")

    def set_ceiling_blender_texture_path(self, path: str):
        """Set ceiling texture path"""
        self.config["ceiling_blender_texture_path"] = path
        print(f"📝 Ceiling texture path updated: {path}")

    def set_hdri_path(self, path: str):
        """Set HDRI environment map path"""
        self.config["hdri_path"] = path
        print(f"📝 HDRI path updated: {path}")

    def set_blender_samples(self, samples: int):
        """Set Blender render sample count"""
        self.config["blender_samples"] = samples
        if self.scene and hasattr(self.scene, "cycles"):
            self.scene.cycles.samples = samples
        print(f"📝 Blender render sample count updated: {samples}")

    def _get_or_create_asset_collection(self, asset_id: int, model_paths: List[str]):
        """Get or create master asset Collection (for instancing)"""
        if asset_id in self.asset_cache:
            return self.asset_cache[asset_id]
        
        # 1. Create a new standalone Collection
        coll_name = f"AssetCollection_{asset_id}"
        if coll_name in bpy.data.collections:
            self.asset_cache[asset_id] = bpy.data.collections[coll_name]
            return self.asset_cache[asset_id]
            
        new_coll = bpy.data.collections.new(coll_name)
        # Temporarily link to scene to allow import
        self.scene.collection.children.link(new_coll)
        
        # 2. Set as active collection and import
        # Note: some Blender import ops depend on active_collection
        orig_collection = bpy.context.view_layer.active_layer_collection
        
        # Recursively find target layer_collection
        def find_layer_collection(layer_coll, name):
            if layer_coll.name == name: return layer_coll
            for child in layer_coll.children:
                res = find_layer_collection(child, name)
                if res: return res
            return None
            
        target_layer_coll = find_layer_collection(bpy.context.view_layer.layer_collection, coll_name)
        if target_layer_coll:
            bpy.context.view_layer.active_layer_collection = target_layer_coll
            
        # 3. Import model
        # Import via util_bpy.load_mesh_to_origin
        # Note: load_mesh_to_origin places objects in the scene main collection
        # Move imported objects into new_coll after import
        mesh_root = util_bpy.load_mesh_to_origin(asset_id, model_paths)
        
        if mesh_root:
            # Move mesh_root and all children into new collection
            objs_to_move = [mesh_root] + list(mesh_root.children_recursive)
            
            # --- [Optimization] Precompute master bounds and cache ---
            master_bounds = util_bpy._get_combined_bounds(objs_to_move)
            self.asset_bounds[asset_id] = master_bounds
            
            for obj in objs_to_move:
                for coll in obj.users_collection:
                    coll.objects.unlink(obj)
                new_coll.objects.link(obj)
            
            # After import, unlink master Collection from scene (keep datablock)
            self.scene.collection.children.unlink(new_coll)
            self.asset_cache[asset_id] = new_coll
            print(f"📦 [Master load] {asset_id} loaded successfully and cached (Size: {master_bounds[1]-master_bounds[0] if master_bounds else 'None'})")
            return new_coll
        else:
            # Import failed; clean up
            self.scene.collection.children.unlink(new_coll)
            bpy.data.collections.remove(new_coll)
            self.asset_cache[asset_id] = None
            return None

    def export_wall_ssl(self, output_dir: str):
        """
        Export wall SSL file (Room and all Walls).
        Includes boundary and partition walls
        """
        output_path = os.path.join(output_dir, 'wall_ssl.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        
        # Generate room_id
        room_id = util.generate_unique_id()
        room_type = self.context["meta"]["scene_type"]
        lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
        
        # Export all walls from context["walls"] (boundary and partitions)
        all_walls = self.context["walls"]
        for wall_id, wall in all_walls.items():
            p = list(wall["s"]) + [0.0]
            q = list(wall["e"]) + [0.0]
            height = wall["height"]
            lines.append(f'Wall(id="{wall_id}", room_id="{room_id}", p={p}, q={q}, height={height})')
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Wall SSL exported: {output_path} ({len(all_walls)} walls)")
        return output_path

    def export_wall_hole_ssl(self, output_dir: str):
        """
        Export wall and opening SSL file (Room, Wall, Door, Window).
        Includes boundary and partition walls
        """
        output_path = os.path.join(output_dir, 'wall_hole_ssl.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        
        # Generate room_id
        room_id = util.generate_unique_id()
        room_type = self.context["meta"]["scene_type"]
        lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
        
        # Iterate all walls and their doors/windows
        for wall_id, wall in self.context["walls"].items():
            p = list(wall["s"]) + [0.0]
            q = list(wall["e"]) + [0.0]
            height = wall["height"]
            lines.append(f'Wall(id="{wall_id}", room_id="{room_id}", p={p}, q={q}, height={height})')
            
            # Export doors on this wall
            for door_id, door in wall.get("doors", {}).items():
                lines.append(f'Door(id="{door_id}", wall_id="{wall_id}", center={door["center"]}, width={door["width"]}, height={door["height"]})')
            
            # Export windows on this wall
            for window_id, window in wall.get("windows", {}).items():
                lines.append(f'Window(id="{window_id}", wall_id="{wall_id}", center={window["center"]}, width={window["width"]}, height={window["height"]})')
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Wall and door/window SSL exported: {output_path}")
        return output_path

    def export_ssl(self, output_dir: str):
        """
        Export full SSL file (Room, Wall, Door, Window, Bbox)
        """
        output_path = os.path.join(output_dir, 'ssl.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        
        # Generate room_id
        room_id = util.generate_unique_id()
        room_type = self.context["meta"]["scene_type"]
        lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
        
        # Export all walls and openings
        for wall_id, wall in self.context["walls"].items():
            p = list(wall["s"]) + [0.0]
            q = list(wall["e"]) + [0.0]
            height = wall["height"]
            lines.append(f'Wall(id="{wall_id}", room_id="{room_id}", p={p}, q={q}, height={height})')
            
            for door_id, door in wall.get("doors", {}).items():
                lines.append(f'Door(id="{door_id}", wall_id="{wall_id}", center={door["center"]}, width={door["width"]}, height={door["height"]})')
            
            for window_id, window in wall.get("windows", {}).items():
                lines.append(f'Window(id="{window_id}", wall_id="{wall_id}", center={window["center"]}, width={window["width"]}, height={window["height"]})')
        
        # Export all furniture (Bbox)
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
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Full SSL exported: {output_path}")
        return output_path

    # ==================== Pure bpy geometry creation (replaces trimesh helpers in util) ====================
    # helper functions now live in util_bpy

    # ==================== Scene construction (logic identical to scenebuilder.py) ====================
    
    def construct_floor(self, show_wall=True, show_window=True, show_door=True, show_ceiling=True, align_height: bool = True):
        """Build floor and walls (logic identical to scenebuilder.py)"""
        # Note: does not create pyrender.Scene(); uses existing bpy scene
        vertices = self.context["meta"]["vertices"]

        if len(vertices) < 3:
            return

        # Create floor
        bounds = self.context["meta"]["bounds"]
        z_max = self.context["meta"]["z_max"]
        wall_thickness = self.config.get("wall_thickness", 0.1)
        floor_vertices = util.calculate_floor_polygon_with_wall_thickness(
            vertices, self.context["walls"], wall_thickness
        )
        floor_bounds = [
            min(p[0] for p in floor_vertices),
            min(p[1] for p in floor_vertices),
            max(p[0] for p in floor_vertices),
            max(p[1] for p in floor_vertices),
        ]
        
        f_scale = self.config.get("floor_texture_scale", 1.0)
        w_scale = self.config.get("wall_texture_scale", 1.0)
        c_scale = self.config.get("ceiling_texture_scale", 1.0)

        floor_obj = util_bpy.create_floor_mesh_bpy(self.scene_collection, floor_vertices, texture_scale=f_scale)
        if floor_obj is None:
            print("⚠️ Floor geometry creation failed, skipping subsequent processing")
            return

        # Load texture
        texture_path = self.config["floor_blender_texture_path"]
        if texture_path and os.path.exists(texture_path):
            mat = util_bpy.create_material_with_texture("Floor_Material", texture_path)
        else:
            mat = util_bpy.create_material_with_color("Floor_Material", [0.8, 0.7, 0.5, 1.0])
        if mat:
            floor_obj.data.materials.append(mat)
        
        self.mesh_nodes["floor"] = {"node": floor_obj, "mesh": floor_obj}  # Keep reference

        # Create walls
        if show_wall:
            wall_color = self.config["wall_color"]
            wall_texture_path = self.config["wall_blender_texture_path"]

            # --- Precompute miter joint offsets ---
            wall_outer_points = util_bpy.calculate_miter_joints(self.context["walls"], wall_thickness)

            # --- Collect all wall objects for optional boolean merge ---
            all_wall_objs = []

            for wall_id, wall in self.context["walls"].items():
                is_partition = wall.get("is_partition", False)
                
                # Collect door/window openings
                openings = []
                for door in wall.get("doors", {}).values():
                    openings.append((door, "door"))
                for window in wall.get("windows", {}).values():
                    openings.append((window, "window"))
                
                # Get precomputed offset points
                outer_s, outer_e = wall_outer_points.get(wall_id, (None, None))

                # Wall height
                wall_h = z_max if align_height else wall["height"]

                # For partitions, check if endpoints connect to other walls to decide extension
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

                # Create wallsmesh
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
                    # Apply texture or color
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

            # Note: bpy version does not implement edge lines (optional feature)
        else:
            print("🚫 Skipping wall creation (show_wall=False)")
        
        # Add doors and windows
        if show_door or show_window:
            print(f"🚪 Processing doors and windows (loading by asset_id)...")
            wall_thickness = self.config.get("wall_thickness", 0.1)
            
            hole_paths = self._hole_asset_search_paths()

            for wall_id, wall in self.context["walls"].items():
                wall_s = np.array(wall["s"])
                wall_e = np.array(wall["e"])
                wall_vec = wall_e - wall_s
                wall_len = np.linalg.norm(wall_vec)
                if wall_len < 1e-6:
                    continue

                wall_dir = wall_vec / wall_len
                # orientation is inward-facing normal
                is_partition = wall.get("is_partition", False)
                orientation = np.array(wall["orientation"])
                
                # Strategy B: build stable virtual normal for partitions for consistent front/back
                if is_partition and np.linalg.norm(orientation) < 1e-4:
                    # Perpendicular to wall direction [-dy, dx]
                    outward_normal = np.array([-wall_dir[1], wall_dir[0]])
                else:
                    # Exterior wall: negate inward normal for outward offset direction
                    outward_normal = -orientation

                # Compute wall rotation angle (degrees)
                wall_angle_deg = np.degrees(np.arctan2(wall_dir[1], wall_dir[0]))

                # Types to process
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

                        # Compute actual center
                        inner_center = np.array(item["center"])
                        real_center = inner_center.copy()
                        
                        # Only exterior walls offset outward by half thickness; partitions sit on centerline
                        if not is_partition:
                            real_center[0] += outward_normal[0] * (wall_thickness / 2)
                            real_center[1] += outward_normal[1] * (wall_thickness / 2)

                        # Build box-format data
                        item_box_data = {
                            "center": real_center.tolist(),
                            # Opening thickness usually from model; wall_thickness passed as reference
                            "scale": [item["width"], wall_thickness, item["height"]],
                            "angle_z": wall_angle_deg,
                            "asset_id": asset_id
                        }

                        # Same pattern as bbox placement
                        # --- [Optimization] Door/window instancing ---
                        asset_coll = self._get_or_create_asset_collection(asset_id, hole_paths)
                        if asset_coll:
                            # Create instance
                            instance_name = f"Instance_{item_type}_{item_id[:4]}"
                            instance_obj = bpy.data.objects.new(instance_name, None)
                            instance_obj.instance_type = 'COLLECTION'
                            instance_obj.instance_collection = asset_coll
                            self.scene_collection.objects.link(instance_obj)
                            
                            try:
                                # Apply transform (same as furniture logic)
                                # [Optimization] Pass precomputed master_bounds
                                master_b = self.asset_bounds.get(asset_id)
                                success_transform = util_bpy.apply_box_transform(instance_obj, item_box_data, master_bounds=master_b)
                            except Exception as e:
                                print(f"  ❌ {item_type} {item_id[:4]} instancing transform error: {e}")
                                success_transform = False

                            if success_transform:
                                self.mesh_nodes[f"{item_type}s"][item_id] = {
                                    "node": instance_obj,
                                    "mesh": instance_obj,
                                    f"{item_type}_data": dict(item),
                                }
                                # print(f"  ✅ {item_type} {item_id[:4]} (asset_id={asset_id}) instancing succeeded")
                            else:
                                print(f"  ❌ {item_type} {item_id[:4]} transform failed")
                                # Clean up on transform failure
                                bpy.data.objects.remove(instance_obj, do_unlink=True)
                        else:
                            print(f"  ❌ {item_type} {item_id[:4]} (asset_id={asset_id}) master load failed")
        else:
            print("🚫 Skipping door/window creation (show_door=False, show_window=False)")
        
        # Create ceiling
        # Outdoor scene (all wall heights 0): skip ceiling even if show_ceiling=True
        wall_max_height = max((w["height"] for w in self.context["walls"].values()), default=0)
        if show_ceiling and wall_max_height > 0:
            z_max = self.context["meta"]["z_max"]
            ceiling_obj = util_bpy.create_ceiling_mesh_bpy(
                self.scene_collection, floor_vertices, z_max, texture_scale=c_scale
            )
            if ceiling_obj:
                ceiling_texture_path = self.config["ceiling_blender_texture_path"]
                if ceiling_texture_path and os.path.exists(ceiling_texture_path):
                    mat = util_bpy.create_material_with_texture("Ceiling_Material", ceiling_texture_path)
                else:
                    mat = util_bpy.create_material_with_color("Ceiling_Material", [0.9, 0.9, 0.9, 1.0])
                if mat:
                    ceiling_obj.data.materials.append(mat)

                self.mesh_nodes["ceiling"] = {"node": ceiling_obj, "mesh": ceiling_obj}
                print(f"✅ Ceiling added (height: {z_max:.2f}m)")
            else:
                print("⚠️ Ceiling geometry creation failed")
        else:
            if not show_ceiling:
                print("🚫 Skipping ceiling (show_ceiling=False)")
            else:
                print("🏞️ Skipping ceiling (outdoor scene, max wall height is 0)")

    def construct_scene(self, show_wall: bool = True, show_window: bool = True, 
                       show_door: bool = True, show_ceiling: bool = True,
                       geometry_mode: str = "gltf", align_height: bool = True,
                       rebuild: bool = False,
                       exclude_box_ids: Optional[Set[str]] = None):
        """
        Build the full scene.
        
        Args:
            show_wall, show_window, show_door, show_ceiling: Whether to render each structure type
            geometry_mode: Geometry mode, one of:
                - "gltf": Load GLTF only; skip objects without asset_id or missing files
                - "mixed": Prefer GLTF; fall back to bbox geometry on failure
                - "bbox": Force bbox geometry for all objects
            align_height: Align wall heights to z_max
            rebuild: Rebuild scene (clear all current objects)
            exclude_box_ids: Furniture box IDs to skip (e.g. ceiling occluders for top-down nav mask)
        """
        if rebuild:
            self.clear_scene()

        self.normalize_scene_data()
        total_start = time.perf_counter()
        skip_box_ids = set(exclude_box_ids or ())

        self.construct_floor(show_wall=show_wall, show_window=show_window, 
                             show_door=show_door, show_ceiling=show_ceiling, align_height=align_height)

        box_total = len(self.context["boxes"])
        box_load_count = box_total - sum(1 for bid in self.context["boxes"] if bid in skip_box_ids)
        print(f"📦 Loading furniture ({box_load_count}/{box_total} objects, mode: {geometry_mode})...")
        success_count = 0
        load_time = 0
        transform_time = 0
        
        model_paths = self._asset_search_paths("model_path")

        for box_id, box in self.context["boxes"].items():
            if box_id in skip_box_ids:
                continue
            asset_id = box.get("asset_id")
            
            # --- Mode selection logic ---
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
            
            # --- Load execution ---
            mesh_root = None
            if should_load_gltf:
                load_start = time.perf_counter()
                
                # --- [Optimization] Use Collection Instance ---
                asset_coll = self._get_or_create_asset_collection(asset_id, model_paths)
                load_time += time.perf_counter() - load_start
                
                if asset_coll:
                    transform_start = time.perf_counter()
                    # Create instance (Empty object)
                    instance_name = f"Instance_{asset_id}_{box_id[:4]}"
                    instance_obj = bpy.data.objects.new(instance_name, None)
                    instance_obj.instance_type = 'COLLECTION'
                    instance_obj.instance_collection = asset_coll
                    self.scene_collection.objects.link(instance_obj)
                    
                    try:
                        # [Optimization] Pass precomputed master_bounds
                        master_b = self.asset_bounds.get(asset_id)
                        success_transform = util_bpy.apply_box_transform(instance_obj, box, master_bounds=master_b)
                    except Exception as e:
                        print(f"  ❌ {box.get('class', 'unknown')} (asset_id={asset_id}) instancing transform error: {e}")
                        success_transform = False
                    transform_time += time.perf_counter() - transform_start
                    
                    if success_transform:
                        self.mesh_nodes["boxes"][box_id] = {
                            "node": instance_obj, "mesh": instance_obj, "box_data": box
                        }
                        success_count += 1
                        name = box.get('label', box.get('class', 'unknown'))
                        # print(f"  ✅ {name} (asset_id={asset_id}) instancing succeeded")
                        continue
                
                # On GLTF load failure, decide whether to fall back
                if geometry_mode == "mixed":
                    should_fallback_bbox = True
                else:
                    print(f"  ❌ {box.get('class', 'unknown')} (asset_id={asset_id}) import failed and hybrid mode not enabled, skipping")
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
                    print(f"  ✅ {name} (bbox geometry) added successfully")
                except Exception as e:
                    name = box.get('label', box.get('class', 'unknown'))
                    print(f"  ❌ {name} (bbox geometry): {e}")

        total_time = time.perf_counter() - total_start
        print(f"✅ Scene build complete! Successfully added {success_count}/{box_load_count} objects")
        print(f"   Build time: {total_time:.2f}s (load: {load_time:.2f}s, transform: {transform_time:.2f}s)")

    def export_glb(self, output_path: str, rebuild: bool = False, **construct_kwargs):
        """Export current Blender scene as GLB."""
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
            # Handle parameter differences across Blender versions; .glb still exports binary glTF.
            bpy.ops.export_scene.gltf(filepath=output_path)

        print(f"✅ GLB exported: {output_path}")
        return output_path

    def export_point_cloud(self, output_dir: str, rebuild: bool = False, **construct_kwargs):
        """Sample surface point clouds per scene object and export as colored PLY."""
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
        all_normals = []

        def export_one(category: str, object_id: str, node, filename: str, sample_count: int,
                       extra: Optional[Dict[str, Any]] = None):
            if node is None:
                return
            use_ses = category in ("boxes", "doors", "windows")
            points, colors, normals = self._sample_bpy_node_surface(
                node,
                sample_count,
                use_ses=use_ses,
            )
            if len(points) == 0:
                print(f"⚠️ Point cloud sampling skipped empty object: {category}/{object_id}")
                return

            path = os.path.join(output_dir, filename)
            self._write_ply(path, points, colors, normals)
            all_points.append(points)
            all_colors.append(colors)
            all_normals.append(normals)

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
            os.makedirs(os.path.join(output_dir, "floor"), exist_ok=True)
            export_one("floor", "floor", floor_info.get("node"), "floor/floor.ply", default_samples)

        ceiling_info = self.mesh_nodes.get("ceiling")
        if ceiling_info:
            os.makedirs(os.path.join(output_dir, "ceiling"), exist_ok=True)
            export_one("ceiling", "ceiling", ceiling_info.get("node"), "ceiling/ceiling.ply", default_samples)

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
            merged_normals = np.vstack(all_normals)
            scene_path = os.path.join(output_dir, "scene_all.ply")
            self._write_ply(scene_path, merged_points, merged_colors, merged_normals)
            metadata["scene_all"] = {
                "path": "scene_all.ply",
                "points": int(len(merged_points)),
            }

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ Point cloud exported: {output_dir}")
        return output_dir

    def export_voxel(self, output_dir: str, rebuild: bool = False, **construct_kwargs):
        """Export full-scene 256³ colored occupancy voxels (no frustum clipping)."""
        wall_max_height = max((w["height"] for w in self.context["walls"].values()), default=0)
        need_ceiling = wall_max_height > 0
        if rebuild or self.mesh_nodes["floor"] is None or (need_ceiling and self.mesh_nodes["ceiling"] is None):
            kwargs = {"show_ceiling": True}
            kwargs.update(construct_kwargs)
            self.construct_scene(rebuild=True, **kwargs)

        try:
            from . import voxel_export as ve
        except ImportError:
            import voxel_export as ve  # type: ignore

        bpy.context.view_layer.update()
        triangles, colors = self._collect_scene_triangle_colors()
        if len(triangles) == 0:
            print("⚠️ Voxel export skipped: scene has no triangles")
            return None

        world_up = [0.0, 0.0, 1.0]
        meta = self.context.get("meta") or {}
        if meta.get("world_up") is not None:
            world_up = [float(x) for x in meta["world_up"]]

        return ve.export_colored_voxel_grids(
            output_dir,
            triangles,
            colors,
            world_up=world_up,
        )

    def export_visible_geometry(
        self,
        output_dir: str,
        camera_obj,
        export_glb: bool = False,
        export_point_cloud: bool = False,
        export_voxel: bool = False,
        transparent_objects: Optional[set] = None,
        panoramic: bool = False,
        view_image_path: Optional[str] = None,
    ):
        """Export visible geometry: mask visibility ratio + frustum clip; write visibility.json."""
        bpy.context.view_layer.update()
        width, height = self._render_resolution()
        visibility_records: List[Dict[str, Any]] = []

        if panoramic:
            visible_entries = self._visible_geometry_entries_pano(
                camera_obj, transparent_objects=transparent_objects or set()
            )
            visibility_records = self._legacy_visibility_records_from_entries(visible_entries)
        else:
            if not view_image_path:
                raise ValueError("export_visible_geometry requires view_image_path (for semantic mask)")
            semantic_rgb, semantic_objects, overall_pixels_map = self.render_semantic_bundle(
                view_image_path,
                save_artifacts=False,
            )
            visible_entries, visibility_records = self._visible_geometry_entries(
                camera_obj,
                transparent_objects=transparent_objects or set(),
                semantic_rgb=semantic_rgb,
                semantic_objects=semantic_objects,
                overall_pixels_map=overall_pixels_map,
                width=width,
                height=height,
            )

        self._write_visibility_json(output_dir, visibility_records, width, height)

        if not visible_entries:
            print("⚠️ No visible geometry detected from current view")
            return

        merge_entries = [e for e in visible_entries if e.get("in_merged_export", True)]

        if export_glb:
            self.export_visible_glb(util_data.visible_glb_path(output_dir), merge_entries)
        if export_point_cloud:
            self.export_visible_point_cloud(output_dir, visible_entries, merge_entries)
        if export_voxel:
            self.export_visible_voxel(output_dir, merge_entries, camera_obj)

    def export_visible_geometry_multi(
        self,
        output_dir: str,
        camera_states: List[Tuple[Any, set]],
        export_glb: bool = False,
        export_point_cloud: bool = False,
        export_voxel: bool = False,
        frame_infos: Optional[List[Dict[str, Any]]] = None,
    ):
        """Multi-camera merged export: keep if visible from any view; frustum clip is union across views."""
        bpy.context.view_layer.update()
        visible_entries = self._visible_geometry_entries_multi(camera_states)
        if not visible_entries:
            print("⚠️ No visible geometry detected in multi-view sequence")
            return
        width, height = self._render_resolution()
        self._write_visibility_json(
            output_dir,
            self._legacy_visibility_records_from_entries(visible_entries),
            width,
            height,
        )
        merge_entries = [e for e in visible_entries if e.get("in_merged_export", True)]
        if export_glb:
            self.export_visible_glb(
                util_data.visible_glb_path(output_dir),
                merge_entries,
                camera_states=camera_states if len(camera_states) > 1 else None,
                frame_infos=frame_infos,
            )
        if export_point_cloud:
            self.export_visible_point_cloud(
                output_dir,
                visible_entries,
                merge_entries,
                camera_states=camera_states if len(camera_states) > 1 else None,
                frame_infos=frame_infos,
            )
        if export_voxel:
            ref_camera = camera_states[0][0] if camera_states else None
            self.export_visible_voxel(output_dir, merge_entries, ref_camera)

    def export_visible_geometry_pano_multi(
        self,
        output_dir: str,
        camera_states: List[Tuple[Any, set]],
        export_glb: bool = False,
        export_point_cloud: bool = False,
        export_voxel: bool = False,
        *,
        pano_resolution: int = 4096,
    ):
        """Multi-panorama merged export: keep full geometry if unobstructed from any camera; no frustum clip."""
        bpy.context.view_layer.update()
        visible_entries = self._visible_geometry_entries_pano_multi(camera_states)
        if not visible_entries:
            print("⚠️ No visible geometry detected in multi-view panorama")
            return
        width, height = self._pano_dimensions(pano_resolution)
        self._write_visibility_json(
            output_dir,
            self._legacy_visibility_records_from_entries(visible_entries),
            width,
            height,
        )
        merge_entries = [e for e in visible_entries if e.get("in_merged_export", True)]
        if export_glb:
            self.export_visible_glb(util_data.visible_glb_path(output_dir), merge_entries)
        if export_point_cloud:
            self.export_visible_point_cloud(output_dir, visible_entries, merge_entries)
        if export_voxel:
            ref_camera = camera_states[0][0] if camera_states else None
            self.export_visible_voxel(output_dir, merge_entries, ref_camera)

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
        panoramic: bool = False,
    ):
        """Export planar inner-surface vertex JSON and line overlay (enabled by default with --ply)."""
        try:
            from . import planar_faces
        except ImportError:
            import planar_faces  # type: ignore

        if panoramic:
            view_ctx = planar_faces.make_bpy_pano_view_context(self.scene, camera_obj, width, height)
        else:
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
            panoramic=panoramic,
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
        """Return (category, obj_id, point)->0/1 using same ray occlusion as visible geometry."""
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

    def export_visible_glb(
        self,
        output_path: str,
        visible_entries: List[Dict[str, Any]],
        camera_states: Optional[List[Tuple[Any, set]]] = None,
        frame_infos: Optional[List[Dict[str, Any]]] = None,
    ):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        temp_objects = []
        prev_selection = list(bpy.context.selected_objects)
        prev_active = bpy.context.view_layer.objects.active
        try:
            bpy.ops.object.select_all(action='DESELECT')
            for entry in visible_entries:
                name_suffix = "_visible"
                if entry.get("frustum_cutted"):
                    name_suffix += "_cutted"
                if entry.get("mostly_occluded"):
                    name_suffix += "_mostly_occluded"
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
                print("⚠️ scene_visible.glb skipped: no geometry after clipping")
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
            print(f"✅ Visible GLB exported: {output_path}")
            ref_camera = visible_entries[0].get("camera_obj") if visible_entries else None
            self._export_glb_opencv_copy(output_path, ref_camera, temp_objects)
            if camera_states and len(camera_states) > 1 and frame_infos:
                view_dir = os.path.dirname(os.path.dirname(output_path))
                centroids, tri_object_indices = self._collect_visible_triangle_centroids_with_indices(
                    visible_entries
                )
                if len(centroids) > 0:
                    self._export_mesh_viewvis_sidecars(
                        view_dir,
                        centroids,
                        tri_object_indices,
                        visible_entries,
                        camera_states,
                        frame_infos,
                    )
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

    def export_visible_point_cloud(
        self,
        view_dir: str,
        visible_entries: List[Dict[str, Any]],
        merge_entries: Optional[List[Dict[str, Any]]] = None,
        camera_states: Optional[List[Tuple[Any, set]]] = None,
        frame_infos: Optional[List[Dict[str, Any]]] = None,
    ):
        """Write per-object and merged PLY under ``{view_dir}/pointcloud/``; GLB under ``{view_dir}/glb/``."""
        os.makedirs(view_dir, exist_ok=True)
        pointcloud_dir = util_data.geometry_pointcloud_dir(view_dir)
        os.makedirs(pointcloud_dir, exist_ok=True)
        merge_entries = merge_entries if merge_entries is not None else visible_entries
        default_samples = 5000
        box_samples = 5000
        metadata = {
            "samples_per_object": default_samples,
            "box_samples_per_object": box_samples,
            "objects": [],
            "visible_geometry": True,
        }
        opencv_ply_count = 0

        for entry in visible_entries:
            sample_count = box_samples if entry["category"] == "boxes" else default_samples
            points, colors, normals = self._sample_visible_records(entry["records"], entry["camera_obj"], sample_count)
            if len(points) == 0:
                continue
            path = os.path.join(pointcloud_dir, entry["visible_ply"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._write_ply_with_opencv_copy(
                path, points, colors, normals, entry.get("camera_obj"), quiet=True
            )
            opencv_ply_count += 1
            metadata["objects"].append({
                "category": entry["category"],
                "id": entry["id"],
                "path": os.path.relpath(path, view_dir),
                "points": int(len(points)),
                "requested_samples": int(sample_count),
                "frustum_cutted": bool(entry.get("frustum_cutted", False)),
                "frustum_in_view_ratio": round(float(entry.get("frustum_in_view_ratio", 1.0)), 4),
                "visibility_ratio": round(float(entry.get("visibility_ratio", 1.0)), 6),
                "mostly_occluded": bool(entry.get("mostly_occluded", False)),
                "in_merged_export": bool(entry.get("in_merged_export", True)),
            })

        merge_points = []
        merge_colors = []
        merge_normals = []
        merge_object_indices: List[int] = []
        for obj_idx, entry in enumerate(merge_entries):
            sample_count = box_samples if entry["category"] == "boxes" else default_samples
            points, colors, normals = self._sample_visible_records(entry["records"], entry["camera_obj"], sample_count)
            if len(points) == 0:
                continue
            merge_points.append(points)
            merge_colors.append(colors)
            merge_normals.append(normals)
            merge_object_indices.append(np.full(len(points), obj_idx, dtype=np.int32))

        if merge_points:
            merged_points = np.vstack(merge_points)
            merged_colors = np.vstack(merge_colors)
            merged_normals = np.vstack(merge_normals)
            merged_object_indices = np.concatenate(merge_object_indices)
            scene_path = util_data.visible_merged_ply_path(view_dir)
            ref_camera = visible_entries[0].get("camera_obj") if visible_entries else None
            self._write_ply_with_opencv_copy(
                scene_path, merged_points, merged_colors, merged_normals, ref_camera, quiet=True
            )
            opencv_ply_count += 1
            metadata["scene_visible"] = {
                "path": os.path.relpath(scene_path, view_dir),
                "points": int(len(merged_points)),
            }
            if camera_states and len(camera_states) > 1 and frame_infos:
                viewvis_meta = self._export_merged_viewvis_sidecars(
                    view_dir,
                    merged_points,
                    merged_object_indices,
                    merge_entries,
                    camera_states,
                    frame_infos,
                )
                if viewvis_meta:
                    metadata["viewvis"] = viewvis_meta

        with open(os.path.join(view_dir, "metadata_visible.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        if opencv_ply_count:
            print(f"✅ OpenCV point cloud copies: {opencv_ply_count} files")
        print(f"✅ Visible point cloud exported: {view_dir}")

    def _compute_object_visibility_per_camera(
        self,
        merge_entries: List[Dict[str, Any]],
        camera_states: List[Tuple[Any, set]],
    ) -> np.ndarray:
        n_objs = len(merge_entries)
        n_cams = len(camera_states)
        obj_vis = np.zeros((n_objs, n_cams), dtype=np.uint8)
        for oi, entry in enumerate(merge_entries):
            node = entry.get("node")
            category = entry.get("category", "")
            records = self._collect_bpy_mesh_records(
                node, use_ses=category in ("boxes", "doors", "windows")
            )
            if not records:
                continue
            target_objects = self._visibility_target_objects(node, records)
            for ki, (camera_obj, transparent_objects) in enumerate(camera_states):
                if self._records_visible_from_camera(
                    records, camera_obj, target_objects, transparent_objects
                ):
                    obj_vis[oi, ki] = 1
        return obj_vis

    def _ray_viewvis_for_points(
        self,
        points: np.ndarray,
        merge_entries: List[Dict[str, Any]],
        object_indices: np.ndarray,
        camera_obj,
        transparent_objects: set,
        in_frustum: np.ndarray,
    ) -> np.ndarray:
        visible = np.zeros(len(points), dtype=bool)
        if not np.any(in_frustum):
            return visible
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache: Dict[int, Tuple[set, List[Dict[str, Any]]]] = {}
        for idx in np.flatnonzero(in_frustum):
            obj_idx = int(object_indices[idx])
            if obj_idx not in cache:
                entry = merge_entries[obj_idx]
                node = entry.get("node")
                category = entry.get("category", "")
                records = self._collect_bpy_mesh_records(
                    node, use_ses=category in ("boxes", "doors", "windows")
                )
                cache[obj_idx] = (
                    self._visibility_target_objects(node, records),
                    records,
                )
            target_objects, records = cache[obj_idx]
            if self._point_visible_from_camera(
                points[idx],
                camera_obj,
                target_objects,
                transparent_objects,
                records,
                depsgraph,
            ):
                visible[idx] = True
        return visible

    def _export_mesh_viewvis_sidecars(
        self,
        view_dir: str,
        triangle_centroids: np.ndarray,
        object_indices: np.ndarray,
        merge_entries: List[Dict[str, Any]],
        camera_states: List[Tuple[Any, set]],
        frame_infos: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        meta = self._export_viewvis_sidecars_for_samples(
            view_dir,
            triangle_centroids,
            object_indices,
            merge_entries,
            camera_states,
            frame_infos,
            write_sidecars=vve.write_mesh_viewvis_sidecars,
            log_label="Mesh triangle view visibility sidecars",
        )
        return meta

    def _export_merged_viewvis_sidecars(
        self,
        view_dir: str,
        merged_points: np.ndarray,
        object_indices: np.ndarray,
        merge_entries: List[Dict[str, Any]],
        camera_states: List[Tuple[Any, set]],
        frame_infos: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return self._export_viewvis_sidecars_for_samples(
            view_dir,
            merged_points,
            object_indices,
            merge_entries,
            camera_states,
            frame_infos,
            write_sidecars=vve.write_viewvis_sidecars,
            log_label="View visibility sidecars",
        )

    def _export_viewvis_sidecars_for_samples(
        self,
        view_dir: str,
        sample_points: np.ndarray,
        object_indices: np.ndarray,
        merge_entries: List[Dict[str, Any]],
        camera_states: List[Tuple[Any, set]],
        frame_infos: List[Dict[str, Any]],
        *,
        write_sidecars: Callable[..., Dict[str, Any]],
        log_label: str,
    ) -> Optional[Dict[str, Any]]:
        if len(sample_points) == 0:
            return None

        object_visible = self._compute_object_visibility_per_camera(merge_entries, camera_states)

        def ray_visible_fn(points, camera_obj, transparent_objects, in_frustum):
            return self._ray_viewvis_for_points(
                points,
                merge_entries,
                object_indices,
                camera_obj,
                transparent_objects,
                in_frustum,
            )

        point_matrix, object_matrix, stats = vve.compute_viewvis_matrices(
            sample_points,
            object_indices,
            object_visible,
            camera_states,
            frame_infos,
            self.scene,
            ray_visible_fn=ray_visible_fn,
        )
        meta = write_sidecars(view_dir, point_matrix, object_matrix, frame_infos)
        meta.update(stats)
        print(
            f"✅ {log_label}: "
            f"{meta['viewvis_point']}, {meta['viewvis_object']} "
            f"({meta['shape'][0]} elements × {meta['shape'][1]} cameras; "
            f"depth={stats.get('depth_frames', 0)}, raycast={stats.get('raycast_frames', 0)})"
        )
        return meta

    def _triangle_rgb_at_centroid(self, tri, color_source, tri_uv=None):
        base_color = color_source.get("base_color", np.array([160, 160, 160], dtype=np.uint8))
        image = color_source.get("image")
        if image is not None and tri_uv is not None:
            uv = np.mean(np.asarray(tri_uv, dtype=float), axis=0)
            mapping = color_source.get("mapping")
            if mapping is not None:
                uv = self._apply_mapping_to_uv(uv, mapping)
            return self._sample_image_color(
                image, uv, base_color, color_source.get("base_factor")
            )
        return base_color

    def _collect_scene_triangle_colors(self):
        triangles = []
        colors = []
        for category, _object_id, node, _filename in self._iter_point_cloud_nodes(visible_suffix=False):
            records = self._collect_bpy_mesh_records(
                node, use_ses=category in ("boxes", "doors", "windows")
            )
            for record in records:
                uvs = record.get("uvs")
                for tri_idx, tri in enumerate(record["triangles"]):
                    tri_uv = uvs[tri_idx] if uvs is not None else None
                    color = self._triangle_rgb_at_centroid(
                        tri, record["color_sources"][tri_idx], tri_uv
                    )
                    triangles.append(np.asarray(tri, dtype=float))
                    colors.append(color)
        if not triangles:
            return np.empty((0, 3, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
        return np.asarray(triangles, dtype=float), np.asarray(colors, dtype=np.uint8)

    def _collect_visible_triangle_colors(self, visible_entries: List[Dict[str, Any]]):
        triangles = []
        colors = []
        for entry in visible_entries:
            for record in entry["records"]:
                uvs = record.get("uvs")
                for tri_idx, tri in enumerate(record["triangles"]):
                    tri_uv = uvs[tri_idx] if uvs is not None else None
                    color = self._triangle_rgb_at_centroid(
                        tri, record["color_sources"][tri_idx], tri_uv
                    )
                    triangles.append(np.asarray(tri, dtype=float))
                    colors.append(color)
        if not triangles:
            return np.empty((0, 3, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
        return np.asarray(triangles, dtype=float), np.asarray(colors, dtype=np.uint8)

    def _collect_visible_triangle_centroids_with_indices(
        self, visible_entries: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Triangle centroids in the same order as export_visible_glb face export."""
        centroids: List[np.ndarray] = []
        object_indices: List[int] = []
        for obj_idx, entry in enumerate(visible_entries):
            for record in entry["records"]:
                for tri in record["triangles"]:
                    tri_arr = np.asarray(tri, dtype=float)
                    centroids.append(tri_arr.mean(axis=0))
                    object_indices.append(obj_idx)
        if not centroids:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=np.int32)
        return np.vstack(centroids), np.asarray(object_indices, dtype=np.int32)

    def export_visible_voxel(
        self,
        output_dir: str,
        visible_entries: List[Dict[str, Any]],
        camera_obj,
    ):
        try:
            from . import voxel_export as ve
        except ImportError:
            import voxel_export as ve  # type: ignore
        try:
            from . import geometry_opencv as geo_cv
        except ImportError:
            import geometry_opencv as geo_cv  # type: ignore

        triangles, colors = self._collect_visible_triangle_colors(visible_entries)
        if len(triangles) == 0:
            print("⚠️ Voxel export skipped: visible geometry has no triangles")
            return

        world_up = [0.0, 0.0, 1.0]
        meta = self.context.get("meta") or {}
        if meta.get("world_up") is not None:
            world_up = [float(x) for x in meta["world_up"]]

        camera_pose = None
        if camera_obj is not None:
            camera_pose = geo_cv.camera_pose_from_matrix(np.array(camera_obj.matrix_world))

        ve.export_colored_voxel_grids(
            output_dir,
            triangles,
            colors,
            world_up=world_up,
            camera_pose=camera_pose,
        )

    def _entity_id_for_category(self, category: str, object_id: str) -> str:
        if category in ("floor", "ceiling"):
            return category
        return str(object_id)

    def _write_visibility_json(
        self,
        output_dir: str,
        records: List[Dict[str, Any]],
        width: int,
        height: int,
    ) -> None:
        try:
            from . import visibility_mask as vm
        except ImportError:
            import visibility_mask as vm  # type: ignore

        settings = vm.visibility_settings(self.config)
        payload = {
            "ratio_if_visible": settings["ratio_if_visible"],
            "min_overall_pixels": settings["min_overall_pixels"],
            "image_size": [int(width), int(height)],
            "objects": records,
        }
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "visibility.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"✅ Visibility stats: {path} ({len(records)} elements)")

    def _legacy_visibility_records_from_entries(
        self, visible_entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        records = []
        for entry in visible_entries:
            records.append({
                "category": entry["category"],
                "id": entry["id"],
                "overall_pixels": None,
                "visible_pixels": None,
                "visibility_ratio": round(float(entry.get("visibility_ratio", 1.0)), 6),
                "mostly_occluded": bool(entry.get("mostly_occluded", False)),
                "in_merged_export": bool(entry.get("in_merged_export", True)),
                "in_frustum": True,
                "export_geometry": True,
            })
        return records

    def _visible_geometry_entries(
        self,
        camera_obj,
        transparent_objects: Optional[set] = None,
        *,
        semantic_rgb: np.ndarray,
        semantic_objects: List[Dict[str, Any]],
        overall_pixels_map: Dict[Tuple[str, str], int],
        width: int,
        height: int,
    ):
        try:
            from . import planar_faces
            from . import visibility_mask as vm
        except ImportError:
            import planar_faces  # type: ignore
            import visibility_mask as vm  # type: ignore

        transparent_objects = transparent_objects or set()
        settings = vm.visibility_settings(self.config)
        color_map = vm.build_entity_color_map(semantic_objects)
        view_ctx = planar_faces.make_bpy_view_context(self.scene, camera_obj, width, height)
        clip_mats = view_ctx["clip_mats"]

        entries = []
        visibility_records: List[Dict[str, Any]] = []

        for category, object_id, node, visible_ply in self._iter_point_cloud_nodes(visible_suffix=True):
            entity_id = self._entity_id_for_category(category, object_id)
            label, asset_id = self._object_export_identity(category, object_id)
            color = color_map.get((category, entity_id))

            records = self._collect_bpy_mesh_records(node, use_ses=category in ("boxes", "doors", "windows"))
            if not records:
                visibility_records.append({
                    "category": category,
                    "id": object_id,
                    "label": label,
                    **vm.evaluate_visibility(
                        category=category,
                        in_frustum=False,
                        overall_pixels=0,
                        visible_pixels=0,
                        ratio_if_visible=settings["ratio_if_visible"],
                        min_overall_pixels=settings["min_overall_pixels"],
                    ),
                })
                continue

            clipped_records = self._clip_records_to_render_frustum(records, camera_obj, clip_mats=clip_mats)
            in_frustum = bool(clipped_records)
            if in_frustum and color is not None:
                overall_pixels = int(overall_pixels_map.get((category, entity_id), 0))
                visible_pixels = vm.count_semantic_color_pixels(semantic_rgb, color)
            else:
                overall_pixels = 0
                visible_pixels = 0
            vis = vm.evaluate_visibility(
                category=category,
                in_frustum=in_frustum,
                overall_pixels=overall_pixels,
                visible_pixels=visible_pixels,
                ratio_if_visible=settings["ratio_if_visible"],
                min_overall_pixels=settings["min_overall_pixels"],
            )
            visibility_records.append({
                "category": category,
                "id": object_id,
                "label": label,
                **vis,
            })

            if not vis["export_geometry"] or not clipped_records:
                continue

            in_view_ratio_value = self._records_frustum_in_view_ratio(records, camera_obj)
            frustum_cutted = in_view_ratio_value < self.FRUSTUM_IN_VIEW_RATIO
            output_ply = self._append_output_suffix(visible_ply, "_cutted") if frustum_cutted else visible_ply
            if vis["mostly_occluded"]:
                output_ply = vm.append_mostly_occluded_suffix(output_ply)

            entries.append({
                "category": category,
                "id": object_id,
                "node": node,
                "records": clipped_records,
                "visible_ply": output_ply,
                "frustum_cutted": frustum_cutted,
                "frustum_in_view_ratio": in_view_ratio_value,
                "visibility_ratio": vis["visibility_ratio"],
                "mostly_occluded": vis["mostly_occluded"],
                "in_merged_export": vis["in_merged_export"],
                "camera_obj": camera_obj,
            })

        return entries, visibility_records

    def _visible_geometry_entries_pano(self, camera_obj, transparent_objects: Optional[set] = None):
        """Panorama visible geometry: occlusion only, no perspective frustum clip, original coordinates."""
        transparent_objects = transparent_objects or set()
        entries = []
        for category, object_id, node, visible_ply in self._iter_point_cloud_nodes(visible_suffix=True):
            records = self._collect_bpy_mesh_records(node, use_ses=category in ("boxes", "doors", "windows"))
            if not records:
                continue
            target_objects = self._visibility_target_objects(node, records)
            if self._records_fully_occluded(records, camera_obj, target_objects, transparent_objects):
                continue
            entries.append({
                "category": category,
                "id": object_id,
                "node": node,
                "records": records,
                "visible_ply": visible_ply,
                "frustum_cutted": False,
                "frustum_in_view_ratio": 1.0,
                "visibility_ratio": 1.0,
                "mostly_occluded": False,
                "in_merged_export": True,
                "camera_obj": camera_obj,
            })
        return entries

    def _visible_geometry_entries_pano_multi(self, camera_states: List[Tuple[Any, set]]):
        """Multi-panorama: keep full records if visible from any camera; no frustum clip."""
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
            primary_camera = camera_states[0][0] if camera_states else None
            entries.append({
                "category": category,
                "id": object_id,
                "node": node,
                "records": records,
                "visible_ply": visible_ply,
                "frustum_cutted": False,
                "frustum_in_view_ratio": 1.0,
                "visibility_ratio": 1.0,
                "mostly_occluded": False,
                "in_merged_export": True,
                "camera_obj": primary_camera,
            })
        return entries

    def _visible_geometry_entries_multi(self, camera_states: List[Tuple[Any, set]]):
        """Multi-camera: keep if visible from any; frustum clip merges kept parts across cameras."""
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
                "visibility_ratio": 1.0,
                "mostly_occluded": False,
                "in_merged_export": True,
                "camera_obj": primary_camera,
            })
        return entries

    @staticmethod
    def _append_output_suffix(path: str, suffix: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}{suffix}{ext}"

    def _iter_point_cloud_nodes(self, visible_suffix: bool = False):
        suffixes = ["visible"] if visible_suffix else []
        floor_info = self.mesh_nodes.get("floor")
        if floor_info:
            filename = util_data.build_pointcloud_ply_relpath("floor", "floor", None, suffixes)
            yield "floor", "floor", floor_info.get("node"), filename
        ceiling_info = self.mesh_nodes.get("ceiling")
        if ceiling_info:
            filename = util_data.build_pointcloud_ply_relpath("ceiling", "ceiling", None, suffixes)
            yield "ceiling", "ceiling", ceiling_info.get("node"), filename
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
        """Clip triangle parts outside frustum using Blender render projection matrix."""
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
        """Triangle in frustum only if all three vertices are inside render viewport (for cutted stats)."""
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
        """Strict ratio of triangles with all three vertices in viewport; cutted only below threshold."""
        return self._records_frustum_in_view_ratio(records, camera_obj) < in_view_ratio

    def _sample_visible_records(self, records: List[Dict[str, Any]], camera_obj, sample_count: int):
        areas = np.array([record["area"] for record in records], dtype=float)
        total_area = float(np.sum(areas))
        if total_area <= 1e-12:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=float),
            )
        counts = np.floor(areas / total_area * sample_count).astype(int)
        remainder = sample_count - int(np.sum(counts))
        if remainder > 0:
            order = np.argsort(-(areas / total_area * sample_count - counts))
            counts[order[:remainder]] += 1
        sampled_points = []
        sampled_colors = []
        sampled_normals = []
        for record, count in zip(records, counts):
            if count <= 0:
                continue
            pts, cols, nrms = self._sample_triangles(
                record["triangles"],
                record["color_sources"],
                count,
                uvs=record.get("uvs"),
            )
            if len(pts) > 0:
                sampled_points.append(pts)
                sampled_colors.append(cols)
                sampled_normals.append(nrms)
        if not sampled_points:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=float),
            )
        return np.vstack(sampled_points), np.vstack(sampled_colors), np.vstack(sampled_normals)

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
        """Collect occlusion sample points. Use triangle centroids not vertices: floor/ceiling n-gon vertices lie on outer contour;
        vertex-only sampling falsely marks visible interior as occluded (same for ceiling; matches scenebuilder.py)."""
        points = []
        for record in records:
            tris = np.asarray(record["triangles"], dtype=float)
            if len(tris) == 0:
                continue
            points.extend(tris.mean(axis=1))
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
        """Visible if at least one sample point is unobstructed (partially or fully visible)."""
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
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=float),
            )

        areas = np.array([record["area"] for record in mesh_records], dtype=float)
        total_area = float(np.sum(areas))
        if total_area <= 1e-12:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=float),
            )

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
        sampled_normals = []
        for record, count in zip(mesh_records, counts):
            if count <= 0:
                continue
            pts, cols, nrms = self._sample_triangles(
                record["triangles"],
                record["color_sources"],
                count,
                uvs=record.get("uvs"),
            )
            if len(pts) > 0:
                sampled_points.append(pts)
                sampled_colors.append(cols)
                sampled_normals.append(nrms)

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
                pts, cols, nrms = self._sample_sharp_edges(record.get("sharp_edges", []), count)
                if len(pts) > 0:
                    sampled_points.append(pts)
                    sampled_colors.append(cols)
                    sampled_normals.append(nrms)

        if not sampled_points:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=float),
            )
        return np.vstack(sampled_points), np.vstack(sampled_colors), np.vstack(sampled_normals)

    def _collect_bpy_mesh_records(self, node, use_ses: bool = False, use_evaluated: bool = True):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        records = []

        def collect_mesh(obj, matrix_world):
            if obj is None or obj.type != "MESH":
                return
            evaluated_obj = None
            if use_evaluated:
                evaluated_obj = obj.evaluated_get(depsgraph)
                mesh = evaluated_obj.to_mesh()
            else:
                mesh = obj.data
            try:
                if mesh is None:
                    return
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
                if evaluated_obj is not None:
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
                "normal": np.asarray(triangle_normals[tri_idx], dtype=float),
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
            nrm = np.zeros(3, dtype=float)
            for e in entries:
                n = np.asarray(e.get("normal", [0.0, 0.0, 0.0]), dtype=float)
                nrm = nrm + n
            nrm_norm = float(np.linalg.norm(nrm))
            if nrm_norm > 1e-12:
                nrm = nrm / nrm_norm
            else:
                nrm = np.asarray(entry.get("normal", [0.0, 0.0, 1.0]), dtype=float)
            sharp_edges.append({
                "p0": p0,
                "p1": p1,
                "uv0": entry.get("uv0"),
                "uv1": entry.get("uv1"),
                "color_source": entry.get("color_source", {"base_color": np.array([160, 160, 160], dtype=np.uint8)}),
                "length": length,
                "normal": nrm,
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
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=float),
            )
        lengths = np.array([edge["length"] for edge in sharp_edges], dtype=float)
        total_length = float(np.sum(lengths))
        if total_length <= 1e-12:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=float),
            )

        edge_indices = np.random.choice(len(sharp_edges), size=sample_count, p=lengths / total_length)
        ts = np.random.random(sample_count)
        points = np.empty((sample_count, 3), dtype=float)
        colors = np.empty((sample_count, 3), dtype=np.uint8)
        normals = np.empty((sample_count, 3), dtype=float)
        for sample_idx, edge_idx in enumerate(edge_indices):
            edge = sharp_edges[edge_idx]
            t = ts[sample_idx]
            points[sample_idx] = (1.0 - t) * edge["p0"] + t * edge["p1"]
            colors[sample_idx] = self._edge_color(edge, t)
            nrm = np.asarray(edge.get("normal", [0.0, 0.0, 1.0]), dtype=float)
            nrm_norm = float(np.linalg.norm(nrm))
            normals[sample_idx] = nrm / nrm_norm if nrm_norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        return points, colors, normals

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
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=float),
            )

        tri_indices = np.random.choice(len(triangles), size=sample_count, p=areas / total_area)
        chosen = triangles[tri_indices]
        r1 = np.sqrt(np.random.random(sample_count))[:, None]
        r2 = np.random.random(sample_count)[:, None]
        points = (1.0 - r1) * chosen[:, 0] + r1 * (1.0 - r2) * chosen[:, 1] + r1 * r2 * chosen[:, 2]
        normals = self._triangle_normals(chosen)
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
        return points, colors, normals

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
    def _write_ply(path: str, points: np.ndarray, colors: np.ndarray, normals: Optional[np.ndarray] = None):
        """Write binary PLY: xyz + nx/ny/nz + rgb."""
        payload = BpySceneCtx._write_ply_binary_payload(points, colors, normals)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(payload)

    @staticmethod
    def _write_ply_binary_payload(
        points: np.ndarray,
        colors: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> bytes:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
        pts = np.asarray(points, dtype=np.float32)
        if normals is None:
            nrm = np.zeros((len(pts), 3), dtype=np.float32)
        else:
            nrm = np.asarray(normals, dtype=np.float32)
            norms = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = np.divide(nrm, norms, out=np.zeros_like(nrm), where=norms > 1e-12)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(pts)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        vertex_dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("nx", "<f4"),
                ("ny", "<f4"),
                ("nz", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        packed = np.empty(len(pts), dtype=vertex_dtype)
        packed["x"] = pts[:, 0]
        packed["y"] = pts[:, 1]
        packed["z"] = pts[:, 2]
        packed["nx"] = nrm[:, 0]
        packed["ny"] = nrm[:, 1]
        packed["nz"] = nrm[:, 2]
        packed["red"] = colors[:, 0]
        packed["green"] = colors[:, 1]
        packed["blue"] = colors[:, 2]
        return header + packed.tobytes()

    def _write_ply_with_opencv_copy(
        self,
        path: str,
        points: np.ndarray,
        colors: np.ndarray,
        normals: np.ndarray,
        camera_obj,
        *,
        quiet: bool = False,
    ):
        """Main PLY: world/normalized SSL; *_opencv.ply: OpenCV camera frame (both binary)."""
        self._write_ply(path, points, colors, normals)
        if camera_obj is None:
            return
        try:
            from . import geometry_opencv as geo_cv
        except ImportError:
            import geometry_opencv as geo_cv  # type: ignore
        pose = geo_cv.camera_pose_from_matrix(np.array(camera_obj.matrix_world))
        opencv_path = geo_cv.opencv_duplicate_path(path)
        pts_o = geo_cv.transform_points_to_opencv(points, pose)
        nrm_o = geo_cv.transform_normals_to_opencv(normals, pose)
        self._write_ply(opencv_path, pts_o, colors, nrm_o)
        if not quiet:
            print(f"✅ OpenCV point cloud copy: {opencv_path}")

    def _export_glb_opencv_copy(self, output_path: str, camera_obj, bpy_objects=None) -> None:
        """Build OpenCV GLB copy from exported world GLB (trimesh transform, avoids second Blender glTF export)."""
        if camera_obj is None or not os.path.isfile(output_path):
            return
        try:
            from . import geometry_opencv as geo_cv
        except ImportError:
            import geometry_opencv as geo_cv  # type: ignore
        pose = geo_cv.camera_pose_from_matrix(np.array(camera_obj.matrix_world))
        geo_cv.export_glb_opencv_copy(output_path, pose)

    def setup_lighting(self, intensity: float = 250,
                      lighting_type: Literal["area", "array", "none"] = "array",
                      ambient_light_color: list = None,
                      ambient_strength: float = 2.0):
        """Set scene lighting: single large area light, array lights, or none"""
        if self.if_set_lights:
            print("⚠️  Lighting already set, skipping duplicate setup")
            return
        
        if lighting_type == "none":
            print("🌑 'none' mode selected, no artificial lights added")
            self.if_set_lights = True
            return

        center = self.context["meta"]["center"]
        z_max = self.context["meta"]["z_max"]
        span = self.context["meta"]["span"]
        
        # Soft warm white
        warm_white = [1.0, 0.95, 0.8]

        # 1. Set ambient/world lighting
        world = self.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            self.scene.world = world
        
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            if ambient_light_color is None:
                # Natural ambient light often has a slight daylight blue tint
                ambient_light_color = [0.95, 0.97, 1.0, 1.0]
            elif len(ambient_light_color) == 3:
                ambient_light_color = ambient_light_color + [1.0]
            bg_node.inputs['Color'].default_value = ambient_light_color
            bg_node.inputs['Strength'].default_value = ambient_strength
        
        # 2. Add indoor artificial lights (height at z_max - 0.1)
        light_z = z_max - 0.1
        
        # Photographic indoor warm white (~3500K)
        # Contrast with bluish ambient light yields realistic indoor mood
        realistic_warm = [1.0, 0.88, 0.75]
        
        if lighting_type == "area":
            # Option 1: single large area light
            light_data = bpy.data.lights.new(name="MainAreaLight", type='AREA')
            light_data.shape = 'RECTANGLE'
            # Cover 80% of room area
            light_data.size = span[0] * 0.8
            light_data.size_y = span[1] * 0.8
            light_data.energy = intensity
            light_data.color = realistic_warm
            
            light_obj = bpy.data.objects.new(name="MainAreaLight", object_data=light_data)
            self.scene_collection.objects.link(light_obj)
            light_obj.location = (center[0], center[1], light_z)
            print(f"💡 Added single large area light (Intensity={intensity})")
            
        elif lighting_type == "array":
            # Option 2: array area lights (downlight style)
            # One every 1.5m, inset from edges
            nx = int(span[0] / 1.5)
            ny = int(span[1] / 1.5)
            
            # Compute X coordinates (centered layout)
            if nx <= 1:
                x_coords = [center[0]]
                nx = 1
            else:
                total_w = (nx - 1) * 1.5
                x_coords = np.linspace(center[0] - total_w/2, center[0] + total_w/2, nx)
                
            # Compute Y coordinates (centered layout)
            if ny <= 1:
                y_coords = [center[1]]
                ny = 1
            else:
                total_h = (ny - 1) * 1.5
                y_coords = np.linspace(center[1] - total_h/2, center[1] + total_h/2, ny)
            
            # Constant per-light intensity so larger rooms get proportionally brighter
            each_intensity = self.config.get("array_light_intensity", 50.0)
            count = 0
            for x in x_coords:
                for y in y_coords:
                    count += 1
                    l_name = f"ArrayLight_{count}"
                    l_data = bpy.data.lights.new(name=l_name, type='AREA')
                    l_data.shape = 'DISK'
                    l_data.size = 0.3  # 0.3m diameter disk
                    l_data.energy = each_intensity
                    l_data.color = realistic_warm
                    
                    l_obj = bpy.data.objects.new(name=l_name, object_data=l_data)
                    self.scene_collection.objects.link(l_obj)
                    l_obj.location = (x, y, light_z)
            print(f"💡 Added dense array area lights x{count} (Each Intensity={each_intensity}, Total={each_intensity*count})")
        
        self.if_set_lights = True

    def _semantic_outputs(self, output_path: str):
        base = os.path.splitext(os.path.basename(output_path))[0]
        out_dir = os.path.dirname(output_path)
        return (
            os.path.join(out_dir, f"{base}_semantic.png"),
            os.path.join(out_dir, f"{base}_semantic.json"),
        )

    def _semantic_color(self, color_key: str, used=None):
        return util.semantic_entity_color(color_key, used=used)

    def _semantic_material(self, color_key: str, color_rgb=None):
        mat_name = f"Semantic_{color_key.replace(':', '_').replace(' ', '_')}"
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(mat_name)
            mat.use_nodes = True
            mat.use_backface_culling = False
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
        mat.use_backface_culling = False
        color = color_rgb if color_rgb is not None else self._semantic_color(color_key)
        if emission is not None:
            emission.inputs["Color"].default_value = [c / 255.0 for c in color] + [1.0]
            emission.inputs["Strength"].default_value = 1.0
        return mat

    def _iter_semantic_scene_nodes(self):
        """Yield scene-level render nodes for semantic pass (never asset master meshes)."""

        def emit(category, color_key, node, object_meta):
            if node is not None:
                yield color_key, node, object_meta

        used_colors = set()
        for category in ("floor", "ceiling"):
            info = self.mesh_nodes.get(category)
            if info:
                object_id = category
                color_key = f"entity:{category}:{object_id}"
                color = list(util.semantic_entity_color(color_key, used=used_colors))
                yield from emit(
                    category,
                    color_key,
                    info.get("node"),
                    {
                        "category": category,
                        "entity_id": object_id,
                        "label": object_id,
                        "color": color,
                    },
                )
        for category in ("walls", "doors", "windows"):
            for object_id, info in self.mesh_nodes.get(category, {}).items():
                color_key = f"entity:{category}:{object_id}"
                color = list(util.semantic_entity_color(color_key, used=used_colors))
                detail_key = {"walls": "wall_data", "doors": "door_data", "windows": "window_data"}[category]
                data = info.get(detail_key, {})
                caption = data.get("caption")
                object_meta = {
                    "category": category,
                    "entity_id": object_id,
                    "label": object_id,
                    "color": color,
                }
                if caption:
                    object_meta["caption"] = caption
                yield from emit(
                    category,
                    color_key,
                    info.get("node"),
                    {
                        "category": category,
                        "entity_id": object_id,
                        "label": object_id,
                        "color": color,
                    },
                )
        for object_id, info in self.mesh_nodes.get("boxes", {}).items():
            data = info.get("box_data", {})
            label = data.get("label", data.get("class", object_id))
            color_key = f"entity:boxes:{object_id}"
            color = list(util.semantic_entity_color(color_key, used=used_colors))
            object_meta = {
                "category": "boxes",
                "entity_id": object_id,
                "label": label,
                "color": color,
            }
            caption = data.get("caption")
            if caption:
                object_meta["caption"] = caption
            yield from emit("boxes", color_key, info.get("node"), object_meta)

    def _create_semantic_proxies(self, node, color_key: str, color_rgb=None):
        """Create render-only mesh copies with semantic materials; originals stay untouched."""
        semantic_mat = self._semantic_material(color_key, color_rgb=color_rgb)
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
            obj_key = (obj.get("category"), obj.get("label"))
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
        backup = {"lights": [], "world_bg": None, "cycles": None, "eevee": None, "render": None}
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

        # Disable AA/denoising so semantic PNG pixels match JSON colors exactly
        cycles = getattr(self.scene, "cycles", None)
        if cycles is not None:
            backup["cycles"] = {
                "samples": int(cycles.samples),
                "filter_width": float(getattr(cycles, "filter_width", 1.5)),
                "use_denoising": bool(getattr(cycles, "use_denoising", False)),
            }
            cycles.samples = 1
            if hasattr(cycles, "filter_width"):
                cycles.filter_width = 0.01
            if hasattr(cycles, "use_denoising"):
                cycles.use_denoising = False
        eevee = getattr(self.scene, "eevee", None)
        if eevee is not None:
            eevee_backup: Dict[str, Any] = {}
            if hasattr(eevee, "taa_render_samples"):
                eevee_backup["taa_render_samples"] = int(eevee.taa_render_samples)
                eevee.taa_render_samples = 1
            for attr in ("use_bloom", "use_ssr", "use_ssr_refraction", "use_gtao"):
                if hasattr(eevee, attr):
                    eevee_backup[attr] = bool(getattr(eevee, attr))
                    setattr(eevee, attr, False)
            if eevee_backup:
                backup["eevee"] = eevee_backup

        render = self.scene.render
        render_backup: Dict[str, Any] = {}
        if hasattr(render, "dither_intensity"):
            render_backup["dither_intensity"] = float(render.dither_intensity)
            render.dither_intensity = 0.0
        if render_backup:
            backup["render"] = render_backup
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
        cycles_bak = backup.get("cycles")
        cycles = getattr(self.scene, "cycles", None)
        if cycles is not None and cycles_bak:
            cycles.samples = cycles_bak["samples"]
            if hasattr(cycles, "filter_width"):
                cycles.filter_width = cycles_bak["filter_width"]
            if hasattr(cycles, "use_denoising"):
                cycles.use_denoising = cycles_bak["use_denoising"]
        eevee_bak = backup.get("eevee")
        eevee = getattr(self.scene, "eevee", None)
        if eevee is not None and eevee_bak:
            if "taa_render_samples" in eevee_bak and hasattr(eevee, "taa_render_samples"):
                eevee.taa_render_samples = eevee_bak["taa_render_samples"]
            for attr, value in eevee_bak.items():
                if attr == "taa_render_samples":
                    continue
                if hasattr(eevee, attr):
                    setattr(eevee, attr, value)
        render_bak = backup.get("render")
        render = self.scene.render
        if render_bak:
            if "dither_intensity" in render_bak and hasattr(render, "dither_intensity"):
                render.dither_intensity = render_bak["dither_intensity"]

    @contextmanager
    def _semantic_render_engine(self):
        """Semantic / isolated mask rendering: EEVEE for perspective; equirectangular panorama requires Cycles."""
        render = self.scene.render
        prev_engine = render.engine
        if self._is_equirectangular_camera(self.scene):
            cycles_engine = "CYCLES"
            if prev_engine != cycles_engine:
                print("🎨 Panoramic semantic rendering uses Cycles (EEVEE does not support EQUIRECTANGULAR projection)")
                render.engine = cycles_engine
            try:
                yield
            finally:
                render.engine = prev_engine
            return

        eevee_engine = "BLENDER_EEVEE"
        if prev_engine != eevee_engine:
            render.engine = eevee_engine
        try:
            yield
        finally:
            render.engine = prev_engine

    def _render_resolution(self) -> Tuple[int, int]:
        render = self.scene.render
        pct = float(render.resolution_percentage) / 100.0
        width = max(int(render.resolution_x * pct), 1)
        height = max(int(render.resolution_y * pct), 1)
        return width, height

    def _load_semantic_bundle_from_disk(
        self,
        output_path: str,
    ) -> Optional[Tuple[np.ndarray, List[Dict[str, Any]], Dict[Tuple[str, str], int]]]:
        try:
            from . import visibility_mask as vm
        except ImportError:
            import visibility_mask as vm  # type: ignore

        view_dir = os.path.dirname(output_path) or "."
        semantic_path, metadata_path = self._semantic_outputs(output_path)
        index_path = vm.semantic_masks_index_path(view_dir)
        if not os.path.isfile(semantic_path) or not os.path.isfile(index_path):
            return None

        semantic_rgb = util.read_image_array(semantic_path)
        if semantic_rgb.ndim == 2:
            semantic_rgb = np.stack([semantic_rgb] * 3, axis=-1)
        semantic_rgb = np.asarray(semantic_rgb[..., :3], dtype=np.uint8)
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            objects = meta.get("objects", [])
        else:
            objects = []
        overall_pixels_map = vm.load_overall_pixels_map(view_dir)
        return semantic_rgb, objects, overall_pixels_map

    def _render_semantic_to_array(
        self,
        temp_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Render full-scene semantic image (caller should have switched to EEVEE)."""
        import imageio
        import tempfile

        temp_dir = temp_dir or "."
        os.makedirs(temp_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix=".scenebuilder_sem_", dir=temp_dir)
        os.close(fd)

        render = self.scene.render
        render_backup = (render.filepath, render.image_settings.file_format, render.film_transparent)
        objects: List[Dict[str, Any]] = []
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

            for _color_key, node, object_meta in self._iter_semantic_scene_nodes():
                objects.append(dict(object_meta))
                temp_proxies.extend(
                    self._create_semantic_proxies(node, _color_key, color_rgb=object_meta.get("color"))
                )
                node.hide_render = True
                hidden_nodes.append(node)

            render.filepath = tmp_path
            render.image_settings.file_format = "PNG"
            render.film_transparent = False
            bpy.ops.render.render(write_still=True)
            semantic_rgb = util.read_image_array(tmp_path)
            if semantic_rgb.ndim == 2:
                semantic_rgb = np.stack([semantic_rgb] * 3, axis=-1)
            semantic_rgb = np.asarray(semantic_rgb[..., :3], dtype=np.uint8)
            util.attach_semantic_bbox_2d(objects, semantic_rgb)
            return semantic_rgb, objects
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
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _render_isolated_semantic_mask(
        self,
        target_node,
        color: Tuple[int, int, int],
        mask_path: str,
        temp_dir: str,
    ) -> int:
        """Render target_node only, save binary mask, return pixel count."""
        if target_node is None or color is None:
            return 0
        try:
            from . import visibility_mask as vm
        except ImportError:
            import visibility_mask as vm  # type: ignore

        import imageio
        import tempfile

        os.makedirs(os.path.dirname(mask_path) or ".", exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix=".scenebuilder_iso_sem_", dir=temp_dir)
        os.close(fd)

        render = self.scene.render
        render_backup = (render.filepath, render.image_settings.file_format, render.film_transparent)
        hidden_nodes = []
        temp_proxies = []
        env_backup = None
        view_backup = None
        rendered = False

        try:
            env_backup = self._begin_semantic_render_env()
            vs = self.scene.view_settings
            view_backup = (vs.view_transform, vs.look, float(vs.exposure), float(vs.gamma))
            vs.view_transform = "Raw"
            vs.look = "None"
            vs.exposure = 0.0
            vs.gamma = 1.0

            for color_key, node, object_meta in self._iter_semantic_scene_nodes():
                if node is None:
                    continue
                node.hide_render = True
                hidden_nodes.append(node)
                if node is not target_node:
                    continue
                temp_proxies.extend(
                    self._create_semantic_proxies(
                        node, color_key, color_rgb=object_meta.get("color")
                    )
                )
                rendered = True

            if not rendered:
                return 0

            render.filepath = tmp_path
            render.image_settings.file_format = "PNG"
            render.film_transparent = False
            bpy.ops.render.render(write_still=True)
            semantic_rgb = util.read_image_array(tmp_path)
            if semantic_rgb.ndim == 2:
                semantic_rgb = np.stack([semantic_rgb] * 3, axis=-1)
            semantic_rgb = np.asarray(semantic_rgb[..., :3], dtype=np.uint8)
            mask = vm.binary_mask_from_isolated_semantic(
                semantic_rgb,
                target_color=color,
                background=util.SEMANTIC_BACKGROUND,
            )
            imageio.imwrite(mask_path, (mask.astype(np.uint8) * 255))
            return int(np.count_nonzero(mask))
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
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def render_semantic_bundle(
        self,
        output_path: str,
        *,
        save_artifacts: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[Tuple[str, str], int]]:
        """Full-scene semantic + per-object isolated masks; EEVEE for perspective, Cycles for equirectangular panorama."""
        try:
            from . import visibility_mask as vm
        except ImportError:
            import visibility_mask as vm  # type: ignore

        import imageio

        cache_key = os.path.abspath(output_path)
        if cache_key in self._semantic_view_cache:
            return self._semantic_view_cache[cache_key]

        loaded = self._load_semantic_bundle_from_disk(output_path)
        if loaded is not None:
            self._semantic_view_cache[cache_key] = loaded
            return loaded

        view_dir = os.path.dirname(output_path) or "."
        masks_dir = vm.semantic_masks_dir(view_dir)
        os.makedirs(masks_dir, exist_ok=True)

        with self._semantic_render_engine():
            semantic_rgb, objects = self._render_semantic_to_array(temp_dir=view_dir)
            overall_pixels_map: Dict[Tuple[str, str], int] = {}
            index_objects: List[Dict[str, Any]] = []
            for _color_key, node, object_meta in self._iter_semantic_scene_nodes():
                category = str(object_meta.get("category", ""))
                entity_id = object_meta.get("entity_id")
                if entity_id is None:
                    entity_id = object_meta.get("label")
                entity_id = str(entity_id)
                color = tuple(int(c) for c in object_meta.get("color", (0, 0, 0)))
                mask_path = vm.semantic_mask_path(view_dir, category, entity_id)
                overall_pixels = self._render_isolated_semantic_mask(
                    node, color, mask_path, temp_dir=view_dir
                )
                overall_pixels_map[(category, entity_id)] = overall_pixels
                index_objects.append({
                    "category": category,
                    "id": entity_id,
                    "label": object_meta.get("label", entity_id),
                    "overall_pixels": int(overall_pixels),
                    "mask": vm.semantic_mask_relpath(category, entity_id),
                })

        with open(vm.semantic_masks_index_path(view_dir), "w", encoding="utf-8") as f:
            json.dump({"objects": index_objects}, f, indent=2, ensure_ascii=False)

        if save_artifacts:
            semantic_path, metadata_path = self._semantic_outputs(output_path)
            imageio.imwrite(semantic_path, semantic_rgb)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._semantic_metadata(objects), f, indent=2, ensure_ascii=False)
            bbox_overlay_path = util.save_bbox_2d_overlay_png(output_path, objects)
            if bbox_overlay_path:
                print(f"✅ Bounding box visualization: {bbox_overlay_path}")
            print(f"✅ Semantic map exported: {semantic_path}")
            print(f"✅ Semantic masks exported: {masks_dir} ({len(index_objects)} items)")

        result = (semantic_rgb, objects, overall_pixels_map)
        self._semantic_view_cache[cache_key] = result
        return result

    def render_semantic_png(self, output_path: str):
        self.render_semantic_bundle(output_path, save_artifacts=True)

    def normalized_topdown_view(
        self,
        output_dir: str,
        geometry_mode: str = "gltf",
        show_wall: bool = True,
        show_window: bool = True,
        show_door: bool = True,
        show_ceiling: bool = False,
        up_vector: list = None,
        auto_transparent: bool = True,
        transparent_alpha: float = 0.0,
        use_HDRI: bool = False,
        hdri_transparent_background: bool = True,
        visible_shadow: bool = True,
        lighting_type: Literal["area", "array", "none"] = "array",
        align_height: bool = True,
        round_decimals: int = 2,
        render_depth: bool = False,
        render_semantic: bool = False,
        align: Optional[Dict[str, Any]] = None,
        write_ssl: bool = True,
        **kwargs,
    ):
        """Pixel-aligned top-down view: translate SSL so image top-left maps to floor (0,0); outputs topdown.png / camera_para.json.

        Resolution is fixed at 1000×1000 (SpatialFactory pixel-align convention); not configurable.
        Skip second translation when align is provided (scene already in pixel-aligned SSL); no ssl.txt when write_ssl=False.
        """
        width = util_data.NORMALIZED_TOPDOWN_WIDTH
        height = util_data.NORMALIZED_TOPDOWN_HEIGHT
        for key in (
            "export_glb", "export_point_cloud", "export_voxel", "visible_geometry",
            "rebuild", "auto_fov", "manual_fov", "glb_path",
        ):
            kwargs.pop(key, None)

        os.makedirs(output_dir, exist_ok=True)
        png_path = os.path.join(output_dir, "topdown.png")
        ssl_path = os.path.join(output_dir, "ssl.txt")
        camera_para_path = os.path.join(output_dir, "camera_para.json")

        align = util_data.prepare_pixel_aligned_topdown_context(
            self.context,
            round_decimals=round_decimals,
        ) if align is None else align
        exclude_box_ids = util.resolve_topdown_exclude_box_ids(
            self.context, self.config, enabled=True, log_prefix="top-down view"
        )
        if write_ssl:
            util_data.write_standard_ssl_to_path(
                self.context, ssl_path, exclude_box_ids=exclude_box_ids
            )

        self.clear_scene()
        self.construct_scene(
            geometry_mode=geometry_mode,
            show_wall=show_wall,
            show_window=show_window,
            show_door=show_door,
            show_ceiling=show_ceiling,
            align_height=align_height,
            rebuild=True,
            exclude_box_ids=exclude_box_ids,
        )

        for wall_info in self.mesh_nodes["walls"].values():
            wall_obj = wall_info.get("node")
            if wall_obj:
                wall_obj.visible_shadow = visible_shadow
        ceiling_info = self.mesh_nodes.get("ceiling")
        if ceiling_info and ceiling_info.get("node"):
            ceiling_info["node"].visible_shadow = visible_shadow

        if not self.if_set_lights:
            self.setup_lighting(lighting_type=lighting_type, ambient_light_color=[1.0, 1.0, 1.0], ambient_strength=3.0)

        self.scene.render.film_transparent = hdri_transparent_background
        if use_HDRI:
            hdri_path = self.config.get("hdri_path")
            if hdri_path and util_bpy.apply_hdri_to_world(self.scene, hdri_path, strength=1.0):
                print(f"🌇 Using HDRI environment lighting: {hdri_path}")

        camera_position = np.array(align["camera_position_ssl"], dtype=float)
        look_at_target = np.array(align["look_at_target_ssl"], dtype=float)
        up_vector = np.array(up_vector if up_vector else [0.0, 1.0, 0.0], dtype=float)
        up_norm = np.linalg.norm(up_vector)
        up_vector = up_vector / up_norm if up_norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=float)
        fov_y = float(align["fov_y"])

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

        camera_data = bpy.data.cameras.new(name=f"NormTopdownCam_{id(self)}")
        camera_obj = bpy.data.objects.new(f"NormTopdownCam_{id(self)}", camera_data)
        self.scene_collection.objects.link(camera_obj)
        self.scene.camera = camera_obj
        camera_obj.matrix_world = Matrix((
            (float(right[0]), float(up[0]), float(-forward[0]), float(camera_position[0])),
            (float(right[1]), float(up[1]), float(-forward[1]), float(camera_position[1])),
            (float(right[2]), float(up[2]), float(-forward[2]), float(camera_position[2])),
            (0.0, 0.0, 0.0, 1.0),
        ))
        camera_data.type = "PERSP"
        camera_data.lens_unit = "FOV"
        camera_data.angle = fov_y

        z_max = self.context["meta"]["z_max"]
        wall_transparency_records = {}
        object_transparency_records = []
        if auto_transparent:
            transparent_wall_ids = util.find_walls_to_make_transparent(
                camera_position.tolist()[:2],
                look_at_target.tolist()[:2],
                self.context["meta"]["vertices"],
                self.context["walls"],
            )
            for wall_id in transparent_wall_ids:
                wall_obj = self.mesh_nodes["walls"].get(wall_id, {}).get("node")
                replacements = util_bpy.apply_wall_transparency(wall_obj, transparent_alpha)
                if replacements:
                    wall_transparency_records[wall_id] = replacements
            camera_z = camera_position[2]
            if camera_z > z_max:
                ceiling_info = self.mesh_nodes.get("ceiling") or {}
                ceiling_node = ceiling_info.get("node")
                if ceiling_node:
                    record = util_bpy.apply_object_transparency(ceiling_node, transparent_alpha)
                    if record:
                        object_transparency_records.append(record)

        self.scene.render.resolution_x = width
        self.scene.render.resolution_y = height
        self.scene.render.filepath = png_path
        self.scene.render.image_settings.color_mode = "RGBA"

        transparent_objects = {
            self.mesh_nodes["walls"].get(wid, {}).get("node") for wid in wall_transparency_records
        }
        transparent_objects.update(record.get("object") for record in object_transparency_records)
        transparent_objects = {obj for obj in transparent_objects if obj is not None}

        depth_scale = None
        try:
            print(f"🎬 Pixel-aligned top-down view rendering ({width}x{height})...")
            if render_depth:
                depth_path = os.path.join(output_dir, "topdown_depth.png")
                depth_scale = util_bpy.render_color_and_depth_png(
                    self.scene, png_path, depth_path, skip_objects=transparent_objects
                )
            else:
                bpy.ops.render.render(write_still=True)
            if render_semantic or visible_geometry:
                self.render_semantic_bundle(png_path, save_artifacts=True)
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

        camera_para = dict(align["camera_para"])
        if depth_scale is not None:
            camera_para["depth_scale"] = float(depth_scale)
            camera_para["depth_unit"] = "meter"
            camera_para["is_metric_depth"] = True
            camera_para.update(util.normal_map_camera_para_fields())
        camera_para.update(
            util.camera_calibration_matrix_fields(
                align["camera_position_ssl"],
                align["look_at_target_ssl"],
                [0.0, 0.0, 1.0],
                align["fov_y"],
                width,
                height,
                aspect_ratio=float(width) / float(height),
            )
        )
        with open(camera_para_path, "w", encoding="utf-8") as f:
            json.dump(camera_para, f, indent=4)
        self.write_opencv_ssl_for_view(
            output_dir,
            align["camera_position_ssl"],
            align["look_at_target_ssl"],
            [0.0, 0.0, 1.0],
            exclude_box_ids=exclude_box_ids,
        )
        print(f"✅ Pixel-aligned top-down view complete: {output_dir}")

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
                     export_voxel: bool = False,
                     visible_geometry: bool = False,
                     render_semantic: bool = False,
                     export_planar_faces: Optional[bool] = None,
                     export_visible_point_cloud: Optional[bool] = None,
                     skip_render: bool = False,
                     write_ssl: bool = True,
                     write_camera_para: bool = True):
        """Top-down render: when output_path is a directory, write single frame and artifacts to {output}/topdown/topdown.png."""
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

        exclude_box_ids = util.resolve_topdown_exclude_box_ids(
            self.context, self.config, enabled=True
        )

        with util_data.ViewSslSession(self, False) as vss:
            vss.setup(world_cam_w, world_look_w, world_up_raw)

            construct_time = 0
            if rebuild or self.mesh_nodes["floor"] is None:
                construct_start = time.perf_counter()
                self.construct_scene(
                    geometry_mode=geometry_mode,
                    show_wall=show_wall,
                    show_window=show_window,
                    show_door=show_door,
                    show_ceiling=show_ceiling,
                    align_height=align_height,
                    rebuild=rebuild,
                    exclude_box_ids=exclude_box_ids,
                )
                construct_time = time.perf_counter() - construct_start

            # Set shadow visibility
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
                    print(f"🌇 Using HDRI environment lighting: {hdri_path}")
                else:
                    print(f"⚠️ HDRI file unavailable or not configured: {hdri_path}")

            center = self.context["meta"]["center"]
            span = self.context["meta"]["span"]
            z_max = self.context["meta"]["z_max"]
            bounds = self.context["meta"]["bounds"]

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
                    print(f"🔍 Detected {len(transparent_wall_ids)} occluding walls, setting to transparent...")
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

            print(f"🎬 Rendering ({width}x{height})..." if not skip_render else "⏭️  Skipping Cycles render (main frame output already exists)")
            depth_scale = None
            transparent_objects = {
                self.mesh_nodes["walls"].get(wall_id, {}).get("node")
                for wall_id in wall_transparency_records
            }
            transparent_objects.update(record.get("object") for record in object_transparency_records)
            transparent_objects = {obj for obj in transparent_objects if obj is not None}
            do_visible_ply = visible_geometry and export_point_cloud
            if export_visible_point_cloud is not None:
                do_visible_ply = bool(export_visible_point_cloud)
            do_planar = export_point_cloud if export_planar_faces is None else bool(export_planar_faces)
            try:
                render_start = time.perf_counter()
                if skip_render:
                    render_time = 0.0
                    para_path = self._camera_para_path(output_path)
                    if os.path.isfile(para_path):
                        try:
                            with open(para_path, "r", encoding="utf-8") as f:
                                depth_scale = json.load(f).get("depth_scale")
                        except (json.JSONDecodeError, OSError):
                            depth_scale = None
                elif render_depth:
                    depth_path = os.path.join(
                        os.path.dirname(output_path),
                        f"{os.path.splitext(os.path.basename(output_path))[0]}_depth.png",
                    )
                    depth_scale = util_bpy.render_color_and_depth_png(
                        self.scene, output_path, depth_path, skip_objects=transparent_objects
                    )
                    render_time = time.perf_counter() - render_start
                else:
                    bpy.ops.render.render(write_still=True)
                    render_time = time.perf_counter() - render_start
                if skip_render and visible_geometry:
                    skipped_geom = []
                    if not export_glb:
                        skipped_geom.append("GLB")
                    if not do_visible_ply:
                        skipped_geom.append("point cloud")
                    if not export_voxel:
                        skipped_geom.append("voxel")
                    if skipped_geom:
                        print(f"⏭️  Skipping visible geometry export (already exists): {', '.join(skipped_geom)}")
                if render_semantic or visible_geometry:
                    self.render_semantic_bundle(output_path, save_artifacts=True)
                if visible_geometry and (export_glb or do_visible_ply or export_voxel):
                    self.export_visible_geometry(
                        os.path.dirname(output_path),
                        camera_obj,
                        export_glb=export_glb,
                        export_point_cloud=do_visible_ply,
                        export_voxel=export_voxel,
                        transparent_objects=transparent_objects,
                        view_image_path=output_path,
                    )
                if do_planar:
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
            if write_camera_para:
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
                    width=width,
                    height=height,
                )
                with open(para_path, 'w') as f:
                    json.dump(camera_para, f, indent=4)
            if write_ssl:
                self.write_opencv_ssl_for_view(
                    view_dir, world_cam_w, world_look_w, vss.world_up,
                    exclude_box_ids=exclude_box_ids,
                )
            save_time = time.perf_counter() - save_start

            total_time = time.perf_counter() - total_start
            print(f"✅ Render complete! Saved to: {output_path}")
            print(f"   Build: {construct_time:.2f}s, Setup: {setup_time:.2f}s, Render: {render_time:.2f}s, Save: {save_time:.2f}s, Total: {total_time:.2f}s")

    @staticmethod
    def _is_nested_point_list(values) -> bool:
        """True for nested list [[x,y,z], ...]; single [x,y,z] returns False."""
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            return False
        first = values[0]
        if isinstance(first, np.ndarray):
            return first.ndim > 0 and len(first) >= 2
        return isinstance(first, (list, tuple)) and len(first) >= 2

    def _is_camera_sequence(self, values) -> bool:
        return self._is_nested_point_list(values)

    def _is_view_sequence(self, camera_position, look_at_target=None, up_vector=None) -> bool:
        """Sequence rendering when camera / look_at / up is a nested list on any side."""
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

    def _resolve_view_output_path(
        self,
        output_path: str,
        view_dir_name: Optional[str] = None,
        *,
        skip_render: bool = False,
    ) -> str:
        """Single camera: when output_path is a directory, use {view_dir_name or dir_stamp}/{image_stamp}.png."""
        if self._is_image_output_path(output_path):
            return output_path
        if skip_render and view_dir_name:
            try:
                from .render_resume import list_primary_frames
            except ImportError:
                from render_resume import list_primary_frames  # type: ignore
            view_dir = os.path.join(output_path, view_dir_name)
            frames = list_primary_frames(view_dir)
            if frames:
                return frames[0]
        return util.resolve_view_image_path(output_path, view_dir_name=view_dir_name)

    def _resolve_topdown_output_path(self, output_path: str) -> str:
        """Top-down: when output_path is a directory, write single frame to {output}/topdown/topdown.png."""
        if self._is_image_output_path(output_path):
            return output_path
        return util.resolve_topdown_image_path(output_path)

    @staticmethod
    def _camera_para_path(output_path: str) -> str:
        """Same basename prefix as main image: {image_basename}_camera_para.json"""
        view_dir = os.path.dirname(output_path) or "."
        base = os.path.splitext(os.path.basename(output_path))[0]
        return os.path.join(view_dir, f"{base}_camera_para.json")

    def write_opencv_ssl_for_view(
        self,
        output_dir: str,
        ref_camera,
        ref_look_at,
        ref_world_up=None,
        *,
        exclude_box_ids=None,
    ) -> str:
        """Export ``ssl_opencv.txt`` (OpenCV camera frame; reference camera ref_*)."""
        from . import ssl_opencv

        return ssl_opencv.write_opencv_ssl(
            self.context,
            output_dir,
            ref_camera,
            ref_look_at,
            ref_world_up if ref_world_up is not None else [0.0, 0.0, 1.0],
            exclude_box_ids=exclude_box_ids,
        )

    def _expand_camera_sequence_args(
        self,
        camera_position,
        look_at_target=None,
        up_vector=None,
    ) -> Tuple[list, list, list]:
        """Expand sequence args: nested list sets frame count; flat [x,y,z] broadcasts (many-to-one / one-to-many / one-to-one)."""
        center = self.context["meta"]["center"]
        z_max = self.context["meta"]["z_max"]
        default_look_at = [center[0], center[1], z_max / 2]

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
            raise ValueError("Sequence mode requires at least one of camera_position / look_at_target / up_vector to be a nested list")

        if len(set(seq_lengths)) != 1:
            raise ValueError(
                f"Nested lists for camera_position / look_at_target / up_vector must have the same length: {seq_lengths}"
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
            ups = [
                geo_cv.resolve_render_view_up_vector(cam, look)
                for cam, look in zip(cam_positions, look_ats)
            ]
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
        up_vector = geo_cv.resolve_render_view_up_vector(
            camera_position, look_at_target, up_vector
        )

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

    def _create_render_camera(self, camera_matrix, fov_y: float, width: int, height: int, panoramic: bool = False):
        camera_data = bpy.data.cameras.new(name="Camera")
        camera_obj = bpy.data.objects.new("Camera", camera_data)
        self.scene_collection.objects.link(camera_obj)
        prev_camera = self.scene.camera
        self.scene.camera = camera_obj
        camera_obj.matrix_world = camera_matrix
        if panoramic:
            camera_data.type = 'PANO'
            pano_settings = camera_data if hasattr(camera_data, "panorama_type") else getattr(camera_data, "cycles", None)
            if pano_settings is None:
                raise RuntimeError("Current Blender camera does not support panorama settings, cannot export equirectangular panorama")
            pano_settings.panorama_type = 'EQUIRECTANGULAR'
            if getattr(pano_settings, "panorama_type", None) != 'EQUIRECTANGULAR':
                raise RuntimeError("Failed to set Blender panorama camera to EQUIRECTANGULAR")
            if hasattr(pano_settings, "longitude_min"):
                pano_settings.longitude_min = -np.pi
            if hasattr(pano_settings, "longitude_max"):
                pano_settings.longitude_max = np.pi
            if hasattr(pano_settings, "latitude_min"):
                pano_settings.latitude_min = -np.pi / 2.0
            if hasattr(pano_settings, "latitude_max"):
                pano_settings.latitude_max = np.pi / 2.0
        else:
            camera_data.type = 'PERSP'
            camera_data.lens_unit = 'FOV'
            camera_data.angle = float(fov_y)
        self.scene.render.resolution_x = width
        self.scene.render.resolution_y = height
        return camera_obj, camera_data, prev_camera

    @staticmethod
    def _is_equirectangular_camera(scene=None) -> bool:
        """Whether scene.camera is a Cycles equirectangular panorama camera."""
        scene = scene or bpy.context.scene
        camera_obj = getattr(scene, "camera", None)
        camera_data = getattr(camera_obj, "data", None) if camera_obj is not None else None
        if camera_data is None or getattr(camera_data, "type", None) != "PANO":
            return False
        pano_settings = (
            camera_data
            if hasattr(camera_data, "panorama_type")
            else getattr(camera_data, "cycles", None)
        )
        return getattr(pano_settings, "panorama_type", None) == "EQUIRECTANGULAR"

    @staticmethod
    def _pano_output_path(output_path: str) -> str:
        view_dir = os.path.dirname(output_path) or "."
        base = os.path.basename(output_path)
        parent = os.path.dirname(view_dir)
        pano_dir = os.path.join(parent, f"{os.path.basename(view_dir)}_pano")
        return os.path.join(pano_dir, base)

    @staticmethod
    def _mark_camera_para_pano(camera_para: Dict[str, Any]) -> Dict[str, Any]:
        camera_para["projection"] = "equirectangular"
        camera_para["pano"] = True
        camera_para["panorama_type"] = "EQUIRECTANGULAR"
        camera_para["horizontal_fov"] = float(2.0 * np.pi)
        camera_para["vertical_fov"] = float(np.pi)
        return camera_para

    @staticmethod
    def _pano_dimensions(pano_resolution: int) -> Tuple[int, int]:
        pano_width = max(1, int(pano_resolution))
        return pano_width, max(1, pano_width // 2)

    def _render_pano_view_pass(
        self,
        output_path: str,
        camera_matrix,
        *,
        pano_resolution: int,
        render_depth: bool,
        render_semantic: bool,
        visible_geometry: bool,
        export_glb: bool,
        export_point_cloud: bool,
        export_voxel: bool,
        transparent_objects: set,
        vss,
        world_cam_w,
        world_look_w,
        world_up_w,
        align_height: bool,
        show_wall: bool,
        show_door: bool,
        show_window: bool,
        show_ceiling: bool,
        hdri_transparent_background: bool,
        reference_frame: bool = True,
        opencv_ref_camera=None,
        opencv_ref_look=None,
        opencv_ref_up=None,
        export_planar_faces: bool = False,
    ) -> str:
        pano_output_path = self._pano_output_path(output_path)
        pano_dir = os.path.dirname(pano_output_path) or "."
        os.makedirs(pano_dir, exist_ok=True)
        width, height = self._pano_dimensions(pano_resolution)
        prev_engine = self.scene.render.engine
        if prev_engine != 'CYCLES':
            print(f"🔄 Panorama rendering switched to Cycles equirectangular (previous engine: {prev_engine})")
            self.scene.render.engine = 'CYCLES'
            if hasattr(self.scene, "cycles"):
                self.scene.cycles.samples = self.config.get("blender_samples", 32)

        camera_obj = None
        camera_data = None
        prev_camera = None
        depth_scale = None
        try:
            camera_obj, camera_data, prev_camera = self._create_render_camera(
                camera_matrix, float(np.pi), width, height, panoramic=True
            )
            print(f"🎬 Panorama rendering ({width}x{height}, EQUIRECTANGULAR 360x180)...")
            self.scene.render.filepath = pano_output_path
            render = self.scene.render
            render.image_settings.color_mode = 'RGBA'
            render.film_transparent = hdri_transparent_background

            if render_depth:
                depth_path = os.path.join(
                    pano_dir,
                    f"{os.path.splitext(os.path.basename(pano_output_path))[0]}_depth.png",
                )
                depth_scale = util_bpy.render_color_and_depth_png(
                    self.scene, pano_output_path, depth_path, skip_objects=transparent_objects
                )
            else:
                bpy.ops.render.render(write_still=True)

            if render_semantic or visible_geometry:
                self.render_semantic_bundle(pano_output_path, save_artifacts=True)
            if visible_geometry and (export_glb or export_point_cloud or export_voxel):
                self.export_visible_geometry(
                    pano_dir,
                    camera_obj,
                    export_glb=export_glb,
                    export_point_cloud=export_point_cloud,
                    export_voxel=export_voxel,
                    transparent_objects=transparent_objects,
                    panoramic=True,
                )
            if export_planar_faces:
                self.export_planar_faces_and_lines(
                    pano_output_path,
                    camera_obj,
                    width,
                    height,
                    align_height=align_height,
                    show_wall=show_wall,
                    show_door=show_door,
                    show_window=show_window,
                    show_ceiling=show_ceiling,
                    transparent_objects=transparent_objects,
                    panoramic=True,
                )

            para_path = self._camera_para_path(pano_output_path)
            camera_para = vss.build_camera_para(
                world_cam_w,
                world_look_w,
                world_up_w,
                float(np.pi),
                float(width) / float(height),
                reference_frame=reference_frame,
                depth_scale=depth_scale,
                include_normal_fields=depth_scale is not None,
                width=width,
                height=height,
                include_intrinsic=False,
            )
            with open(para_path, 'w') as f:
                json.dump(self._mark_camera_para_pano(camera_para), f, indent=4)
            self.write_opencv_ssl_for_view(
                pano_dir,
                opencv_ref_camera if opencv_ref_camera is not None else world_cam_w,
                opencv_ref_look if opencv_ref_look is not None else world_look_w,
                opencv_ref_up if opencv_ref_up is not None else world_up_w,
            )
            print(f"✅ Panorama render complete! Saved to: {pano_output_path}")
            return pano_output_path
        finally:
            self._destroy_render_camera(camera_obj, camera_data, prev_camera, self.scene)
            self.scene.render.engine = prev_engine
            util_bpy.cleanup_bpy_render_memory(self.scene)

    @staticmethod
    def _remove_camera_blocks(camera_obj, camera_data):
        """Remove temporary camera; data may already be removed with the object."""
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
            print(f"🔍 Detected {len(transparent_wall_ids)} occluding walls, setting to transparent...")
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
        export_voxel: bool = False,
        visible_geometry: bool = False,
        render_semantic: bool = False,
        pano: bool = False,
        pano_resolution: int = 4096,
        pano_only: bool = False,
        view_dir_name: Optional[str] = None,
        export_planar_faces: Optional[bool] = None,
        export_visible_point_cloud: Optional[bool] = None,
        skip_render: bool = False,
        write_ssl: bool = True,
        write_camera_para: bool = True,
    ):
        """Multi-camera sequence: all frames in one sequence directory; visible geometry merged once there (multi-camera union).

        Default directory ``{timestamp}_seq``; ``view_dir_name`` can fix the name (e.g. ``auto_path_0008_seq``).
        When ``pano_only=True`` (``--video``), skip perspective RGB and render equirectangular pano per frame only.
        """
        if pano_only:
            pano = True
        total_start = time.perf_counter()
        seq_dir, dir_stamp = util.allocate_sequence_view_output_dir(output_root, view_dir_name)
        used_frame_stamps = {dir_stamp}

        world_cam0 = list(camera_positions[0])
        world_look0 = list(look_at_targets[0])
        world_up0 = list(up_vectors[0])

        with util_data.ViewSslSession(self, False) as vss:
            vss.setup(world_cam0, world_look0, world_up0)

            if rebuild or self.mesh_nodes["floor"] is None:
                self.construct_scene(
                    geometry_mode=geometry_mode,
                    show_wall=show_wall,
                    show_window=show_window,
                    show_door=show_door,
                    show_ceiling=show_ceiling,
                    align_height=align_height,
                    rebuild=rebuild,
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
                    print(f"🌇 Using HDRI environment lighting: {hdri_path}")
                    self.scene.render.film_transparent = hdri_transparent_background
                else:
                    print(f"⚠️ HDRI file unavailable or not configured: {hdri_path}")

            z_max = self.context["meta"]["z_max"]
            frame_states = []
            pano_frame_states = []
            n_frames = len(camera_positions)
            do_visible_ply = visible_geometry and export_point_cloud
            if export_visible_point_cloud is not None:
                do_visible_ply = bool(export_visible_point_cloud)
            do_planar = export_point_cloud if export_planar_faces is None else bool(export_planar_faces)
            existing_frames: List[str] = []
            if skip_render:
                for name in sorted(os.listdir(seq_dir)):
                    if not name.lower().endswith(".png"):
                        continue
                    if any(m in name for m in ("_depth", "_semantic", "_normal", "_lines", "_bbox_2d")):
                        continue
                    png_path = os.path.join(seq_dir, name)
                    base = os.path.splitext(name)[0]
                    para = os.path.join(seq_dir, f"{base}_camera_para.json")
                    if os.path.isfile(para) and os.path.getsize(png_path) > 0:
                        existing_frames.append(png_path)

            for frame_idx, (cam_pos, look_at, up_vec) in enumerate(
                zip(camera_positions, look_at_targets, up_vectors)
            ):
                world_cam = list(cam_pos)
                world_look = list(look_at)
                world_up = list(up_vec)
                render_cam, render_look, render_up = cam_pos, look_at, up_vec

                if skip_render and frame_idx < len(existing_frames):
                    output_path = existing_frames[frame_idx]
                    image_stamp = os.path.splitext(os.path.basename(output_path))[0]
                    print(f"⏭️  Skipping sequence frame {frame_idx + 1}/{n_frames} Cycles render -> {output_path}")
                else:
                    image_stamp = util.allocate_millis_stamp(exclude=used_frame_stamps)
                    used_frame_stamps.add(image_stamp)
                    output_path = os.path.join(seq_dir, f"{image_stamp}.png")
                    if pano_only:
                        print(f"🎬 Video pano frame {frame_idx + 1}/{n_frames} -> {self._pano_output_path(output_path)}")
                    else:
                        print(f"🎬 Sequence frame {frame_idx + 1}/{n_frames} -> {output_path}")
                view_dir = seq_dir

                camera_position, look_at_target, camera_matrix, fov_y = self._build_view_camera_matrix_and_fov(
                    render_cam, render_look, render_up, auto_fov=auto_fov, manual_fov=manual_fov
                )

                wall_transparency_records = {}
                object_transparency_records = []
                if auto_transparent:
                    wall_transparency_records, object_transparency_records = self._apply_view_auto_transparency(
                        camera_position, look_at_target, transparent_alpha, z_max
                    )

                transparent_objects = self._collect_transparent_objects(
                    wall_transparency_records, object_transparency_records
                )

                if pano_only:
                    try:
                        self._render_pano_view_pass(
                            output_path,
                            camera_matrix,
                            pano_resolution=pano_resolution,
                            render_depth=False,
                            render_semantic=False,
                            visible_geometry=False,
                            export_glb=False,
                            export_point_cloud=False,
                            export_voxel=False,
                            transparent_objects=transparent_objects,
                            vss=vss,
                            world_cam_w=world_cam,
                            world_look_w=world_look,
                            world_up_w=world_up,
                            align_height=align_height,
                            show_wall=show_wall,
                            show_door=show_door,
                            show_window=show_window,
                            show_ceiling=show_ceiling,
                            hdri_transparent_background=hdri_transparent_background,
                            reference_frame=(frame_idx == 0),
                            opencv_ref_camera=world_cam0,
                            opencv_ref_look=world_look0,
                            opencv_ref_up=world_up0,
                            export_planar_faces=False,
                        )
                        pano_frame_states.append({
                            "camera_matrix": camera_matrix.copy(),
                            "transparent_objects": transparent_objects,
                            "image_stamp": image_stamp,
                        })
                    finally:
                        self._restore_view_transparency(wall_transparency_records, object_transparency_records)
                    continue

                if skip_render:
                    frame_states.append({
                        "camera_matrix": camera_matrix.copy(),
                        "fov_y": fov_y,
                        "transparent_objects": transparent_objects,
                        "image_stamp": image_stamp,
                    })
                    if pano:
                        pano_frame_states.append({
                            "camera_matrix": camera_matrix.copy(),
                            "transparent_objects": transparent_objects,
                        })
                    continue

                camera_obj, camera_data, prev_camera = self._create_render_camera(
                    camera_matrix, fov_y, width, height
                )
                self.scene.render.filepath = output_path
                render = self.scene.render
                render.image_settings.color_mode = 'RGBA'
                render.film_transparent = hdri_transparent_background

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

                    if do_planar:
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
                    if render_semantic or visible_geometry:
                        self.render_semantic_bundle(output_path, save_artifacts=True)

                    frame_states.append({
                        "camera_matrix": camera_matrix.copy(),
                        "fov_y": fov_y,
                        "transparent_objects": transparent_objects,
                        "image_stamp": image_stamp,
                    })

                    if write_camera_para:
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
                            width=width,
                            height=height,
                        )
                        with open(para_path, 'w') as f:
                            json.dump(camera_para, f, indent=4)
                    if pano:
                        self._render_pano_view_pass(
                            output_path,
                            camera_matrix,
                            pano_resolution=pano_resolution,
                            render_depth=render_depth,
                            render_semantic=render_semantic,
                            visible_geometry=False,
                            export_glb=False,
                            export_point_cloud=do_visible_ply,
                            export_voxel=export_voxel,
                            transparent_objects=transparent_objects,
                            vss=vss,
                            world_cam_w=world_cam,
                            world_look_w=world_look,
                            world_up_w=world_up,
                            align_height=align_height,
                            show_wall=show_wall,
                            show_door=show_door,
                            show_window=show_window,
                            show_ceiling=show_ceiling,
                            hdri_transparent_background=hdri_transparent_background,
                            reference_frame=(frame_idx == 0),
                            opencv_ref_camera=world_cam0,
                            opencv_ref_look=world_look0,
                            opencv_ref_up=world_up0,
                            export_planar_faces=do_planar,
                        )
                        pano_frame_states.append({
                            "camera_matrix": camera_matrix.copy(),
                            "transparent_objects": transparent_objects,
                        })
                finally:
                    self._restore_view_transparency(wall_transparency_records, object_transparency_records)
                    self._destroy_render_camera(camera_obj, camera_data, prev_camera, self.scene)
                    util_bpy.cleanup_bpy_render_memory(self.scene)

            if visible_geometry and (export_glb or do_visible_ply or export_voxel) and frame_states and not pano_only:
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
                    frame_infos = vve.build_frame_infos(
                        seq_dir,
                        [state["image_stamp"] for state in frame_states],
                    )
                    self.export_visible_geometry_multi(
                        seq_dir,
                        camera_states,
                        export_glb=export_glb,
                        export_point_cloud=do_visible_ply,
                        export_voxel=export_voxel,
                        frame_infos=frame_infos,
                    )
                finally:
                    for cam_obj, cam_data in reversed(temp_cameras):
                        if cam_obj and self.scene.camera == cam_obj:
                            if original_camera is not None and getattr(original_camera, "name", None) in bpy.data.objects:
                                self.scene.camera = original_camera
                            else:
                                self.scene.camera = None
                        self._remove_camera_blocks(cam_obj, cam_data)

            if pano and visible_geometry and (export_glb or export_point_cloud or export_voxel) and pano_frame_states:
                pano_dir = f"{seq_dir}_pano"
                original_camera = self.scene.camera
                camera_states = []
                temp_cameras = []
                pano_width, pano_height = self._pano_dimensions(pano_resolution)
                for state in pano_frame_states:
                    cam_obj, cam_data, _prev = self._create_render_camera(
                        state["camera_matrix"], float(np.pi), pano_width, pano_height, panoramic=True
                    )
                    temp_cameras.append((cam_obj, cam_data))
                    camera_states.append((cam_obj, state["transparent_objects"]))
                try:
                    self.export_visible_geometry_pano_multi(
                        pano_dir,
                        camera_states,
                        export_glb=export_glb,
                        export_point_cloud=export_point_cloud,
                        export_voxel=export_voxel,
                        pano_resolution=pano_resolution,
                    )
                finally:
                    for cam_obj, cam_data in reversed(temp_cameras):
                        if cam_obj and self.scene.camera == cam_obj:
                            if original_camera is not None and getattr(original_camera, "name", None) in bpy.data.objects:
                                self.scene.camera = original_camera
                            else:
                                self.scene.camera = None
                        self._remove_camera_blocks(cam_obj, cam_data)

            if write_ssl:
                self.write_opencv_ssl_for_view(seq_dir, world_cam0, world_look0, world_up0)
            pano_ssl_dir = f"{seq_dir}_pano"
            if write_ssl and os.path.isdir(pano_ssl_dir):
                self.write_opencv_ssl_for_view(pano_ssl_dir, world_cam0, world_look0, world_up0)

        total_time = time.perf_counter() - total_start
        print(f"✅ Sequence render complete ({n_frames} frames), sequence dir: {seq_dir}, total time: {total_time:.2f}s")

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
                    export_voxel: bool = False,
                    visible_geometry: bool = False,
                    render_semantic: bool = False,
                    pano: bool = False,
                    pano_resolution: int = 4096,
                    pano_only: bool = False,
                    view_dir_name: Optional[str] = None,
                    export_planar_faces: Optional[bool] = None,
                    export_visible_point_cloud: Optional[bool] = None,
                    skip_render: bool = False,
                    write_ssl: bool = True,
                    write_camera_para: bool = True):
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
                export_voxel=export_voxel,
                visible_geometry=visible_geometry,
                render_semantic=render_semantic,
                pano=pano,
                pano_resolution=pano_resolution,
                pano_only=pano_only,
                view_dir_name=view_dir_name,
                export_planar_faces=export_planar_faces,
                export_visible_point_cloud=export_visible_point_cloud,
                skip_render=skip_render,
                write_ssl=write_ssl,
                write_camera_para=write_camera_para,
            )

        output_path = self._resolve_view_output_path(
            output_path, view_dir_name=view_dir_name, skip_render=skip_render
        )
        view_dir = os.path.dirname(output_path) or "."
        center = self.context["meta"]["center"]
        z_max = self.context["meta"]["z_max"]
        if look_at_target is None:
            world_look_w = [center[0], center[1], z_max / 2]
        else:
            world_look_w = list(look_at_target)
        world_cam_w = list(camera_position)
        world_up_raw = geo_cv.resolve_render_view_up_vector(
            world_cam_w, world_look_w, up_vector
        )

        total_start = time.perf_counter()

        with util_data.ViewSslSession(self, False) as vss:
            vss.setup(world_cam_w, world_look_w, world_up_raw)

            construct_time = 0.0
            if rebuild or self.mesh_nodes["floor"] is None:
                construct_start = time.perf_counter()
                self.construct_scene(
                    geometry_mode=geometry_mode,
                    show_wall=show_wall,
                    show_window=show_window,
                    show_door=show_door,
                    show_ceiling=show_ceiling,
                    align_height=align_height,
                    rebuild=rebuild,
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
                    print(f"🌇 Using HDRI environment lighting: {hdri_path}")
                    self.scene.render.film_transparent = hdri_transparent_background
                else:
                    print(f"⚠️ HDRI file unavailable or not configured: {hdri_path}")
            setup_time = time.perf_counter() - setup_start

            bounds = self.context["meta"]["bounds"]

            if look_at_target is None:
                look_at_target = [center[0], center[1], z_max / 2]
            up_vector = geo_cv.resolve_render_view_up_vector(
                camera_position, look_at_target, up_vector
            )
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
                    print(f"🔍 Detected {len(transparent_wall_ids)} occluding walls, setting to transparent...")
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

            camera_obj, camera_data, prev_camera = self._create_render_camera(
                camera_matrix, fov_y, width, height
            )
            self.scene.render.filepath = output_path

            render = self.scene.render
            render.image_settings.color_mode = 'RGBA'
            render.film_transparent = hdri_transparent_background

            print(f"🎬 Rendering ({width}x{height})..." if not skip_render else "⏭️  Skipping Cycles render (main frame output already exists)")
            depth_scale = None
            transparent_objects = {
                self.mesh_nodes["walls"].get(wall_id, {}).get("node")
                for wall_id in wall_transparency_records
            }
            transparent_objects.update(record.get("object") for record in object_transparency_records)
            transparent_objects = {obj for obj in transparent_objects if obj is not None}
            do_visible_ply = visible_geometry and export_point_cloud
            if export_visible_point_cloud is not None:
                do_visible_ply = bool(export_visible_point_cloud)
            do_planar = export_point_cloud if export_planar_faces is None else bool(export_planar_faces)
            try:
                render_start = time.perf_counter()
                if skip_render:
                    render_time = 0.0
                    para_path = self._camera_para_path(output_path)
                    if os.path.isfile(para_path):
                        try:
                            with open(para_path, "r", encoding="utf-8") as f:
                                depth_scale = json.load(f).get("depth_scale")
                        except (json.JSONDecodeError, OSError):
                            depth_scale = None
                elif render_depth:
                    depth_path = os.path.join(
                        os.path.dirname(output_path),
                        f"{os.path.splitext(os.path.basename(output_path))[0]}_depth.png",
                    )
                    depth_scale = util_bpy.render_color_and_depth_png(
                        self.scene, output_path, depth_path, skip_objects=transparent_objects
                    )
                    render_time = time.perf_counter() - render_start
                else:
                    bpy.ops.render.render(write_still=True)
                    render_time = time.perf_counter() - render_start
                if skip_render and visible_geometry:
                    skipped_geom = []
                    if not export_glb:
                        skipped_geom.append("GLB")
                    if not do_visible_ply:
                        skipped_geom.append("point cloud")
                    if not export_voxel:
                        skipped_geom.append("voxel")
                    if skipped_geom:
                        print(f"⏭️  Skipping visible geometry export (already exists): {', '.join(skipped_geom)}")
                if render_semantic or visible_geometry:
                    self.render_semantic_bundle(output_path, save_artifacts=True)
                if visible_geometry and (export_glb or do_visible_ply or export_voxel):
                    self.export_visible_geometry(
                        os.path.dirname(output_path),
                        camera_obj,
                        export_glb=export_glb,
                        export_point_cloud=do_visible_ply,
                        export_voxel=export_voxel,
                        transparent_objects=transparent_objects,
                        view_image_path=output_path,
                    )
                if do_planar:
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
                if pano:
                    self._render_pano_view_pass(
                        output_path,
                        camera_matrix,
                        pano_resolution=pano_resolution,
                        render_depth=render_depth,
                        render_semantic=render_semantic,
                        visible_geometry=visible_geometry,
                        export_glb=export_glb,
                        export_point_cloud=do_visible_ply,
                        export_voxel=export_voxel,
                        transparent_objects=transparent_objects,
                        vss=vss,
                        world_cam_w=world_cam_w,
                        world_look_w=world_look_w,
                        world_up_w=vss.world_up,
                        align_height=align_height,
                        show_wall=show_wall,
                        show_door=show_door,
                        show_window=show_window,
                        show_ceiling=show_ceiling,
                        hdri_transparent_background=hdri_transparent_background,
                        reference_frame=True,
                        export_planar_faces=do_planar,
                    )
                total_time = time.perf_counter() - total_start
                print(f"✅ Render complete! Saved to: {output_path}")
                print(f"   Build: {construct_time:.2f}s, Setup: {setup_time:.2f}s, Render: {render_time:.2f}s, Total: {total_time:.2f}s")
            except Exception as e:
                print(f"❌ Render failed: {e}")
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

            if write_camera_para:
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
                    width=width,
                    height=height,
                )
                with open(para_path, 'w') as f:
                    json.dump(camera_para, f, indent=4)
            if write_ssl:
                self.write_opencv_ssl_for_view(view_dir, world_cam_w, world_look_w, vss.world_up)
    
# ==================== Test code ====================




if __name__ == "__main__":
    print("="*60)
    print("🔍 Checking Blender Python environment")
    print("="*60)
    print(f"✅ bpy version: {bpy.app.version_string}")
    print(f"✅ Blender version: {'.'.join(map(str, bpy.app.version))}")
    print()
    
    # List render devices
    print("🖥️  Available render devices:")
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for backend in ("OPTIX", "CUDA"):
        for device in prefs.get_devices_for_type(backend):
            status = "✅ Enabled" if device.use else "⚪ Disabled"
            print(f"   - {device.name} (backend: {backend}, type: {device.type}) {status}")
    print(f"   Selected backend: {configure_cycles_gpu_devices(prefs)}")
    print()
    
    # Test scene rendering
    line = 15
    jsonl_path = '/data-nas/data/experiments/mushui/datasets/manycore/spatiallm_raw.jsonl'
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    test_dir = os.path.join(base_dir, 'test_bpy')
    os.makedirs(test_dir, exist_ok=True)
    topdown_path = os.path.join(test_dir, 'scene_topdown.png')
    
    try:
        data = util.read_jsonl_line(jsonl_path, line)
        print(f"📁 Loading scene: {data['room']['room_type']} ({len(data['bbox'])} objects)")
        
        ctx = BpySceneCtx(data['room']['room_type'])
        ctx.add_walls(data['wall'])
        ctx.add_doors(data.get('door', []))
        ctx.add_windows(data.get('window', []))
        ctx.add_boxes(data['bbox'])
        
        print("\n" + "="*60)
        print("Rendering top-down view")
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
            print(f"\n📷 Rendering {view_name} view...")
            print(f"   Camera position: {camera_pos}")
            print(f"   Look-at target: {look_at}")
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
        
        print("\n✅ All done!")
        print(ctx.get_context())
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
