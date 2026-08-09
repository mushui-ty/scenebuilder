import os
# Set PyOpenGL platform to EGL BEFORE importing pyrender
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import json
import numpy as np
import pyrender
import trimesh
import time
import colorsys
import hashlib
import imageio
from typing import List, Dict, Any, Optional

try:
    from . import util, util_data, geometry_opencv as geo_cv
    from .config_utils import CONFIG_PATH, load_config
except ImportError:
    import util  # type: ignore
    import util_data  # type: ignore
    import geometry_opencv as geo_cv  # type: ignore
    from config_utils import CONFIG_PATH, load_config  # type: ignore

class SceneCtx:
    """Scene context manager."""
    # Cubemap face axes (right/up/forward), aligned with standard cubemap convention
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

    def __init__(self, scene_type: str, model_extra_path: Optional[str] = None,
                 hole_extra_path: Optional[str] = None):
        self.context = {
            "meta": {"scene_type": scene_type},
            "walls": {},
            "boxes": {}
        }
        self.scene = None
        self.if_set_lights = False
        self.model_extra_path = model_extra_path
        self.hole_extra_path = hole_extra_path

        # Keep all node references keyed by unique ID
        self.mesh_nodes = {
            "walls": {},      # wall_id -> node
            "doors": {},      # door_id -> node
            "windows": {},    # window_id -> node
            "boxes": {},      # box_id -> node
            "floor": None,    # floor node
            "ceiling": None,  # ceiling node
        }
        self._scene_show_ceiling = None

        # Load config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        self.config = load_config(config_path)

    def _asset_search_paths(self, default_key: str = "model_path") -> List[str]:
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

    def normalize_scene_data(self) -> None:
        util_data.normalize_scene_context(
            self.context,
            model_paths=self._asset_search_paths("model_path"),
            hole_paths=self._hole_asset_search_paths(),
        )

    def _object_export_identity(self, category: str, object_id: str):
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

    def _reset_render_state(self):
        self.scene = None
        self.if_set_lights = False
        self.mesh_nodes = {
            "walls": {},
            "doors": {},
            "windows": {},
            "boxes": {},
            "floor": None,
            "ceiling": None,
        }
        self._scene_show_ceiling = None

    def clear_scene(self):
        """Match BpySceneCtx: clear built scene after temporary view SSL transforms."""
        self._reset_render_state()

    def add_walls(self, walls: List[Dict[str, Any]]):
        """Add walls and compute scene metadata."""
        # Convert format: p/q -> s/e
        walls_converted = [{"s": w["p"][:2], "e": w["q"][:2], "height": w["height"]} for w in walls]

        # Endpoint snapping enabled by default, threshold 0.3m
        do_snap = True
        if do_snap:
            walls_converted = util.snap_wall_endpoints(walls_converted, threshold=0.3)

        # Compute bounds and vertices
        all_points = [p for w in walls_converted for p in [w["s"], w["e"]]]
        x_coords, y_coords = zip(*all_points)

        # vertices: [(x1, y1), (x2, y2), ...] outer polygon boundary vertex sequence
        # partitions: [(xs, ys, xe, ye, height), ...] interior partition wall segments (with height)
        vertices, partitions = util.calculate_minimum_area_polygon_and_partitions(walls_converted)
        
        self.context["meta"].update({
            "bounds": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
            "center": [(min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2],
            "span": [max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)],
            "z_max": max(w["height"] for w in walls_converted),
            "vertices": vertices
        })
        self.context["meta"].pop("_scene_normalized", None)
        
        # Debug info
        print(f"📐 Input wall count: {len(walls_converted)}")
        print(f"📐 Boundary wall segments: {len(vertices)}")
        print(f"📐 Interior partition walls: {len(partitions)}")
        
        # 1. Add boundary walls
        # Boundary walls form a closed loop from vertices so floor and wall bases align
        for i in range(len(vertices)):
            v_s = vertices[i]
            v_e = vertices[(i + 1) % len(vertices)]
            
            # Look up original height for this boundary segment
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
                # Boundary walls need orientation (pointing into the room)
                "orientation": util.calculate_wall_orientation(v_s, v_e, vertices),
                "is_partition": False,
                "doors": {},
                "windows": {}
            }

        # 2. Add interior partition walls
        for p in partitions:
            wall_id = util.generate_unique_id()
            self.context["walls"][wall_id] = {
                "s": [p[0], p[1]],
                "e": [p[2], p[3]],
                "height": p[4],
                # Partition walls do not need orientation
                "orientation": (0.0, 0.0),
                "is_partition": True,
                "doors": {},
                "windows": {}
            }

    def add_wall(self, s: List[float], e: List[float], height: float):
        """Add a single wall."""
        # Simple implementation: rebuild via add_walls
        current_walls = [{"p": w["s"] + [0], "q": w["e"] + [0], "height": w["height"]}
                        for w in self.context["walls"].values()]
        current_walls.append({"p": s[:2] + [0], "q": e[:2] + [0], "height": height})
        self.context["walls"] = {}  # clear
        self.add_walls(current_walls)

    def add_doors(self, doors: List[Dict[str, Any]]):
        """Add doors in batch."""
        self.context["meta"].pop("_scene_normalized", None)
        for door in doors:
            self.add_door(door["center"], door["width"], door["height"], door.get("asset_id"))

    def add_door(self, center: List[float], width: float, height: float, asset_id: Optional[int] = None):
        """Add a single door."""
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
        """Add windows in batch."""
        self.context["meta"].pop("_scene_normalized", None)
        for window in windows:
            self.add_window(window["center"], window["width"], window["height"], window.get("asset_id"))

    def add_window(self, center: List[float], width: float, height: float, asset_id: Optional[int] = None):
        """Add a single window."""
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
        """Add furniture boxes in batch."""
        self.context["meta"].pop("_scene_normalized", None)
        for box in boxes:
            self.add_box(box["center"], box["angle_z"], box["scale"], box.get("class"),
                        box.get("label"), box.get("caption"), box.get("asset_id"))

    def add_box(self, center: List[float], angle_z: float, scale: List[float],
                class_name: Optional[str] = None, label: Optional[str] = None,
                caption: Optional[str] = None, asset_id: Optional[int] = None) -> str:
        """Add a single furniture box; returns its ID."""
        box_id = util.generate_unique_id()
        box_data = {"center": center, "angle_z": angle_z, "scale": scale}

        if class_name: box_data["class"] = class_name
        if label: box_data["label"] = label
        if caption: box_data["caption"] = caption
        if asset_id: box_data["asset_id"] = asset_id

        self.context["boxes"][box_id] = box_data

        # Update z_max
        box_top = center[2] + scale[2] / 2
        if box_top > self.context["meta"].get("z_max", 0):
            self.context["meta"]["z_max"] = box_top

        return box_id

    def delete_box(self, box_id: str):
        """Remove a furniture box and recompute z_max."""
        if box_id not in self.context["boxes"]:
            raise ValueError(f"Box ID {box_id} not found")

        del self.context["boxes"][box_id]
        self.scene = None  # mark scene for rebuild

        # Recompute z_max: max height across walls and remaining boxes
        wall_max = max((w["height"] for w in self.context["walls"].values()), default=0)

        box_max = 0
        for box in self.context["boxes"].values():
            box_top = box["center"][2] + box["scale"][2] / 2
            box_max = max(box_max, box_top)

        self.context["meta"]["z_max"] = max(wall_max, box_max)

    def get_context(self) -> Dict:
        """Return the scene context."""
        return self.context
    
    def get_boxes(self) -> Dict:
        """Return all furniture boxes."""
        return self.context["boxes"]

    def export_wall_ssl(self, output_dir: str):
        """
        Export wall SSL (Room and all Wall entries).
        Includes boundary and interior partition walls.
        """
        output_path = os.path.join(output_dir, 'wall_ssl.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        
        # Generate room_id
        room_id = util.generate_unique_id()
        room_type = self.context["meta"]["scene_type"]
        lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
        
        # Export all walls directly from context["walls"]
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
        Export wall and opening SSL (Room, Wall, Door, Window).
        """
        output_path = os.path.join(output_dir, 'wall_hole_ssl.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        
        # Generate room_id
        room_id = util.generate_unique_id()
        room_type = self.context["meta"]["scene_type"]
        lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
        
        # Record all walls
        all_walls = self.context["walls"]
        for wall_id, wall in all_walls.items():
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
        
        print(f"✅ Wall and opening SSL exported: {output_path}")
        return output_path

    def export_ssl(self, output_dir: str):
        """
        Export full SSL (Room, Wall, Door, Window, Bbox).
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
        
        # Export all objects (Bbox)
        for box_id, box in self.context["boxes"].items():
            lines.append(f'Bbox(id="{box_id}", room_id="{room_id}", center={box["center"]}, scale={box["scale"]}, angle_z={box["angle_z"]}, class="{box["class"]}")')
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Full SSL exported: {output_path}")
        return output_path
    
    def construct_floor(self, show_wall=True, show_window=True, show_door=True, show_ceiling=True):
        """Build floor and walls."""
        self.scene = pyrender.Scene()
        vertices = self.context["meta"]["vertices"]

        if len(vertices) < 3:
            return

        # Create floor
        bounds = self.context["meta"]["bounds"]
        texture_scale = self.config.get("texture_scale", None)
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
        floor_mesh = util.create_floor_mesh(floor_vertices, floor_bounds, texture_scale=texture_scale)

        # Load texture
        texture_path = self.config.get("floor_texture_path")
        if texture_path and os.path.exists(texture_path):
            try:
                import PIL.Image
                texture = PIL.Image.open(texture_path).convert('RGB')
                floor_mesh.visual = trimesh.visual.TextureVisuals(uv=floor_mesh.visual.uv, image=texture)
            except:
                floor_mesh.visual.material = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[0.8, 0.7, 0.5, 1.0])

        floor_node = self.scene.add(pyrender.Mesh.from_trimesh(floor_mesh))
        self.mesh_nodes["floor"] = {"node": floor_node, "mesh": floor_mesh}

        # Create walls
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
                    print(f"⚠️ Failed to load wall texture: {e}")
            
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
        
        # Add real door/window meshes: walls already cut openings above; place assets into holes here.
        if show_door or show_window:
            config_with_extra = self.config.copy()
            if self.model_extra_path:
                config_with_extra["model_extra_path"] = self.model_extra_path
            if self.hole_extra_path:
                config_with_extra["hole_extra_path"] = self.hole_extra_path
            print("🚪 Processing doors and windows (loading real meshes by asset_id)...")

            for wall_id, wall in self.context["walls"].items():
                wall_vec = np.array(wall["e"], dtype=float) - np.array(wall["s"], dtype=float)
                wall_len = float(np.linalg.norm(wall_vec))
                if wall_len < 1e-6:
                    continue

                wall_dir = wall_vec / wall_len
                orientation = np.array(wall.get("orientation", [0.0, 0.0]), dtype=float)
                is_partition = wall.get("is_partition", False)
                if is_partition and np.linalg.norm(orientation) < 1e-4:
                    outward_normal = np.array([-wall_dir[1], wall_dir[0]], dtype=float)
                else:
                    outward_normal = -orientation
                wall_angle_deg = float(np.degrees(np.arctan2(wall_dir[1], wall_dir[0])))

                types_to_process = []
                if show_door:
                    types_to_process.append(("door", "doors"))
                if show_window:
                    types_to_process.append(("window", "windows"))

                for item_type, key in types_to_process:
                    for item_id, item in wall.get(key, {}).items():
                        asset_id = item.get("asset_id")
                        if not asset_id:
                            print(f"  ⚠️ Skipping {item_type} {item_id}: missing asset_id, cannot load real mesh")
                            continue

                        center = np.array(item["center"], dtype=float)
                        real_center = center.copy()
                        if not is_partition:
                            real_center[:2] += outward_normal * (wall_thickness / 2)

                        item_box_data = {
                            "center": real_center.tolist(),
                            "scale": [item["width"], wall_thickness, item["height"]],
                            "angle_z": wall_angle_deg,
                        }
                        loaded = self._load_and_transform_asset(
                            asset_id,
                            item_box_data,
                            config_with_extra,
                            default_key="model_hole_path",
                        )
                        if loaded is None:
                            print(f"  ❌ {item_type} {item_id} (asset_id={asset_id}) load failed")
                            continue

                        nodes = self._add_trimesh_to_scene(loaded)
                        if nodes:
                            self.mesh_nodes[f"{item_type}s"][item_id] = {
                                "nodes": nodes,
                                "mesh": loaded,
                                f"{item_type}_data": dict(item),
                            }
                        else:
                            print(f"  ❌ {item_type} {item_id} (asset_id={asset_id}) failed to add to scene")
        
        if show_ceiling:
            # Create ceiling (same outer outline as floor, including wall thickness)
            z_max = self.context["meta"]["z_max"]
            ceiling_mesh = util.create_ceiling_mesh(floor_vertices, floor_bounds, z_max)
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
                    print(f"⚠️ Failed to load ceiling texture: {e}")
            
            if ceiling_texture:
                ceiling_mesh.visual = trimesh.visual.TextureVisuals(uv=ceiling_mesh.visual.uv, image=ceiling_texture)
            else:
                ceiling_mesh.visual.material = trimesh.visual.material.PBRMaterial(baseColorFactor=ceiling_color)
            
            ceiling_node = self.scene.add(pyrender.Mesh.from_trimesh(ceiling_mesh))
            self.mesh_nodes["ceiling"] = {"node": ceiling_node, "mesh": ceiling_mesh}
        else:
            self.mesh_nodes["ceiling"] = None

    def _load_and_transform_asset(
        self,
        asset_id,
        box_data: Dict[str, Any],
        config: Dict[str, Any],
        default_key: str = "model_path",
    ):
        loaded = self._load_asset_mesh(asset_id, config, default_key=default_key)
        if loaded is None:
            return None

        try:
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
            scale_factors = np.array(box_data["scale"], dtype=float) / new_size
            theta = np.radians(float(box_data["angle_z"]))
            rotation = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ])
            translation = np.array(box_data["center"], dtype=float)

            if isinstance(loaded, trimesh.Scene):
                for geom in loaded.geometry.values():
                    if hasattr(geom, 'vertices'):
                        geom.vertices = (geom.vertices * scale_factors).dot(rotation.T) + translation
            elif hasattr(loaded, 'vertices'):
                loaded.vertices = (loaded.vertices * scale_factors).dot(rotation.T) + translation
            return loaded
        except Exception as e:
            print(f"  ❌ asset {asset_id} transform failed: {e}")
            return None

    @staticmethod
    def _load_asset_mesh(asset_id, config: Dict[str, Any], default_key: str = "model_path"):
        search_paths = util.build_asset_search_paths(config, default_key=default_key)
        for base_path in search_paths:
            for ext in ("glb", "gltf"):
                path = os.path.join(base_path, f"{asset_id}.{ext}")
                if os.path.exists(path):
                    try:
                        return trimesh.load(path)
                    except Exception:
                        continue
        return None

    def _add_trimesh_to_scene(self, mesh_obj):
        if isinstance(mesh_obj, trimesh.Scene):
            mesh_scene = pyrender.Scene.from_trimesh_scene(mesh_obj)
            return [self.scene.add(node.mesh) for node in mesh_scene.get_nodes() if node.mesh]
        if hasattr(mesh_obj, 'vertices') and len(mesh_obj.vertices) > 0:
            return [self.scene.add(pyrender.Mesh.from_trimesh(mesh_obj))]
        return []

    def construct_scene(self, show_wall: bool = True, show_window: bool = True,
                        show_door: bool = True, use_bbox_geometry: bool = False,
                        show_ceiling: bool = True):
        """Build the full scene."""
        total_start = time.perf_counter()
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.normalize_scene_data()

        self.construct_floor(
            show_wall=show_wall,
            show_window=show_window,
            show_door=show_door,
            show_ceiling=show_ceiling,
        )
        self._scene_show_ceiling = bool(show_ceiling)

        print(f"📦 Loading furniture ({len(self.context['boxes'])} objects)...")
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
                # GLTF assets are Y-up. Blender converts them to this repo's Z-up
                # convention on import; trimesh/pyrender does not, so do it here
                # before bbox scaling, yaw rotation, and translation.
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

        print(f"✅ Build complete! {success_count}/{len(self.context['boxes'])} objects succeeded (elapsed: {time.perf_counter() - total_start:.2f}s)")

    def export_glb(self, output_path: str, rebuild: bool = False, **construct_kwargs):
        """Export the current scene as a GLB file."""
        if rebuild or self.scene is None or self.mesh_nodes.get("floor") is None:
            self.construct_scene(**construct_kwargs)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        export_scene = trimesh.Scene()

        def add_mesh(name: str, mesh_obj):
            if mesh_obj is None:
                return
            if isinstance(mesh_obj, trimesh.Scene):
                try:
                    dumped_meshes = mesh_obj.dump(concatenate=False)
                except Exception:
                    dumped_meshes = []
                if dumped_meshes:
                    for idx, geom in enumerate(dumped_meshes):
                        if hasattr(geom, "vertices") and len(geom.vertices) > 0:
                            export_scene.add_geometry(geom.copy(), node_name=f"{name}_{idx}")
                else:
                    for geom_name, geom in mesh_obj.geometry.items():
                        if hasattr(geom, "vertices") and len(geom.vertices) > 0:
                            export_scene.add_geometry(geom.copy(), node_name=f"{name}_{geom_name}")
            elif isinstance(mesh_obj, trimesh.Trimesh):
                if len(mesh_obj.vertices) > 0:
                    export_scene.add_geometry(mesh_obj.copy(), node_name=name)

        floor_info = self.mesh_nodes.get("floor")
        if floor_info:
            add_mesh("floor", floor_info.get("mesh"))

        ceiling_info = self.mesh_nodes.get("ceiling")
        if ceiling_info:
            add_mesh("ceiling", ceiling_info.get("mesh"))

        for wall_id, wall_info in self.mesh_nodes["walls"].items():
            add_mesh(f"wall_{wall_id}", wall_info.get("mesh"))

        for door_id, door_info in self.mesh_nodes["doors"].items():
            add_mesh(f"door_{door_id}", door_info.get("mesh"))

        for window_id, window_info in self.mesh_nodes["windows"].items():
            add_mesh(f"window_{window_id}", window_info.get("mesh"))

        for box_id, box_info in self.mesh_nodes["boxes"].items():
            add_mesh(f"box_{box_id}", box_info.get("mesh"))

        if not export_scene.geometry:
            raise ValueError("No exportable meshes in current scene; build or render the scene first")

        export_scene.export(output_path)
        print(f"✅ GLB exported: {output_path}")
        return output_path

    def export_point_cloud(self, output_dir: str, rebuild: bool = False, **construct_kwargs):
        """Sample per-object surface point clouds and export colored PLY files."""
        if rebuild or self.scene is None or self.mesh_nodes.get("floor") is None:
            self.construct_scene(**construct_kwargs)

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

        def export_one(category: str, object_id: str, mesh_obj, filename: str, sample_count: int,
                       extra: Optional[Dict[str, Any]] = None):
            meshes = self._flatten_trimesh_meshes(mesh_obj)
            if not meshes:
                return
            points, colors = self._sample_trimesh_meshes(
                meshes,
                sample_count,
                use_ses=(category in ("boxes", "doors", "windows")),
            )
            if len(points) == 0:
                print(f"⚠️ Point cloud sampling skipped empty object: {category}/{object_id}")
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
            os.makedirs(os.path.join(output_dir, "floor"), exist_ok=True)
            export_one("floor", "floor", floor_info.get("mesh"), "floor/floor.ply", default_samples)

        ceiling_info = self.mesh_nodes.get("ceiling")
        if ceiling_info:
            os.makedirs(os.path.join(output_dir, "ceiling"), exist_ok=True)
            export_one("ceiling", "ceiling", ceiling_info.get("mesh"), "ceiling/ceiling.ply", default_samples)

        for category in ["walls", "doors", "windows", "boxes"]:
            os.makedirs(os.path.join(output_dir, category), exist_ok=True)
            for object_id, info in self.mesh_nodes[category].items():
                label, asset_id = self._object_export_identity(category, object_id)
                filename = util_data.build_pointcloud_ply_relpath(category, label, asset_id)
                extra = {"label": label}
                if asset_id is not None:
                    extra["asset_id"] = asset_id
                sample_count = box_samples if category == "boxes" else default_samples
                export_one(category, object_id, info.get("mesh"), filename, sample_count, extra)

        if all_points:
            merged_points = np.vstack(all_points)
            merged_colors = np.vstack(all_colors)
            scene_path = os.path.join(output_dir, "scene_all.ply")
            self._write_ply(scene_path, merged_points, merged_colors)
            metadata["scene_all"] = {
                "path": "scene_all.ply",
                "points": int(len(merged_points)),
            }

        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ Point cloud exported: {output_dir}")
        return output_dir

    def export_visible_geometry(self, output_dir: str, camera_pose: np.ndarray, fov_y: float, aspect: float,
                                export_glb: bool = False, export_point_cloud: bool = False,
                                transparent_keys: Optional[set] = None,
                                clip_start: float = 0.05, clip_end: Optional[float] = None):
        """Export object-level visible geometry clipped to the current camera frustum."""
        transparent_keys = transparent_keys or set()
        entries = self._visible_trimesh_entries(camera_pose, fov_y, aspect, transparent_keys, clip_start, clip_end)
        if not entries:
            print("⚠️ No visible geometry detected in current pyrender view")
            return
        if export_glb:
            self.export_visible_glb(os.path.join(output_dir, "scene_visible.glb"), entries)
        if export_point_cloud:
            self.export_visible_point_cloud(os.path.join(output_dir, "pointcloud"), entries)

    def export_visible_glb(self, output_path: str, entries: List[Dict[str, Any]]):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        scene = trimesh.Scene()
        for entry in entries:
            for idx, mesh in enumerate(entry["meshes"]):
                if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                    name_suffix = "_visible_cutted" if entry.get("frustum_cutted") else "_visible"
                    scene.add_geometry(
                        mesh.copy(),
                        node_name=f"{entry['category']}_{entry['id']}{name_suffix}_{idx}",
                    )
        if not scene.geometry:
            print("⚠️ scene_visible.glb skipped: no geometry after clipping")
            return
        scene.export(output_path)
        print(f"✅ Visible GLB exported: {output_path}")

    def export_visible_point_cloud(self, output_dir: str, entries: List[Dict[str, Any]]):
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

        for entry in entries:
            sample_count = box_samples if entry["category"] == "boxes" else default_samples
            points, colors = self._sample_trimesh_meshes(entry["meshes"], sample_count, use_ses=False)
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
        print(f"✅ Visible point cloud exported: {output_dir}")

    def _visible_trimesh_entries(
        self,
        camera_pose: np.ndarray,
        fov_y: float,
        aspect: float,
        transparent_keys: set,
        clip_start: float,
        clip_end: Optional[float],
    ):
        entries = []
        for category, object_id, mesh_obj, visible_ply in self._iter_trimesh_nodes(visible_suffix=True):
            source_meshes = list(self._flatten_trimesh_meshes(mesh_obj))
            if not source_meshes:
                continue
            clipped_meshes = []
            for mesh in source_meshes:
                clipped = self._clip_trimesh_to_camera_frustum(mesh, camera_pose, fov_y, aspect, clip_start, clip_end)
                if clipped is not None and len(clipped.vertices) > 0 and len(clipped.faces) > 0:
                    clipped_meshes.append(clipped)
            if not clipped_meshes:
                continue
            frustum_cutted = self._trimesh_meshes_frustum_cutted(
                source_meshes, camera_pose, fov_y, aspect
            )
            in_view_ratio_value = self._trimesh_meshes_frustum_in_view_ratio(
                source_meshes, camera_pose, fov_y, aspect
            )
            output_ply = self._append_output_suffix(visible_ply, "_cutted") if frustum_cutted else visible_ply
            entries.append({
                "category": category,
                "id": object_id,
                "key": (category, object_id),
                "meshes": clipped_meshes,
                "visible_ply": output_ply,
                "frustum_cutted": frustum_cutted,
                "frustum_in_view_ratio": in_view_ratio_value,
            })
        return entries

    @staticmethod
    def _append_output_suffix(path: str, suffix: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}{suffix}{ext}"

    FRUSTUM_IN_VIEW_RATIO = 0.95

    def _vertex_in_pyrender_view(
        self,
        point,
        camera_pose: np.ndarray,
        fov_y: float,
        aspect: float,
        margin: float = 0.002,
    ) -> bool:
        world_to_camera = np.linalg.inv(camera_pose)
        cam = world_to_camera @ np.array([point[0], point[1], point[2], 1.0], dtype=float)
        depth = float(-cam[2])
        if depth <= 1e-6:
            return False
        tan_y = float(np.tan(fov_y * 0.5))
        tan_x = tan_y * float(aspect)
        u = float(cam[0] / (depth * tan_x)) * 0.5 + 0.5
        v = float(cam[1] / (depth * tan_y)) * 0.5 + 0.5
        return (-margin <= u <= 1.0 + margin) and (-margin <= v <= 1.0 + margin)

    def _triangle_fully_in_pyrender_view(
        self,
        triangle: np.ndarray,
        camera_pose: np.ndarray,
        fov_y: float,
        aspect: float,
        margin: float = 0.002,
    ) -> bool:
        tri = np.asarray(triangle, dtype=float)
        return all(
            self._vertex_in_pyrender_view(p, camera_pose, fov_y, aspect, margin=margin) for p in tri
        )

    def _trimesh_meshes_frustum_in_view_ratio(
        self,
        meshes,
        camera_pose: np.ndarray,
        fov_y: float,
        aspect: float,
    ) -> float:
        total = 0
        in_view = 0
        for mesh in meshes:
            if mesh is None or len(mesh.faces) == 0:
                continue
            vertices = np.asarray(mesh.vertices, dtype=float)
            for face in mesh.faces:
                total += 1
                if self._triangle_fully_in_pyrender_view(vertices[face], camera_pose, fov_y, aspect):
                    in_view += 1
        if total == 0:
            return 1.0
        return in_view / total

    def _trimesh_meshes_frustum_cutted(
        self,
        meshes,
        camera_pose: np.ndarray,
        fov_y: float,
        aspect: float,
        in_view_ratio: float = FRUSTUM_IN_VIEW_RATIO,
    ) -> bool:
        return self._trimesh_meshes_frustum_in_view_ratio(meshes, camera_pose, fov_y, aspect) < in_view_ratio

    def _trimesh_frustum_planes(self, fov_y: float, aspect: float, clip_start: float, clip_end: Optional[float]):
        tan_y = float(np.tan(fov_y * 0.5))
        tan_x = tan_y * float(aspect)
        clip_start = max(float(clip_start or 0.0), 0.0)
        clip_end_value = float(clip_end) if clip_end is not None else 0.0
        planes = [
            (np.array([0.0, 0.0, -1.0], dtype=float), -clip_start),
            (np.array([1.0, 0.0, -tan_x], dtype=float), 0.0),
            (np.array([-1.0, 0.0, -tan_x], dtype=float), 0.0),
            (np.array([0.0, 1.0, -tan_y], dtype=float), 0.0),
            (np.array([0.0, -1.0, -tan_y], dtype=float), 0.0),
        ]
        if clip_end_value > clip_start:
            planes.append((np.array([0.0, 0.0, 1.0], dtype=float), clip_end_value))
        return planes

    def _iter_trimesh_nodes(self, visible_suffix: bool = False):
        suffixes = ["visible"] if visible_suffix else []
        floor_info = self.mesh_nodes.get("floor")
        if floor_info:
            filename = util_data.build_pointcloud_ply_relpath("floor", "floor", None, suffixes)
            yield "floor", "floor", floor_info.get("mesh"), filename
        ceiling_info = self.mesh_nodes.get("ceiling")
        if ceiling_info:
            filename = util_data.build_pointcloud_ply_relpath("ceiling", "ceiling", None, suffixes)
            yield "ceiling", "ceiling", ceiling_info.get("mesh"), filename
        for category in ["walls", "doors", "windows", "boxes"]:
            for object_id, info in self.mesh_nodes[category].items():
                label, asset_id = self._object_export_identity(category, object_id)
                suffixes = ["visible"] if visible_suffix else []
                filename = util_data.build_pointcloud_ply_relpath(
                    category, label, asset_id, suffixes,
                )
                yield category, object_id, info.get("mesh"), filename

    def _clip_trimesh_to_camera_frustum(
        self,
        mesh: trimesh.Trimesh,
        camera_pose: np.ndarray,
        fov_y: float,
        aspect: float,
        clip_start: float = 0.05,
        clip_end: Optional[float] = None,
    ):
        if mesh is None or len(mesh.faces) == 0:
            return None
        world_to_camera = np.linalg.inv(camera_pose)
        tan_y = float(np.tan(fov_y * 0.5))
        tan_x = tan_y * float(aspect)
        clip_start = max(float(clip_start or 0.0), 0.0)
        clip_end_value = float(clip_end) if clip_end is not None else 0.0
        planes = [
            (np.array([0.0, 0.0, -1.0], dtype=float), -clip_start),
            (np.array([1.0, 0.0, -tan_x], dtype=float), 0.0),
            (np.array([-1.0, 0.0, -tan_x], dtype=float), 0.0),
            (np.array([0.0, 1.0, -tan_y], dtype=float), 0.0),
            (np.array([0.0, -1.0, -tan_y], dtype=float), 0.0),
        ]
        if clip_end_value > clip_start:
            planes.append((np.array([0.0, 0.0, 1.0], dtype=float), clip_end_value))

        vertices = []
        faces = []
        uvs = []
        source_face_indices = []
        mesh_vertices = np.asarray(mesh.vertices, dtype=float)
        visual_uv = getattr(mesh.visual, "uv", None)
        has_uv = visual_uv is not None and len(visual_uv) >= len(mesh_vertices)
        visual_uv = np.asarray(visual_uv, dtype=float) if has_uv else None
        for face_idx, face in enumerate(mesh.faces):
            tri = mesh_vertices[face]
            cam_tri = tri @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
            tri_uv = visual_uv[face] if visual_uv is not None else None
            polygon = [
                {"cam": cam_tri[idx], "uv": None if tri_uv is None else tri_uv[idx]}
                for idx in range(3)
            ]
            for plane in planes:
                polygon = self._clip_camera_points_against_plane(polygon, plane)
                if len(polygon) < 3:
                    break
            if len(polygon) < 3:
                continue
            for i in range(1, len(polygon) - 1):
                items = [polygon[0], polygon[i], polygon[i + 1]]
                cam_clipped = np.array([item["cam"] for item in items], dtype=float)
                world_clipped = cam_clipped @ camera_pose[:3, :3].T + camera_pose[:3, 3]
                base = len(vertices)
                vertices.extend(world_clipped)
                faces.append([base, base + 1, base + 2])
                if tri_uv is not None:
                    uvs.extend([item["uv"] for item in items])
                source_face_indices.append(face_idx)

        if not faces:
            return None
        clipped = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        target_uvs = np.asarray(uvs, dtype=float) if uvs and len(uvs) == len(vertices) else None
        self._copy_trimesh_visual(mesh, clipped, source_face_indices, target_uvs)
        return clipped

    @staticmethod
    def _clip_camera_points_against_plane(points: List[Dict[str, Any]], plane):
        normal, offset = plane
        clipped = []
        prev = points[-1]
        prev_dist = float(np.dot(normal, prev["cam"]) + offset)
        prev_inside = prev_dist >= -1e-8
        for curr in points:
            curr_dist = float(np.dot(normal, curr["cam"]) + offset)
            curr_inside = curr_dist >= -1e-8
            if curr_inside != prev_inside:
                denom = prev_dist - curr_dist
                t = 0.0 if abs(denom) <= 1e-12 else prev_dist / denom
                cam = prev["cam"] + t * (curr["cam"] - prev["cam"])
                if prev["uv"] is None or curr["uv"] is None:
                    uv = None
                else:
                    uv = prev["uv"] + t * (curr["uv"] - prev["uv"])
                clipped.append({"cam": cam, "uv": uv})
            if curr_inside:
                clipped.append(curr)
            prev = curr
            prev_dist = curr_dist
            prev_inside = curr_inside
        return clipped

    @staticmethod
    def _copy_trimesh_visual(
        source: trimesh.Trimesh,
        target: trimesh.Trimesh,
        source_face_indices: List[int],
        target_uvs: Optional[np.ndarray] = None,
    ):
        material = getattr(source.visual, "material", None)
        if material is not None and target_uvs is not None and len(target_uvs) == len(target.vertices):
            try:
                target.visual = trimesh.visual.TextureVisuals(uv=target_uvs, material=material)
                return
            except Exception:
                pass
        face_colors = getattr(source.visual, "face_colors", None)
        if face_colors is not None and len(face_colors) > 0:
            colors = np.asarray(face_colors)[source_face_indices]
            target.visual.face_colors = colors
            return
        vertex_colors = getattr(source.visual, "vertex_colors", None)
        if vertex_colors is not None and len(vertex_colors) > 0:
            color = np.asarray(vertex_colors)[0]
            target.visual.vertex_colors = np.tile(color, (len(target.vertices), 1))
            return
        if material is not None:
            try:
                target.visual = trimesh.visual.TextureVisuals(material=material)
            except Exception:
                pass

    @staticmethod
    def _entry_triangles(entry: Dict[str, Any]):
        triangles = []
        for mesh in entry["meshes"]:
            if len(mesh.faces) > 0:
                triangles.append(mesh.triangles)
        if not triangles:
            return np.empty((0, 3, 3), dtype=float)
        return np.vstack(triangles)

    def _entry_visible_from_camera(self, entry: Dict[str, Any], camera_pose: np.ndarray,
                                   opaque_triangles: List[Any], transparent_keys: set):
        target_triangles = self._entry_triangles(entry)
        if len(target_triangles) == 0:
            return False
        centroids = np.mean(target_triangles, axis=1)
        if len(centroids) > 64:
            step = max(1, len(centroids) // 64)
            centroids = centroids[::step][:64]
        origin = np.asarray(camera_pose[:3, 3], dtype=float)
        occluders = [
            tris for key, tris in opaque_triangles
            if key != entry["key"] and key not in transparent_keys and len(tris) > 0
        ]
        occluder_triangles = np.vstack(occluders) if occluders else np.empty((0, 3, 3), dtype=float)
        for point in centroids:
            direction = point - origin
            distance = float(np.linalg.norm(direction))
            if distance <= 1e-8:
                return True
            if len(occluder_triangles) == 0:
                return True
            hit_dist = self._nearest_ray_triangle_distance(origin, direction / distance, occluder_triangles)
            if hit_dist is None or hit_dist >= distance - 1e-4:
                return True
        return False

    @staticmethod
    def _nearest_ray_triangle_distance(origin: np.ndarray, direction: np.ndarray, triangles: np.ndarray):
        eps = 1e-8
        v0 = triangles[:, 0]
        v1 = triangles[:, 1]
        v2 = triangles[:, 2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        a = np.einsum("ij,ij->i", edge1, h)
        mask = np.abs(a) > eps
        if not np.any(mask):
            return None
        f = np.zeros_like(a)
        f[mask] = 1.0 / a[mask]
        s = origin - v0
        u = f * np.einsum("ij,ij->i", s, h)
        mask &= (u >= 0.0) & (u <= 1.0)
        if not np.any(mask):
            return None
        q = np.cross(s, edge1)
        v = f * np.einsum("j,ij->i", direction, q)
        mask &= (v >= 0.0) & ((u + v) <= 1.0)
        if not np.any(mask):
            return None
        t = f * np.einsum("ij,ij->i", edge2, q)
        t = t[mask & (t > eps)]
        if len(t) == 0:
            return None
        return float(np.min(t))

    def _flatten_trimesh_meshes(self, mesh_obj):
        if mesh_obj is None:
            return []
        if isinstance(mesh_obj, trimesh.Trimesh):
            return [mesh_obj]
        if isinstance(mesh_obj, trimesh.Scene):
            try:
                dumped = mesh_obj.dump(concatenate=False)
                return [mesh for mesh in dumped if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0]
            except Exception:
                return [
                    mesh for mesh in mesh_obj.geometry.values()
                    if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0
                ]
        return []

    def _sample_trimesh_meshes(self, meshes: List[trimesh.Trimesh], sample_count: int, use_ses: bool = False):
        areas = np.array([float(mesh.area) for mesh in meshes], dtype=float)
        total_area = float(np.sum(areas))
        if total_area <= 1e-12:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)

        sharp_lengths = np.array([self._trimesh_sharp_edge_length(mesh) for mesh in meshes], dtype=float)
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
        for mesh, count in zip(meshes, counts):
            if count <= 0:
                continue
            points, face_indices = trimesh.sample.sample_surface(mesh, count)
            color = self._trimesh_color(mesh)
            colors = np.tile(color, (len(points), 1))
            if len(face_indices) == len(points):
                face_colors = self._trimesh_face_colors(mesh, face_indices)
                if face_colors is not None:
                    colors = face_colors
            sampled_points.append(points)
            sampled_colors.append(colors)

        if sharp_sample_count > 0:
            total_sharp_length = float(np.sum(sharp_lengths))
            sharp_counts = np.floor(sharp_lengths / total_sharp_length * sharp_sample_count).astype(int)
            sharp_remainder = sharp_sample_count - int(np.sum(sharp_counts))
            if sharp_remainder > 0:
                order = np.argsort(-(sharp_lengths / total_sharp_length * sharp_sample_count - sharp_counts))
                sharp_counts[order[:sharp_remainder]] += 1

            for mesh, count in zip(meshes, sharp_counts):
                if count <= 0:
                    continue
                points, colors = self._sample_trimesh_sharp_edges(mesh, count)
                if len(points) > 0:
                    sampled_points.append(points)
                    sampled_colors.append(colors)

        if not sampled_points:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
        return np.vstack(sampled_points), np.vstack(sampled_colors).astype(np.uint8)

    def _trimesh_sharp_edges(self, mesh: trimesh.Trimesh, angle_threshold_degrees: float = 30.0):
        if mesh is None or len(mesh.faces) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)
        face_adjacency = getattr(mesh, "face_adjacency", None)
        adjacency_edges = getattr(mesh, "face_adjacency_edges", None)
        if face_adjacency is None or adjacency_edges is None:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)
        normals = np.asarray(mesh.face_normals)
        if len(normals) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)
        n0 = normals[face_adjacency[:, 0]]
        n1 = normals[face_adjacency[:, 1]]
        cos_angles = np.clip(np.sum(n0 * n1, axis=1), -1.0, 1.0)
        angles = np.arccos(cos_angles)
        sharp_edges = np.asarray(adjacency_edges)[angles >= np.radians(angle_threshold_degrees)]
        if len(sharp_edges) == 0:
            return sharp_edges, np.empty((0,), dtype=float)
        edge_vectors = mesh.vertices[sharp_edges[:, 1]] - mesh.vertices[sharp_edges[:, 0]]
        lengths = np.linalg.norm(edge_vectors, axis=1)
        valid = lengths > 1e-12
        return sharp_edges[valid], lengths[valid]

    def _trimesh_sharp_edge_length(self, mesh: trimesh.Trimesh):
        _, lengths = self._trimesh_sharp_edges(mesh)
        return float(np.sum(lengths))

    def _sample_trimesh_sharp_edges(self, mesh: trimesh.Trimesh, sample_count: int):
        sharp_edges, lengths = self._trimesh_sharp_edges(mesh)
        if len(sharp_edges) == 0 or sample_count <= 0:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
        total_length = float(np.sum(lengths))
        if total_length <= 1e-12:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
        edge_indices = np.random.choice(len(sharp_edges), size=sample_count, p=lengths / total_length)
        ts = np.random.random(sample_count)[:, None]
        chosen = sharp_edges[edge_indices]
        points = (1.0 - ts) * mesh.vertices[chosen[:, 0]] + ts * mesh.vertices[chosen[:, 1]]
        color = self._trimesh_color(mesh)
        colors = np.tile(color, (len(points), 1))
        return points, colors

    @staticmethod
    def _trimesh_color(mesh: trimesh.Trimesh):
        default = np.array([160, 160, 160], dtype=np.uint8)
        material = getattr(mesh.visual, "material", None)
        color = getattr(material, "baseColorFactor", None) if material is not None else None
        if color is None:
            color = getattr(material, "diffuse", None) if material is not None else None
        if color is None:
            return default
        arr = np.array(color[:3], dtype=float)
        if np.max(arr) <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    @staticmethod
    def _trimesh_face_colors(mesh: trimesh.Trimesh, face_indices):
        face_colors = getattr(mesh.visual, "face_colors", None)
        if face_colors is None or len(face_colors) == 0:
            return None
        colors = np.asarray(face_colors)[face_indices, :3]
        return np.clip(colors, 0, 255).astype(np.uint8)

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

    @staticmethod
    def _view_aux_paths(output_path: str, suffix: str, ext: str):
        base = os.path.splitext(os.path.basename(output_path))[0]
        return os.path.join(os.path.dirname(output_path), f"{base}_{suffix}.{ext}")

    def _write_depth_outputs(self, output_path: str, depth: np.ndarray) -> float:
        png_path = self._view_aux_paths(output_path, "depth", "png")
        depth_scale = util.compute_depth_encode_scale(depth)
        depth_png = util.encode_depth_uint16(depth, depth_scale)
        imageio.imwrite(png_path, depth_png)
        print(f"✅ Depth map exported: {png_path} (uint16, depth_m * {depth_scale:.6f})")
        return depth_scale

    def _semantic_entries(self):
        used_colors = set()
        floor_info = self.mesh_nodes.get("floor")
        if floor_info:
            color_key = "entity:floor:floor"
            yield "floor", color_key, floor_info.get("mesh"), {
                "category": "floor",
                "label": "floor",
                "color": list(util.semantic_entity_color(color_key, used=used_colors)),
            }
        ceiling_info = self.mesh_nodes.get("ceiling")
        if ceiling_info:
            color_key = "entity:ceiling:ceiling"
            yield "ceiling", color_key, ceiling_info.get("mesh"), {
                "category": "ceiling",
                "label": "ceiling",
                "color": list(util.semantic_entity_color(color_key, used=used_colors)),
            }
        for category in ("walls", "doors", "windows"):
            for object_id, info in self.mesh_nodes.get(category, {}).items():
                color_key = f"entity:{category}:{object_id}"
                detail_key = {"walls": "wall_data", "doors": "door_data", "windows": "window_data"}[category]
                data = info.get(detail_key, {})
                caption = data.get("caption")
                object_meta = {
                    "category": category,
                    "label": object_id,
                    "color": list(util.semantic_entity_color(color_key, used=used_colors)),
                }
                if caption:
                    object_meta["caption"] = caption
                yield category, color_key, info.get("mesh"), {
                    **object_meta,
                }
        for object_id, info in self.mesh_nodes.get("boxes", {}).items():
            data = info.get("box_data", {})
            label = data.get("label", data.get("class", object_id))
            color_key = f"entity:boxes:{object_id}"
            object_meta = {
                "category": "boxes",
                "label": label,
                "color": list(util.semantic_entity_color(color_key, used=used_colors)),
            }
            caption = data.get("caption")
            if caption:
                object_meta["caption"] = caption
            yield "boxes", color_key, info.get("mesh"), object_meta

    def _semantic_metadata(self, objects: List[Dict[str, Any]]):
        return {
            "semantic": True,
            "format": "per_entity_color_png",
            "background": list(util.SEMANTIC_BACKGROUND),
            "objects": objects,
        }

    def _semantic_color(self, color_key: str):
        return util.semantic_entity_color(color_key)

    def render_semantic_png(self, output_path: str, camera_pose: np.ndarray, fov_y: float,
                            aspect: float, width: int, height: int,
                            znear: float = 0.05, zfar: Optional[float] = None,
                            renderer: Optional[pyrender.OffscreenRenderer] = None):
        semantic_path = self._view_aux_paths(output_path, "semantic", "png")
        metadata_path = self._view_aux_paths(output_path, "semantic", "json")
        semantic_scene = pyrender.Scene(bg_color=[0, 0, 0, 255], ambient_light=[1.0, 1.0, 1.0])
        mesh_count = 0
        objects = []
        for _, color_key, mesh_obj, object_meta in self._semantic_entries():
            color = np.array(list(object_meta["color"]) + [255], dtype=np.uint8)
            objects.append(object_meta)
            for mesh in self._flatten_trimesh_meshes(mesh_obj):
                if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    continue
                colored = mesh.copy()
                colored.visual = trimesh.visual.ColorVisuals(
                    mesh=colored,
                    vertex_colors=np.tile(color, (len(colored.vertices), 1)),
                )
                semantic_scene.add(pyrender.Mesh.from_trimesh(colored, smooth=False))
                mesh_count += 1
        if mesh_count == 0:
            return
        zfar_value = zfar if zfar is not None else 1000.0
        semantic_scene.add(
            pyrender.PerspectiveCamera(yfov=float(fov_y), aspectRatio=float(aspect), znear=znear, zfar=zfar_value),
            pose=camera_pose,
        )
        own_renderer = renderer is None
        if renderer is None:
            renderer = pyrender.OffscreenRenderer(width, height)
        try:
            color, _ = renderer.render(semantic_scene, flags=pyrender.RenderFlags.FLAT)
            imageio.imwrite(semantic_path, color)
            util.attach_semantic_bbox_2d(objects, color)
        finally:
            if own_renderer:
                renderer.delete()
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._semantic_metadata(objects), f, indent=2, ensure_ascii=False)
        bbox_overlay_path = util.save_bbox_2d_overlay_png(output_path, objects)
        if bbox_overlay_path:
            print(f"✅ Bounding box visualization: {bbox_overlay_path}")
        print(f"✅ Semantic map exported: {semantic_path}")

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

    def write_opencv_ssl_for_view(
        self,
        output_dir: str,
        ref_camera,
        ref_look_at,
        ref_world_up=None,
    ) -> str:
        from . import ssl_opencv

        return ssl_opencv.write_opencv_ssl(
            self.context,
            output_dir,
            ref_camera,
            ref_look_at,
            ref_world_up if ref_world_up is not None else [0.0, 0.0, 1.0],
        )

    def normalized_topdown_view(
        self,
        output_dir: str,
        show_ceiling: bool = False,
        round_decimals: int = 2,
        render_depth: bool = False,
        render_semantic: bool = False,
        transparent_alpha: float = 0.0,
        align: Optional[Dict[str, Any]] = None,
        write_ssl: bool = True,
        **kwargs,
    ):
        """Pixel-aligned top-down view (pyrender): translate SSL, render, and write camera_para.json.

        Resolution is fixed at 1000×1000; not configurable.
        """
        width = util_data.NORMALIZED_TOPDOWN_WIDTH
        height = util_data.NORMALIZED_TOPDOWN_HEIGHT
        for key in (
            "export_glb", "export_point_cloud", "visible_geometry",
            "rebuild", "use_HDRI", "geometry_mode", "show_wall", "show_window", "show_door",
            "align_height", "auto_fov", "manual_fov", "glb_path",
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
        if write_ssl:
            util_data.write_standard_ssl_to_path(self.context, ssl_path)

        self.clear_scene()
        self.construct_scene(show_ceiling=show_ceiling)
        self.setup_lighting()

        meta = self.context["meta"]
        camera_pos = np.array(align["camera_position_ssl"], dtype=float)
        look_at = list(align["look_at_target_ssl"])
        fov_y = float(align["fov_y"])

        camera_pose = np.eye(4)
        camera_pose[:3, 1], camera_pose[:3, 2], camera_pose[:3, 3] = [0, 1, 0], [0, 0, 1], camera_pos

        trans_ids = util.find_walls_to_make_transparent(
            camera_pos[:2], look_at[:2], meta["vertices"], self.context["walls"]
        )
        for wid in trans_ids:
            util.set_mesh_alpha(self.mesh_nodes, "walls", wid, transparent_alpha)

        znear = 0.05
        zfar = max(float(camera_pos[2] + meta["z_max"] + max(meta["span"]) * 2.0), znear + 1.0)
        camera_node = self.scene.add(
            pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=width / height, znear=znear, zfar=zfar),
            pose=camera_pose,
        )
        renderer = pyrender.OffscreenRenderer(width, height)
        color, depth = renderer.render(self.scene)
        imageio.imwrite(png_path, color)

        depth_scale = None
        if render_depth:
            depth_scale = self._write_depth_outputs(png_path, depth)
        if render_semantic:
            self.render_semantic_png(
                png_path, camera_pose, fov_y, width / height, width, height, znear, zfar, renderer=renderer
            )

        for wid in trans_ids:
            util.reset_mesh_alpha(self.mesh_nodes, "walls", wid)
        self.scene.remove_node(camera_node)
        renderer.delete()

        camera_para = dict(align["camera_para"])
        if depth_scale is not None:
            camera_para["depth_scale"] = float(depth_scale)
            camera_para["depth_unit"] = "meter"
            camera_para["is_metric_depth"] = True
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
        )
        print(f"✅ Pixel-aligned top-down view complete: {output_dir}")

    def topdown_view(self, output_path: str, width: int = 1024, height: int = 1024, **kwargs):
        if not output_path.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".webp")):
            output_path = util.resolve_topdown_image_path(output_path)
        view_dir = os.path.dirname(output_path) or "."
        export_glb = kwargs.pop("export_glb", False)
        glb_path = kwargs.pop("glb_path", None)
        export_point_cloud = kwargs.pop("export_point_cloud", False)
        visible_geometry = kwargs.pop("visible_geometry", False)
        show_ceiling = kwargs.pop("show_ceiling", False)
        rebuild = bool(kwargs.pop("rebuild", False))
        kwargs.pop("use_HDRI", None)
        render_depth = bool(kwargs.pop("render_depth", False))
        kwargs.pop("render_depth_scale", None)
        render_semantic = bool(kwargs.pop("render_semantic", False))
        kwargs.pop("view_transform", None)
        transparent_alpha = kwargs.pop("transparent_alpha", 0.0)
        construct_kwargs = {
            key: kwargs[key]
            for key in ("show_wall", "show_window", "show_door", "use_bbox_geometry")
            if key in kwargs
        }
        construct_kwargs["show_ceiling"] = show_ceiling
        meta = self.context["meta"]
        world_cam_w = [
            meta["center"][0],
            meta["center"][1],
            meta["z_max"] + max(meta["span"]) * 1.5,
        ]
        world_look_w = [meta["center"][0], meta["center"][1], 0.0]
        world_up_raw = [0.0, 1.0, 0.0]

        with util_data.ViewSslSession(self, False) as vss:
            vss.setup(world_cam_w, world_look_w, world_up_raw)

            if rebuild or self._scene_show_ceiling != bool(show_ceiling):
                self._reset_render_state()
            if rebuild or self.scene is None:
                self.construct_scene(**construct_kwargs)
            self.setup_lighting()

            camera_height = meta["z_max"] + max(meta["span"]) * 1.5
            camera_pos = np.array([meta["center"][0], meta["center"][1], camera_height])
            look_at = [meta["center"][0], meta["center"][1], 0.0]
            world_up = np.array(world_up_raw, dtype=float)

            fov_y = util.calculate_optimal_fov(
                camera_pos, look_at,
                meta["vertices"], meta["z_max"], meta["bounds"],
                self.config.get('indoor_fov', 160), self.config.get('outdoor_fov_scale', 1.05),
            )

            camera_pose = np.eye(4)
            camera_pose[:3, 1], camera_pose[:3, 2], camera_pose[:3, 3] = [0, 1, 0], [0, 0, 1], camera_pos

            trans_ids = util.find_walls_to_make_transparent(
                camera_pos[:2], meta["center"][:2], meta["vertices"], self.context["walls"]
            )
            for wid in trans_ids:
                util.set_mesh_alpha(self.mesh_nodes, "walls", wid, transparent_alpha)

            znear = 0.05
            zfar = max(float(camera_pos[2] + meta["z_max"] + max(meta["span"]) * 2.0), znear + 1.0)
            camera_node = self.scene.add(
                pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=width / height, znear=znear, zfar=zfar),
                pose=camera_pose,
            )
            renderer = pyrender.OffscreenRenderer(width, height)
            color, depth = renderer.render(self.scene)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            imageio.imwrite(output_path, color)
            depth_scale = None
            if render_depth:
                depth_scale = self._write_depth_outputs(output_path, depth)
            if visible_geometry and (export_glb or export_point_cloud):
                self.export_visible_geometry(
                    os.path.dirname(output_path),
                    camera_pose,
                    fov_y,
                    width / height,
                    export_glb=export_glb,
                    export_point_cloud=export_point_cloud,
                    transparent_keys={("walls", wid) for wid in trans_ids},
                    clip_start=znear,
                    clip_end=zfar,
                )
            if render_semantic:
                self.render_semantic_png(
                    output_path, camera_pose, fov_y, width / height, width, height, znear, zfar, renderer=renderer
                )

            for wid in trans_ids:
                util.reset_mesh_alpha(self.mesh_nodes, "walls", wid)
            self.scene.remove_node(camera_node)
            renderer.delete()
            para_path = os.path.join(
                os.path.dirname(output_path) or ".",
                f"{os.path.splitext(os.path.basename(output_path))[0]}_camera_para.json",
            )
            camera_para = vss.build_camera_para(
                world_cam_w,
                world_look_w,
                vss.world_up,
                float(fov_y),
                float(width) / float(height),
                reference_frame=True,
                depth_scale=depth_scale,
                width=width,
                height=height,
            )
            with open(para_path, 'w') as f:
                json.dump(camera_para, f, indent=4)
            self.write_opencv_ssl_for_view(view_dir, world_cam_w, world_look_w, vss.world_up)
            if export_glb and not visible_geometry:
                self.export_glb(glb_path or os.path.splitext(output_path)[0] + ".glb")
        print(f"✅ Top-down view saved to: {output_path}")

    def render_view(self, output_path: str, camera_position: list, look_at_target: list = None, **kwargs):
        export_glb = kwargs.pop("export_glb", False)
        glb_path = kwargs.pop("glb_path", None)
        export_point_cloud = kwargs.pop("export_point_cloud", False)
        visible_geometry = kwargs.pop("visible_geometry", False)
        width = int(kwargs.pop("width", 1024))
        height = int(kwargs.pop("height", 1024))
        up_vector_kw = kwargs.pop("up_vector", None)
        auto_fov = kwargs.pop("auto_fov", True)
        manual_fov = kwargs.pop("manual_fov", None)
        auto_transparent = kwargs.pop("auto_transparent", True)
        transparent_alpha = kwargs.pop("transparent_alpha", 0.0)
        kwargs.pop("use_HDRI", None)
        render_depth = bool(kwargs.pop("render_depth", False))
        kwargs.pop("render_depth_scale", None)
        render_semantic = bool(kwargs.pop("render_semantic", False))
        kwargs.pop("view_transform", None)
        rebuild = bool(kwargs.pop("rebuild", False))
        show_ceiling = kwargs.pop("show_ceiling", True)

        if not output_path.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".webp")):
            output_path = util.resolve_view_image_path(output_path)
        view_dir = os.path.dirname(output_path) or "."

        meta = self.context["meta"]
        center = meta["center"]
        z_max = meta["z_max"]
        if look_at_target is None:
            world_look_w = [center[0], center[1], z_max / 2]
        else:
            world_look_w = list(look_at_target)
        world_cam_w = list(camera_position)
        world_up_raw = geo_cv.resolve_render_view_up_vector(
            world_cam_w, world_look_w, up_vector_kw
        )

        construct_kwargs = {
            key: kwargs[key]
            for key in ("show_wall", "show_window", "show_door", "use_bbox_geometry")
            if key in kwargs
        }
        construct_kwargs["show_ceiling"] = show_ceiling

        with util_data.ViewSslSession(self, False) as vss:
            vss.setup(world_cam_w, world_look_w, world_up_raw)

            if rebuild or self._scene_show_ceiling != bool(show_ceiling):
                self._reset_render_state()
            if rebuild or self.scene is None:
                self.construct_scene(**construct_kwargs)
            self.setup_lighting()

            bounds = self.context["meta"]["bounds"]
            if look_at_target is None:
                look_at_target = [center[0], center[1], z_max / 2]
            up_vector = np.array(
                geo_cv.resolve_render_view_up_vector(
                    camera_position, look_at_target, up_vector_kw
                ),
                dtype=float,
            )

            if np.linalg.norm(up_vector) < 1e-6:
                up_vector = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                up_vector = up_vector / np.linalg.norm(up_vector)

            camera_position = np.array(camera_position, dtype=float)
            look_at_target = np.array(look_at_target, dtype=float)

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

            camera_pose = np.eye(4)
            camera_pose[:3, 0] = right
            camera_pose[:3, 1] = up
            camera_pose[:3, 2] = -forward
            camera_pose[:3, 3] = camera_position

            if manual_fov is not None:
                fov_y = np.radians(float(manual_fov))
            elif auto_fov:
                indoor_fov = self.config.get('indoor_fov', 120)
                outdoor_fov_scale = self.config.get('outdoor_fov_scale', 1.05)
                fov_y = util.calculate_optimal_fov(
                    camera_position,
                    look_at_target,
                    meta["vertices"],
                    z_max,
                    bounds,
                    indoor_fov,
                    outdoor_fov_scale,
                )
            else:
                fov_y = np.radians(70.0)

            trans_ids = []
            if auto_transparent:
                trans_ids = util.find_walls_to_make_transparent(
                    camera_position[:2],
                    look_at_target[:2],
                    meta["vertices"],
                    self.context["walls"],
                )
                for wid in trans_ids:
                    util.set_mesh_alpha(self.mesh_nodes, "walls", wid, transparent_alpha)

            znear = 0.05
            scene_diag = float(np.linalg.norm([bounds[2] - bounds[0], bounds[3] - bounds[1], z_max]))
            zfar = max(float(np.linalg.norm(camera_position - look_at_target) + scene_diag * 2.0), znear + 1.0)
            camera_node = self.scene.add(
                pyrender.PerspectiveCamera(yfov=float(fov_y), aspectRatio=width / height, znear=znear, zfar=zfar),
                pose=camera_pose,
            )
            renderer = pyrender.OffscreenRenderer(width, height)
            depth_scale = None
            try:
                color, depth = renderer.render(self.scene)
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                imageio.imwrite(output_path, color)
                if render_depth:
                    depth_scale = self._write_depth_outputs(output_path, depth)
                if visible_geometry and (export_glb or export_point_cloud):
                    self.export_visible_geometry(
                        os.path.dirname(output_path),
                        camera_pose,
                        float(fov_y),
                        width / height,
                        export_glb=export_glb,
                        export_point_cloud=export_point_cloud,
                        transparent_keys={("walls", wid) for wid in trans_ids},
                        clip_start=znear,
                        clip_end=zfar,
                    )
                if render_semantic:
                    self.render_semantic_png(
                        output_path, camera_pose, float(fov_y), width / height, width, height, znear, zfar, renderer=renderer
                    )
            finally:
                renderer.delete()
                self.scene.remove_node(camera_node)
                for wid in trans_ids:
                    util.reset_mesh_alpha(self.mesh_nodes, "walls", wid)

            para_path = os.path.join(
                os.path.dirname(output_path) or ".",
                f"{os.path.splitext(os.path.basename(output_path))[0]}_camera_para.json",
            )
            camera_para = vss.build_camera_para(
                world_cam_w,
                world_look_w,
                vss.world_up,
                float(fov_y),
                float(width) / float(height),
                reference_frame=True,
                depth_scale=depth_scale if render_depth else None,
                width=width,
                height=height,
            )
            with open(para_path, 'w') as f:
                json.dump(camera_para, f, indent=4)
            view_dir = os.path.dirname(output_path) or "."
            self.write_opencv_ssl_for_view(view_dir, world_cam_w, world_look_w, vss.world_up)

            if export_glb and not visible_geometry:
                self.export_glb(glb_path or os.path.splitext(output_path)[0] + ".glb")
            print(f"✅ View image saved to: {output_path}")

if __name__ == "__main__":
    pass
