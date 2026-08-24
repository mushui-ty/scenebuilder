"""
Blender helper utilities for scenebuilder_bpy.py.
"""

import os
import colorsys
import glob
import hashlib
import contextlib
import logging
import math
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import imageio
except ImportError:  # pragma: no cover
    imageio = None

import numpy as np

try:
    from . import util
except ImportError:  # pragma: no cover
    import util  # type: ignore


def _preload_bpy() -> None:
    """Silently preload bpy, suppressing harmless Swig ABI warnings from NumPy 1.x/2.x."""
    import sys
    import warnings

    if "bpy" in sys.modules:
        return

    @contextlib.contextmanager
    def _quiet_stdio():
        devnull = open(os.devnull, "w")
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            devnull.close()

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*SwigPy.*")
            warnings.filterwarnings("ignore", message=".*swigvarlink.*")
            with _quiet_stdio():
                import bpy  # noqa: F401
    except ImportError:
        pass


_preload_bpy()

try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix  # type: ignore[import]
except ImportError:  # pragma: no cover
    bpy = None
    bmesh = None
    Vector = None
    Matrix = None


STRUCTURAL_SUBDIVIDE_CELL = 0.18
STRUCTURAL_SUBDIVIDE_MAX_CUTS = 20
STRUCTURAL_SUBDIVIDE_MIN_CUTS = 2

# Floor API aliases (same tuning as walls).
FLOOR_TOP_SUBDIVIDE_CELL = STRUCTURAL_SUBDIVIDE_CELL
FLOOR_TOP_SUBDIVIDE_MAX_CUTS = STRUCTURAL_SUBDIVIDE_MAX_CUTS
FLOOR_TOP_SUBDIVIDE_MIN_CUTS = STRUCTURAL_SUBDIVIDE_MIN_CUTS


def _structural_subdivide_cuts(span: float, cell: float, min_cuts: int, max_cuts: int) -> int:
    """Pick subdiv cuts from a characteristic length (m); yields hundreds–~2000 tris per large face."""
    if span <= 1e-3 or cell <= 1e-3:
        return min_cuts
    return max(min_cuts, min(max_cuts, int(round(span / cell))))


def _floor_top_subdivide_cuts(span: float, cell: float, min_cuts: int, max_cuts: int) -> int:
    return _structural_subdivide_cuts(span, cell, min_cuts, max_cuts)


def _subdivide_mesh_bpy(
    mesh,
    span: float,
    *,
    cell: float = STRUCTURAL_SUBDIVIDE_CELL,
    min_cuts: int = STRUCTURAL_SUBDIVIDE_MIN_CUTS,
    max_cuts: int = STRUCTURAL_SUBDIVIDE_MAX_CUTS,
) -> None:
    """Triangulate and subdivide mesh faces for visibility ray sampling (UVs interpolate)."""
    if bpy is None or bmesh is None or mesh is None:
        return
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    subdivide_cuts = _structural_subdivide_cuts(span, float(cell), int(min_cuts), int(max_cuts))
    bmesh.ops.subdivide_edges(
        bm,
        edges=bm.edges[:],
        cuts=subdivide_cuts,
        use_grid_fill=True,
    )
    bm.to_mesh(mesh)
    bm.free()
    mesh.update(calc_edges=True)


def create_floor_mesh_bpy(
    scene_collection,
    vertices,
    thickness=0.1,
    name="Floor",
    texture_scale=1.0,
    top_subdivide_cell=FLOOR_TOP_SUBDIVIDE_CELL,
    top_subdivide_max_cuts=FLOOR_TOP_SUBDIVIDE_MAX_CUTS,
    top_subdivide_min_cuts=FLOOR_TOP_SUBDIVIDE_MIN_CUTS,
):
    """Create floor mesh with a subdivided top for per-view ray visibility tests.

    The room outline starts as one n-gon (only a handful of triangles after triangulation).
    That is too coarse for ray sampling: every centroid can sit under furniture. We
    triangulate and subdivide the top face before extrusion so centroids spread across
    the floor; typical rooms end up with hundreds to ~2000 top triangles (~10× coarser
    defaults) — still negligible vs furniture GLBs. Visibility uses the same ray-cast
    path as all other categories.
    """
    if bpy is None or len(vertices) < 3:
        return None

    pts2_raw = [(float(v[0]), float(v[1])) for v in vertices]
    pts_arr = np.array(pts2_raw)
    xs, ys = pts_arr[:, 0], pts_arr[:, 1]
    area2 = np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
    if area2 < 0:
        pts2_raw = pts2_raw[::-1]

    verts = [(x, y, 0.0) for x, y in pts2_raw]
    faces = [tuple(range(len(verts)))]

    mesh = bpy.data.meshes.new(name=name)
    obj = bpy.data.objects.new(name, mesh)
    scene_collection.objects.link(obj)

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            v_idx = mesh.loops[loop_index].vertex_index
            vx, vy, _ = mesh.vertices[v_idx].co
            uv_layer.data[loop_index].uv = (vx * texture_scale, vy * texture_scale)

    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1e-3)
    _subdivide_mesh_bpy(
        mesh,
        span,
        cell=top_subdivide_cell,
        min_cuts=top_subdivide_min_cuts,
        max_cuts=top_subdivide_max_cuts,
    )
    bm = bmesh.new()
    bm.from_mesh(mesh)
    extruded = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    extruded_verts = [elem for elem in extruded["geom"] if isinstance(elem, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=(0.0, 0.0, -float(thickness)), verts=extruded_verts)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update(calc_edges=True)
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")

    return obj


def apply_hdri_to_world(scene, hdri_path: str, strength: float = 1.0):
    if bpy is None or scene is None or not hdri_path:
        return False

    if not os.path.exists(hdri_path):
        return False

    world = scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    bg_node = nodes.get("Background")
    if not bg_node:
        bg_node = nodes.new("ShaderNodeBackground")
    env_node = nodes.get("EnvTex")
    if not env_node:
        env_node = nodes.new("ShaderNodeTexEnvironment")
        env_node.name = "EnvTex"

    try:
        image = bpy.data.images.load(hdri_path)
    except RuntimeError:
        image = bpy.data.images.get(os.path.basename(hdri_path))
    if image is None:
        return False

    env_node.image = image
    bg_node.inputs["Strength"].default_value = strength

    if not any(link.from_node == env_node and link.to_node == bg_node for link in links):
        links.new(env_node.outputs["Color"], bg_node.inputs["Color"])

    output_node = nodes.get("World Output")
    if not output_node:
        output_node = nodes.new("ShaderNodeOutputWorld")

    if not any(link.from_node == bg_node and link.to_node == output_node for link in links):
        links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

    return True


def create_single_wall_mesh_bpy(scene_collection, s, e, height, orientation,
                                chip=False, openings=None, wall_thickness=0.1, name="Wall",
                                outer_s=None, outer_e=None, texture_scale=1.0,
                                is_partition=False, extend_s=False, extend_e=False):
    if bpy is None:
        return None

    s_arr = np.array(s, dtype=float)[:2]
    e_arr = np.array(e, dtype=float)[:2]
    wall_vec = e_arr - s_arr
    wall_length = np.linalg.norm(wall_vec)
    if wall_length < 1e-6:
        return None

    wall_dir = wall_vec / wall_length
    # normal points into the room interior
    normal = np.array(orientation, dtype=float)[:2]
    
    # For partitions without a valid normal, construct one
    if is_partition and np.linalg.norm(normal) < 1e-4:
        normal = np.array([-wall_dir[1], wall_dir[0]])

    # Endpoint extension: if an endpoint connects, extend outward by half thickness to embed into adjacent walls
    s_base = s_arr - wall_dir * (wall_thickness / 2.0) if extend_s else s_arr
    e_base = e_arr + wall_dir * (wall_thickness / 2.0) if extend_e else e_arr

    if is_partition:
        # Partition wall: centered mode, offset half thickness along both normal directions
        inner_s_p = s_base + normal * (wall_thickness / 2.0)
        inner_e_p = e_base + normal * (wall_thickness / 2.0)
        outer_s_p = s_base - normal * (wall_thickness / 2.0)
        outer_e_p = e_base - normal * (wall_thickness / 2.0)
    else:
        # Exterior wall: one-sided offset (inner face on centerline, outer face offset outward)
        inner_s_p = s_base
        inner_e_p = e_base
        outward_normal = -normal
        if outer_s is None:
            outer_s_p = s_base + outward_normal * wall_thickness
        else:
            outer_s_p = np.array(outer_s, dtype=float)[:2]

        if outer_e is None:
            outer_e_p = e_base + outward_normal * wall_thickness
        else:
            outer_e_p = np.array(outer_e, dtype=float)[:2]

    mesh = bpy.data.meshes.new(name=name)
    obj = bpy.data.objects.new(name, mesh)
    scene_collection.objects.link(obj)

    # Vertices: 0-3 on one side, 4-7 on the other
    verts = [
        (inner_s_p[0], inner_s_p[1], 0.0),        # 0
        (inner_e_p[0], inner_e_p[1], 0.0),        # 1
        (inner_e_p[0], inner_e_p[1], height),     # 2
        (inner_s_p[0], inner_s_p[1], height),     # 3
        (outer_s_p[0], outer_s_p[1], 0.0),        # 4
        (outer_e_p[0], outer_e_p[1], 0.0),        # 5
        (outer_e_p[0], outer_e_p[1], height),     # 6
        (outer_s_p[0], outer_s_p[1], height),     # 7
    ]

    # Define 6 faces
    faces = [
        (0, 3, 2, 1), # side A
        (4, 5, 6, 7), # side B
        (3, 7, 6, 2), # top
        (0, 1, 5, 4), # bottom
        (0, 4, 7, 3), # end S
        (1, 2, 6, 5), # end E
    ]

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # Add wall UV coordinates
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for f_idx, poly in enumerate(mesh.polygons):
        for loop_index in poly.loop_indices:
            v_idx = mesh.loops[loop_index].vertex_index
            vx, vy, vz = verts[v_idx]
            if f_idx == 0: # side A
                u = np.dot(np.array([vx, vy]) - inner_s_p, wall_dir)
                uv_layer.data[loop_index].uv = (u * texture_scale, vz * texture_scale)
            elif f_idx == 1: # side B
                u = np.dot(np.array([vx, vy]) - outer_s_p, wall_dir)
                uv_layer.data[loop_index].uv = (u * texture_scale, vz * texture_scale)
            else:
                uv_layer.data[loop_index].uv = ((vx + vy) * texture_scale, vz * texture_scale)

    wall_span = max(float(wall_length), float(height), 1e-3)
    _subdivide_mesh_bpy(mesh, wall_span)

    if openings:
        for opening_data, opening_type in openings:
            opening_box = create_opening_box_bpy(
                opening_data, s_arr, e_arr, wall_dir, normal, height, 
                wall_thickness, opening_type, is_partition=is_partition
            )
            if opening_box:
                mod = obj.modifiers.new(name="Boolean", type='BOOLEAN')
                mod.operation = 'DIFFERENCE'
                mod.object = opening_box
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_apply(modifier=mod.name)
                bpy.data.objects.remove(opening_box, do_unlink=True)

    return obj


def calculate_miter_joints(walls: Dict[str, Any], wall_thickness: float) -> Dict[str, Any]:
    """Precompute miter-joint offset points for all exterior walls."""
    from . import util
    return util.calculate_miter_joints(walls, wall_thickness)


def create_opening_box_bpy(opening, wall_s, wall_e, wall_dir, normal, wall_height, wall_thickness=0.1, opening_type="window", is_partition=False):
    if bpy is None:
        return None

    center = np.array(opening["center"])
    width = opening["width"]
    height = opening["height"]

    # Base projection point
    proj = wall_s + wall_dir * np.dot(center[:2] - wall_s, wall_dir)
    
    if is_partition:
        # Partition opening: fully centered, extra thickness to cut through both sides
        final_center_2d = proj
        cutter_thickness = wall_thickness * 2.0
    else:
        # Exterior wall opening: original logic
        outward_normal = -np.array(normal)
        # Slightly offset opening center outward
        if opening_type == "door":
            center_offset = outward_normal * (wall_thickness / 2.0) * 0.90
            cutter_thickness = wall_thickness * 2
        else:
            center_offset = outward_normal * (wall_thickness / 2.0) * 0.90
            cutter_thickness = wall_thickness * 2
        final_center_2d = proj + center_offset

    z_bottom = center[2] - height / 2
    z_top = center[2] + height / 2

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    box_obj = bpy.context.object
    box_obj.name = f"OpeningBox_{opening_type}"
    box_obj.scale = (width, cutter_thickness, height)
    
    final_z = (z_bottom + z_top) / 2
    if opening_type == "door":
        final_z -= 0.01  # Lower door opening slightly so wall base is fully cut
    box_obj.location = (final_center_2d[0], final_center_2d[1], final_z)
    angle = np.arctan2(wall_dir[1], wall_dir[0])
    box_obj.rotation_euler = (0, 0, angle)

    return box_obj


def create_windows_and_doors_bpy(scene_collection, walls, door_texture_path, window_texture_path):
    if bpy is None:
        return []

    result = []
    for wall_id, wall in walls.items():
        wall_s = np.array(wall["s"])
        wall_e = np.array(wall["e"])
        wall_vec = wall_e - wall_s
        wall_length = np.linalg.norm(wall_vec)
        if wall_length < 1e-6:
            continue

        wall_dir = wall_vec / wall_length

        if door_texture_path:
            for door_id, door in wall.get("doors", {}).items():
                door_mesh = create_door_or_window_mesh_bpy(
                    scene_collection, door, wall_s, wall_e, wall_dir, door_texture_path, "door"
                )
                if door_mesh:
                    result.append({
                        "type": "door",
                        "id": door_id,
                        "wall_id": wall_id,
                        "mesh": door_mesh
                    })

        if window_texture_path:
            for window_id, window in wall.get("windows", {}).items():
                window_mesh = create_door_or_window_mesh_bpy(
                    scene_collection, window, wall_s, wall_e, wall_dir, window_texture_path, "window"
                )
                if window_mesh:
                    result.append({
                        "type": "window",
                        "id": window_id,
                        "wall_id": wall_id,
                        "mesh": window_mesh
                    })

    return result


def create_door_or_window_mesh_bpy(scene_collection, opening, wall_s, wall_e, wall_dir, texture_path, item_type):
    if bpy is None:
        return None

    center = np.array(opening["center"])
    width = opening["width"]
    height = opening["height"]
    proj = wall_s + wall_dir * np.dot(center[:2] - wall_s, wall_dir)
    z_bottom = center[2] - height / 2
    z_center = center[2]

    mesh = bpy.data.meshes.new(name=f"{item_type}_mesh")
    obj = bpy.data.objects.new(f"{item_type}", mesh)
    scene_collection.objects.link(obj)

    verts = [
        (-width / 2, 0, -height / 2),
        (width / 2, 0, -height / 2),
        (width / 2, 0, height / 2),
        (-width / 2, 0, height / 2),
    ]
    faces = [(0, 1, 2, 3)]

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for i, loop_idx in enumerate(mesh.polygons[0].loop_indices):
        uv_layer[loop_idx].uv = uvs[i]

    obj.location = (proj[0], proj[1], z_center)
    angle = np.arctan2(wall_dir[1], wall_dir[0])
    obj.rotation_euler = (0, 0, angle)

    if texture_path and os.path.exists(texture_path):
        mat = create_material_with_texture(f"{item_type}_mat", texture_path)
        if mat:
            obj.data.materials.append(mat)

    return obj


def create_ceiling_mesh_bpy(scene_collection, vertices, z_max, thickness=0.2, name="Ceiling", texture_scale=1.0):
    if bpy is None or len(vertices) < 3:
        return None

    pts2_raw = [(float(v[0]), float(v[1])) for v in vertices]
    pts_arr = np.array(pts2_raw)
    xs, ys = pts_arr[:, 0], pts_arr[:, 1]
    area2 = np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
    if area2 < 0:
        pts2_raw = pts2_raw[::-1]

    verts = [(x, y, z_max) for x, y in pts2_raw]
    faces = [tuple(range(len(verts)))]

    mesh = bpy.data.meshes.new(name=name)
    obj = bpy.data.objects.new(name, mesh)
    scene_collection.objects.link(obj)

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # Add ceiling UV coordinates (x, y)
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            v_idx = mesh.loops[loop_index].vertex_index
            vx, vy, _ = verts[v_idx]
            uv_layer.data[loop_index].uv = (vx * texture_scale, vy * texture_scale)

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0.0, 0.0, thickness)})
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh.update(calc_edges=True)
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")

    return obj


def create_bbox_geometry(center, scale, angle_z, color=None):
    if bpy is None:
        return None

    center_arr = np.array(center)
    scale_arr = np.array(scale)
    if color is None:
        color = get_class_color(None)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center_arr.tolist())
    cube = bpy.context.object
    cube.scale = (scale_arr[0], scale_arr[1], scale_arr[2])
    cube.rotation_euler = (0, 0, np.radians(angle_z))

    arrow_length = max(1e-3, min(scale_arr[0], scale_arr[1]) * 0.3)
    arrow_radius = max(arrow_length * 0.05, 0.02)

    bpy.ops.mesh.primitive_cylinder_add(
        radius=arrow_radius, depth=arrow_length * 0.75, location=center_arr.tolist()
    )
    arrow_body = bpy.context.object

    bpy.ops.mesh.primitive_cone_add(
        radius1=arrow_radius * 3, depth=arrow_length * 0.25, location=center_arr.tolist()
    )
    arrow_head = bpy.context.object

    arrow_body.rotation_euler = (np.pi / 2, 0, 0)
    arrow_head.rotation_euler = (np.pi / 2, 0, 0)

    arrow_offset = scale_arr[2] / 2 + arrow_length * 0.375
    arrow_body.location = (
        center_arr[0],
        center_arr[1] - arrow_length * 0.375,
        center_arr[2] + arrow_offset,
    )
    arrow_head.location = (
        center_arr[0],
        center_arr[1] - arrow_length * 0.875,
        center_arr[2] + arrow_offset,
    )

    arrow_body.rotation_euler = (np.pi / 2, 0, np.radians(angle_z))
    arrow_head.rotation_euler = (np.pi / 2, 0, np.radians(angle_z))

    bpy.ops.object.select_all(action='DESELECT')
    cube.select_set(True)
    arrow_body.select_set(True)
    arrow_head.select_set(True)
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.join()

    combined_obj = bpy.context.object
    mat = bpy.data.materials.new(name="BboxMaterial")
    mat.use_nodes = True
    combined_obj.data.materials.append(mat)

    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color

    return combined_obj


def _get_combined_bounds(objects: List[Any]):
    """Return combined bounding box (min/max corners) for a list of objects."""
    if Vector is None or not objects:
        return None, None

    corners = []
    for obj in objects:
        if obj.type != 'MESH':
            continue
        bbox = getattr(obj, "bound_box", None)
        if not bbox:
            continue
        for corner in bbox:
            try:
                world_corner = obj.matrix_world @ Vector(corner)
            except Exception:
                continue
            corners.append(world_corner)

    if not corners:
        return None, None

    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    zs = [corner[2] for corner in corners]

    min_corner = Vector((min(xs), min(ys), min(zs)))
    max_corner = Vector((max(xs), max(ys), max(zs)))
    return min_corner, max_corner


def _calculate_center(objects: List[Any]):
    """Compute the center of an object's bounding box."""
    bounds = _get_combined_bounds(objects)
    if not bounds or bounds[0] is None or bounds[1] is None:
        return None
    return (bounds[0] + bounds[1]) / 2


def load_mesh_to_origin(asset_id: int, model_root: Union[str, List[str]]):
    """
    Import the GLTF/GLB model for the given asset_id and move geometry center to origin.
    Supports lookup across multiple root directories.
    """
    if bpy is None or Vector is None or not model_root:
        return None

    # Normalize to list for uniform handling
    if isinstance(model_root, str):
        roots = [model_root]
    else:
        roots = model_root

    candidates = []
    for root in roots:
        candidates.append(os.path.join(root, f"{asset_id}.glb"))
        candidates.append(os.path.join(root, f"{asset_id}.gltf"))

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue

        # Record object pointers before import to identify newly added objects afterward
        before_ids = {obj.as_pointer() for obj in bpy.data.objects}
        try:
            with quiet_stdio():
                bpy.ops.import_scene.gltf(
                    filepath=candidate,
                )
        except Exception:
            continue

        new_objects = [
            obj for obj in bpy.data.objects
            if obj.as_pointer() not in before_ids
        ]
        if not new_objects:
            continue

        # Compute geometry center of new objects first (before setting parent)
        bounds = _get_combined_bounds(new_objects)
        if bounds and bounds[0] is not None and bounds[1] is not None:
            center = (bounds[0] + bounds[1]) / 2
        else:
            center = Vector((0.0, 0.0, 0.0))
        
        # Create an empty container as the root for this imported mesh
        container = bpy.data.objects.new(f"mesh_{asset_id}_root", None)
        bpy.context.scene.collection.objects.link(container)

        
        for obj in new_objects:
            obj.parent = container
        # Set parent with default parent inverse (children follow container movement)
        # Overall logic: container defaults to location 0, so imported children have no displacement relative to their original positions
        # Parent can be set before or after; then:
        # Set container location to -center so children shift by -center relative to their original positions
        # This ensures imported child bbox geometric center is at the origin
        container.location = -center
        # Force scene update after changing node properties so transforms take effect
        bpy.context.view_layer.update()
        
        return container

    return None


def apply_box_transform(container_obj: Any, box: Dict, master_bounds: Optional[tuple] = None):
    """
    Apply scale, rotation, and translation to an imported mesh root (or instance object).
    master_bounds: if provided, use this bbox directly for scaling instead of scanning children.
    """
    if bpy is None or Vector is None or container_obj is None:
        return False

    if master_bounds and master_bounds[0] is not None:
        min_corner, max_corner = master_bounds
    else:
        # Collection instance without master_bounds: try instance_collection
        if container_obj.instance_type == 'COLLECTION' and container_obj.instance_collection:
            child_objects = [obj for obj in container_obj.instance_collection.objects if obj.parent is None]
        else:
            child_objects = list(container_obj.children)
            
        if not child_objects:
            # Fallback: no children but object itself is MESH
            if container_obj.type == 'MESH':
                child_objects = [container_obj]
            else:
                return False

        bounds = _get_combined_bounds(child_objects)
        if not bounds or bounds[0] is None or bounds[1] is None:
            return False
        min_corner, max_corner = bounds

    extent = np.array((max_corner - min_corner), dtype=float)
    # ... remaining logic unchanged ...
    current_center = (min_corner + max_corner) / 2
    extent = np.where(extent == 0.0, 1.0, extent)

    target_scale = np.array(box.get("scale", [1.0, 1.0, 1.0]), dtype=float)
    scale_factors = target_scale / extent

    if Matrix is None:
        return False

    angle_z = float(box.get("angle_z", 0.0))
    theta = np.radians(angle_z)

    target_center = box.get("center", [0.0, 0.0, 0.0])
    center_vec = Vector((float(target_center[0]),
                         float(target_center[1]),
                         float(target_center[2])))

    # Debug output
    # print(f"    [debug] object label: {box.get('label', 'unknown')}")
    # print(f"    [debug] current geometry center: ({current_center[0]:.3f}, {current_center[1]:.3f}, {current_center[2]:.3f})")
    # print(f"    [debug] current extent: ({extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f})")
    # print(f"    [debug] target center: ({target_center[0]:.3f}, {target_center[1]:.3f}, {target_center[2]:.3f})")
    # print(f"    [debug] target scale: ({target_scale[0]:.3f}, {target_scale[1]:.3f}, {target_scale[2]:.3f})")
    # print(f"    [debug] bottom z (min): {min_corner[2]:.3f}, top z (max): {max_corner[2]:.3f}")

    rotation_mat = Matrix.Rotation(theta, 4, 'Z')
    # rotation_mat_fixed = Matrix.Rotation(np.radians(-90), 4, 'X')
    scale_mat = Matrix.Identity(4)
    scale_mat[0][0] = float(scale_factors[0])
    scale_mat[1][1] = float(scale_factors[1])
    scale_mat[2][2] = float(scale_factors[2])

    stage2_transform = Matrix.Translation(center_vec) @ rotation_mat @ scale_mat
    previous_matrix = container_obj.matrix_world.copy()
    container_obj.matrix_world = stage2_transform @ previous_matrix
    
    # Force scene update so transforms take effect immediately
    bpy.context.view_layer.update()
    
    # Verify final position (non-instanced objects or those with child_objects defined)
    if 'child_objects' in locals() and child_objects:
        final_bounds = _get_combined_bounds(child_objects)
        if final_bounds and final_bounds[0] is not None:
            final_center = (final_bounds[0] + final_bounds[1]) / 2
            # print(f"    [debug] geometry center after transform: ({final_center[0]:.3f}, {final_center[1]:.3f}, {final_center[2]:.3f})")
            
            # Check for floating (height mismatch)
            expected_bottom = target_center[2] - target_scale[2] / 2
            actual_bottom = final_bounds[0][2]
            height_diff = actual_bottom - expected_bottom
            if abs(height_diff) > 0.01:  # deviation over 1 cm
                print(f"    ⚠️  Height mismatch: bottom should be at {expected_bottom:.3f}m, actual {actual_bottom:.3f}m, delta {height_diff:.3f}m")
    
    return True


def append_material_from_blend(blend_path):
    if bpy is None:
        return None

    abs_path = os.path.abspath(blend_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Blend file not found: {abs_path}")

    loaded_names = []
    with bpy.data.libraries.load(abs_path, link=False) as (data_from, data_to):
        data_to.materials = [name for name in data_from.materials if name]
        data_to.images = [name for name in data_from.images if name]
        loaded_names = list(data_to.materials)

    for name in loaded_names:
        mat = bpy.data.materials.get(name)
        if mat:
            return mat
    return None


def create_material_with_texture(name, texture_path):
    if bpy is None:
        return None

    mat = None
    if texture_path and texture_path.lower().endswith(".blend"):
        try:
            mat = append_material_from_blend(texture_path)
        except FileNotFoundError:
            mat = None

    if mat:
        return mat

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    tex_image = nodes.new('ShaderNodeTexImage')
    emit = nodes.new('ShaderNodeEmission')
    mix_shader = nodes.new('ShaderNodeAddShader')

    if texture_path and os.path.exists(texture_path):
        # [optimization] skip if texture already loaded
        img_name = os.path.basename(texture_path)
        img = bpy.data.images.get(img_name)
        
        # Even with same name, verify filepath matches
        if img and (not hasattr(img, 'filepath') or bpy.path.abspath(img.filepath) != bpy.path.abspath(texture_path)):
            img = None
            
        if img is None:
            try:
                img = bpy.data.images.load(texture_path)
            except Exception:
                img = None
        
        if img:
            tex_image.image = img

    links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(tex_image.outputs['Color'], emit.inputs['Color'])
    emit.inputs['Strength'].default_value = 1.0
    links.new(bsdf.outputs['BSDF'], mix_shader.inputs[0])
    links.new(emit.outputs['Emission'], mix_shader.inputs[1])
    links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])
    mat.use_backface_culling = False

    return mat


def create_material_with_color(name, color):
    if bpy is None:
        return None

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    emit = nodes.new('ShaderNodeEmission')
    mix_shader = nodes.new('ShaderNodeAddShader')
    output = nodes.get('Material Output')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color
    emit.inputs['Color'].default_value = color
    emit.inputs['Strength'].default_value = 1.0
    if output:
        links = mat.node_tree.links
        links.new(bsdf.outputs['BSDF'], mix_shader.inputs[0])
        links.new(emit.outputs['Emission'], mix_shader.inputs[1])
        links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])
    mat.use_backface_culling = False
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = 'NONE'
    return mat


def disable_shadows_for_all(scene_collection):
    if bpy is None:
        return

    for obj in scene_collection.all_objects:
        if hasattr(obj, "cycles_visibility"):
            obj.cycles_visibility.shadow = False
        for mat in obj.data.materials if hasattr(obj.data, "materials") else []:
            if mat and hasattr(mat, "shadow_method"):
                mat.shadow_method = 'NONE'


def apply_wall_transparency(obj, alpha):
    if bpy is None or obj is None:
        return None

    replacements = []
    for slot_idx, slot in enumerate(obj.material_slots):
        mat = slot.material
        if mat is None:
            continue

        transparent_mat = mat.copy()
        transparent_mat.name = f"{mat.name}_transparent"
        transparent_mat.blend_method = 'BLEND'
        if hasattr(transparent_mat, 'shadow_method'):
            transparent_mat.shadow_method = 'NONE'
        transparent_mat.use_backface_culling = False
        transparent_mat.use_nodes = True

        principled = next(
            (node for node in transparent_mat.node_tree.nodes if node.type == 'BSDF_PRINCIPLED'),
            None
        )
        if not principled:
            output_node = transparent_mat.node_tree.nodes.get('Material Output')
            if output_node is None:
                output_node = transparent_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
            principled = transparent_mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
            transparent_mat.node_tree.links.new(principled.outputs['BSDF'], output_node.inputs['Surface'])

        principled.inputs['Alpha'].default_value = alpha
        slot.material = transparent_mat

        replacements.append({
            "slot_idx": slot_idx,
            "original": mat,
            "replacement": transparent_mat
        })

    return replacements if replacements else None


def restore_wall_transparency(obj, records):
    if bpy is None or obj is None or not records:
        return

    for slot in records:
        slot_idx = slot["slot_idx"]
        original = slot["original"]
        replacement = slot["replacement"]
        if slot_idx < len(obj.material_slots):
            obj.material_slots[slot_idx].material = original

        try:
            bpy.data.materials.remove(replacement, do_unlink=True)
        except Exception:
            pass


def apply_object_transparency(obj, alpha):
    if bpy is None or obj is None:
        return None

    replacements = []
    for slot_idx, slot in enumerate(obj.material_slots):
        mat = slot.material
        if mat is None:
            continue

        transparent_mat = mat.copy()
        transparent_mat.name = f"{mat.name}_transparent"
        transparent_mat.blend_method = 'BLEND'
        if hasattr(transparent_mat, 'shadow_method'):
            transparent_mat.shadow_method = 'NONE'
        transparent_mat.use_backface_culling = False
        transparent_mat.use_nodes = True

        principled = next(
            (node for node in transparent_mat.node_tree.nodes if node.type == 'BSDF_PRINCIPLED'),
            None
        )
        if not principled:
            output_node = transparent_mat.node_tree.nodes.get('Material Output')
            if output_node is None:
                output_node = transparent_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
            principled = transparent_mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
            transparent_mat.node_tree.links.new(principled.outputs['BSDF'], output_node.inputs['Surface'])

        principled.inputs['Alpha'].default_value = alpha
        slot.material = transparent_mat

        replacements.append({
            "slot_idx": slot_idx,
            "original": mat,
            "replacement": transparent_mat
        })

    if not replacements:
        return None

    return {"object": obj, "slots": replacements}


def restore_object_transparency(record):
    if bpy is None or not record:
        return

    obj = record.get("object")
    slots = record.get("slots", [])
    if obj is None:
        return

    for slot in slots:
        slot_idx = slot["slot_idx"]
        original = slot["original"]
        replacement = slot["replacement"]
        if slot_idx < len(obj.material_slots):
            obj.material_slots[slot_idx].material = original

        try:
            bpy.data.materials.remove(replacement, do_unlink=True)
        except Exception:
            pass


@contextlib.contextmanager
def quiet_stdio(*, suppress_logging: bool = True):
    """Suppress stdout/stderr, including Blender C-level logs (Saved:, cycles progress)."""
    sys.stdout.flush()
    sys.stderr.flush()

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull_file = open(os.devnull, "w")
    loggers: List[Tuple[logging.Logger, int]] = []
    root_old_level = logging.root.level

    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        if suppress_logging:
            logging.root.setLevel(logging.CRITICAL + 1)
            for name in ("io_scene_gltf2", "io_scene_gltf", "bpy"):
                lg = logging.getLogger(name)
                loggers.append((lg, lg.level))
                lg.setLevel(logging.CRITICAL + 1)
        with contextlib.redirect_stdout(devnull_file), contextlib.redirect_stderr(devnull_file):
            yield
    finally:
        devnull_file.close()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull_fd)
        logging.root.setLevel(root_old_level)
        for lg, old_level in loggers:
            lg.setLevel(old_level)


def export_gltf(filepath: str, *, quiet: bool = True, **kwargs) -> None:
    """Export glTF/GLB via Blender addon, optionally suppressing verbose addon logs."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")

    def _export() -> None:
        try:
            bpy.ops.export_scene.gltf(filepath=filepath, **kwargs)
        except TypeError:
            fallback = dict(kwargs)
            for key in ("export_format", "export_apply"):
                fallback.pop(key, None)
            if fallback:
                bpy.ops.export_scene.gltf(filepath=filepath, **fallback)
            else:
                bpy.ops.export_scene.gltf(filepath=filepath)

    if quiet:
        with quiet_stdio():
            _export()
    else:
        _export()


def render_frame(*, write_still: bool = True, quiet: bool = True) -> None:
    """Run bpy.ops.render.render, optionally silencing Blender progress output."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    if quiet:
        with quiet_stdio():
            bpy.ops.render.render(write_still=write_still)
    else:
        bpy.ops.render.render(write_still=write_still)


def render_still_rgb_uint8(scene=None, *, quiet: bool = True) -> np.ndarray:
    """Render current frame to uint8 RGB (top-down) without writing a file when possible."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    scene = scene or bpy.context.scene

    render_frame(write_still=False, quiet=quiet)

    img = bpy.data.images.get("Render Result")
    if img is not None:
        width, height = img.size
        if width > 0 and height > 0:
            pixels = np.array(img.pixels[:], dtype=np.float32).reshape(height, width, 4)
            rgb = (np.clip(pixels[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
            # Blender stores pixels bottom-up; flip to match PNG / PIL convention.
            return rgb[::-1, :, :]

    # Cycles cleanup may remove Render Result; fall back to a quiet temp PNG.
    import tempfile

    fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix=".scenebuilder_sem_")
    os.close(fd)
    render = scene.render
    backup = (render.filepath, render.image_settings.file_format)
    try:
        render.filepath = tmp_path
        render.image_settings.file_format = "PNG"
        render_frame(write_still=True, quiet=quiet)
        semantic_rgb = util.read_image_array(tmp_path)
        if semantic_rgb.ndim == 2:
            semantic_rgb = np.stack([semantic_rgb] * 3, axis=-1)
        return np.asarray(semantic_rgb[..., :3], dtype=np.uint8)
    finally:
        render.filepath, render.image_settings.file_format = backup
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def render_depth_exr(scene, depth_path: str):
    if bpy is None or scene is None:
        return False

    render = scene.render
    image_settings = render.image_settings
    backup = (
        render.filepath,
        image_settings.file_format,
        image_settings.color_mode,
        getattr(image_settings, "color_depth", None),
    )

    try:
        render.filepath = depth_path
        image_settings.file_format = 'OPEN_EXR'
        image_settings.color_mode = 'BW'
        if hasattr(image_settings, "color_depth"):
            image_settings.color_depth = '32'
        render_frame(write_still=True)
        return True
    finally:
        render.filepath = backup[0]
        image_settings.file_format = backup[1]
        image_settings.color_mode = backup[2]
        if hasattr(image_settings, "color_depth") and backup[3] is not None:
            image_settings.color_depth = backup[3]


def camera_frustum_tangents(camera_obj, scene=None, margin: float = 1.0) -> Tuple[float, float, float, float]:
    """Return (tan_x, tan_y, clip_start, clip_end) aligned with Blender render frustum."""
    if bpy is None or camera_obj is None:
        return 1.0, 1.0, 0.01, 1000.0
    scene = scene or bpy.context.scene
    cam = camera_obj.data
    render = scene.render
    aspect = (
        float(render.resolution_x) / float(max(render.resolution_y, 1))
        * float(getattr(render, "pixel_aspect_x", 1.0))
        / float(max(getattr(render, "pixel_aspect_y", 1.0), 1e-6))
    )
    clip_start = max(float(getattr(cam, "clip_start", 0.01)), 1e-6)
    clip_end = float(getattr(cam, "clip_end", 1000.0))

    angle_x = float(getattr(cam, "angle_x", 0.0))
    angle_y = float(getattr(cam, "angle_y", 0.0))
    if angle_x > 1e-6 and angle_y > 1e-6:
        return (
            math.tan(angle_x * 0.5) * margin,
            math.tan(angle_y * 0.5) * margin,
            clip_start,
            clip_end,
        )

    angle = float(getattr(cam, "angle", math.radians(70.0)))
    sensor_fit = str(getattr(cam, "sensor_fit", "AUTO"))
    if sensor_fit == "VERTICAL":
        tan_y = math.tan(angle * 0.5) * margin
        tan_x = tan_y * aspect
    elif sensor_fit == "HORIZONTAL":
        tan_x = math.tan(angle * 0.5) * margin
        tan_y = tan_x / max(aspect, 1e-6)
    else:
        if aspect >= 1.0:
            tan_x = math.tan(angle * 0.5) * margin
            tan_y = tan_x / aspect
        else:
            tan_y = math.tan(angle * 0.5) * margin
            tan_x = tan_y * aspect
    return tan_x, tan_y, clip_start, clip_end


def cleanup_bpy_render_memory(scene=None) -> None:
    """Best-effort cleanup of Cycles render buffers before bpy process exit."""
    _free_cycles_render_buffers()


def _free_cycles_render_buffers() -> None:
    """Release Cycles render buffers that linger until process exit."""
    if bpy is None:
        return
    scene = getattr(bpy.context, "scene", None)
    if scene is not None:
        scene.render.use_compositing = False
    for name in ("Render Result", "Viewer Node"):
        img = bpy.data.images.get(name)
        if img is None:
            continue
        try:
            while img.users > 0:
                img.user_clear()
            bpy.data.images.remove(img, do_unlink=True)
        except Exception:
            pass


_REUSABLE_DEPTH_COMPOSITOR = "SceneBuilderDepthCompositor"


def _clear_compositor_tree(tree) -> None:
    if tree is None or bpy is None:
        return
    try:
        for node in list(tree.nodes):
            tree.nodes.remove(node)
    except Exception:
        pass


def _dispose_compositor_node_group(scene, tree) -> None:
    if tree is None or bpy is None:
        return
    try:
        if hasattr(scene, "compositing_node_group") and scene.compositing_node_group == tree:
            scene.compositing_node_group = None
    except Exception:
        pass
    _clear_compositor_tree(tree)
    try:
        if tree.name in bpy.data.node_groups and tree.name != _REUSABLE_DEPTH_COMPOSITOR:
            bpy.data.node_groups.remove(tree, do_unlink=True)
    except Exception:
        pass


def _get_compositor_tree(scene, create: bool = True):
    if hasattr(scene, "compositing_node_group"):
        if create:
            tree = bpy.data.node_groups.get(_REUSABLE_DEPTH_COMPOSITOR)
            if tree is None:
                tree = bpy.data.node_groups.new(_REUSABLE_DEPTH_COMPOSITOR, "CompositorNodeTree")
            scene.compositing_node_group = tree
            return tree
        return scene.compositing_node_group
    if getattr(scene, "use_nodes", False) or create:
        scene.use_nodes = True
    return scene.node_tree


def _load_exr_channel_names(header) -> List[str]:
    channels = header.get("channels", {})
    if hasattr(channels, "keys"):
        return list(channels.keys())
    return list(channels)


def _pick_exr_rgb_channels(channel_names: List[str]) -> Tuple[str, str, str]:
    candidates = [
        ("Normal.R", "Normal.G", "Normal.B"),
        ("R", "G", "B"),
        ("Normal.X", "Normal.Y", "Normal.Z"),
        ("X", "Y", "Z"),
    ]
    for r_name, g_name, b_name in candidates:
        if all(name in channel_names for name in (r_name, g_name, b_name)):
            return r_name, g_name, b_name
    rgb_like = [name for name in channel_names if name.endswith((".R", ".G", ".B"))]
    if len(rgb_like) >= 3:
        prefix = rgb_like[0].rsplit(".", 1)[0]
        triplet = (f"{prefix}.R", f"{prefix}.G", f"{prefix}.B")
        if all(name in channel_names for name in triplet):
            return triplet
    flat = [name for name in channel_names if "." not in name and name not in {"A", "Alpha"}]
    if len(flat) >= 3:
        return flat[0], flat[1], flat[2]
    if len(channel_names) >= 3:
        return channel_names[0], channel_names[1], channel_names[2]
    raise RuntimeError(f"Cannot find RGB channels in EXR: {channel_names}")


def _load_depth_exr(path: str) -> np.ndarray:
    try:
        import OpenEXR
        import Imath
    except ImportError:
        OpenEXR = None

    if OpenEXR is not None:
        exr = OpenEXR.InputFile(path)
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        channels = _load_exr_channel_names(header)
        channel_name = "Depth.V" if "Depth.V" in channels else ("V" if "V" in channels else channels[0])
        raw = exr.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(raw, dtype=np.float32).reshape(height, width).astype(np.float64)

    depth = util.read_image_array(path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return np.asarray(depth, dtype=np.float64)


def _load_rgb_exr_via_bpy(path: str) -> np.ndarray:
    """Decode EXR with Blender built-in decoder (no OpenEXR Python package required)."""
    if bpy is None:
        raise RuntimeError("bpy unavailable for EXR load")
    img = bpy.data.images.load(path, check_existing=False)
    try:
        width, height = img.size
        if width <= 0 or height <= 0:
            raise RuntimeError(f"invalid EXR image size: {width}x{height}")
        channels = getattr(img, "channels", 4) or 4
        pixels = np.array(img.pixels[:], dtype=np.float64)
        planes = pixels.reshape(height, width, channels)
        if planes.shape[-1] >= 3:
            rgb = planes[..., :3]
        else:
            rgb = np.stack([planes[..., 0]] * 3, axis=-1)
        # Blender pixels are bottom-up; flip to top-down to match PNG/depth maps
        return rgb[::-1, :, :]
    finally:
        bpy.data.images.remove(img)


def _load_rgb_exr(path: str) -> np.ndarray:
    try:
        import OpenEXR
        import Imath
    except ImportError:
        OpenEXR = None

    if OpenEXR is not None:
        try:
            exr = OpenEXR.InputFile(path)
            header = exr.header()
            dw = header["dataWindow"]
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            channels = _load_exr_channel_names(header)
            r_name, g_name, b_name = _pick_exr_rgb_channels(channels)
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            planes = []
            for name in (r_name, g_name, b_name):
                raw = exr.channel(name, pt)
                planes.append(np.frombuffer(raw, dtype=np.float32).reshape(height, width))
            return np.stack(planes, axis=-1).astype(np.float64)
        except Exception:
            pass

    if bpy is not None:
        return _load_rgb_exr_via_bpy(path)

    img = util.read_image_array(path)
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr[..., :3]


def _add_cycles_exr_output(tree, render_layers, socket_name: str, aov_dir: str, file_name: str):
    """Add a compositor file-output node for a single AOV EXR."""
    output = tree.nodes.new("CompositorNodeOutputFile")
    aov_dir = aov_dir if aov_dir.endswith(os.sep) else aov_dir + os.sep
    is_normal = socket_name == "Normal"

    if hasattr(output, "directory"):
        output.directory = aov_dir
        output.file_name = file_name
        output.format.file_format = "OPEN_EXR_MULTILAYER"
        output.format.color_depth = "32"
        item = output.file_output_items.new("RGBA" if is_normal else "FLOAT", file_name.capitalize())
        item.override_node_format = True
        item.format.file_format = "OPEN_EXR"
        item.format.color_depth = "32"
        if is_normal:
            item.format.color_mode = "RGB"
    else:
        output.base_path = aov_dir
        output.file_slots[0].path = file_name
        output.file_slots[0].use_node_format = True
        output.format.file_format = "OPEN_EXR"
        output.format.color_depth = "32"
        output.format.color_mode = "RGB" if is_normal else "BW"

    tree.links.new(render_layers.outputs[socket_name], output.inputs[0])
    return output


def _prepare_cycles_aov_compositor(scene, aov_dir: str):
    """Cycles compositor: output Depth + Normal EXR in one render."""
    render = scene.render
    render.use_compositing = True
    tree = _get_compositor_tree(scene, create=True)
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    render_layers = tree.nodes.new("CompositorNodeRLayers")
    aov_dir = aov_dir if aov_dir.endswith(os.sep) else aov_dir + os.sep

    try:
        composite = tree.nodes.new("CompositorNodeComposite")
        tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])
    except RuntimeError:
        composite = tree.nodes.new("NodeGroupOutput")
        tree.links.new(render_layers.outputs["Image"], composite.inputs[0])

    _add_cycles_exr_output(tree, render_layers, "Depth", aov_dir, "depth")
    _add_cycles_exr_output(tree, render_layers, "Normal", aov_dir, "normal")


def _prepare_cycles_depth_compositor(scene, depth_dir: str):
    """Legacy alias: equivalent to _prepare_cycles_aov_compositor."""
    _prepare_cycles_aov_compositor(scene, depth_dir)


def _normal_path_from_depth(depth_path: str) -> str:
    if depth_path.endswith("_depth.png"):
        return depth_path[:-10] + "_normal.png"
    base, _ = os.path.splitext(depth_path)
    return f"{base}_normal.png"


def _restore_render_compositor(scene, backup: Dict[str, Any]):
    render = scene.render
    render.use_compositing = backup["use_compositing"]
    if hasattr(scene, "compositing_node_group"):
        prev_group = backup.get("compositing_node_group")
        temp_group = backup.get("temp_compositing_node_group")
        scene.compositing_node_group = prev_group
        if temp_group is not None and temp_group != prev_group:
            _clear_compositor_tree(temp_group)
        return
    scene.use_nodes = backup["use_nodes"]
    if not backup["use_nodes"] and scene.node_tree:
        for node in list(scene.node_tree.nodes):
            scene.node_tree.nodes.remove(node)


def _cycles_depth_to_metric(depth_raw: np.ndarray, clip_end: float) -> np.ndarray:
    depth_m = np.nan_to_num(depth_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    depth_m = np.abs(depth_m)
    invalid = (~np.isfinite(depth_raw)) | (depth_m <= 1e-6) | (depth_m >= clip_end * 0.999)
    depth_m[invalid] = 0.0
    return depth_m


def _encode_opencv_normal_png_from_cycles_exr(normal_exr: np.ndarray, scene) -> np.ndarray:
    """Cycles Normal pass (world frame) → OpenCV camera frame → uint8 RGB PNG."""
    try:
        from . import util
        from . import geometry_opencv as geo_cv
    except ImportError:
        import util  # type: ignore
        import geometry_opencv as geo_cv  # type: ignore

    arr = np.asarray(normal_exr, dtype=np.float64)
    if scene is not None and getattr(scene, "camera", None) is not None:
        pose = geo_cv.camera_pose_from_matrix(np.array(scene.camera.matrix_world))
        flat = arr.reshape(-1, 3)
        norms = np.linalg.norm(flat, axis=1)
        valid = norms > 1e-6
        flat_cam = flat.copy()
        if np.any(valid):
            flat_cam[valid] = geo_cv.transform_normals_to_opencv(flat[valid], pose)
        arr = flat_cam.reshape(arr.shape[0], arr.shape[1], 3)
    else:
        print("⚠️ Scene has no camera; normal map still exported in world coordinates")
    return util.encode_normal_directions_uint8_png(arr)


def render_color_and_depth_png(
    scene,
    color_path: str,
    depth_path: str,
    skip_objects=None,
    normal_path: Optional[str] = None,
):
    """Single Cycles render: RGB PNG + metric depth PNG + OpenCV-camera normal PNG."""
    if bpy is None or scene is None or imageio is None:
        return None

    try:
        from . import util
    except ImportError:
        import util  # type: ignore

    view_layer = bpy.context.view_layer
    if view_layer is None:
        return None

    if normal_path is None:
        normal_path = _normal_path_from_depth(depth_path)

    os.makedirs(os.path.dirname(color_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(depth_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(normal_path) or ".", exist_ok=True)

    render = scene.render
    image_settings = render.image_settings
    prev_view_pass_z = view_layer.use_pass_z
    prev_view_pass_normal = getattr(view_layer, "use_pass_normal", False)
    prev_render_state = (
        render.filepath,
        render.use_compositing,
        image_settings.file_format,
        image_settings.color_mode,
        getattr(image_settings, "color_depth", None),
    )
    prev_scene_nodes = scene.use_nodes if hasattr(scene, "use_nodes") else False
    compositor_backup = {
        "use_compositing": render.use_compositing,
        "use_nodes": prev_scene_nodes,
        "compositing_node_group": getattr(scene, "compositing_node_group", None),
        "temp_compositing_node_group": None,
    }
    temp_dir = tempfile.mkdtemp(prefix="scenebuilder_depth_")
    clip_end = float(getattr(scene.camera.data, "clip_end", 1000.0)) if scene.camera else 1000.0

    try:
        view_layer.use_pass_z = True
        if hasattr(view_layer, "use_pass_normal"):
            view_layer.use_pass_normal = True
        _prepare_cycles_aov_compositor(scene, temp_dir)
        if hasattr(scene, "compositing_node_group"):
            compositor_backup["temp_compositing_node_group"] = scene.compositing_node_group

        render.filepath = color_path
        image_settings.file_format = "PNG"
        image_settings.color_mode = "RGBA"
        if hasattr(image_settings, "color_depth"):
            image_settings.color_depth = "8"

        render_frame(write_still=True)

        depth_files = sorted(glob.glob(os.path.join(temp_dir, "depth*.exr")))
        if not depth_files:
            depth_files = [p for p in sorted(glob.glob(os.path.join(temp_dir, "*.exr"))) if "normal" not in os.path.basename(p).lower()]
        if not depth_files:
            raise RuntimeError("Cycles depth EXR was not produced by compositor")
        depth_m = _cycles_depth_to_metric(_load_depth_exr(depth_files[-1]), clip_end)

        depth_scale = util.compute_depth_encode_scale(depth_m)
        depth_u16 = util.encode_depth_uint16(depth_m, depth_scale)
        imageio.imwrite(depth_path, depth_u16)
        print(
            f"✅ Depth map exported: {depth_path} "
            f"(uint16, depth_m * {depth_scale:.6f}, Cycles Z pass)"
        )

        normal_files = sorted(glob.glob(os.path.join(temp_dir, "normal*.exr")))
        if not normal_files:
            print("⚠️ Cycles normal EXR not found; skipping normal export")
        else:
            try:
                normal_raw = _load_rgb_exr(normal_files[-1])
                normal_u8 = _encode_opencv_normal_png_from_cycles_exr(normal_raw, scene)
                imageio.imwrite(normal_path, normal_u8)
                print(
                    f"✅ Normal map exported: {normal_path} "
                    f"(uint8 RGB, OpenCV camera normal, Cycles Normal pass)"
                )
            except Exception as normal_exc:
                print(f"⚠️ Normal EXR read/write failed; skipped normal export: {normal_exc}")

        return depth_scale
    except Exception as exc:
        print(f"⚠️ Cycles Z pass depth failed; falling back to per-pixel ray_cast: {exc}")
        if not os.path.exists(color_path):
            render.filepath = color_path
            image_settings.file_format = "PNG"
            image_settings.color_mode = "RGBA"
            render_frame(write_still=True)
        print("⚠️ ray_cast fallback path does not export normal map (Cycles Normal pass only)")
        return render_depth_png(scene, depth_path, skip_objects=skip_objects)
    finally:
        view_layer.use_pass_z = prev_view_pass_z
        if hasattr(view_layer, "use_pass_normal"):
            view_layer.use_pass_normal = prev_view_pass_normal
        render.filepath = prev_render_state[0]
        render.use_compositing = prev_render_state[1]
        image_settings.file_format = prev_render_state[2]
        image_settings.color_mode = prev_render_state[3]
        if hasattr(image_settings, "color_depth") and prev_render_state[4] is not None:
            image_settings.color_depth = prev_render_state[4]
        _restore_render_compositor(scene, compositor_backup)
        _free_cycles_render_buffers()
        shutil.rmtree(temp_dir, ignore_errors=True)


def render_depth_png(scene, depth_path: str, skip_objects=None):
    """Fallback: bpy ray_cast depth (slow, used only when Z pass fails)."""
    if bpy is None or scene is None or imageio is None:
        return None

    try:
        from . import util
    except ImportError:
        import util  # type: ignore

    os.makedirs(os.path.dirname(depth_path) or ".", exist_ok=True)
    camera = scene.camera
    if camera is None:
        return None

    width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    if width <= 0 or height <= 0:
        return None

    skip_ids = {id(obj) for obj in (skip_objects or []) if obj is not None}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frame = camera.data.view_frame(scene=scene)
    top_right, bottom_right, bottom_left, top_left = frame
    camera_matrix = camera.matrix_world
    camera_rotation = camera_matrix.to_3x3()
    world_to_camera = camera_matrix.inverted()
    camera_origin = camera_matrix.translation
    clip_end = float(getattr(camera.data, "clip_end", 1000.0))

    depth_m = np.zeros((height, width), dtype=np.float64)
    is_ortho = getattr(camera.data, "type", "PERSP") == "ORTHO"
    ortho_direction = (camera_rotation @ Vector((0.0, 0.0, -1.0))).normalized()

    for y in range(height):
        v = (y + 0.5) / height
        left = top_left.lerp(bottom_left, v)
        right = top_right.lerp(bottom_right, v)
        for x in range(width):
            u = (x + 0.5) / width
            local_point = left.lerp(right, u)
            if is_ortho:
                ray_origin = camera_matrix @ local_point
                ray_direction = ortho_direction
            else:
                ray_origin = camera_origin
                ray_direction = (camera_rotation @ local_point).normalized()

            hit, location, _, _, hit_obj, _ = scene.ray_cast(
                depsgraph, ray_origin, ray_direction, distance=clip_end,
            )
            if not hit or (hit_obj is not None and id(hit_obj) in skip_ids):
                continue
            camera_space_location = world_to_camera @ location
            depth_value = float(-camera_space_location.z)
            if depth_value > 0.0:
                depth_m[y, x] = depth_value

    depth_scale = util.compute_depth_encode_scale(depth_m)
    depth_u16 = util.encode_depth_uint16(depth_m, depth_scale)
    imageio.imwrite(depth_path, depth_u16)
    print(
        f"✅ Depth map exported: {depth_path} "
        f"(uint16, depth_m * {depth_scale:.6f}, bpy ray_cast fallback)"
    )
    return depth_scale


def get_class_color(class_name: Optional[str]):
    label = (class_name or "default").lower()
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    h = int.from_bytes(digest[0:2], "big") / 65535.0
    s = 0.55 + (digest[2] / 255.0) * (0.85 - 0.55)
    v = 0.65 + (digest[3] / 255.0) * (0.85 - 0.65)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [float(r), float(g), float(b), 0.9]


def build_render_frustum_clip_mats(scene, camera_obj, width: int, height: int):
    """Same as visible geometry: Blender calc_matrix_camera projection + camera modelview."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    render = scene.render
    pct = float(render.resolution_percentage) / 100.0
    rw = max(int(width * pct), 1)
    rh = max(int(height * pct), 1)
    modelview = np.array(camera_obj.matrix_world.inverted(), dtype=float)
    calc_fn = getattr(camera_obj, "calc_matrix_camera", None)
    if calc_fn is None:
        calc_fn = camera_obj.data.calc_matrix_camera
    proj = np.array(
        calc_fn(
            depsgraph,
            x=rw,
            y=rh,
            scale_x=float(getattr(render, "pixel_aspect_x", 1.0)),
            scale_y=float(getattr(render, "pixel_aspect_y", 1.0)),
        ),
        dtype=float,
    )
    return proj, modelview


def render_frustum_clip_planes():
    """Blender clip-space frustum half-spaces (consistent with scenebuilder_bpy visible geometry)."""
    eps = 1e-5
    return [
        (np.array([0.0, 0.0, 0.0, 1.0], dtype=float), eps),
        (np.array([1.0, 0.0, 0.0, 1.0], dtype=float), 0.0),
        (np.array([-1.0, 0.0, 0.0, 1.0], dtype=float), 0.0),
        (np.array([0.0, 1.0, 0.0, 1.0], dtype=float), 0.0),
        (np.array([0.0, -1.0, 0.0, 1.0], dtype=float), 0.0),
    ]


def _clip_homogeneous_polygon_against_plane(polygon, plane_normal, plane_offset: float):
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
            })
        if curr_inside:
            clipped.append(curr)
        prev = curr
        prev_dist = curr_dist
        prev_inside = curr_inside
    return clipped


def clip_polygon_to_render_frustum(vertices, proj, modelview):
    """Clip a planar polygon in homogeneous clip space, interpolating world coords along edges (keeps coplanarity)."""
    poly = [np.asarray(v, dtype=float) for v in vertices]
    if len(poly) < 3:
        return np.empty((0, 3), dtype=float)

    payload = []
    for v in poly:
        ph = np.array([v[0], v[1], v[2], 1.0], dtype=float)
        payload.append({
            "clip": proj @ (modelview @ ph),
            "world": v,
        })

    polygon = payload
    for plane_normal, plane_offset in render_frustum_clip_planes():
        polygon = _clip_homogeneous_polygon_against_plane(polygon, plane_normal, plane_offset)
        if len(polygon) < 3:
            return np.empty((0, 3), dtype=float)

    ring = np.asarray([item["world"] for item in polygon], dtype=float)
    return _dedupe_consecutive_ring_vertices(ring)


def _dedupe_consecutive_ring_vertices(vertices: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if len(vertices) == 0:
        return vertices
    out = [vertices[0]]
    for v in vertices[1:]:
        if np.linalg.norm(v - out[-1]) > eps:
            out.append(v)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) < eps:
        out.pop()
    return np.asarray(out, dtype=float)

