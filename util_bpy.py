"""
Blender 版本的辅助工具函数，供 fast_scene_bpy.py 使用。
"""

import os
import colorsys
import hashlib
import contextlib
from typing import Any, Dict, List, Optional, Union

try:
    import imageio
except ImportError:  # pragma: no cover
    imageio = None

import numpy as np

try:
    import bpy
    from mathutils import Vector, Matrix  # type: ignore[import]
except ImportError:  # pragma: no cover
    bpy = None
    Vector = None
    Matrix = None


def create_floor_mesh_bpy(scene_collection, vertices, thickness=0.1, name="Floor", texture_scale=1.0):
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

    # 添加地板 UV 坐标 (x, y)
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
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0.0, 0.0, -thickness)})
    bpy.ops.object.mode_set(mode='OBJECT')

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
    # normal 指向房间内部
    normal = np.array(orientation, dtype=float)[:2]
    
    # 如果是隔断且没有有效法向量，构造一个
    if is_partition and np.linalg.norm(normal) < 1e-4:
        normal = np.array([-wall_dir[1], wall_dir[0]])

    # 处理端点延长逻辑：如果端点有连接，向外延伸半个厚度以嵌入相邻墙体
    s_base = s_arr - wall_dir * (wall_thickness / 2.0) if extend_s else s_arr
    e_base = e_arr + wall_dir * (wall_thickness / 2.0) if extend_e else e_arr

    if is_partition:
        # 隔断墙：居中模式，向法线正负方向各偏移一半厚度
        inner_s_p = s_base + normal * (wall_thickness / 2.0)
        inner_e_p = e_base + normal * (wall_thickness / 2.0)
        outer_s_p = s_base - normal * (wall_thickness / 2.0)
        outer_e_p = e_base - normal * (wall_thickness / 2.0)
    else:
        # 外墙：单向偏移模式（内侧在中心线上，外侧向外偏移）
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

    # 顶点定义：0-3 为一侧，4-7 为另一侧
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

    # 定义 6 个面
    faces = [
        (0, 3, 2, 1), # 侧面 A
        (4, 5, 6, 7), # 侧面 B
        (3, 7, 6, 2), # 顶面
        (0, 1, 5, 4), # 底面
        (0, 4, 7, 3), # 端面 S
        (1, 2, 6, 5), # 端面 E
    ]

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # 添加墙体 UV 坐标
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for f_idx, poly in enumerate(mesh.polygons):
        for loop_index in poly.loop_indices:
            v_idx = mesh.loops[loop_index].vertex_index
            vx, vy, vz = verts[v_idx]
            if f_idx == 0: # 侧面 A
                u = np.dot(np.array([vx, vy]) - inner_s_p, wall_dir)
                uv_layer.data[loop_index].uv = (u * texture_scale, vz * texture_scale)
            elif f_idx == 1: # 侧面 B
                u = np.dot(np.array([vx, vy]) - outer_s_p, wall_dir)
                uv_layer.data[loop_index].uv = (u * texture_scale, vz * texture_scale)
            else:
                uv_layer.data[loop_index].uv = ((vx + vy) * texture_scale, vz * texture_scale)

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
    """
    预计算所有外墙的斜接（Miter Joint）偏移点。
    """
    # 过滤：斜接只适用于外墙
    boundary_walls = {wid: w for wid, w in walls.items() if not w.get("is_partition", False)}
    
    # 1. 建立顶点到墙体的映射
    pt_to_walls = {}
    for wall_id, wall in boundary_walls.items():
        s = tuple(wall["s"])
        e = tuple(wall["e"])
        for pt in [s, e]:
            if pt not in pt_to_walls:
                pt_to_walls[pt] = []
            pt_to_walls[pt].append(wall_id)

    # 2. 预计算每面墙的外侧法线
    wall_out_normals = {}
    for wall_id, wall in boundary_walls.items():
        wall_out_normals[wall_id] = -np.array(wall["orientation"])

    def _get_miter_point(pt, wid1, wid2):
        n1 = wall_out_normals[wid1]
        n2 = wall_out_normals[wid2]
        n_avg = n1 + n2
        n_avg_norm = np.linalg.norm(n_avg)
        if n_avg_norm < 1e-4:
            return np.array(pt) + n1 * wall_thickness
        n_avg /= n_avg_norm
        cos_half_theta = np.dot(n_avg, n1)
        if abs(cos_half_theta) < 1e-4:
            return np.array(pt) + n1 * wall_thickness
        length = wall_thickness / cos_half_theta
        length = min(length, wall_thickness * 10)
        return np.array(pt) + n_avg * length

    wall_outer_points = {}
    for wall_id, wall in boundary_walls.items():
        s = tuple(wall["s"])
        e = tuple(wall["e"])
        outer_s, outer_e = None, None
        neighbors_s = [wid for wid in pt_to_walls.get(s, []) if wid != wall_id]
        if neighbors_s:
            outer_s = _get_miter_point(s, wall_id, neighbors_s[0])
        neighbors_e = [wid for wid in pt_to_walls.get(e, []) if wid != wall_id]
        if neighbors_e:
            outer_e = _get_miter_point(e, wall_id, neighbors_e[0])
        wall_outer_points[wall_id] = (outer_s, outer_e)
    return wall_outer_points


def create_opening_box_bpy(opening, wall_s, wall_e, wall_dir, normal, wall_height, wall_thickness=0.1, opening_type="window", is_partition=False):
    if bpy is None:
        return None

    center = np.array(opening["center"])
    width = opening["width"]
    height = opening["height"]

    # 基础投影点
    proj = wall_s + wall_dir * np.dot(center[:2] - wall_s, wall_dir)
    
    if is_partition:
        # 隔断墙洞口：完全居中，厚度加大以确保两边切透
        final_center_2d = proj
        cutter_thickness = wall_thickness * 2.0
    else:
        # 外墙洞口：原有的逻辑
        outward_normal = -np.array(normal)
        # 洞口中心稍微向外偏移
        center_offset = outward_normal * (wall_thickness / 2.0) * 0.90
        final_center_2d = proj + center_offset
        if opening_type == "door":
            cutter_thickness = wall_thickness
        else:
            cutter_thickness = wall_thickness * 2

    z_bottom = center[2] - height / 2
    z_top = center[2] + height / 2

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    box_obj = bpy.context.object
    box_obj.name = f"OpeningBox_{opening_type}"
    box_obj.scale = (width, cutter_thickness, height)
    
    final_z = (z_bottom + z_top) / 2
    if opening_type == "door":
        final_z += 0.01 # 门底部稍微抬高
        
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

    # 添加天花板 UV 坐标 (x, y)
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
    """返回对象列表的联合包围盒（最小/最大点）"""
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
    """计算对象包围盒中点"""
    bounds = _get_combined_bounds(objects)
    if not bounds or bounds[0] is None or bounds[1] is None:
        return None
    return (bounds[0] + bounds[1]) / 2


def load_mesh_to_origin(asset_id: int, model_root: Union[str, List[str]]):
    """
    导入指定 asset_id 的 GLTF/GLB 模型，并将几何体中心移至原点。
    支持从多个根目录中查找。
    """
    if bpy is None or Vector is None or not model_root:
        return None

    # 统一转换为列表处理
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

        # 记录导入前场景中已有对象的指针，导入完成后用来识别新加入的对象
        before_ids = {obj.as_pointer() for obj in bpy.data.objects}
        try:
            # 抑制 Blender 内部的 glTF 导入日志
            with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
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

        # 首先计算新对象的几何中心（在设置parent之前）
        bounds = _get_combined_bounds(new_objects)
        if bounds and bounds[0] is not None and bounds[1] is not None:
            center = (bounds[0] + bounds[1]) / 2
        else:
            center = Vector((0.0, 0.0, 0.0))
        
        # 创建一个空的容器对象作为本次导入网格的根节点
        container = bpy.data.objects.new(f"mesh_{asset_id}_root", None)
        bpy.context.scene.collection.objects.link(container)

        
        for obj in new_objects:
            obj.parent = container
        # 设置parent关系，使用默认的parent inverse（让子对象跟随容器移动）
        # 整体逻辑: container 默认的 location 为 0, 在这种情况下导入的子节点相对于子节点原本的位置没有位移
        # 现在我们可以在挂载子节点前/后均可, 进入如下操作: 
        # 让container的位置为-center, 这样导入的子节点相对于子节点原本的位置有-center的位移
        # 因此可以保证导入的子节点bbox 的几何中心在原点
        container.location = -center
        # 更改节点属性后要强制更新场景，确保变换生效
        bpy.context.view_layer.update()
        
        return container

    return None


def apply_box_transform(container_obj: Any, box: Dict, master_bounds: Optional[tuple] = None):
    """
    对导入的 mesh 根节点（或实例对象）依次执行 scale、rotation、translation。
    master_bounds: 如果提供，则直接使用该包围盒进行缩放计算，不再扫描子对象。
    """
    if bpy is None or Vector is None or container_obj is None:
        return False

    if master_bounds and master_bounds[0] is not None:
        min_corner, max_corner = master_bounds
    else:
        # 如果是 Collection Instance 且未提供 master_bounds，尝试从其 instance_collection 获取
        if container_obj.instance_type == 'COLLECTION' and container_obj.instance_collection:
            child_objects = [obj for obj in container_obj.instance_collection.objects if obj.parent is None]
        else:
            child_objects = list(container_obj.children)
            
        if not child_objects:
            # 容错：如果还是没有子对象，但本身是 MESH
            if container_obj.type == 'MESH':
                child_objects = [container_obj]
            else:
                return False

        bounds = _get_combined_bounds(child_objects)
        if not bounds or bounds[0] is None or bounds[1] is None:
            return False
        min_corner, max_corner = bounds

    extent = np.array((max_corner - min_corner), dtype=float)
    # ... 后续逻辑保持不变 ...
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

    # 调试输出
    # print(f"    [调试] 物体 Label: {box.get('label', 'unknown')}")
    # print(f"    [调试] 当前几何中心: ({current_center[0]:.3f}, {current_center[1]:.3f}, {current_center[2]:.3f})")
    # print(f"    [调试] 当前extent: ({extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f})")
    # print(f"    [调试] 目标center: ({target_center[0]:.3f}, {target_center[1]:.3f}, {target_center[2]:.3f})")
    # print(f"    [调试] 目标scale: ({target_scale[0]:.3f}, {target_scale[1]:.3f}, {target_scale[2]:.3f})")
    # print(f"    [调试] 底部z (min): {min_corner[2]:.3f}, 顶部z (max): {max_corner[2]:.3f}")

    rotation_mat = Matrix.Rotation(theta, 4, 'Z')
    # rotation_mat_fixed = Matrix.Rotation(np.radians(-90), 4, 'X')
    scale_mat = Matrix.Identity(4)
    scale_mat[0][0] = float(scale_factors[0])
    scale_mat[1][1] = float(scale_factors[1])
    scale_mat[2][2] = float(scale_factors[2])

    stage2_transform = Matrix.Translation(center_vec) @ rotation_mat @ scale_mat
    previous_matrix = container_obj.matrix_world.copy()
    container_obj.matrix_world = stage2_transform @ previous_matrix
    
    # 强制更新场景，让变换立即生效
    bpy.context.view_layer.update()
    
    # 验证最终位置（仅针对非实例化对象或已定义 child_objects 的对象）
    if 'child_objects' in locals() and child_objects:
        final_bounds = _get_combined_bounds(child_objects)
        if final_bounds and final_bounds[0] is not None:
            final_center = (final_bounds[0] + final_bounds[1]) / 2
            # print(f"    [调试] 变换后几何中心: ({final_center[0]:.3f}, {final_center[1]:.3f}, {final_center[2]:.3f})")
            
            # 检查是否有浮空问题
            expected_bottom = target_center[2] - target_scale[2] / 2
            actual_bottom = final_bounds[0][2]
            height_diff = actual_bottom - expected_bottom
            if abs(height_diff) > 0.01:  # 超过1cm的偏差
                print(f"    ⚠️  高度偏差: 底部应该在 {expected_bottom:.3f}m，实际在 {actual_bottom:.3f}m，偏差 {height_diff:.3f}m")
    
    return True


def append_material_from_blend(blend_path):
    if bpy is None:
        return None

    abs_path = os.path.abspath(blend_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Blend 文件不存在：{abs_path}")

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
        # [优化] 检查是否已经加载过该纹理
        img_name = os.path.basename(texture_path)
        img = bpy.data.images.get(img_name)
        
        # 即使名字相同，也检查路径是否一致
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

    # For fully transparent (alpha=0), hide the object from render entirely
    if alpha <= 0.01:
        obj.hide_render = True
        obj.hide_viewport = True
        return [{"slot_idx": -1, "original": None, "replacement": None, "hide": True}]

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
        if slot.get("hide"):
            obj.hide_render = False
            obj.hide_viewport = False
            continue
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
        image_settings.color_mode = 'RGB'
        if hasattr(image_settings, "color_depth"):
            image_settings.color_depth = '16'
        bpy.ops.render.render(write_still=True)
        return True
    finally:
        render.filepath = backup[0]
        image_settings.file_format = backup[1]
        image_settings.color_mode = backup[2]
        if hasattr(image_settings, "color_depth") and backup[3] is not None:
            image_settings.color_depth = backup[3]


def get_class_color(class_name: Optional[str]):
    label = (class_name or "default").lower()
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    h = int.from_bytes(digest[0:2], "big") / 65535.0
    s = 0.55 + (digest[2] / 255.0) * (0.85 - 0.55)
    v = 0.65 + (digest[3] / 255.0) * (0.85 - 0.65)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [float(r), float(g), float(b), 0.9]

