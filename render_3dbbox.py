import os
import json
import numpy as np
import trimesh
import pyrender

def build_bbox_mesh(center, scale, angle_z_deg,
                    box_color=(0.2, 0.5, 0.8, 1.0),
                    arrow_color=(0.0, 0.8, 0.0, 1.0)):
    """
    构建一个实心bbox立方体，顶部中心放置一个箭头，箭头在0°时指向-y，逆时针绕z旋转。
    - 使用顶点颜色(ColorVisuals)避免材质覆盖
    - 合并后统一进行旋转和平移
    """
    center = np.asarray(center, dtype=float)
    scale = np.asarray(scale, dtype=float)
    theta = np.radians(angle_z_deg)

    # 立方体（局部坐标原点，尺寸=scale）
    box = trimesh.creation.box(extents=scale)
    box_rgba = (np.asarray(box_color) * 255).astype(np.uint8)
    box.visual = trimesh.visual.ColorVisuals(
        box, vertex_colors=np.tile(box_rgba, (len(box.vertices), 1))
    )

    # 箭头（圆柱+圆锥），占顶面最小边的80%
    top_min = float(min(scale[0], scale[1]))
    arrow_total = max(top_min * 0.80, 0.05)
    body_len = arrow_total * 0.65
    head_len = arrow_total * 0.35
    radius = max(0.02, arrow_total * 0.06)

    # 几何方向约定：沿局部 -y 指向；整体以顶面中心为对称中心
    minus_y = np.array([0, -1, 0], dtype=float)
    plus_y = np.array([0,  1, 0], dtype=float)
    top_center = np.array([0.0, 0.0, scale[2] / 2.0], dtype=float)

    # 以顶面中心为中点，整支箭头长度为 arrow_total，朝 -y 放置
    p_tail = top_center + plus_y * (arrow_total * 0.5)  # 箭尾端（靠 +y）
    p_body_end = p_tail + minus_y * body_len            # 圆柱末端（与圆锥底面相接）
    body = trimesh.creation.cylinder(segment=[p_tail, p_body_end], radius=radius, sections=24)

    # 圆锥：默认底面在(0,0,0)，高度沿+z；对齐到 -y 并把底面放到 p_body_end
    head = trimesh.creation.cone(radius=radius * 2.5, height=head_len, sections=24)
    z_axis = np.array([0, 0, 1], dtype=float)
    align_to_minus_y = trimesh.geometry.align_vectors(z_axis, minus_y)
    head.apply_transform(align_to_minus_y)
    head.apply_translation(p_body_end)

    arrow_rgba = (np.asarray(arrow_color) * 255).astype(np.uint8)
    body.visual = trimesh.visual.ColorVisuals(
        body, vertex_colors=np.tile(arrow_rgba, (len(body.vertices), 1))
    )
    head.visual = trimesh.visual.ColorVisuals(
        head, vertex_colors=np.tile(arrow_rgba, (len(head.vertices), 1))
    )

    # 合并为一个mesh（trimesh.util.concatenate）
    combined = trimesh.util.concatenate([box, body, head])

    # 全局旋转（绕z逆时针），然后平移到center
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1],
    ], dtype=float)
    combined.apply_transform(Rz)
    combined.apply_translation(center)

    return combined
def render_scene(scene, camera_pos, target_pos, output_path):
    """
    渲染场景，使用指定的相机位置和目标位置
    """
    # 移除已有的方向光和相机
    nodes_to_remove = []
    for node in scene.nodes:
        if hasattr(node, 'light') and isinstance(node.light, pyrender.DirectionalLight):
            nodes_to_remove.append(node)
        if hasattr(node, 'camera') and node.camera is not None:
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        scene.remove_node(node)

    # 设置相机 (俯拍视角)
    cam = pyrender.PerspectiveCamera(yfov=np.radians(120), aspectRatio=1.0)

    # 计算相机姿态 (看向下方)
    world_up = np.array([0.0, 1.0, 0.0])  # Y 轴向上

    forward = target_pos - camera_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / np.linalg.norm(right)

    up = np.cross(right, forward)

    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up
    cam_pose[:3, 2] = -forward  # OpenGL 相机朝向 -Z
    cam_pose[:3, 3] = camera_pos

    scene.add(cam, pose=cam_pose)

    # 添加方向光照（跟随相机）
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=cam_pose)

    # 渲染
    r = pyrender.OffscreenRenderer(1024, 1024)
    try:
        color, depth = r.render(
            scene,
            flags=(pyrender.constants.RenderFlags.RGBA |
                   pyrender.constants.RenderFlags.VERTEX_NORMALS)
        )

        import imageio
        imageio.imwrite(output_path, color)
        print(f"✅ 保存渲染结果: {os.path.abspath(output_path)}")

    finally:
        r.delete()


def render_all_bboxes(json_path):
    """
    渲染 JSON 文件中的所有 3D bbox，使用俯拍视角
    """
    # 读取 JSON 数据 (跳过 markdown 标记)
    json_dir = os.path.dirname(json_path)
    output_path = os.path.join(json_dir, 'rendered_3d_bboxes.png')
    # with open(json_path, 'r', encoding='utf-8') as f:
    #     content = f.read()

    # # 如果文件以 ```json 开头，提取 JSON 部分
    # import re
    # matches = re.findall(r"```json\n([\s\S]*?)\n```", content)
    # json_content = matches[-1]

    # bboxes = json.loads(json_content)
    with open(json_path, 'r', encoding='utf-8') as f:
        bboxes = json.load(f)
    if not bboxes:
        print("没有找到 bbox 数据")
        return

    # 将所有 bbox 的 y 坐标取负
    for bbox in bboxes:
        bbox['position'][1] = -bbox['position'][1]

    # 创建场景
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])

    # 记录场景 bounds
    min_bounds = np.array([float('inf'), float('inf'), float('inf')], dtype=float)
    max_bounds = np.array([float('-inf'), float('-inf'), float('-inf')], dtype=float)

    # 添加所有 bbox
    for bbox in bboxes:
        x, y, z = bbox['position']
        w, h, d = bbox['size']
        orientation = bbox['orientation']

        # 转换为 build_bbox_mesh 的参数格式
        center = [x, y, z]
        scale = [w, h, d]  # 注意：JSON 中的 size 是 [width, height, depth]

        mesh = build_bbox_mesh(center=center, scale=scale, angle_z_deg=orientation)

        # 更新场景 bounds
        mesh_bounds = mesh.bounds  # 形如 [[min_x,min_y,min_z],[max_x,max_y,max_z]]
        min_bounds = np.minimum(min_bounds, mesh_bounds[0])
        max_bounds = np.maximum(max_bounds, mesh_bounds[1])

        pm = pyrender.Mesh.from_trimesh(mesh, smooth=False)

        # # 强制双面渲染
        # if hasattr(pm, 'primitives'):
        #     for prim in pm.primitives:
        #         if prim.material:
        #             prim.material.doubleSided = True

        scene.add(pm)

    # 添加环境光照
    scene.ambient_light = np.array([0.6, 0.6, 0.6], dtype=np.float32)

    if not np.all(np.isfinite(min_bounds)) or not np.all(np.isfinite(max_bounds)):
        print("无法计算场景 bounds")
        return

    # 计算场景中心
    center_x = (min_bounds[0] + max_bounds[0]) / 2
    center_y = (min_bounds[1] + max_bounds[1]) / 2
    min_z = min_bounds[2]
    max_z = max_bounds[2]

    # 设置相机位置和目标
    camera_height = min_z + 2 * (max_z - min_z)
    camera_pos = np.array([center_x, center_y, camera_height])
    target_pos = np.array([center_x, center_y, 0.0])

    # 设置第二个相机位置
    scene_y_length = max_bounds[1] - min_bounds[1]
    camera_height2 = min_z + 1.5 * (max_z - min_z)
    camera_pos2 = np.array([center_x-2, center_y - scene_y_length*0.7, camera_height2])
    print("camera_pos2, camera_pos", camera_pos2, camera_pos)
    # 第一个相机位置渲染 (俯拍)
    render_scene(scene, camera_pos, target_pos, output_path)
    # 第二个相机位置渲染 (前视图)
    output_path2 = os.path.join(json_dir, 'rendered_3d_bboxes_front.png')
    render_scene(scene, camera_pos2, target_pos, output_path2)


if __name__ == '__main__':
    # 测试渲染
    json_file = '/root/projects/utils/fast-scene/test_3dbbox.txt'
    render_all_bboxes(json_file)
