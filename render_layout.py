#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取spatial-layout.jsonl并渲染场景
"""

import os
import sys
import json
import argparse

from .fast_scene import SceneCtx
from .retrieve import retrieve
from . import util as util


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


def check_and_retrieve_mesh(box_data, config):
    """
    检查mesh文件是否存在，如果不存在则使用retrieve获取新的asset_id
    
    Args:
        box_data: bbox数据字典
        config: 配置字典（包含model_path）
    
    Returns:
        bool: 是否成功（文件存在或retrieve成功）
    """
    asset_id = box_data.get('asset_id')
    
    if not asset_id:
        print(f"   ⚠️  {box_data.get('label', 'unknown')}: 缺少asset_id")
        return False
    
    # 检查mesh文件是否存在
    model_path = config.get('model_path', '')
    glb_path = os.path.join(model_path, f"{asset_id}.glb")
    gltf_path = os.path.join(model_path, f"{asset_id}.gltf")
    
    if os.path.exists(glb_path) or os.path.exists(gltf_path):
        format_type = 'GLB' if os.path.exists(glb_path) else 'GLTF'
        print(f"   ✅ {box_data.get('label', 'unknown')} (asset_id={asset_id}): {format_type}文件存在")
        return True
    
    # 文件不存在，使用retrieve
    print(f"   ⚠️  {box_data.get('label', 'unknown')} (asset_id={asset_id}): 文件不存在，开始retrieve...")
    
    try:
        # 使用label和caption进行retrieve
        label = box_data.get('label', '')
        caption = box_data.get('caption', '')
        scale = box_data.get('scale', [1, 1, 1])
        
        if not label:
            print(f"      ❌ 缺少label，无法retrieve")
            return False
        
        # 使用label作为class_name，caption作为描述，size用于精确匹配
        new_asset_id = retrieve(
            class_name=label,
            caption=caption if caption else "",
            size=scale,
            k=5  # 在前5个相似结果中选择size最接近的
        )
        
        # 更新asset_id
        box_data['asset_id'] = new_asset_id
        print(f"      ✅ Retrieve成功: 新asset_id={new_asset_id}")
        return True
        
    except Exception as e:
        print(f"      ❌ Retrieve失败: {e}")
        return False


def render_empty_room_topdown(data, output_path, width=1024, height=1024):
    """
    渲染空房间的俯视图（只包含地板和墙体，不包含家具）
    
    Args:
        data: 原始JSON数据
        output_path: 输出图片路径
        width: 渲染宽度
        height: 渲染高度
    """
    print(f"🏠 构建空房间场景: {data['room']['room_type']}")
    
    # 创建场景上下文
    ctx = SceneCtx(data['room']['room_type'])
    
    # 添加墙体
    if data.get('wall'):
        ctx.add_walls(data['wall'])
    
    # 添加门
    if data.get('door'):
        ctx.add_doors(data['door'])
    
    # 添加窗
    if data.get('window'):
        ctx.add_windows(data['window'])
    
    # 只构建地板和墙体，不添加家具
    print("🏗️  构建空房间（仅地板和墙体）...")
    ctx.construct_floor(show_wall=True, show_window=True, show_door=True)  # 只构建地板和墙体
    
    # 渲染俯视图
    print("🎨 渲染空房间俯视图...")
    ctx.topdown_view(output_path, width=width, height=height)
    
    print(f"✅ 空房间俯视图已保存: {output_path}")


def render_empty_room_views(data, output_dir, width=1024, height=1024):
    """
    渲染空房间的4个方向视角（只包含地板和墙体，不包含家具）
    
    Args:
        data: 原始JSON数据
        output_dir: 输出目录
        width: 渲染宽度
        height: 渲染高度
    """
    print(f"🏠 构建空房间场景用于多视角渲染: {data['room']['room_type']}")
    
    # 创建场景上下文
    ctx = SceneCtx(data['room']['room_type'])
    
    # 添加墙体
    if data.get('wall'):
        ctx.add_walls(data['wall'])
    
    # 添加门
    if data.get('door'):
        ctx.add_doors(data['door'])
    
    # 添加窗
    if data.get('window'):
        ctx.add_windows(data['window'])
    
    # 只构建地板和墙体，不添加家具
    print("🏗️  构建空房间（仅地板和墙体）...")
    ctx.construct_floor(show_wall=True, show_window=True, show_door=True)  # 只构建地板和墙体
    
    # 计算相机参数
    center = ctx.context["meta"]["center"]
    span = ctx.context["meta"]["span"]
    z_max = ctx.context["meta"]["z_max"]
    
    # 看向的目标点（房间中心，高度为z_max的一半）
    look_at = [center[0], center[1], z_max / 2]
    
    # 4个方向的相机位置
    views = [
        ("empty_right", [center[0] + span[0]/2+2, center[1], z_max * 7/6]),
        ("empty_left", [center[0] - span[0]/2-2, center[1], z_max * 7/6]),
        ("empty_front", [center[0], center[1] + span[1]/2+2, z_max * 7/6]),
        ("empty_back", [center[0], center[1] - span[1]/2-2, z_max * 7/6]),
    ]
    
    print("📸 渲染空房间四个方向视角...")
    for view_name, camera_pos in views:
        output_file = os.path.join(output_dir, f'{view_name}.png')
        print(f"\n📷 渲染 {view_name} 视角...")
        
        ctx.render_view(
            output_path=output_file,
            camera_position=camera_pos,
            look_at_target=look_at,
            width=width,
            height=height,
        )
        print(f"✅ {view_name} 视角已保存: {output_file}")


def render_full_room_no_doors_windows_topdown(data, output_path, width=1024, height=1024):
    """
    渲染不显示门窗的满房间俯视图（包含家具，但不显示门窗）
    
    Args:
        data: 原始JSON数据
        output_path: 输出图片路径
        width: 渲染宽度
        height: 渲染高度
    """
    print(f"🏠 构建不显示门窗的满房间场景: {data['room']['room_type']}")
    
    # 创建场景上下文
    ctx = SceneCtx(data['room']['room_type'])
    
    # 添加墙体
    if data.get('wall'):
        ctx.add_walls(data['wall'])
    
    # 添加门（数据添加但不渲染）
    if data.get('door'):
        ctx.add_doors(data['door'])
    
    # 添加窗（数据添加但不渲染）
    if data.get('window'):
        ctx.add_windows(data['window'])
    
    # 添加家具
    if data.get('bbox'):
        ctx.add_boxes(data['bbox'])
    
    # 构建完整场景（不显示门窗，需加载家具）
    print("🏗️  构建满房间（不显示门窗，含家具）...")
    ctx.construct_scene(show_wall=True, show_window=False, show_door=False)
    
    # 渲染俯视图
    print("🎨 渲染不显示门窗的满房间俯视图...")
    ctx.topdown_view(output_path, width=width, height=height)
    
    print(f"✅ 不显示门窗的满房间俯视图已保存: {output_path}")


def render_empty_room_no_doors_windows_topdown(data, output_path, width=1024, height=1024):
    """
    渲染不显示门窗的空房间俯视图（不包含家具，不显示门窗）
    
    Args:
        data: 原始JSON数据
        output_path: 输出图片路径
        width: 渲染宽度
        height: 渲染高度
    """
    print(f"🏠 构建不显示门窗的空房间场景: {data['room']['room_type']}")
    
    # 创建场景上下文
    ctx = SceneCtx(data['room']['room_type'])
    
    # 添加墙体
    if data.get('wall'):
        ctx.add_walls(data['wall'])
    
    # 添加门（数据添加但不渲染）
    if data.get('door'):
        ctx.add_doors(data['door'])
    
    # 添加窗（数据添加但不渲染）
    if data.get('window'):
        ctx.add_windows(data['window'])
    
    # 只构建地板和墙体，不添加家具，不显示门窗
    print("🏗️  构建空房间（不显示门窗）...")
    ctx.construct_floor(show_wall=True, show_window=False, show_door=False)
    
    # 渲染俯视图
    print("🎨 渲染不显示门窗的空房间俯视图...")
    ctx.topdown_view(output_path, width=width, height=height)
    
    print(f"✅ 不显示门窗的空房间俯视图已保存: {output_path}")


def render_empty_room_no_walls_doors_windows_topdown(data, output_path, width=1024, height=1024):
    """
    渲染不显示门窗墙的空房间俯视图（只显示地板）
    
    Args:
        data: 原始JSON数据
        output_path: 输出图片路径
        width: 渲染宽度
        height: 渲染高度
    """
    print(f"🏠 构建不显示门窗墙的空房间场景: {data['room']['room_type']}")
    
    # 创建场景上下文
    ctx = SceneCtx(data['room']['room_type'])
    
    # 添加墙体（数据添加但不渲染）
    if data.get('wall'):
        ctx.add_walls(data['wall'])
    
    # 添加门（数据添加但不渲染）
    if data.get('door'):
        ctx.add_doors(data['door'])
    
    # 添加窗（数据添加但不渲染）
    if data.get('window'):
        ctx.add_windows(data['window'])
    
    # 只构建地板，不显示墙体、门窗
    print("🏗️  构建空房间（仅地板）...")
    ctx.construct_floor(show_wall=False, show_window=False, show_door=False)
    
    # 渲染俯视图
    print("🎨 渲染不显示门窗墙的空房间俯视图...")
    ctx.topdown_view(output_path, width=width, height=height)
    
    print(f"✅ 不显示门窗墙的空房间俯视图已保存: {output_path}")


def render_full_room_no_walls_doors_windows_topdown(data, output_path, width=1024, height=1024):
    """
    渲染不显示门窗墙的满房间俯视图（包含家具，但只显示地板）
    
    Args:
        data: 原始JSON数据
        output_path: 输出图片路径
        width: 渲染宽度
        height: 渲染高度
    """
    print(f"🏠 构建不显示门窗墙的满房间场景: {data['room']['room_type']}")
    
    # 创建场景上下文
    ctx = SceneCtx(data['room']['room_type'])
    
    # 添加墙体（数据添加但不渲染）
    if data.get('wall'):
        ctx.add_walls(data['wall'])
    
    # 添加门（数据添加但不渲染）
    if data.get('door'):
        ctx.add_doors(data['door'])
    
    # 添加窗（数据添加但不渲染）
    if data.get('window'):
        ctx.add_windows(data['window'])
    
    # 添加家具
    if data.get('bbox'):
        ctx.add_boxes(data['bbox'])
    
    # 构建完整场景（只显示地板，但加载家具）
    print("🏗️  构建满房间（仅地板，含家具）...")
    ctx.construct_scene(show_wall=False, show_window=False, show_door=False)
    
    # 渲染俯视图
    print("🎨 渲染不显示门窗墙的满房间俯视图...")
    ctx.topdown_view(output_path, width=width, height=height)
    
    print(f"✅ 不显示门窗墙的满房间俯视图已保存: {output_path}")


def generate_ssl_format(data, ctx):
    """
    生成SSL格式的文本输出
    
    Args:
        data: 原始JSON数据
        ctx: SceneCtx对象，包含构建后的场景信息
    
    Returns:
        str: SSL格式的文本内容
    """
    lines = []
    
    # 生成房间ID（使用4位随机ID）
    room_id = util.generate_unique_id()
    
    # 1. Room信息
    room_type = data['room']['room_type']
    lines.append(f'Room(id="{room_id}", room_type="{room_type}")')
    
    # 2. Wall信息
    walls_data = ctx.context.get("walls", {})
    for wall_id, wall in walls_data.items():
        s = wall['s']
        e = wall['e'] 
        height = wall['height']
        lines.append(f'Wall(id="{wall_id}", room_id="{room_id}", p=[{s[0]}, {s[1]}, 0.0], q=[{e[0]}, {e[1]}, 0.0], height={height})')
    
    # 3. Door信息
    for wall_id, wall in walls_data.items():
        doors = wall.get("doors", {})
        for door_id, door in doors.items():
            center = door['center']
            width = door['width']
            height = door['height']
            lines.append(f'Door(id="{door_id}", wall_id="{wall_id}", center=[{center[0]}, {center[1]}, {center[2]}], width={width}, height={height})')
    
    # 4. Window信息
    for wall_id, wall in walls_data.items():
        windows = wall.get("windows", {})
        for window_id, window in windows.items():
            center = window['center']
            width = window['width']
            height = window['height']
            lines.append(f'Window(id="{window_id}", wall_id="{wall_id}", center=[{center[0]}, {center[1]}, {center[2]}], width={width}, height={height})')
    
    # 5. Bbox信息
    boxes_data = ctx.context.get("boxes", {})
    for box_id, box in boxes_data.items():
        center = box['center']
        angle_z = box['angle_z']
        scale = box['scale']
        label = box.get('label', 'unknown')
        asset_id = box.get('asset_id')
        if asset_id:
            lines.append(f'Bbox(id="{box_id}", room_id="{room_id}", asset_id="{asset_id}", label="{label}", center=[{center[0]}, {center[1]}, {center[2]}], angle_z={angle_z}, scale=[{scale[0]}, {scale[1]}, {scale[2]}])')
        else:
            lines.append(f'Bbox(id="{box_id}", room_id="{room_id}", label="{label}", center=[{center[0]}, {center[1]}, {center[2]}], angle_z={angle_z}, scale=[{scale[0]}, {scale[1]}, {scale[2]}])')
    
    return '\n'.join(lines)


def render_layout(line_number, output_dir='/root/projects/utils/fast-scene/output/layout', width=1024, height=1024):
    print("=" * 60)
    print(f"🏠 渲染Layout场景 (第 {line_number} 行)")
    print("=" * 60)

    # 读取数据
    jsonl_path = '/root/datasets/manycore/spatiallm_raw.jsonl'
    print(f"\n📂 读取数据: {jsonl_path}")
    print(f"   行号: {line_number}")

    try:
        data = read_jsonl_line(jsonl_path, line_number)
        print(data)
    except ValueError as e:
        print(f"❌ 错误: {e}")
        return
    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在 {jsonl_path}")
        return

    print(f"✅ 数据读取成功")
    print(f"   - 场景ID: {data.get('id', 'unknown')}")
    print(f"   - 房间类型: {data['room']['room_type']}")
    print(f"   - 墙体数量: {len(data.get('wall', []))}")
    print(f"   - 门数量: {len(data.get('door', []))}")
    print(f"   - 窗数量: {len(data.get('window', []))}")
    print(f"   - 家具数量: {len(data.get('bbox', []))}")

    # 加载配置
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 检查mesh文件并进行retrieve
    print("\n🔍 检查mesh文件...")
    print(f"   配置的model_path: {config.get('model_path', '')}")

    valid_boxes = []
    for box in data.get('bbox', []):
        if check_and_retrieve_mesh(box, config):
            valid_boxes.append(box)
        else:
            print(f"   ⚠️  跳过 {box.get('label', 'unknown')} (无可用mesh)")

    print(f"\n✅ 可用家具: {len(valid_boxes)}/{len(data.get('bbox', []))}")

    # 构建场景
    print(f"\n🏗️  构建场景: {data['room']['room_type']}")
    ctx = SceneCtx(data['room']['room_type'])

    # 添加墙体
    if data.get('wall'):
        ctx.add_walls(data['wall'])

    # 添加门
    if data.get('door'):
        ctx.add_doors(data['door'])

    # 添加窗
    if data.get('window'):
        ctx.add_windows(data['window'])

    # 添加家具（只添加有效的）
    if valid_boxes:
        ctx.add_boxes(valid_boxes)

    # 创建以行号命名的输出目录（补全到5位）
    line_number_str = f"{line_number:05d}"
    scene_output_dir = os.path.join(output_dir, line_number_str)
    os.makedirs(scene_output_dir, exist_ok=True)

    # 渲染俯视图
    output_topdown = os.path.join(scene_output_dir, 'topdown.png')

    print(f"\n🎨 渲染俯视图...")
    ctx.topdown_view(output_topdown, width=width, height=height)

    # 渲染空房间俯视图
    output_empty_topdown = os.path.join(scene_output_dir, 'empty_topdown.png')
    print(f"\n🏠 渲染空房间俯视图...")
    render_empty_room_topdown(data, output_empty_topdown, width=width, height=height)

    # 渲染空房间四个方向的视角
    print(f"\n🏠 渲染空房间四个方向视角...")
    render_empty_room_views(data, scene_output_dir, width=1024, height=1024)

    # 渲染四种新的俯视图
    print(f"\n🎨 渲染四种新的俯视图...")

    # 1. 不显示门窗的满房间俯视图
    output_full_no_doors_windows = os.path.join(scene_output_dir, 'full_no_doors_windows_topdown.png')
    render_full_room_no_doors_windows_topdown(data, output_full_no_doors_windows, width=1024, height=1024)

    # 2. 不显示门窗的空房间俯视图
    output_empty_no_doors_windows = os.path.join(scene_output_dir, 'empty_no_doors_windows_topdown.png')
    render_empty_room_no_doors_windows_topdown(data, output_empty_no_doors_windows, width=1024, height=1024)

    # 3. 不显示门窗墙的空房间俯视图
    output_empty_no_walls = os.path.join(scene_output_dir, 'empty_no_walls_topdown.png')
    render_empty_room_no_walls_doors_windows_topdown(data, output_empty_no_walls, width=1024, height=1024)

    # 4. 不显示门窗墙的满房间俯视图
    output_full_no_walls = os.path.join(scene_output_dir, 'full_no_walls_topdown.png')
    render_full_room_no_walls_doors_windows_topdown(data, output_full_no_walls, width=1024, height=1024)

    # 渲染四个方向的视角
    print("\n📸 渲染四个方向视角...")
    center = ctx.context["meta"]["center"]
    span = ctx.context["meta"]["span"]
    z_max = ctx.context["meta"]["z_max"]

    # 看向的目标点（房间中心，高度为z_max的一半）
    look_at = [center[0], center[1], z_max / 2]

    # 4个方向的相机位置
    views = [
        ("right", [center[0] + span[0]/2-0.2, center[1], z_max * 2/3]),
        ("left", [center[0] - span[0]/2+0.2, center[1], z_max * 2/3]),
        ("front", [center[0], center[1] + span[1]/2-0.2, z_max * 2/3]),
        ("back", [center[0], center[1] - span[1]/2+0.2, z_max * 2/3]),
    ]

    view_width = 1024
    view_height = 1024

    for view_name, camera_pos in views:
        output_file = os.path.join(scene_output_dir, f'{view_name}.png')
        print(f"\n📷 渲染 {view_name} 视角...")

        ctx.render_view(
            output_path=output_file,
            camera_position=camera_pos,
            look_at_target=look_at,
            width=view_width,
            height=view_height,
            render_depth=True,
        )

    # 保存完整的数据JSON（包含meta，位于id之后）
    output_json = os.path.join(scene_output_dir, 'data.json')
    ordered = {}
    if isinstance(data, dict):
        # 先放 id
        if 'id' in data:
            ordered['id'] = data['id']
        # 紧跟 meta
        ordered['meta'] = ctx.context.get('meta', {})
        # 再把其余字段按原顺序补齐（跳过已放入的键与可能存在的原meta）
        for k, v in data.items():
            if k in ('id', 'meta'):
                continue
            ordered[k] = v
    else:
        # 回退：若 data 非字典，直接包装
        ordered = { 'meta': ctx.context.get('meta', {}), 'data': data }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)

    # 保存只有boxes的JSON
    boxes_only = {
        "id": data.get('id'),
        "room": data.get('room'),
        "bbox": data.get('bbox', [])
    }
    output_boxes_json = os.path.join(scene_output_dir, 'boxes.json')
    with open(output_boxes_json, 'w', encoding='utf-8') as f:
        json.dump(boxes_only, f, indent=2, ensure_ascii=False)

    # 生成并保存SSL格式文件
    print("\n📝 生成SSL格式文件...")
    ssl_content = generate_ssl_format(data, ctx)
    output_ssl = os.path.join(scene_output_dir, 'ssl.txt')
    with open(output_ssl, 'w', encoding='utf-8') as f:
        f.write(ssl_content)
    print(f"✅ SSL格式文件已保存: {output_ssl}")

    print("\n" + "=" * 60)
    print("✅ 渲染完成!")
    print("=" * 60)
    print(f"\n输出目录: {scene_output_dir}")
    print(f"图像文件 (共14个，分辨率1024x1024):")
    print(f"满房间视角 (5个):")
    print(f"  - topdown.png (完整场景俯视图)")
    for view_name, _ in views:
        print(f"  - {view_name}.png")
    print(f"空房间视角 (5个):")
    print(f"  - empty_topdown.png (空房间俯视图)")
    print(f"  - empty_right.png (空房间右视角)")
    print(f"  - empty_left.png (空房间左视角)")
    print(f"  - empty_front.png (空房间前视角)")
    print(f"  - empty_back.png (空房间后视角)")
    print(f"特殊俯视图 (4个):")
    print(f"  - full_no_doors_windows_topdown.png (不显示门窗的满房间俯视图)")
    print(f"  - empty_no_doors_windows_topdown.png (不显示门窗的空房间俯视图)")
    print(f"  - empty_no_walls_topdown.png (不显示门窗墙的空房间俯视图)")
    print(f"  - full_no_walls_topdown.png (不显示门窗墙的满房间俯视图)")
    print(f"数据文件:")
    print(f"  - data.json (完整数据)")
    print(f"  - boxes.json (仅包含boxes)")
    print(f"  - ssl.txt (SSL格式)")


def main():
    """主函数 (CLI)"""
    parser = argparse.ArgumentParser(description='渲染spatial-layout.jsonl中的场景')
    parser.add_argument('line_number', type=int, help='要渲染的行号（从0开始）')

    args = parser.parse_args()
    render_layout(args.line_number)


if __name__ == "__main__":
    main()



