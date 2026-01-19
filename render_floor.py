from .render_layout import render_empty_room_no_doors_windows_topdown
from .fast_scene import SceneCtx
from .retrieve import retrieve
from . import util as util
import json
import os
import yaml
import argparse

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

def render_floor(line_number, output_dir='/root/projects/react/output/floor', width=1024, height=1024):
    print("=" * 60)
    print(f"🏠 渲染Layout场景 (第 {line_number} 行)")
    print("=" * 60)

    # 读取数据
    jsonl_path = '/root/projects/utils/fast-scene/data-agent/spatial-layout.jsonl'
    print(f"\n📂 读取数据: {jsonl_path}")
    print(f"   行号: {line_number}")

    try:
        data = read_jsonl_line(jsonl_path, line_number)
    except ValueError as e:
        print(f"❌ 错误: {e}")
        return
    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在 {jsonl_path}")
        return

    # 加载配置
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # 构建场景
    print(f"\n🏗️  构建场景: {data['room']['label']}")
    ctx = SceneCtx(data['room']['label'])

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
    if data.get('bbox'):
        ctx.add_boxes(data['bbox'])

    # 创建以行号命名的输出目录（补全到5位）
    line_number_str = f"{line_number:05d}"
    scene_output_dir = os.path.join(output_dir, line_number_str)
    os.makedirs(scene_output_dir, exist_ok=True)

    # 渲染俯视图
    output_topdown = os.path.join(scene_output_dir, 'topdown.png')


    ctx.construct_floor(show_wall=True, show_window=False, show_door=False)
    print(f"\n🎨 渲染俯视图...")
    ctx.topdown_view(output_topdown, width=width, height=height)

    ctx.export_wall_ssl(scene_output_dir)
    print(f"✅ 墙体SSL已导出: {scene_output_dir}/wall.txt")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='渲染spatial-layout.jsonl中的场景')
    parser.add_argument('line_number', type=int, help='要渲染的行号（从0开始）')

    args = parser.parse_args()
    render_floor(args.line_number, output_dir='/root/projects/utils/fast-scene/output')
   