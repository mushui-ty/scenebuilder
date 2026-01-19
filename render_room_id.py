


import os
from util import read_jsonl_line
from fast_scene_bpy import BpySceneCtx
import bpy



if __name__ == "__main__":
    room_id = "115612732_1"
    design_id = room_id.split("_")[0]
    line = int(room_id.split("_")[1])

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
    jsonl_root = '/data-nas/data/experiments/mushui/datasets/manycore/spatialllm_raw'
    jsonl_path = os.path.join(jsonl_root, f'{design_id}.jsonl')
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    test_dir = os.path.join(base_dir, 'test_bpy_roomid')
    os.makedirs(test_dir, exist_ok=True)
    topdown_path = os.path.join(test_dir, 'scene_topdown.png')
    
    try:
        data = read_jsonl_line(jsonl_path, line)
        print(f"📁 加载场景: {data['room']['room_type']} ({len(data['bbox'])} 个物体)")
        
        ctx = BpySceneCtx(data['room']['room_type'])
        ctx.add_walls(data['wall'])
        ctx.add_doors(data.get('door', []))
        ctx.add_windows(data.get('window', []))
        ctx.add_boxes(data['bbox'])
        
        print("\n" + "="*60)
        print("渲染俯视图")
        print("="*60)
        ctx.topdown_view(topdown_path, show_wall=True, show_window=True, show_door=True, show_ceiling=False, auto_fov=True, auto_transparent=False, use_HDRI=True, render_depth=True, hdri_transparent_background=True)

        look_at = [
            (ctx.context["meta"]["center"][0], ctx.context["meta"]["center"][1], ctx.context["meta"]["z_max"] / 2)
        ][0]
        views = [
            ("right", [look_at[0] + ctx.context["meta"]["span"][0]/2 - 0.1, look_at[1], ctx.context["meta"]["z_max"] * 2/3]),
            ("left", [look_at[0] - ctx.context["meta"]["span"][0]/2 + 0.1, look_at[1], ctx.context["meta"]["z_max"] * 2/3]),
            ("front", [look_at[0], look_at[1] + ctx.context["meta"]["span"][1]/2 - 0.1, ctx.context["meta"]["z_max"] * 2/3]),
            ("back", [look_at[0], look_at[1] - ctx.context["meta"]["span"][1]/2 + 0.1, ctx.context["meta"]["z_max"] * 2/3]),
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
                show_wall=True,
                show_window=False,
                show_door=False, 
                show_ceiling=False, 
                auto_fov=True, 
                auto_transparent=False, 
                use_HDRI=True, 
                render_depth=False,
                hdri_transparent_background=True
            )
        
        print("\n✅ 全部完成!")
        print(ctx.get_context())
        
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()