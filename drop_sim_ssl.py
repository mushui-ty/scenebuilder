"""
物理掉落仿真：读取带 supported_by 的 SSL，执行重力仿真，导出更新后的 SSL。

用法：
    from fast_scene.drop_sim_ssl import drop_sim_ssl
    new_ssl = drop_sim_ssl(ssl_text, asset_dir="/path/to/assets")
"""

import os
import re
from typing import Optional

try:
    from .util_data import parse_ssl_to_json
except (ImportError, ValueError):
    from util_data import parse_ssl_to_json  # type: ignore


PASSIVE_SUPPORTS = {"floor", "wall", "ceiling"}


def drop_sim_ssl(
    ssl_text: str,
    asset_dir: Optional[str] = None,
    sim_time: float = 1.0,
) -> str:
    """执行物理掉落仿真，返回更新后的 SSL 文本。

    流程：
      1. 解析 SSL（含 supported_by 属性）
      2. 判断 active：supported_by 不在 {floor, wall, ceiling} 且不是 chip → active
      3. 如果没有 active 物体 → 直接返回原 SSL（去掉 supported_by）
      4. 构建场景 + 执行 drop_sim
      5. 导出更新后的 SSL（不含 supported_by）

    Args:
        ssl_text: 带 supported_by 属性的 SSL 文本
        asset_dir: 资产目录（GLB 文件所在目录）
        sim_time: 仿真时长（秒）

    Returns:
        更新后的 SSL 文本（不含 supported_by，z 已修正）
    """
    # 1. 解析
    scene_json = parse_ssl_to_json(ssl_text)

    # 2. 判断 active
    has_active = False
    for bbox in scene_json["bbox"]:
        supported_by = bbox.get("supported_by", "floor")
        # chip 物体（门窗嵌墙）永远 passive
        is_chip = bbox.get("scale", [1, 1, 1])[1] == 0  # b=0 是 chip 的特征
        if supported_by not in PASSIVE_SUPPORTS and not is_chip:
            bbox["active"] = True
            has_active = True
        else:
            bbox["active"] = False

    # 3. 没有 active → 直接返回去掉 supported_by 的 SSL
    if not has_active:
        print("⚡ drop_sim_ssl: 没有 active 物体，跳过物理仿真")
        return _strip_supported_by(ssl_text)

    print(f"⚡ drop_sim_ssl: 发现 {sum(1 for b in scene_json['bbox'] if b.get('active'))} 个 active 物体")

    # 4. 构建场景
    try:
        from .fast_scene_bpy import BpySceneCtx
    except (ImportError, ValueError):
        from fast_scene_bpy import BpySceneCtx  # type: ignore

    room_type = scene_json["room"].get("room_type", "unknown")
    ctx = BpySceneCtx(room_type, asset_dir)

    ctx.add_walls(scene_json["wall"])
    if scene_json["door"]:
        ctx.add_doors(scene_json["door"])
    if scene_json["window"]:
        ctx.add_windows(scene_json["window"])

    # add_boxes 会读取 active 属性
    # 同时需要把 supported_by 传进去给 drop_sim 穿透检测用
    for bbox in scene_json["bbox"]:
        ctx.add_box(
            center=bbox["center"],
            angle_z=bbox["angle_z"],
            scale=bbox["scale"],
            label=bbox.get("label"),
            caption=bbox.get("caption"),
            asset_id=bbox.get("asset_id"),
            active=bbox.get("active", False),
        )
        # 把 supported_by 存到 context 的 box_data 里，给 drop_sim 穿透检测用
        # add_box 返回 box_id，最后一个加入的就是最新的
        last_box_id = list(ctx.context["boxes"].keys())[-1]
        if bbox.get("supported_by"):
            ctx.context["boxes"][last_box_id]["supported_by"] = bbox["supported_by"]

    # 构建场景几何体
    geometry_mode = "gltf" if asset_dir else "bbox"
    ctx.construct_scene(geometry_mode=geometry_mode, show_ceiling=False)

    # 5. 执行物理仿真
    ctx.drop_sim(sim_time=sim_time)

    # 6. 导出更新后的 SSL
    #    export_ssl 写文件，我们读回来返回
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx.export_ssl(tmpdir)
        ssl_path = os.path.join(tmpdir, "ssl.txt")
        with open(ssl_path, "r", encoding="utf-8") as f:
            new_ssl = f.read()

    return new_ssl


def _strip_supported_by(ssl_text: str) -> str:
    """从 SSL 文本中移除 supported_by 属性。"""
    # 匹配 , supported_by="xxx" 或 supported_by="xxx",
    return re.sub(r',?\s*supported_by="[^"]*"', '', ssl_text)
