#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import sys
import json
import torch
import lancedb
import uuid
from typing import Dict, Any, Optional, Literal, List
from pathlib import Path
from PIL import Image

# 确保能找到 Qwen3-VL-Embedding
sys.path.insert(0, str(Path(__file__).parent.parent / "Qwen3-VL-Embedding"))
try:
    from qwen3_vl_embedding import Qwen3VLEmbedder
except ImportError:
    Qwen3VLEmbedder = None

def parse_ssl_to_json(ssl_text: str) -> Dict[str, Any]:
    """
    解析SSL格式场景描述并返回指定的JSON结构
    """
    data = {
        "wall": [],
        "door": [],
        "window": [],
        "bbox": [],
        "room": {"room_type": "unknown"}
    }

    # 提取属性的正则辅助函数
    def get_attr(pattern, text, default=None):
        match = re.search(pattern, text)
        return match.group(1) if match else default

    def get_list_attr(pattern, text, default=None):
        match = re.search(pattern, text)
        if match:
            return [float(x.strip()) for x in match.group(1).split(',')]
        return default

    def add_if_present(target_dict, key, value):
        if value is not None:
            target_dict[key] = value

    for line in ssl_text.strip().split('\n'):
        line = line.strip()
        if not line: continue

        # Room(id="...", room_type="...")
        if line.startswith('Room('):
            add_if_present(data["room"], "id", get_attr(r'id="([^"]+)"', line))
            add_if_present(data["room"], "room_type", get_attr(r'room_type="([^"]+)"', line))

        # Wall(id="...", room_id="...", p=[...], q=[...], height=...)
        elif line.startswith('Wall('):
            wall_item = {
                "p": get_list_attr(r'p=\[([^\]]+)\]', line, []),
                "q": get_list_attr(r'q=\[([^\]]+)\]', line, []),
                "height": float(get_attr(r'height=([\d.]+)', line, "2.8"))
            }
            add_if_present(wall_item, "id", get_attr(r'id="([^"]+)"', line))
            add_if_present(wall_item, "room_id", get_attr(r'room_id="([^"]+)"', line))
            data["wall"].append(wall_item)

        # Door(id="...", wall_id="...", center=[...], width=..., height=..., mesh_id="...", category="...", label="...", caption="...", bbox_2d=[...])
        elif line.startswith('Door(') or line.startswith('Window('):
            is_door = line.startswith('Door(')
            item = {
                "center": get_list_attr(r'center=\[([^\]]+)\]', line, []),
                "width": float(get_attr(r'width=([\d.]+)', line, "0")),
                "height": float(get_attr(r'height=([\d.]+)', line, "0")),
            }
            add_if_present(item, "category", get_attr(r'category="([^"]+)"', line))
            add_if_present(item, "id", get_attr(r'id="([^"]+)"', line))
            add_if_present(item, "wall_id", get_attr(r'wall_id="([^"]+)"', line))
            add_if_present(item, "mesh_id", get_attr(r'mesh_id="([^"]+)"', line))
            add_if_present(item, "label", get_attr(r'label="([^"]+)"', line))
            add_if_present(item, "caption", get_attr(r'caption="([^"]+)"', line))
            add_if_present(item, "bbox_2d", get_list_attr(r'bbox_2d=\[([^\]]+)\]', line))

            if is_door:
                data["door"].append(item)
            else:
                data["window"].append(item)

        # Bbox(id="...", room_id="...", mesh_id="...", center=[...], angle_z=..., scale=[...], caption="...", label="...", bbox_2d=[...])
        elif line.startswith('Bbox('):
            bbox_item = {
                "center": get_list_attr(r'center=\[([^\]]+)\]', line, []),
                "angle_z": float(get_attr(r'angle_z=([\d.-]+)', line, "0")),
                "scale": get_list_attr(r'scale=\[([^\]]+)\]', line, []),
                "label": get_attr(r'label="([^"]+)"', line, "")
            }
            add_if_present(bbox_item, "id", get_attr(r'id="([^"]+)"', line))
            add_if_present(bbox_item, "room_id", get_attr(r'room_id="([^"]+)"', line))
            add_if_present(bbox_item, "mesh_id", get_attr(r'mesh_id="([^"]+)"', line))
            add_if_present(bbox_item, "asset_id", get_attr(r'asset_id="([^"]+)"', line))
            add_if_present(bbox_item, "caption", get_attr(r'caption="([^"]+)"', line))
            add_if_present(bbox_item, "bbox_2d", get_list_attr(r'bbox_2d=\[([^\]]+)\]', line))
            data["bbox"].append(bbox_item)

    return data

def process_image_for_generation(image: Image.Image) -> Image.Image:
    """
    图像补全和超分辨工具占位函数
    """
    # TODO: 实现图像补全和超分辨逻辑
    return image

def generate_3d_mesh(image: Image.Image, mesh_id: str) -> str:
    """
    3D 生成工具占位函数
    返回生成的 .glb 文件路径
    """
    save_path = f"/data-nas/data/dataset/qunhe/Manycore-Future/generate/{mesh_id}.glb"
    # TODO: 调用 Trellis 或其他工具生成 3D 模型并保存到 save_path
    # 实际实现时应在这里执行写入操作
    return save_path

def get_mesh(
    scene_json: Dict[str, Any], 
    image_path: Optional[str] = None, 
    retrieve_hole: bool = True,
    asset_mode: Literal["none", "retrieve", "generate"] = "none"
) -> Dict[str, Any]:
    """
    根据 asset_mode 处理资产并更新 scene_json
    """
    if asset_mode == "none":
        return scene_json

    # 预加载图片
    base_image = None
    if image_path and os.path.exists(image_path):
        try:
            base_image = Image.open(image_path).convert("RGB").resize((1000, 1000))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    if asset_mode == "generate":
        assert base_image is not None, "Image must be provided for generate mode"
        
        # 按 asset_id 分组，只处理没有 mesh_id 的
        asset_groups = {}
        for bbox in scene_json.get("bbox", []):
            if bbox.get("mesh_id"):
                continue
            aid = bbox.get("asset_id")
            if aid:
                if aid not in asset_groups:
                    asset_groups[aid] = []
                asset_groups[aid].append(bbox)
        
        for aid, bboxes in asset_groups.items():
            if not bboxes: continue
            # 取第一个 bbox 进行处理
            first_bbox = bboxes[0]
            bbox_2d = first_bbox.get("bbox_2d")
            if bbox_2d and len(bbox_2d) == 4:
                y1, x1, y2, x2 = bbox_2d
                # PIL crop: (left, top, right, bottom) -> (x1, y1, x2, y2)
                cropped_img = base_image.crop((x1, y1, x2, y2))
                
                # 图像增强
                processed_img = process_image_for_generation(cropped_img)
                
                # 生成 3D 资产
                mesh_id = str(uuid.uuid4())[:8]
                print(f"✨ 为 {first_bbox.get('label')} 生成新 mesh_id: {mesh_id} (asset_id: {aid})")
                glb_path = generate_3d_mesh(processed_img, mesh_id)
                
                # 更新这一组所有 bbox 的 mesh_id
                for b in bboxes:
                    b["mesh_id"] = mesh_id
                    
        return scene_json

    # 下面是 asset_mode == "retrieve" 的逻辑
    db_uri = "/data-nas/data/experiments/mushui/projects/utils/fast-scene/fast_scene/manycore"
    db = lancedb.connect(db_uri)
    
    # 1. 处理 hole (door/window)
    if retrieve_hole:
        for hole_type in ["door", "window"]:
            if hole_type in scene_json and scene_json[hole_type]:
                try:
                    table = db.open_table(hole_type)
                    for item in scene_json[hole_type]:
                        if not item.get("mesh_id"):
                            query_vector = [float(item.get("width", 0)), float(item.get("height", 0))]
                            result = table.search(query_vector).where("exist = true").metric("l2").limit(1).to_list()
                            if result:
                                item["mesh_id"] = result[0]["mesh_id"]
                                print(f"🔍 检索到 {hole_type} mesh_id: {item['mesh_id']} (width: {item.get('width')}, height: {item.get('height')})")
                except Exception as e:
                    print(f"Error retrieving {hole_type}: {e}")

    # 2. 处理 bbox
    if "bbox" in scene_json and scene_json["bbox"]:
        bboxes_to_process = [b for b in scene_json["bbox"] if not b.get("mesh_id")]
        if not bboxes_to_process:
            return scene_json

        embedder = None
        if Qwen3VLEmbedder:
            embedder = Qwen3VLEmbedder(
                model_name_or_path="/data-nas/data/experiments/mushui/.cache/huggingface/hub/Qwen/Qwen3-VL-Embedding-2B",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
        
        if not embedder:
            return scene_json

        try:
            furniture_table = db.open_table("furniture")
        except Exception as e:
            print(f"Error opening furniture table: {e}")
            return scene_json

        # 准备批量计算的输入和分组信息
        batch_inputs = []
        groups_info = [] # 记录每个 input 对应的 bboxes 列表

        if base_image:
            # 情况一：有图片，按 asset_id 分组
            asset_groups = {}
            for bbox in bboxes_to_process:
                aid = bbox.get("asset_id", f"no_id_{id(bbox)}") # 如果没 ID 就当做独立个体
                if aid not in asset_groups:
                    asset_groups[aid] = []
                asset_groups[aid].append(bbox)
            
            for aid, group_bboxes in asset_groups.items():
                first_bbox = group_bboxes[0]
                label = first_bbox.get("label")
                if not label: # label 是必须有的
                    continue
                
                caption = first_bbox.get("caption")
                prompt = f"this is a {label}"
                if caption:
                    prompt += f", the caption is {caption}"
                
                emb_input = {"text": prompt}
                bbox_2d = first_bbox.get("bbox_2d")
                if bbox_2d and len(bbox_2d) == 4:
                    y1, x1, y2, x2 = bbox_2d
                    emb_input["image"] = base_image.crop((x1, y1, x2, y2))
                
                batch_inputs.append(emb_input)
                groups_info.append(group_bboxes)
        else:
            # 情况二：无图片，每个 bbox 独立，并行计算
            for bbox in bboxes_to_process:
                label = bbox.get("label")
                if not label: # label 是必须有的
                    continue
                    
                caption = bbox.get("caption")
                prompt = f"this is a {label}"
                if caption:
                    prompt += f", the caption is {caption}"
                
                batch_inputs.append({"text": prompt})
                groups_info.append([bbox])

        # Batch Embedding 计算
        if batch_inputs:
            try:
                all_embeddings = embedder.process(batch_inputs)
                if isinstance(all_embeddings, torch.Tensor):
                    if all_embeddings.dtype == torch.bfloat16:
                        all_embeddings = all_embeddings.float()
                    all_embeddings = all_embeddings.cpu().numpy().tolist()
                
                # 并行查询
                for i, query_vector in enumerate(all_embeddings):
                    result = furniture_table.search(query_vector).where("exist = true").distance_type("cosine").limit(1).to_list()
                    if result:
                        mesh_id = result[0]["mesh_id"]
                        first_bbox = groups_info[i][0]
                        print(f"🔍 检索到 {first_bbox.get('label')} mesh_id: {mesh_id} (asset_id: {first_bbox.get('asset_id')})")
                        for b in groups_info[i]:
                            b["mesh_id"] = mesh_id
            except Exception as e:
                print(f"Error processing batch embedding/search: {e}")

    return scene_json

def update_ssl_with_mesh_id(ssl_text: str, scene_json: Dict[str, Any]) -> str:
    """
    将 scene_json 中的 mesh_id 填回原始 ssl_text 中
    """
    # 建立 (type, id) -> mesh_id 的映射
    mesh_map = {}
    for item in scene_json.get("door", []):
        if item.get("id") and item.get("mesh_id"):
            mesh_map[("Door", item["id"])] = item["mesh_id"]
    for item in scene_json.get("window", []):
        if item.get("id") and item.get("mesh_id"):
            mesh_map[("Window", item["id"])] = item["mesh_id"]
    for item in scene_json.get("bbox", []):
        if item.get("id") and item.get("mesh_id"):
            mesh_map[("Bbox", item["id"])] = item["mesh_id"]

    new_lines = []
    for line in ssl_text.splitlines():
        trimmed_line = line.strip()
        if not trimmed_line:
            new_lines.append(line)
            continue
            
        # 识别类型和 id
        obj_type = None
        for t in ["Door", "Window", "Bbox"]:
            if trimmed_line.startswith(t + "("):
                obj_type = t
                break
        
        if obj_type:
            id_match = re.search(r'id="([^"]+)"', trimmed_line)
            if id_match:
                obj_id = id_match.group(1)
                mesh_id = mesh_map.get((obj_type, obj_id))
                if mesh_id:
                    # 更新或添加 mesh_id
                    if 'mesh_id=' in trimmed_line:
                        # 替换已有的 mesh_id
                        line = re.sub(r'mesh_id="[^"]*"', f'mesh_id="{mesh_id}"', line)
                    else:
                        # 在最后一个 ')' 之前插入 mesh_id
                        r_idx = line.rfind(')')
                        if r_idx != -1:
                            # 检查前面是否有参数，有的话加逗号
                            prefix = line[:r_idx].strip()
                            if prefix.endswith('('):
                                line = line[:r_idx] + f'mesh_id="{mesh_id}"' + line[r_idx:]
                            else:
                                line = line[:r_idx] + f', mesh_id="{mesh_id}"' + line[r_idx:]
        new_lines.append(line)
    
    return "\n".join(new_lines)
