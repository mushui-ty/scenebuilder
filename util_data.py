#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import sys
import json
import torch
import lancedb
import uuid
import shutil
from typing import Dict, Any, Optional, Literal, List
from pathlib import Path
from PIL import Image

# 确保能找到 Qwen3-VL-Embedding
sys.path.insert(0, str(Path(__file__).parent.parent / "Qwen3-VL-Embedding"))
sys.path.insert(0, "/data-nas/data/experiments/mushui/projects/react")

try:
    from qwen3_vl_embedding import Qwen3VLEmbedder
except ImportError:
    Qwen3VLEmbedder = None

try:
    from qunhe_api.nano_gemini import edit_image_with_qunhe
except ImportError:
    edit_image_with_qunhe = None

try:
    from qunhe_api.gen_hunyuan import hunyuan_gen
except ImportError:
    hunyuan_gen = None


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

    # 存储解析出的 caption，以便后续匹配
    captions_map = {}

    for line in ssl_text.strip().split('\n'):
        line = line.strip()
        if not line: continue

        # 解析 Caption(id="...", caption="...")
        if line.startswith('Caption('):
            cid = get_attr(r'id="([^"]+)"', line)
            cval = get_attr(r'caption="([^"]+)"', line)
            if cid and cval:
                captions_map[cid] = cval
            continue

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

    # 将独立的 Caption 匹配回对应的物体
    for category in ["bbox", "door", "window"]:
        if category in data:
            for item in data[category]:
                item_id = item.get("id")
                if item_id in captions_map:
                    item["caption"] = captions_map[item_id]

    return data

def process_image_for_generation(image: Image.Image, label: str, caption: str, output_dir: str) -> str:
    """
    使用 nanobanana (edit_image_with_qunhe) 将裁剪后的俯视图转换为物体正视图。
    """
    u_id = uuid.uuid4().hex[:8]
    if edit_image_with_qunhe is None:
        print("Warning: edit_image_with_qunhe not found, saving original.")
        tmp_path = os.path.join(output_dir, f"{u_id}.png")
        image.save(tmp_path)
        return tmp_path

    # 构建 prompt
    nano_prompt = f"现在是一幅从俯视图 crop 出来的物体图片, 上面用箭头标有物体的方向, 请你生成从箭头指向的方向也就是物体的正面看向这个物体的清晰的, 真实的, 带有细节的图像, 为了突出实体, 背景为黑色, 这个物体的描述为 一个{label}, {caption}"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存输入图作为临时文件
    input_tmp_path = os.path.join(output_dir, f"{u_id}_input_to_nano.png")
    image.save(input_tmp_path)
    
    # 调用 nanobanana
    print(f"🚀 调用 nanobanana 处理 {label}...")
    try:
        raw_res_path = edit_image_with_qunhe(
            prompt=nano_prompt,
            input_image=input_tmp_path,
            output_dir=output_dir,
        )
        
        if raw_res_path and os.path.exists(raw_res_path):
            # 复制并重命名到目标路径 (output_dir/uuid.png)
            final_output_path = os.path.join(output_dir, f"{u_id}_{label}.png")
            shutil.copy2(raw_res_path, final_output_path)
            print(f"✅ 结果已保存至: {final_output_path}")
            
            # 安全删除 nano_gemini 产生的临时时间戳目录
            temp_dir = os.path.dirname(raw_res_path)
            if temp_dir != output_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            return return_image_path
        else:
            return return_image_path
            
    except Exception as e:
        print(f"Error calling nanobanana: {e}")
        # 如果失败，尝试将输入图改名为 uuid.png 作为备份
        final_output_path = os.path.join(output_dir, f"{u_id}_{label}.png")
        if os.path.exists(input_tmp_path):
            shutil.copy2(input_tmp_path, final_output_path)
            return_image_path = final_output_path
        else:
            return_image_path = input_tmp_path
    
    # 清理临时输入文件
    if os.path.exists(input_tmp_path):
        os.remove(input_tmp_path)
    
    return return_image_path

def generate_3d_mesh(image_path_for_gen: str, mesh_id: str, gen_asset_dir: str, scale: List[float], gen_model: Literal["hunyuan-3d-rapid", "hunyuan-3d-pro"] = "hunyuan-3d-pro") -> str:
    """
    使用 nano_gen 生成 3D 资产，并将结果复制到目标路径并清理临时文件夹。
    """
    print(f"📦 正在为图片 {image_path_for_gen} 生成 3D 资产 (scale={scale})...")
    
    # 确保输出目录存在
    os.makedirs(gen_asset_dir, exist_ok=True)
    
    # 最终路径
    final_glb_path = os.path.join(gen_asset_dir, f"{mesh_id}.glb")
    
    try:
        # 调用资产生成工具
        if hunyuan_gen is None:
            print("Warning: hunyuan_gen not found.")
            if os.path.exists(image_path_for_gen):
                shutil.copy2(image_path_for_gen, final_glb_path)
            return final_glb_path

        glb_raw_path = hunyuan_gen(image_path=image_path_for_gen, output_dir=gen_asset_dir, model=gen_model)
        
        if glb_raw_path and os.path.exists(glb_raw_path):

            # 根据scale对glb_raw_path的位姿进行矫正

            shutil.copy2(glb_raw_path, final_glb_path)
            print(f"✅ 3D 资产已保存至: {final_glb_path}")
            
            # 删除 nano_gen 产生的时间戳中间文件夹
            temp_dir = os.path.dirname(glb_raw_path)
            if temp_dir != gen_asset_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            return final_glb_path
        else:
            # 如果生成失败，将输入图片复制为 .glb 作为占位
            if os.path.exists(image_path_for_gen):
                shutil.copy2(image_path_for_gen, final_glb_path)
                print(f"⚠️ 生成失败，已将输入图片作为占位符复制到: {final_glb_path}")
            return final_glb_path
            
    except Exception as e:
        print(f"Error in generate_3d_mesh: {e}")
        if os.path.exists(image_path_for_gen):
            shutil.copy2(image_path_for_gen, final_glb_path)
        return final_glb_path

def get_mesh(
    scene_json: Dict[str, Any], 
    image_path: Optional[str] = None, 
    retrieve_hole: bool = True,
    asset_mode: Literal["none", "retrieve", "generate"] = "none",
    outpaint_image_dir: Optional[str] = None,
    gen_asset_dir: Optional[str] = "/data-nas/data/dataset/qunhe/Manycore-Future/generate",
    gen_3d_model: Literal["hunyuan-3d-rapid", "hunyuan-3d-pro"] = "hunyuan-3d-pro"
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

    # --- 通用预处理：建立 asset_id 到 最佳代表物索引 的映射，并加载 Mask ---
    asset_to_orig_idx = {}
    masks_data = []
    if base_image:
        dir_name = os.path.dirname(image_path)
        
        # 1. 建立映射 (基于体积最大原则), 检索和生成都是基于 asset_id 拿属于这个asset_id的最大 bbox 的 mask
        ref_json_path = os.path.join(dir_name, "g_asset_resp.json")
        if os.path.exists(ref_json_path):
            try:
                with open(ref_json_path, "r", encoding="utf-8") as f:
                    ref_data = json.load(f)
                asset_max_volumes = {} 
                for i, item in enumerate(ref_data):
                    aid = item.get("asset_id")
                    if aid is None: continue
                    bbox = item.get("bbox", [0,0,0,0])
                    hr = item.get("h_range", [0,0])
                    vol = (bbox[3]-bbox[1]) * (bbox[2]-bbox[0]) * (hr[1]-hr[0])
                    try: aid_key = int(aid)
                    except: aid_key = aid
                    if aid_key not in asset_max_volumes or vol > asset_max_volumes[aid_key]:
                        asset_max_volumes[aid_key] = vol
                        asset_to_orig_idx[aid_key] = i
            except Exception as e:
                print(f"Error building asset mapping: {e}")

        # 2. 加载 Mask 数据
        mask_path = os.path.join(dir_name, "h_masks_amodal.pkl")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(dir_name, "d_masks.pkl")
        if os.path.exists(mask_path):
            try:
                import pickle
                with open(mask_path, 'rb') as f:
                    masks_data = pickle.load(f)
            except Exception as e:
                print(f"Error loading masks: {e}")

    # ------------------ 分支逻辑开始 ------------------

    if asset_mode == "generate":
        assert base_image is not None, "Image must be provided for generate mode"
        
        # 按 asset_id 分组
        asset_groups = {}
        for bbox in scene_json.get("bbox", []):
            if bbox.get("mesh_id"): continue
            aid = bbox.get("asset_id")
            if aid is not None:
                try: aid_key = int(aid)
                except: aid_key = aid
                if aid_key not in asset_groups: asset_groups[aid_key] = []
                asset_groups[aid_key].append(bbox)
        
        for aid, bboxes in asset_groups.items():
            if not bboxes: continue
            first_bbox = bboxes[0]
            label = first_bbox.get("label", "object")
            caption = first_bbox.get("caption", "")
            bbox_2d = first_bbox.get("bbox_2d")
            scale = first_bbox.get("scale", [1.0, 1.0, 1.0])
            
            if bbox_2d and len(bbox_2d) == 4:
                orig_idx = asset_to_orig_idx.get(aid)
                # 使用 Mask 抠图
                if masks_data and orig_idx is not None and orig_idx < len(masks_data):
                    import numpy as np
                    mask_np = masks_data[orig_idx]
                    mask_np = (mask_np > 127) if mask_np.max() > 1 else mask_np.astype(bool)
                    base_np = np.array(base_image)
                    mask_3d = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2) if mask_np.ndim == 2 else mask_np.astype(bool)
                    masked_np = np.where(mask_3d, base_np, np.zeros_like(base_np))
                    masked_image = Image.fromarray(masked_np.astype(np.uint8))
                else:
                    masked_image = base_image

                # 1.5 倍扩张裁剪
                cx, cy = (bbox_2d[1] + bbox_2d[3]) / 2, (bbox_2d[0] + bbox_2d[2]) / 2
                w, h = (bbox_2d[3] - bbox_2d[1]), (bbox_2d[2] - bbox_2d[0])
                nx1, ny1, nx2, ny2 = int(cx - w*0.75), int(cy - h*0.75), int(cx + w*0.75), int(cy + h*0.75)
                cropped_img = masked_image.crop((nx1, ny1, nx2, ny2))
                
                gen_dir = outpaint_image_dir if outpaint_image_dir else "."
                image_path_for_gen = process_image_for_generation(cropped_img, label, caption, gen_dir)
                
                mesh_id = os.path.basename(image_path_for_gen).split("_")[0]
                glb_path = generate_3d_mesh(image_path_for_gen, mesh_id, gen_asset_dir, scale, gen_3d_model)
                cropped_img.save(os.path.join(gen_dir, f"{mesh_id}_{label}_asset_{aid}_cropped.png"))
                
                for b in bboxes: b["mesh_id"] = mesh_id if glb_path else None
                    
        return scene_json


    # asset_mode == "retrieve" 逻辑
    db_uri = "/data-nas/data/experiments/mushui/projects/utils/fast-scene/fast_scene/manycore"
    db = lancedb.connect(db_uri)
    
    if retrieve_hole:
        for hole_type in ["door", "window"]:
            if hole_type in scene_json and scene_json[hole_type]:
                try:
                    table = db.open_table(hole_type)
                    for item in scene_json[hole_type]:
                        if not item.get("mesh_id"):
                            query_vector = [float(item.get("width", 0)), float(item.get("height", 0))]
                            result = table.search(query_vector).where("exist = true").metric("l2").limit(1).to_list()
                            if result: item["mesh_id"] = result[0]["mesh_id"]
                except Exception as e: print(f"Error retrieving {hole_type}: {e}")

    if "bbox" in scene_json and scene_json["bbox"]:
        bboxes_to_process = [b for b in scene_json["bbox"] if not b.get("mesh_id")]
        if not bboxes_to_process: return scene_json

        if not Qwen3VLEmbedder: return scene_json
        embedder = Qwen3VLEmbedder(model_name_or_path="/data-nas/data/experiments/mushui/.cache/huggingface/hub/Qwen/Qwen3-VL-Embedding-2B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        try: furniture_table = db.open_table("furniture")
        except: return scene_json

        batch_inputs, groups_info = [], []
        if base_image:
            asset_groups = {}
            for bbox in bboxes_to_process:
                aid = bbox.get("asset_id", f"no_id_{id(bbox)}")
                if aid not in asset_groups: asset_groups[aid] = []
                asset_groups[aid].append(bbox)
            
            for aid, group_bboxes in asset_groups.items():
                first_bbox = group_bboxes[0]
                label, caption = first_bbox.get("label"), first_bbox.get("caption")
                if not label: continue
                
                # --- Retrieve 同样使用精准代表物索引和 Mask 抠图 ---
                orig_idx = asset_to_orig_idx.get(aid)
                if masks_data and orig_idx is not None and orig_idx < len(masks_data):
                    import numpy as np
                    mask_np = masks_data[orig_idx]
                    mask_np = (mask_np > 127) if mask_np.max() > 1 else mask_np.astype(bool)
                    base_np, mask_3d = np.array(base_image), np.repeat(mask_np[:, :, np.newaxis], 3, axis=2) if mask_np.ndim == 2 else mask_np.astype(bool)
                    masked_np = np.where(mask_3d, base_np, np.zeros_like(base_np))
                    masked_image = Image.fromarray(masked_np.astype(np.uint8))
                else: masked_image = base_image

                prompt = f"this is a {label}" + (f", the caption is {caption}" if caption else "")
                bbox_2d = first_bbox.get("bbox_2d")
                if bbox_2d and len(bbox_2d) == 4:
                    # 检索也使用 1.2 倍扩张以获取更多形状信息，但核心是黑色背景
                    cx, cy = (bbox_2d[1] + bbox_2d[3]) / 2, (bbox_2d[0] + bbox_2d[2]) / 2
                    w, h = (bbox_2d[3] - bbox_2d[1]), (bbox_2d[2] - bbox_2d[0])
                    cropped = masked_image.crop((int(cx - w*0.6), int(cy - h*0.6), int(cx + w*0.6), int(cy + h*0.6)))
                    batch_inputs.append({"text": prompt, "image": cropped})
                    groups_info.append(group_bboxes)
        else:
            for bbox in bboxes_to_process:
                label, caption = bbox.get("label"), bbox.get("caption")
                if not label: continue
                batch_inputs.append({"text": f"this is a {label}" + (f", the caption is {caption}" if caption else "")})
                groups_info.append([bbox])

        if batch_inputs:
            try:
                all_embeddings = embedder.process(batch_inputs)
                if isinstance(all_embeddings, torch.Tensor):
                    all_embeddings = all_embeddings.float().cpu().numpy().tolist()
                for i, query_vector in enumerate(all_embeddings):
                    result = furniture_table.search(query_vector).where("exist = true").distance_type("cosine").limit(1).to_list()
                    if result:
                        mesh_id = result[0]["mesh_id"]
                        for b in groups_info[i]: b["mesh_id"] = mesh_id
            except Exception as e: print(f"Error in batch search: {e}")

    return scene_json

def update_ssl_with_mesh_id(ssl_text: str, scene_json: Dict[str, Any]) -> str:
    mesh_map = {}
    for cat in ["door", "window", "bbox"]:
        for item in scene_json.get(cat, []):
            if item.get("id") and item.get("mesh_id"):
                mesh_map[(cat.capitalize() if cat != "bbox" else "Bbox", item["id"])] = item["mesh_id"]
    new_lines = []
    for line in ssl_text.splitlines():
        trimmed = line.strip()
        if not trimmed: new_lines.append(line); continue
        obj_type = next((t for t in ["Door", "Window", "Bbox"] if trimmed.startswith(t + "(")), None)
        if obj_type:
            id_match = re.search(r'id="([^"]+)"', trimmed)
            if id_match:
                mesh_id = mesh_map.get((obj_type, id_match.group(1)))
                if mesh_id:
                    if 'mesh_id=' in trimmed: line = re.sub(r'mesh_id="[^"]*"', f'mesh_id="{mesh_id}"', line)
                    else:
                        r_idx = line.rfind(')')
                        if r_idx != -1:
                            prefix = line[:r_idx].strip()
                            line = line[:r_idx] + ('' if prefix.endswith('(') else ', ') + f'mesh_id="{mesh_id}"' + line[r_idx:]
        new_lines.append(line)
    return "\n".join(new_lines)
