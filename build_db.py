#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建mesh检索数据库
从mesh_info.csv读取数据，构建Faiss索引并保存
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def parse_size(size_str):
    """解析size字符串为numpy数组"""
    if pd.isna(size_str) or size_str == 'null' or size_str == '':
        return None
    try:
        size_str = str(size_str).strip().strip('[]')
        if not size_str:
            return None
        size_list = [float(x.strip()) for x in size_str.split(',')]
        return np.array(size_list, dtype=np.float32) if len(size_list) == 3 else None
    except:
        return None


def parse_feature(feature_str):
    """解析feature字符串为numpy数组"""
    if pd.isna(feature_str) or feature_str == 'null' or feature_str == '':
        return None
    try:
        feature_str = str(feature_str).strip().strip('[]')
        if not feature_str:
            return None
        feature_list = [float(x.strip()) for x in feature_str.split(',')]
        return np.array(feature_list, dtype=np.float32) if len(feature_list) == 512 else None
    except:
        return None


def build_database(
    csv_path="/data-nas/data/experiments/mushui/projects/utils/fast-scene/mesh-download/mesh_info.csv",
    mesh_base_path="/data-nas/data/dataset/qunhe/Manycore-Future/processed",
    output_path="/data-nas/data/experiments/mushui/projects/utils/fast-scene/fast_scene/mesh_database.pkl",
    use_gpu=True
):
    """
    构建mesh检索数据库
    
    Args:
        csv_path: mesh_info.csv路径
        mesh_base_path: mesh文件基础路径
        output_path: 输出数据库路径
        use_gpu: 是否使用GPU构建Faiss索引
        
    Returns:
        database: 构建好的数据库字典
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mesh-download", "mesh_info.csv")
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "mesh_database.pkl")

    print("=" * 60)
    print("开始构建Mesh检索数据库")
    print("=" * 60)
    
    # 读取CSV
    print(f"\n1. 读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"   总条目数: {total_rows}")
    
    # 解析数据
    print("\n2. 解析数据...")
    mesh_ids = []
    classes = []
    classes_zh = []
    labels = []
    sizes = []
    features = []
    valid_mask = []
    
    valid_count = 0
    feature_null_count = 0
    size_null_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="   解析数据"):
        mesh_id = row['brandgood_id']
        class_name = row['class']
        class_zh = row.get('class_zh', '')
        label = row.get('label', '')
        result_key = row.get('resultKey', '')
        size = parse_size(row.get('size', 'null'))
        feature = parse_feature(row.get('feature', 'null'))
        
        # 检查mesh文件是否存在（两个条件：resultKey不为null 且 文件存在）
        result_key_exists = pd.notna(result_key) and result_key != '' and result_key != 'null'
        mesh_path = os.path.join(mesh_base_path, f"{mesh_id}.glb")
        file_exists = os.path.exists(mesh_path)
        mesh_exists = result_key_exists and file_exists
        
        # 只保留有feature的条目（feature是必需的）
        if feature is None:
            feature_null_count += 1
            continue
        
        if size is None:
            size_null_count += 1
            size = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 使用零向量占位
        
        mesh_ids.append(mesh_id)
        classes.append(class_name)
        classes_zh.append(class_zh if pd.notna(class_zh) else '')
        labels.append(label if pd.notna(label) else '')
        sizes.append(size)
        features.append(feature)
        valid_mask.append(mesh_exists)
        
        if mesh_exists:
            valid_count += 1
    
    # 转换为numpy数组
    mesh_ids = np.array(mesh_ids, dtype=np.int64)
    sizes = np.stack(sizes, axis=0)  # (N, 3)
    features = np.stack(features, axis=0)  # (N, 512)
    valid_mask = np.array(valid_mask, dtype=bool)
    
    print(f"\n   有效数据统计:")
    print(f"   - 保留条目数: {len(mesh_ids)}")
    print(f"   - Feature为null(已跳过): {feature_null_count}")
    print(f"   - Size为null(使用零向量): {size_null_count}")
    print(f"   - Mesh文件存在: {valid_count} ({valid_count/len(mesh_ids)*100:.2f}%)")
    print(f"   - Mesh文件缺失: {len(mesh_ids) - valid_count}")
    
    # 构建class索引（包括中英文）
    print("\n3. 构建class索引...")
    class_to_indices = defaultdict(list)
    for idx, (class_name, class_zh) in enumerate(zip(classes, classes_zh)):
        # 英文class
        class_to_indices[class_name].append(idx)
        # 中文class（如果存在）
        if class_zh:
            class_to_indices[class_zh].append(idx)
    
    # 转换为numpy数组
    for key in class_to_indices:
        class_to_indices[key] = np.array(class_to_indices[key], dtype=np.int32)
    
    class_to_indices = dict(class_to_indices)
    
    print(f"   - 类别总数（含中英文）: {len(class_to_indices)}")
    print(f"   - 类别示例: {list(class_to_indices.keys())[:8]}")
    
    # 构建label索引（包括中英文）
    print("\n4. 构建label索引...")
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        if label:  # 只处理非空label
            label_to_indices[label].append(idx)
    
    # 转换为numpy数组
    for label in label_to_indices:
        label_to_indices[label] = np.array(label_to_indices[label], dtype=np.int32)
    
    label_to_indices = dict(label_to_indices)
    
    print(f"   - Label总数: {len(label_to_indices)}")
    print(f"   - Label示例: {list(label_to_indices.keys())[:5]}")
    
    # 预计算所有label的CLIP特征
    print("\n5. 预计算label的CLIP特征...")
    from .clip_encoder import CLIPTextEncoder
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   使用设备: {device}")
    text_encoder = CLIPTextEncoder(device=device)
    
    # 获取所有唯一的label
    unique_labels = list(label_to_indices.keys())
    print(f"   需要编码的label数量: {len(unique_labels)}")
    
    # 批量编码
    batch_size = 32
    label_features_dict = {}
    
    for i in tqdm(range(0, len(unique_labels), batch_size), desc="   编码label"):
        batch_labels = unique_labels[i:i+batch_size]
        _, text_embeds = text_encoder(batch_labels)
        text_embeds = text_embeds.cpu().numpy()
        
        for j, label in enumerate(batch_labels):
            label_features_dict[label] = text_embeds[j]
    
    print(f"   Label特征编码完成")
    
    # 构建Faiss索引
    print("\n6. 构建Faiss索引...")
    
    # 检测GPU
    gpu_available = faiss.get_num_gpus() > 0
    use_gpu = use_gpu and gpu_available
    
    if use_gpu:
        print(f"   使用GPU加速 (可用GPU数: {faiss.get_num_gpus()})")
        res = faiss.StandardGpuResources()
    else:
        print(f"   使用CPU")
    
    # 为每个class构建索引
    class_faiss_indices = {}
    
    for class_key, indices in tqdm(class_to_indices.items(), desc="   构建class索引"):
        class_features = features[indices]  # (M, 512)
        
        # 使用IndexFlatIP (内积索引，因为特征已归一化)
        index = faiss.IndexFlatIP(512)
        
        if use_gpu:
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        index.add(class_features)
        
        # 转回CPU以便保存
        if use_gpu:
            index = faiss.index_gpu_to_cpu(index)
        
        class_faiss_indices[class_key] = index
    
    # 为每个label构建索引
    label_faiss_indices = {}
    
    for label, indices in tqdm(label_to_indices.items(), desc="   构建label索引"):
        label_features = features[indices]  # (M, 512)
        
        # 使用IndexFlatIP (内积索引，因为特征已归一化)
        index = faiss.IndexFlatIP(512)
        
        if use_gpu:
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        index.add(label_features)
        
        # 转回CPU以便保存
        if use_gpu:
            index = faiss.index_gpu_to_cpu(index)
        
        label_faiss_indices[label] = index
    
    # 构建全局索引（用于class不匹配时）
    print("   构建全局索引...")
    global_index = faiss.IndexFlatIP(512)
    
    if use_gpu:
        global_index = faiss.index_cpu_to_gpu(res, 0, global_index)
    
    global_index.add(features)
    
    if use_gpu:
        global_index = faiss.index_gpu_to_cpu(global_index)
    
    print(f"   - 全局索引大小: {global_index.ntotal}")
    
    # 构建数据库字典
    database = {
        # 元数据
        "class_to_indices": class_to_indices,
        "label_to_indices": label_to_indices,
        "all_classes": list(set([c for c in classes if c])),  # 唯一的class列表（英文）
        "all_labels": list(label_to_indices.keys()),
        
        # 核心数据
        "mesh_ids": mesh_ids,
        "classes": classes,
        "labels": labels,
        "sizes": sizes,
        "features": features,
        "valid_mask": valid_mask,
        
        # Label CLIP特征
        "label_features": label_features_dict,
        
        # Faiss索引
        "class_faiss_indices": class_faiss_indices,
        "label_faiss_indices": label_faiss_indices,
        "global_faiss_index": global_index,
        
        # 统计信息
        "stats": {
            "total_entries": len(mesh_ids),
            "valid_meshes": valid_count,
            "num_classes": len(class_to_indices),
            "num_labels": len(label_to_indices),
            "feature_dim": 512,
        }
    }
    
    # 保存数据库
    print(f"\n7. 保存数据库: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
    print(f"   数据库大小: {file_size:.2f} MB")
    
    print("\n" + "=" * 60)
    print("数据库构建完成！")
    print("=" * 60)
    print(f"输出路径: {output_path}")
    print(f"总条目数: {len(mesh_ids)}")
    print(f"有效mesh数: {valid_count}")
    print(f"类别数（含中英文）: {len(class_to_indices)}")
    print(f"Label数: {len(label_to_indices)}")
    print("=" * 60)
    
    return database


if __name__ == "__main__":
    # 构建数据库
    database = build_database(use_gpu=True)
    
    # 打印一些统计信息
    print("\n类别分布 (前10个):")
    class_counts = {}
    for class_name, indices in database["class_to_indices"].items():
        class_counts[class_name] = len(indices)
    
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for class_name, count in sorted_classes[:10]:
        print(f"  {class_name}: {count}")

