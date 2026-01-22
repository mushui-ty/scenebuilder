#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh检索功能
提供基于CLIP特征和size的mesh检索
"""

import os
import sys
import pickle
import numpy as np
import random as random_module
from typing import Optional, List, Tuple, Dict
from difflib import get_close_matches
import warnings
warnings.filterwarnings('ignore')

from .clip_encoder import CachedCLIPEncoder


class MeshRetriever:
    """Mesh检索器"""
    
    def __init__(self, db_path=None):
        """
        初始化检索器
        
        Args:
            db_path: 数据库文件路径
        """
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "mesh_database.pkl")
        self.db_path = db_path
        
        # 检查数据库是否存在
        if not os.path.exists(db_path):
            print(f"数据库不存在: {db_path}")
            print(f"正在构建数据库...")
            
            # 动态导入build_db避免循环依赖
            from .build_db import build_database
            build_database(output_path=db_path)
            
            print(f"数据库构建完成!")
        
        # 加载数据库
        print(f"加载数据库: {db_path}")
        with open(db_path, 'rb') as f:
            self.db = pickle.load(f)
        
        # 提取数据
        self.class_to_indices = self.db["class_to_indices"]
        self.label_to_indices = self.db["label_to_indices"]
        self.all_classes = self.db["all_classes"]
        self.all_labels = self.db["all_labels"]
        self.mesh_ids = self.db["mesh_ids"]
        self.classes_list = self.db["classes"]
        self.labels_list = self.db["labels"]
        self.sizes = self.db["sizes"]
        self.features = self.db["features"]
        self.valid_mask = self.db["valid_mask"]
        self.label_features = self.db["label_features"]
        self.class_faiss_indices = self.db["class_faiss_indices"]
        self.label_faiss_indices = self.db["label_faiss_indices"]
        self.global_faiss_index = self.db["global_faiss_index"]
        
        # 初始化CLIP编码器（延迟初始化，只在需要时创建）
        self.clip_encoder = None
        
        # 统计信息
        stats = self.db["stats"]
        print(f"数据库加载完成:")
        print(f"  - 总条目数: {stats['total_entries']}")
        print(f"  - 有效mesh数: {stats['valid_meshes']}")
        print(f"  - 类别数（含中英文）: {stats['num_classes']}")
        print(f"  - Label数: {stats['num_labels']}")
    
    def _init_clip_encoder(self):
        """延迟初始化CLIP编码器"""
        if self.clip_encoder is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"初始化CLIP编码器 (device={device})...")
            self.clip_encoder = CachedCLIPEncoder(device=device)
    
    def _fuzzy_match(self, query: str, candidates: List[str], threshold: float = 0.6) -> Optional[str]:
        """
        模糊匹配
        
        Args:
            query: 查询字符串
            candidates: 候选列表
            threshold: 匹配阈值
            
        Returns:
            匹配到的字符串，如果没有匹配返回None
        """
        matches = get_close_matches(query, candidates, n=1, cutoff=threshold)
        if matches:
            matched = matches[0]
            print(f"  模糊匹配: '{query}' -> '{matched}'")
            return matched
        return None
    
    def _semantic_match_class(self, query_class: str) -> Tuple[Optional[str], Optional[float]]:
        """
        使用CLIP语义匹配最相似的class（阈值在上层判断）
        """
        # 初始化CLIP编码器
        self._init_clip_encoder()
        
        # 懒加载/缓存 class 文本特征
        if not hasattr(self, "_class_text_features") or self._class_text_features is None:
            self._class_text_features = {}
            for cls in self.all_classes:
                self._class_text_features[cls] = self.clip_encoder.encode(cls)
        
        query_feature = self.clip_encoder.encode(query_class)
        best_cls = None
        best_similarity = -1.0
        for cls, feat in self._class_text_features.items():
            similarity = np.dot(query_feature, feat)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cls = cls
        if best_cls is not None:
            print(f"  Class语义匹配: '{query_class}' -> '{best_cls}' (相似度={best_similarity:.4f})")
        return best_cls, float(best_similarity) if best_cls is not None else (None, None)
    
    def _compute_size_distance(self, size1: np.ndarray, size2: np.ndarray) -> float:
        """
        计算size的欧氏距离平方（不开根号）
        
        Args:
            size1: 尺寸1 [x, y, z]
            size2: 尺寸2 [x, y, z]
            
        Returns:
            距离平方
        """
        return np.sum((size1 - size2) ** 2)
    
    def retrieve(
        self,
        class_name: str,
        caption: str = "",
        size: Optional[List[float]] = None,
        k: int = 1,
        random: bool = False,
        only_valid: bool = True
    ) -> int:
        """
        检索mesh_id
        
        Args:
            class_name: 家具类别（支持中英文）
            caption: 描述文本（可选）
            size: 期望的尺寸 [x, y, z]（可选）
            k: 返回前k个相似结果中size最接近的（仅normal模式有caption时）
            random: 是否使用随机模式
            only_valid: 是否只在有效mesh中筛选（mesh文件存在）
            
        Returns:
            mesh_id (brandgood_id)
        """
        # ========== 阶段选择：新三阶段逻辑 ==========
        candidate_indices = None
        faiss_index = None
        match_type = None  # 'label' 或 'class'

        # 阶段1：Label 精确匹配
        if class_name in self.label_to_indices:
            candidate_indices = self.label_to_indices[class_name]
            faiss_index = self.label_faiss_indices[class_name]
            match_type = 'label'
            print(f"Label精确匹配: '{class_name}' (候选数: {len(candidate_indices)})")

        # 阶段2：Class 精确匹配
        elif class_name in self.class_to_indices:
            candidate_indices = self.class_to_indices[class_name]
            faiss_index = self.class_faiss_indices[class_name]
            match_type = 'class'
            print(f"Class精确匹配: '{class_name}' (候选数: {len(candidate_indices)})")

        # 阶段3：Class 语义匹配（阈值0.7）
        else:
            matched_class, sim = self._semantic_match_class(class_name)
            if matched_class is not None and sim is not None and sim >= 0.9:
                candidate_indices = self.class_to_indices[matched_class]
                faiss_index = self.class_faiss_indices[matched_class]
                match_type = 'class'
                print(f"使用Class语义匹配: '{matched_class}' (相似度={sim:.4f}, 候选数: {len(candidate_indices)})")
            else:
                print(f"Class语义匹配失败或相似度过低 (<0.7): '{class_name}'。返回 None。")
                return None
        
        # ========== 第二阶段：在候选集中检索 ==========
        
        # 2.1 Random模式：直接随机返回（需要先过滤valid）
        if random:
            if only_valid:
                valid_candidates = candidate_indices[self.valid_mask[candidate_indices]]
                if len(valid_candidates) == 0:
                    raise ValueError(f"'{class_name}' 中没有有效的mesh文件")
                candidate_indices = valid_candidates
            
            random_idx = random_module.choice(candidate_indices)
            mesh_id = self.mesh_ids[random_idx]
            print(f"Random模式: 返回 mesh_id={mesh_id}")
            return int(mesh_id)
        
        # 2.2 Normal模式 - 情况1：没有caption也没有size -> 随机返回（保持原行为）
        if not caption and size is None:
            if only_valid:
                valid_candidates = candidate_indices[self.valid_mask[candidate_indices]]
                if len(valid_candidates) == 0:
                    raise ValueError(f"'{class_name}' 中没有有效的mesh文件")
                candidate_indices = valid_candidates
            
            random_idx = random_module.choice(candidate_indices)
            mesh_id = self.mesh_ids[random_idx]
            print(f"无caption无size: 随机返回 mesh_id={mesh_id}")
            return int(mesh_id)
        
        # 2.3 Normal模式 - 情况2：只有size没有caption -> 直接size匹配
        if not caption and size is not None:
            if only_valid:
                valid_candidates = candidate_indices[self.valid_mask[candidate_indices]]
                if len(valid_candidates) == 0:
                    raise ValueError(f"'{class_name}' 中没有有效的mesh文件")
                candidate_indices = valid_candidates
            
            if len(candidate_indices) == 0:
                raise ValueError(f"候选集为空，无法进行size匹配")
            
            query_size = np.array(size, dtype=np.float32)
            best_idx = None
            min_dist = float('inf')
            
            for idx in candidate_indices:
                mesh_size = self.sizes[idx]
                dist = self._compute_size_distance(query_size, mesh_size)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            
            mesh_id = self.mesh_ids[best_idx]
            print(f"只有size: 返回size最接近的 mesh_id={mesh_id} (距离={min_dist:.4f})")
            return int(mesh_id)
        
        # 2.4 Normal模式 - 情况3：有caption -> 在候选集中用CLIP相似度匹配（无阈值），然后按size择优
        self._init_clip_encoder()
        
        # 计算query特征
        query_feature = self.clip_encoder.encode(caption)  # (512,)
        query_feature = query_feature.reshape(1, -1)  # (1, 512)
        
        # Faiss搜索（在所有候选中搜索，不提前过滤）
        # 注意：search_k不能超过faiss索引的大小
        search_k = min(k * 10, faiss_index.ntotal)
        
        # 边界情况：如果search_k为0，说明没有可用的mesh
        if search_k == 0:
            raise ValueError(f"候选集中没有可用的mesh")
        
        distances, indices = faiss_index.search(query_feature, search_k)
        
        # indices是在faiss_index中的索引，需要映射回全局索引
        top_k_indices = candidate_indices[indices[0]]
        
        # 过滤掉无效的mesh
        if only_valid:
            top_k_indices = top_k_indices[self.valid_mask[top_k_indices]]
        
        # 只保留前k个（如果候选数少于k，会返回所有候选）
        actual_k = min(k, len(top_k_indices))
        top_k_indices = top_k_indices[:actual_k]
        
        if len(top_k_indices) == 0:
            raise ValueError(f"没有找到匹配的mesh")
        
        # 2.5 如果没有size或k=1，直接返回相似度最高的
        if size is None or k == 1:
            best_idx = top_k_indices[0]
            mesh_id = self.mesh_ids[best_idx]
            print(f"返回相似度最高的 mesh_id={mesh_id}")
            return int(mesh_id)
        
        # 2.6 从top_k中选择size最接近的
        query_size = np.array(size, dtype=np.float32)
        best_idx = None
        min_dist = float('inf')
        
        for idx in top_k_indices:
            mesh_size = self.sizes[idx]
            dist = self._compute_size_distance(query_size, mesh_size)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        
        mesh_id = self.mesh_ids[best_idx]
        print(f"返回CLIP+size最佳匹配 mesh_id={mesh_id} (距离={min_dist:.4f})")
        return int(mesh_id)
    
    def get_mesh_info(self, mesh_id: int) -> Dict:
        """
        获取mesh的详细信息
        
        Args:
            mesh_id: mesh_id
            
        Returns:
            mesh信息字典
        """
        idx = np.where(self.mesh_ids == mesh_id)[0]
        if len(idx) == 0:
            return None
        
        idx = idx[0]
        return {
            "mesh_id": int(self.mesh_ids[idx]),
            "class": self.classes_list[idx],
            "size": self.sizes[idx].tolist(),
            "valid": bool(self.valid_mask[idx]),
        }
    
    def list_classes(self) -> List[str]:
        """列出所有类别"""
        return self.all_classes
    
    def get_class_stats(self, class_name: str) -> Dict:
        """
        获取某个class的统计信息
        
        Args:
            class_name: 类别名称
            
        Returns:
            统计信息字典
        """
        if class_name not in self.class_to_indices:
            return None
        
        indices = self.class_to_indices[class_name]
        valid_count = np.sum(self.valid_mask[indices])
        
        return {
            "class": class_name,
            "total_count": len(indices),
            "valid_count": int(valid_count),
            "valid_ratio": valid_count / len(indices),
        }


# 全局单例（可选，避免重复加载）
_global_retriever = None

def get_retriever(db_path="/data-nas/data/experiments/mushui/projects/utils/fast-scene/fast_scene/mesh_database.pkl") -> MeshRetriever:
    """获取全局检索器单例"""
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = MeshRetriever(db_path)
    return _global_retriever


def retrieve(
    class_name: str,
    caption: str = "",
    size: Optional[List[float]] = None,
    k: int = 1,
    random: bool = False
) -> int:
    """
    便捷函数：直接检索mesh_id
    
    Args:
        class_name: 家具类别
        caption: 描述文本
        size: 期望的尺寸 [x, y, z]
        k: 返回前k个相似结果中size最接近的
        random: 是否使用随机模式
        
    Returns:
        mesh_id
    """
    retriever = get_retriever()
    return retriever.retrieve(class_name, caption, size, k, random)


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("测试Mesh检索功能")
    print("=" * 60)
    
    # 初始化检索器
    retriever = MeshRetriever()
    
    # 测试1: 正常检索
    print("\n【测试1】正常检索 - 精确class匹配")
    mesh_id = retriever.retrieve(
        class_name="bed",
        caption="white tufted bed",
        size=[2.0, 2.0, 1.0],
        k=1
    )
    print(f"结果: {mesh_id}")
    info = retriever.get_mesh_info(mesh_id)
    print(f"信息: {info}")
    
    # 测试2: 模糊class匹配
    print("\n【测试2】模糊class匹配")
    mesh_id = retriever.retrieve(
        class_name="beds",  # 故意写错
        caption="modern bed",
        size=[2.0, 2.0, 1.0],
        k=1
    )
    print(f"结果: {mesh_id}")
    
    # 测试3: k>1，size筛选
    print("\n【测试3】k=5，size筛选")
    mesh_id = retriever.retrieve(
        class_name="sofa",
        caption="leather sofa",
        size=[1.8, 0.9, 0.8],
        k=5
    )
    print(f"结果: {mesh_id}")
    
    # 测试4: Random模式
    print("\n【测试4】Random模式")
    mesh_id = retriever.retrieve(
        class_name="chair",
        random=True
    )
    print(f"结果: {mesh_id}")
    
    # 测试5: 列出类别
    print("\n【测试5】列出前10个类别")
    classes = retriever.list_classes()[:10]
    print(f"类别: {classes}")
    
    # 测试6: 类别统计
    print("\n【测试6】类别统计")
    stats = retriever.get_class_stats("bed")
    print(f"bed类别统计: {stats}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

