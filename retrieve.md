# Mesh检索系统使用文档

## 概述

本系统提供基于CLIP语义特征和尺寸的家具mesh检索功能，支持精确匹配、模糊匹配和随机模式。

---

## 快速开始

### 1. 基本使用

```python
from retrieve import MeshRetriever

# 初始化检索器（首次使用会自动构建数据库）
retriever = MeshRetriever()

# 检索mesh（支持中英文class）
mesh_id = retriever.retrieve(
    class_name="bed",  # 也可以用中文 "床"
    caption="white tufted bed",
    size=[2.0, 2.0, 1.0],
    k=1
)

print(f"检索到的mesh_id: {mesh_id}")
```

### 2. 便捷函数

```python
from retrieve import retrieve

# 直接调用，无需手动初始化
mesh_id = retrieve(
    class_name="sofa",
    caption="modern leather sofa",
    size=[1.8, 0.9, 0.8],
    k=5  # 从前5个相似结果中选size最接近的
)
```

### 3. Random模式

```python
# 随机选择一个mesh（不考虑相似度和size）
mesh_id = retriever.retrieve(
    class_name="chair",
    random=True
)
```

---

## API参考

### MeshRetriever.retrieve()

```python
def retrieve(
    class_name: str,
    caption: str = "",
    size: Optional[List[float]] = None,
    k: int = 1,
    random: bool = False,
    only_valid: bool = True
) -> int
```

**参数说明：**

- `class_name` (str): 家具类别名称（**支持中英文**）
  - 支持Label精确匹配（如 "bathroom vanity"）
  - 支持Class精确匹配（如 "bed" 或 "床"）
  - 支持Label模糊匹配（如 "vanities" → "bathroom vanity"）
  - 支持Class模糊匹配（如 "beds" → "bed"）
  - 支持CLIP语义匹配（输入任意描述，自动匹配最相似的label）
  - 都不匹配时使用全局搜索

- `caption` (str): 文本描述（**可选**）
  - 有caption时使用CLIP相似度匹配
  - 无caption但有size时使用尺寸匹配
  - 都没有时随机返回
  - Random模式下忽略

- `size` (List[float]): 期望尺寸 `[x, y, z]`（**可选**）
  - 有caption且k>1时：从前k个相似结果中选size最接近的
  - 无caption但有size时：直接选size最接近的
  - Random模式下忽略

- `k` (int): 相似度Top-K，默认=1
  - k=1: 直接返回最相似的mesh（忽略size）
  - k>1: 从前k个相似结果中选size最接近的

- `random` (bool): 是否随机模式，默认=False
  - True: 在候选集中随机选择
  - False: 根据caption和size智能匹配

- `only_valid` (bool): 是否只返回有效mesh，默认=True
  - True: 只从文件存在的mesh中选择
  - False: 包含文件缺失的mesh

**返回值：**
- `int`: mesh_id (brandgood_id)

---

## 辅助方法

### 获取mesh详细信息

```python
info = retriever.get_mesh_info(mesh_id=343325455)
print(info)
# {'mesh_id': 343325455, 'class': 'bed', 'size': [2.22, 2.40, 1.17], 'valid': True}
```

### 列出所有类别

```python
classes = retriever.list_classes()
print(f"共有 {len(classes)} 个类别")
```

### 获取类别统计

```python
stats = retriever.get_class_stats("bed")
print(stats)
# {'class': 'bed', 'total_count': 6495, 'valid_count': 6148, 'valid_ratio': 0.946}
```

---

## 数据库管理

### 手动构建/更新数据库

当 `mesh_info.csv` 更新后，需要手动重建数据库：

```bash
cd /root/projects/utils/fast-scene
python src/build_db.py
```

构建过程会：
1. 读取 `mesh-download/mesh_info.csv`
2. 解析 size 和 feature 列
3. 检查每个mesh的 `.glb` 文件是否存在
4. 构建class索引和Faiss向量索引
5. 保存到 `src/mesh_database.pkl` (约471MB)

**数据库路径：** `/root/projects/utils/fast-scene/fast_scene/mesh_database.pkl`

---

## 完整示例

```python
from retrieve import MeshRetriever

# 初始化
retriever = MeshRetriever()

# 示例1: 中文class + CLIP相似度
mesh_id = retriever.retrieve(
    class_name="床",  # 中文class
    caption="white modern bed with storage",
    k=1
)

# 示例2: Label精确匹配 + size筛选
mesh_id = retriever.retrieve(
    class_name="bathroom vanity",  # label精确匹配
    caption="modern bathroom vanity",
    size=[1.2, 0.6, 0.8],
    k=10  # 从前10个相似结果中选size最接近的
)

# 示例3: 只有size，无caption
mesh_id = retriever.retrieve(
    class_name="sofa",
    size=[2.0, 0.9, 0.8]  # 只匹配size最接近的
)

# 示例4: CLIP语义匹配label
mesh_id = retriever.retrieve(
    class_name="洗手盆",  # 会语义匹配到 "bathroom vanity"
    caption="white sink",
    k=5
)

# 示例5: Random模式
mesh_id = retriever.retrieve(
    class_name="chair",
    random=True
)

# 获取详细信息
info = retriever.get_mesh_info(mesh_id)
print(f"选中的mesh: {info}")
```

---

# 系统架构详解

## 一、数据库构建流程

### 1.1 输入数据

**CSV文件：** `/root/projects/utils/fast-scene/mesh-download/mesh_info.csv`

**格式：**
```
brandgood_id,class,label,caption,class_zh,caption_zh,resultKey,size,feature
318510122,bathroom-cabinet,bathroom vanity,bathroom vanity,浴室柜,灰色的浴室柜,,null,"[0.013,0.030,...]"
37890831,toilet,floor-mounted toilet,white wall-hung toilet,马桶,白色的马桶,glb/...,null,"[-0.034,-0.006,...]"
...
```

**关键列说明：**
- `brandgood_id`: mesh唯一ID
- `class`: 家具类别（英文）
- `class_zh`: 家具类别（中文）
- `label`: 更细粒度的标签（英文）
- `resultKey`: mesh数据存储路径，**为null表示mesh不存在**
- `size`: 尺寸 `[x, y, z]`，可能为 `null`
- `feature`: CLIP特征向量（512维），可能为 `null`

### 1.2 数据解析

```python
# 伪代码（中文注释）

读取CSV文件 -> DataFrame (80175条记录)

初始化临时存储:
    mesh_ids = []        # mesh ID列表
    classes = []         # 英文类别列表
    classes_zh = []      # 中文类别列表
    labels = []          # 英文标签列表
    sizes = []           # 尺寸列表 (N, 3)
    features = []        # CLIP特征列表 (N, 512)
    valid_mask = []      # mesh文件存在性标记

对每条记录:
    读取字段:
        mesh_id = row['brandgood_id']
        class_name = row['class']           # 英文class
        class_zh = row['class_zh']          # 中文class
        label = row['label']                # 英文label
    
    解析 size:
        如果 size == "null" 或解析失败:
            size = [0.0, 0.0, 0.0]  # 使用零向量占位
        否则:
            size = 解析字符串 "[x,y,z]" -> numpy数组 (3,)
    
    解析 feature:
        如果 feature == "null" 或解析失败:
            跳过此条记录  # feature是必需的，没有则丢弃
            continue
        否则:
            feature = 解析字符串 "[f1,f2,...,f512]" -> numpy数组 (512,)
    
    检查mesh文件是否存在（双重验证）:
        # 第一重：检查CSV中的resultKey字段
        result_key = row.get('resultKey', '')
        result_key_exists = (resultKey 不为 null 且 不为空字符串)
        
        # 第二重：检查物理文件是否存在
        mesh_path = f"/data-nas/data/dataset/qunhe/Manycore-Future/processed/{mesh_id}.glb"
        file_exists = os.path.exists(mesh_path)
        
        # 两个条件都满足才标记为有效
        mesh_exists = result_key_exists AND file_exists
    
    添加到临时存储:
        mesh_ids.append(mesh_id)
        classes.append(class_name)
        classes_zh.append(class_zh if 存在 else '')
        labels.append(label if 存在 else '')
        sizes.append(size)
        features.append(feature)
        valid_mask.append(mesh_exists)

转换为numpy数组:
    mesh_ids = np.array(mesh_ids, dtype=int64)         # (N,)
    sizes = np.stack(sizes)                            # (N, 3)
    features = np.stack(features)                      # (N, 512)
    valid_mask = np.array(valid_mask, dtype=bool)     # (N,)
```

**输出数据格式：**
- `mesh_ids`: `numpy.ndarray(N,)` - int64类型的mesh ID数组
- `classes`: `list[str]` - 英文类别列表
- `classes_zh`: `list[str]` - 中文类别列表
- `labels`: `list[str]` - 英文标签列表
- `sizes`: `numpy.ndarray(N, 3)` - float32类型的尺寸矩阵
- `features`: `numpy.ndarray(N, 512)` - float32类型的CLIP特征矩阵（已L2归一化）
- `valid_mask`: `numpy.ndarray(N,)` - bool类型的有效性标记

**统计示例：**
- 总条目数: 80175
- Feature为null(已跳过): 0
- Size为null(使用零向量): 7788
- Mesh文件存在: 72590 (90.54%)

### 1.3 构建Class索引（支持中英文）

```python
# 伪代码

初始化 class_to_indices = {}  # 字典: class名称 -> 索引数组

对每个 (index, (class_name, class_zh)) in enumerate(zip(classes, classes_zh)):
    # 英文class建立索引
    class_to_indices[class_name].append(index)
    
    # 中文class建立索引（指向同一个候选集）
    如果 class_zh 存在:
        class_to_indices[class_zh].append(index)

# 转换为numpy数组
对每个 key in class_to_indices:
    class_to_indices[key] = np.array(
        class_to_indices[key], 
        dtype=int32
    )
```

**输出数据格式：**
```python
class_to_indices = {
    "bed": np.array([5, 23, 47, ...], dtype=int32),         # 6495个索引
    "床": np.array([5, 23, 47, ...], dtype=int32),          # 同一个候选集
    "sofa": np.array([1, 8, 19, ...], dtype=int32),         # 6170个索引
    "沙发": np.array([1, 8, 19, ...], dtype=int32),         # 同一个候选集
    "chair": np.array([3, 12, 28, ...], dtype=int32),       # 5441个索引
    "椅子": np.array([3, 12, 28, ...], dtype=int32),        # 同一个候选集
    ...
}

all_classes = ["bathroom-cabinet", "toilet", "bed", ...]  # 46个唯一英文类别
```

### 1.4 构建Label索引

```python
# 伪代码

初始化 label_to_indices = {}  # 字典: label名称 -> 索引数组

对每个 (index, label) in enumerate(labels):
    如果 label 不为空:
        label_to_indices[label].append(index)

# 转换为numpy数组
对每个 label in label_to_indices:
    label_to_indices[label] = np.array(
        label_to_indices[label], 
        dtype=int32
    )
```

**输出数据格式：**
```python
label_to_indices = {
    "bathroom vanity": np.array([12, 45, ...], dtype=int32),
    "floor-mounted toilet": np.array([23, 89, ...], dtype=int32),
    "queen bed": np.array([5, 103, ...], dtype=int32),
    ...
}

all_labels = ["bathroom vanity", "floor-mounted toilet", ...]  # 数千个label
```

### 1.5 预计算Label的CLIP特征

```python
# 伪代码

初始化CLIP编码器:
    device = "cuda" if GPU可用 else "cpu"
    text_encoder = CLIPTextEncoder(device=device)

获取所有唯一的label:
    unique_labels = list(label_to_indices.keys())

批量编码label特征:
    batch_size = 32
    label_features_dict = {}
    
    对每个 batch in batches(unique_labels, batch_size):
        # 批量编码
        _, text_embeds = text_encoder(batch)  # (batch_size, 512)
        text_embeds = text_embeds.cpu().numpy()
        
        # 保存到字典
        对每个 (label, feature) in zip(batch, text_embeds):
            label_features_dict[label] = feature  # (512,)
```

**输出数据格式：**
```python
label_features_dict = {
    "bathroom vanity": np.array([0.012, -0.034, ...], dtype=float32),  # (512,)
    "floor-mounted toilet": np.array([-0.023, 0.056, ...], dtype=float32),
    ...
}
```

**用途**：用于输入class与所有label进行语义匹配

### 1.6 构建Faiss索引

**Faiss简介：** Facebook开源的向量相似度搜索库，支持高效的KNN（K近邻）搜索。

**索引类型：** `IndexFlatIP` (Inner Product Index)
- 内积索引，因为CLIP特征已L2归一化，内积等价于余弦相似度
- 精确搜索，无损压缩

```python
# 伪代码

# 为每个class单独构建索引（包括中英文）
class_faiss_indices = {}

对每个 (class_key, indices) in class_to_indices.items():
    # 提取该class的所有特征
    class_features = features[indices]  # (M, 512)，M是该class的mesh数量
    
    # 创建Faiss索引
    index = faiss.IndexFlatIP(512)  # 512是特征维度
    
    # 添加特征向量
    index.add(class_features)  # 将M个512维向量添加到索引
    
    # 保存索引
    class_faiss_indices[class_key] = index

# 为每个label单独构建索引
label_faiss_indices = {}

对每个 (label, indices) in label_to_indices.items():
    # 提取该label的所有特征
    label_features = features[indices]  # (M, 512)
    
    # 创建Faiss索引
    index = faiss.IndexFlatIP(512)
    index.add(label_features)
    
    # 保存索引
    label_faiss_indices[label] = index

# 构建全局索引（用于所有匹配都失败时）
global_index = faiss.IndexFlatIP(512)
global_index.add(features)  # 添加所有N个特征向量
```

**输出数据格式：**
```python
class_faiss_indices = {
    "bed": <faiss.IndexFlatIP对象，包含6495个向量>,
    "床": <faiss.IndexFlatIP对象，包含6495个向量>,  # 同一个候选集
    "sofa": <faiss.IndexFlatIP对象，包含6170个向量>,
    ...
}

label_faiss_indices = {
    "bathroom vanity": <faiss.IndexFlatIP对象>,
    "queen bed": <faiss.IndexFlatIP对象>,
    ...
}

global_faiss_index = <faiss.IndexFlatIP对象，包含80175个向量>
```

### 1.7 保存数据库

```python
# 伪代码

database = {
    # 元数据
    "class_to_indices": class_to_indices,      # dict[str, np.ndarray]
    "label_to_indices": label_to_indices,      # dict[str, np.ndarray]
    "all_classes": all_classes,                # list[str] 唯一英文类别
    "all_labels": all_labels,                  # list[str] 所有label
    
    # 核心数据（按索引对齐）
    "mesh_ids": mesh_ids,                      # np.ndarray(N,)
    "classes": classes,                        # list[str]
    "labels": labels,                          # list[str]
    "sizes": sizes,                            # np.ndarray(N, 3)
    "features": features,                      # np.ndarray(N, 512)
    "valid_mask": valid_mask,                  # np.ndarray(N,)
    
    # Label CLIP特征（用于语义匹配）
    "label_features": label_features_dict,     # dict[str, np.ndarray]
    
    # Faiss索引
    "class_faiss_indices": class_faiss_indices,   # dict[str, faiss.Index]
    "label_faiss_indices": label_faiss_indices,   # dict[str, faiss.Index]
    "global_faiss_index": global_faiss_index,     # faiss.Index
    
    # 统计信息
    "stats": {
        "total_entries": 80175,
        "valid_meshes": 72590,
        "num_classes": 92,      # 含中英文
        "num_labels": 数千,      # 所有label数量
        "feature_dim": 512,
    }
}

# 保存为pickle文件
保存 database 到 "/root/projects/utils/fast-scene/fast_scene/mesh_database.pkl"
文件大小: 约500+ MB（增加了label索引）
```

---

## 二、检索流程详解

### 2.1 检索输入

```python
输入参数:
    class_name: str         例如 "bed"
    caption: str            例如 "white tufted bed"
    size: [x, y, z]         例如 [2.0, 2.0, 1.0]
    k: int                  例如 5
    random: bool            例如 False
    only_valid: bool        例如 True
```

### 2.2 步骤1：确定候选集（多级匹配）

```python
# 伪代码 - 六级匹配逻辑

# 级别1: Label精确匹配（最细粒度）
如果 class_name 在 label_to_indices 中:
    candidate_indices = label_to_indices[class_name]
    faiss_index = label_faiss_indices[class_name]
    match_type = 'label'
    
    输出: "Label精确匹配: 'bathroom vanity' (候选数: xxx)"
    跳转到: 过滤有效mesh

# 级别2: Class精确匹配（支持中英文）
否则如果 class_name 在 class_to_indices 中:
    candidate_indices = class_to_indices[class_name]
    faiss_index = class_faiss_indices[class_name]
    match_type = 'class'
    
    输出: "Class精确匹配: 'bed' 或 '床' (候选数: 6495)"
    跳转到: 过滤有效mesh

# 级别3: Label模糊匹配
否则:
    matched_label = 模糊匹配(class_name, all_labels, threshold=0.6)
    
    如果 matched_label 不为空:
        candidate_indices = label_to_indices[matched_label]
        faiss_index = label_faiss_indices[matched_label]
        match_type = 'label'
        
        输出: "Label模糊匹配: 'vanities' -> 'bathroom vanity'"
        跳转到: 过滤有效mesh

# 级别4: Class模糊匹配
    否则:
        matched_class = 模糊匹配(class_name, all_classes, threshold=0.6)
        
        如果 matched_class 不为空:
            candidate_indices = class_to_indices[matched_class]
            faiss_index = class_faiss_indices[matched_class]
            match_type = 'class'
            
            输出: "Class模糊匹配: 'beds' -> 'bed'"
            跳转到: 过滤有效mesh

# 级别5: CLIP语义匹配label
        否则:
            matched_label = CLIP语义匹配(class_name, label_features_dict)
            
            如果 matched_label 不为空:
                candidate_indices = label_to_indices[matched_label]
                faiss_index = label_faiss_indices[matched_label]
                match_type = 'label'
                
                输出: "语义匹配: '洗手盆' -> 'bathroom vanity' (相似度=0.85)"
                跳转到: 过滤有效mesh

# 级别6: 全局搜索（所有匹配都失败）
            否则:
                candidate_indices = np.arange(N)  # 所有索引
                faiss_index = global_faiss_index
                match_type = 'global'
                
                输出: "所有匹配失败，使用全局搜索 (候选数: 80175)"

---

更新：新的三阶段匹配规则（移除两类模糊匹配与全局随机）

1) Label 精确匹配（优先级最高）
   - 命中 label 后，仅在该 label 的候选集中检索：
     - 有 caption：用 CLIP 对 caption 相似度检索（无阈值）得到 top_k，若提供 size 则在 top_k 中选 size 最接近；否则取相似度最高者。
     - 无 caption 但有 size：直接按 size 欧氏距离平方最小者返回。
     - 无 caption 且无 size：在候选集中随机返回（保持历史行为）。

2) Class 精确匹配
   - 未命中 label 时，尝试命中 class；检索规则与 (1) 相同。

3) Class 语义匹配（阈值 0.7）
   - Label 和 Class 精确匹配都失败时，使用 CLIP 对所有 class 文本做语义匹配，取相似度最高的 class。
   - 若相似度 ≥ 0.7：在该 class 的候选集中按 (2) 的规则检索；
   - 若相似度 < 0.7：打印“Class 语义匹配失败或相似度过低 (<0.7)”，返回 None（由上层以 bbox 回退）。

# 过滤有效mesh
如果 only_valid == True:
    valid_candidates = candidate_indices[valid_mask[candidate_indices]]
    candidate_indices = valid_candidates
```

**数据格式：**
- `candidate_indices`: `numpy.ndarray(M,)` - 候选mesh的全局索引数组
- `faiss_index`: `faiss.IndexFlatIP` - 对应的Faiss索引对象
- `match_type`: `str` - 'label' / 'class' / 'global'
- `M`: 候选集大小

**匹配算法详解：**

**1. 模糊匹配（字符串相似度）**
```python
from difflib import get_close_matches

def 模糊匹配(query, candidates, threshold=0.6):
    matches = get_close_matches(query, candidates, n=1, cutoff=threshold)
    return matches[0] if matches else None

# 示例:
# 模糊匹配("vanities", ["bathroom vanity", ...], 0.6) -> "bathroom vanity"
# 模糊匹配("beds", ["bed", "sofa", ...], 0.6) -> "bed"
```

**2. CLIP语义匹配（向量相似度）**
```python
def CLIP语义匹配(query_class, label_features_dict):
    # 初始化CLIP编码器
    self._init_clip_encoder()
    
    # 编码查询文本
    query_feature = clip_encoder.encode(query_class)  # (512,)
    
    # 计算与所有label的相似度
    best_label = None
    best_similarity = -1.0
    
    对每个 (label, label_feature) in label_features_dict.items():
        similarity = np.dot(query_feature, label_feature)  # 内积
        
        如果 similarity > best_similarity:
            best_similarity = similarity
            best_label = label
    
    返回 best_label

# 示例:
# CLIP语义匹配("洗手盆", label_features) -> "bathroom vanity" (相似度=0.85)
# CLIP语义匹配("sink", label_features) -> "bathroom vanity" (相似度=0.92)
```

### 2.3 步骤2：Random模式分支

```python
# 伪代码

如果 random == True:
    # 从候选集中随机选择一个
    random_idx = random.choice(candidate_indices)
    mesh_id = mesh_ids[random_idx]
    
    输出: f"Random模式: 返回 mesh_id={mesh_id}"
    返回 mesh_id
    
    # Random模式直接返回，不执行后续步骤
```

**数据格式：**
- `random_idx`: `int` - 随机选中的全局索引
- `mesh_id`: `int` - 对应的mesh ID

### 2.4 步骤3：Normal模式 - 情况分类

```python
# 伪代码

# 情况1: 无caption无size -> 随机返回
如果 caption == "" 且 size is None:
    random_idx = random.choice(candidate_indices)
    mesh_id = mesh_ids[random_idx]
    
    输出: "无caption无size: 随机返回 mesh_id={mesh_id}"
    返回 mesh_id

# 情况2: 只有size，无caption -> 直接size匹配
如果 caption == "" 且 size 不为 None:
    # 边界检查：确保候选集不为空
    如果 len(candidate_indices) == 0:
        抛出错误: "候选集为空，无法进行size匹配"
    
    query_size = np.array(size, dtype=float32)
    best_idx = None
    min_dist = +∞
    
    对每个 idx in candidate_indices:
        mesh_size = sizes[idx]
        dist = sum((query_size - mesh_size) ** 2)  # 欧氏距离平方
        
        如果 dist < min_dist:
            min_dist = dist
            best_idx = idx
    
    mesh_id = mesh_ids[best_idx]
    输出: "只有size: 返回size最接近的 mesh_id={mesh_id} (距离={min_dist})"
    返回 mesh_id

# 情况3: 有caption -> CLIP相似度匹配（继续下一步）
```

### 2.5 步骤4：计算Query特征（有caption时）

```python
# 伪代码

# 初始化CLIP编码器（延迟初始化，只初始化一次）
如果 clip_encoder 为 None:
    device = "cuda" if GPU可用 else "cpu"
    clip_encoder = CachedCLIPEncoder(device=device)
    
    输出: "初始化CLIP编码器 (device={device})..."

# 编码caption（带缓存）
query_feature = clip_encoder.encode(caption)  # (512,)

# 调整形状以适配Faiss
query_feature = query_feature.reshape(1, -1)  # (1, 512)
```

**数据格式：**
- `caption`: `str` - 输入文本，如 "white tufted bed"
- `query_feature`: `numpy.ndarray(1, 512)` - float32类型的查询特征向量（已L2归一化）

**CLIP编码过程：**
```python
# CLIP编码内部流程
1. Tokenization: "white tufted bed" -> [token_ids]
2. Text Encoder: [token_ids] -> (1, 77, 512) 隐藏状态
3. Projection: (1, 77, 512) -> (1, 512) 文本嵌入
4. L2 Normalize: 特征向量归一化，使 ||feature|| = 1
```

**缓存机制：**
- 使用LRU缓存，默认缓存1024个最近使用的编码结果
- 相同caption重复查询时直接返回缓存，避免重复计算
- 缓存命中可节省约50ms的编码时间

### 2.6 步骤5：Faiss向量搜索

```python
# 伪代码

# 步骤4.1: 确定搜索数量（边界检查）
search_k = min(k * 10, len(candidate_indices), faiss_index.ntotal)
# 注意：search_k不能超过候选集大小，也不能超过faiss索引的实际大小

# 步骤4.1.1: 边界情况 - 空候选集检查
如果 search_k == 0:
    抛出错误: "候选集中没有可用的mesh"
    # 这种情况通常不会发生，因为前面已经过滤了

# 步骤4.2: Faiss KNN搜索
distances, indices = faiss_index.search(query_feature, search_k)
# distances: (1, search_k) - 相似度分数（内积值，越大越相似）
# indices: (1, search_k) - 在faiss_index中的局部索引

# 步骤4.3: 映射回全局索引
top_k_indices = candidate_indices[indices[0]]  # (search_k,)

# 步骤4.4: 再次过滤有效mesh（保险起见）
如果 only_valid:
    top_k_indices = top_k_indices[valid_mask[top_k_indices]]

# 步骤4.5: 只保留前k个（边界处理）
actual_k = min(k, len(top_k_indices))  # 如果候选数少于k，返回所有候选
top_k_indices = top_k_indices[:actual_k]  # (actual_k,)

# 步骤4.6: 最终检查
如果 len(top_k_indices) == 0:
    抛出错误: "没有找到匹配的mesh"
```

**数据格式：**

**Faiss搜索输入：**
- `query_feature`: `numpy.ndarray(1, 512)` - 查询特征
- `search_k`: `int` - 要搜索的数量

**Faiss搜索输出：**
- `distances`: `numpy.ndarray(1, search_k)` - 相似度分数（内积值）
  - 取值范围：[-1, 1]，因为特征已归一化
  - 越大越相似，1表示完全相同
  
- `indices`: `numpy.ndarray(1, search_k)` - 局部索引
  - 在`faiss_index`中的位置，不是全局索引
  - 需要通过`candidate_indices`映射回全局索引

**示例：**
```python
# 假设在bed类别中搜索，候选集有6495个mesh
candidate_indices = [5, 23, 47, 51, 89, ...]  # 长度6495

# Faiss搜索返回
distances = [[0.95, 0.92, 0.88, 0.85, 0.82]]  # 前5个最相似的分数
indices = [[0, 1, 2, 3, 4]]                   # 在bed类别内的局部索引

# 映射回全局索引
top_k_indices = candidate_indices[indices[0]]
# = [5, 23, 47, 51, 89]  # 这些是在整个数据库中的全局索引
```

### 2.7 步骤6：Size筛选（有caption时）

```python
# 伪代码

# 情况3a: 有caption但无size，或k=1 -> 直接返回相似度最高的
如果 size is None 或 k == 1:
    best_idx = top_k_indices[0]
    mesh_id = mesh_ids[best_idx]
    
    输出: "返回相似度最高的 mesh_id={mesh_id}"
    返回 mesh_id

# 情况3b: 既有caption又有size -> CLIP + Size组合筛选
query_size = np.array(size, dtype=float32)  # (3,)

best_idx = None
min_dist = +∞

对每个 idx in top_k_indices:
    mesh_size = sizes[idx]  # (3,)
    
    # 计算欧氏距离平方（不开根号）
    dist = sum((query_size - mesh_size) ** 2)
    # dist = (x1-x2)² + (y1-y2)² + (z1-z2)²
    
    如果 dist < min_dist:
        min_dist = dist
        best_idx = idx

mesh_id = mesh_ids[best_idx]

输出: "返回CLIP+size最佳匹配 mesh_id={mesh_id} (距离={min_dist:.4f})"
返回 mesh_id
```

**数据格式：**

**输入：**
- `top_k_indices`: `numpy.ndarray(k,)` - Top-K个全局索引，如 `[23, 47, 89, 102, 156]`
- `query_size`: `numpy.ndarray(3,)` - 查询尺寸，如 `[2.0, 2.0, 1.0]`

**过程：**
```python
对每个候选mesh:
    idx = 23
    mesh_size = sizes[23] = [2.22, 2.40, 1.17]  # 从数据库读取
    
    距离计算:
        diff = [2.0-2.22, 2.0-2.40, 1.0-1.17]
             = [-0.22, -0.40, -0.17]
        
        dist = (-0.22)² + (-0.40)² + (-0.17)²
             = 0.0484 + 0.1600 + 0.0289
             = 0.2373  # 不开根号
```

**输出：**
- `best_idx`: `int` - 最佳匹配的全局索引
- `min_dist`: `float` - 最小距离平方值
- `mesh_id`: `int` - 最终返回的mesh ID

---

## 三、数据流示例

### 示例1：中文class + CLIP + size筛选

**输入：**
```python
class_name = "床"  # 中文
caption = "white tufted bed"
size = [2.0, 2.0, 1.0]
k = 5
random = False
```

**步骤1 - 多级匹配确定候选集：**
```
输入: class_name="床"

级别1 - Label精确匹配: "床" 不在 label_to_indices  ❌
级别2 - Class精确匹配: "床" 在 class_to_indices  ✅
  输出: candidate_indices = [5, 23, 47, ..., 79856]  # 6495个索引
        faiss_index = <IndexFlatIP，包含6495个特征>
        match_type = 'class'
```

**步骤2 - 跳过Random模式**

**步骤3 - 计算Query特征：**
```
输入: caption = "white tufted bed"
CLIP编码: 
    Tokenize -> [101, 2317, 22299, 2793, 102, ...]
    Encode   -> (1, 512) float32
    Normalize -> L2范数 = 1.0
输出: query_feature = [[0.0137, 0.0308, -0.0030, ...]]  # (1, 512)
```

**步骤4 - Faiss搜索：**
```
输入: query_feature = (1, 512)
      faiss_index = bed类别的索引（6495个向量）
      search_k = min(5*10, 6495) = 50

Faiss搜索:
    计算query与6495个特征的内积
    排序得到Top-50
    
输出: distances = [[0.95, 0.92, 0.90, 0.88, 0.87, ...]]  # (1, 50)
      indices = [[0, 123, 456, 789, 1024, ...]]           # (1, 50) 局部索引

映射全局索引:
    top_k_indices = candidate_indices[[0, 123, 456, 789, 1024]]
                  = [5, 1823, 5467, 8923, 12456]  # 全局索引

过滤有效 + 取前k:
    top_k_indices = [5, 1823, 5467, 8923, 12456][:5]
                  = [5, 1823, 5467, 8923, 12456]  # 刚好5个
```

**步骤5 - Size筛选：**
```
输入: top_k_indices = [5, 1823, 5467, 8923, 12456]
      query_size = [2.0, 2.0, 1.0]

对每个候选:
    idx=5:     size=[2.22, 2.40, 1.17]  dist=0.2373
    idx=1823:  size=[1.95, 2.10, 1.05]  dist=0.0150  ← 最小
    idx=5467:  size=[2.50, 2.60, 1.30]  dist=0.7300
    idx=8923:  size=[1.80, 1.85, 0.95]  dist=0.0750
    idx=12456: size=[2.15, 2.25, 1.20]  dist=0.1250

选择最小距离:
    best_idx = 1823
    min_dist = 0.0150

输出: mesh_id = mesh_ids[1823] = 343325455
```

**最终输出：**
```python
mesh_id = 343325455
```

### 示例2：CLIP语义匹配 + 只有size

**输入：**
```python
class_name = "洗手盆"  # 中文描述
size = [1.2, 0.6, 0.8]
caption = ""  # 无caption
k = 1
random = False
```

**步骤1 - 多级匹配确定候选集：**
```
输入: class_name="洗手盆"

级别1 - Label精确匹配: "洗手盆" 不在 label_to_indices  ❌
级别2 - Class精确匹配: "洗手盆" 不在 class_to_indices  ❌
级别3 - Label模糊匹配: 模糊匹配("洗手盆", all_labels)  ❌
级别4 - Class模糊匹配: 模糊匹配("洗手盆", all_classes)  ❌
级别5 - CLIP语义匹配: ✅
  CLIP编码 "洗手盆"
  计算与所有label的相似度
  最佳匹配: "bathroom vanity" (相似度=0.857)
  
  输出: candidate_indices = [12, 45, 89, ...]  # bathroom vanity的mesh
        faiss_index = label_faiss_indices["bathroom vanity"]
        match_type = 'label'
```

**步骤2 - Normal模式分类：**
```
caption = ""
size = [1.2, 0.6, 0.8]

=> 情况2: 只有size，无caption
```

**步骤3 - Size直接匹配：**
```
query_size = [1.2, 0.6, 0.8]

对每个候选:
  idx=12:  size=[1.15, 0.58, 0.75]  dist=0.0074  ← 最小
  idx=45:  size=[1.30, 0.65, 0.85]  dist=0.0149
  idx=89:  size=[1.10, 0.55, 0.80]  dist=0.0125
  ...

best_idx = 12
mesh_id = mesh_ids[12] = 318510122
```

**最终输出：**
```python
mesh_id = 318510122
```

---

## 四、性能分析

### 时间复杂度

| 步骤 | 操作 | 时间复杂度 | 实际耗时 |
|------|------|-----------|---------|
| Class匹配 | 字典查询 | O(1) | <1ms |
| 模糊匹配 | difflib | O(N×M) | <10ms |
| CLIP编码 | Transformer | O(L×D²) | ~50ms (GPU) |
| Faiss搜索 | 内积+排序 | O(N×D+N×log k) | ~10ms |
| Size筛选 | 线性遍历 | O(k) | <1ms |
| **总计** | | | **~70ms** |

**说明：**
- N: 候选集大小（6495 for bed, 80175 for global）
- D: 特征维度（512）
- L: 文本长度（通常<20）
- k: Top-K数量（通常1-10）

### 空间复杂度

| 数据结构 | 大小 | 说明 |
|---------|------|------|
| mesh_ids | 80175 × 8 bytes ≈ 0.6 MB | int64数组 |
| sizes | 80175 × 3 × 4 bytes ≈ 1.0 MB | float32矩阵 |
| features | 80175 × 512 × 4 bytes ≈ 160 MB | float32矩阵 |
| Faiss索引 | ~300 MB | 所有class索引+全局索引 |
| 元数据 | ~10 MB | class映射、统计等 |
| **总计** | **~471 MB** | 数据库文件大小 |

### 优化建议

1. **CLIP编码缓存：** 已实现LRU缓存，相同caption避免重复计算
2. **批量检索：** 一次检索多个mesh时可批量编码caption
3. **Faiss GPU：** 使用faiss-gpu可进一步加速（需Python 3.11以下）
4. **索引压缩：** 对于海量数据可使用IVF或PQ索引压缩

---

## 五、常见问题

### Q1: 数据库何时需要重建？

**需要重建的情况：**
- `mesh_info.csv` 有新增或修改
- 运行了 `compute_size_feature.py` 更新了size或feature
- mesh文件目录有变化（新增/删除.glb文件）

**重建命令：**
```bash
python src/build_db.py
```

### Q2: 为什么有些mesh检索不到？

可能原因：
1. mesh的feature为null（CSV中未计算CLIP特征）
2. **mesh的resultKey为null**（mesh数据不存在）
3. mesh文件不存在且设置了`only_valid=True`
4. class不匹配且模糊匹配失败

**解决方法：**
- 运行 `compute_size_feature.py` 计算缺失的feature
- 检查CSV中的`resultKey`字段，为null表示mesh数据不存在
- 设置 `only_valid=False` 允许返回文件缺失的mesh（不推荐）
- 检查class名称是否正确

**有效性检查逻辑：**
一个mesh被标记为有效需要同时满足：
- ✅ `resultKey` 字段不为null
- ✅ `.glb` 文件在指定路径存在

### Q3: Random模式和Normal模式有什么区别？

| 特性 | Normal模式 | Random模式 |
|------|-----------|-----------|
| 需要caption | ✅ 必需 | ❌ 可选 |
| 需要size | 可选 | ❌ 忽略 |
| CLIP搜索 | ✅ 执行 | ❌ 跳过 |
| Size筛选 | ✅ 执行（k>1时） | ❌ 跳过 |
| 结果 | 最相似+最接近尺寸 | 随机 |

### Q4: k参数如何选择？

- **k=1:** 直接返回最相似的mesh，忽略size差异
- **k>1:** 从前k个相似结果中选size最接近的
  - 推荐：k=5-10，平衡相似度和尺寸匹配
  - k过大会降低相似度质量
  - k过小可能找不到尺寸合适的

**边界情况处理：**
- 如果候选集总数 m < k：系统会自动返回所有 m 个候选
- 例如：k=10 但候选集只有 3 个，则返回这 3 个候选中最优的
- 系统会自动处理，无需手动调整 k 值

### Q5: 如何查看数据库统计信息？

```python
retriever = MeshRetriever()

# 查看整体统计
print(retriever.db["stats"])

# 查看某个类别
stats = retriever.get_class_stats("bed")
print(stats)

# 查看所有类别
classes = retriever.list_classes()
print(f"共有 {len(classes)} 个类别")
```

### Q6: 系统如何处理边界情况和异常？

**边界情况自动处理：**

1. **候选集少于k个**
   - 输入：k=10，但候选集只有 3 个有效mesh
   - 处理：自动返回这 3 个中的最优结果
   - 无需调整参数

2. **候选集为空**
   - 情况：所有匹配都失败，且全局索引也为空
   - 处理：抛出清晰的错误信息
   - 提示：检查class名称或数据库是否正确加载

3. **无有效mesh**
   - 情况：候选集中所有mesh的文件都不存在
   - 处理：如果`only_valid=True`，会抛出错误
   - 解决：设置`only_valid=False`或检查mesh文件

**错误信息示例：**
```python
# 候选集为空
ValueError: "候选集中没有可用的mesh"

# 没有找到匹配
ValueError: "没有找到匹配的mesh"

# size匹配时候选集为空
ValueError: "候选集为空，无法进行size匹配"
```

**健壮性保证：**
- ✅ 所有边界情况都有检查
- ✅ 错误信息清晰明确
- ✅ 不会因数据问题而崩溃

---

## 六、技术栈

- **CLIP模型：** `openai/clip-vit-base-patch32`
  - 文本编码器：Transformer (12层, 512维输出)
  - 特征归一化：L2归一化
  
- **向量搜索：** Faiss (Facebook AI Similarity Search)
  - 索引类型：IndexFlatIP（精确内积搜索）
  - 搜索算法：暴力搜索（保证精确）
  
- **模糊匹配：** Python标准库 `difflib`
  - 算法：序列匹配（基于编辑距离）
  - 阈值：0.6（可调整）

- **数据存储：** Pickle
  - 格式：二进制序列化
  - 大小：~471 MB

---

## 七、未来扩展

### 可能的改进方向

1. **增量更新：** 支持部分更新数据库，无需全量重建
2. **多模态检索：** 支持图片输入（使用CLIP图像编码器）
3. **索引优化：** 使用IVF或HNSW索引加速大规模搜索
4. **分布式部署：** 支持多机分布式检索
5. **在线学习：** 根据用户反馈微调检索策略

---

**文档版本：** v1.1  
**最后更新：** 2025-10-14  
**作者：** Claude AI Assistant  

**v1.1 更新内容：**
- ✅ 添加 resultKey 字段双重验证逻辑
- ✅ 增强边界情况处理（topK、空候选集）
- ✅ 新增 Q6：边界情况和异常处理说明

