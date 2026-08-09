import json
import sys
import os
import pickle
from pathlib import Path
import lancedb
import torch
from tqdm import tqdm
import asyncio
import aiohttp
from PIL import Image
from io import BytesIO

# 添加 Qwen3-VL-Embedding 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "Qwen3-VL-Embedding"))
from qwen3_vl_embedding import Qwen3VLEmbedder


async def download_image_async(session: aiohttp.ClientSession, url: str) -> Image.Image:
    """
    异步下载单个图片并返回PIL Image对象
    """
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                image_data = await response.read()
                return Image.open(BytesIO(image_data))
    except Exception:
        pass
    return None


async def download_batch_async(urls: list, max_concurrent: int = 16) -> dict:
    """
    异步并行下载一批图片
    
    参数:
        urls: 图片URL列表
        max_concurrent: 最大并发数
    
    返回:
        {url: PIL.Image} 字典
    """
    # 创建连接池（限制并发数）
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=10)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 创建所有下载任务
        tasks = [download_image_async(session, url) for url in urls]
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 构建结果字典
        url_to_image = {}
        for url, result in zip(urls, results):
            if isinstance(result, Image.Image):
                url_to_image[url] = result
        
        return url_to_image


def prepare_batch_with_images(inputs_batch: list, download_workers: int = 16) -> list:
    """
    使用异步方式下载batch的所有图片
    
    参数:
        inputs_batch: 包含 {"text": ..., "image": url} 的列表
        download_workers: 最大并发数
    
    返回:
        processed_batch: 将URL替换为PIL Image对象的列表
    """
    # 收集所有需要下载的URL
    urls = []
    for item in inputs_batch:
        if isinstance(item.get('image'), str):
            urls.append(item['image'])
    
    # 异步下载所有图片
    if urls:
        url_to_image = asyncio.run(download_batch_async(urls, download_workers))
    else:
        url_to_image = {}
    
    # 构建处理后的batch
    processed_batch = []
    for item in inputs_batch:
        new_item = {"text": item["text"]}
        if isinstance(item.get('image'), str):
            # 如果下载成功，使用PIL Image；否则保留URL（让模型自己处理）
            if item['image'] in url_to_image:
                new_item['image'] = url_to_image[item['image']]
            else:
                new_item['image'] = item['image']  # 下载失败，保留URL
        else:
            new_item['image'] = item.get('image')
        
        processed_batch.append(new_item)
    
    return processed_batch


def build_asset_db(
    mesh_info_path: str = "/data-nas/data/experiments/mushui/datasets/manycore/mesh_info.json",
    bgids_data_path: str = "/data-nas/data/experiments/mushui/datasets/manycore/bgids_data.json",
    db_uri: str = "/data-nas/data/experiments/mushui/projects/utils/fast-scene/scenebuilder/manycore",
    table_name: str = "furniture",
    model_path: str = "/data-nas/data/experiments/mushui/.cache/huggingface/hub/Qwen/Qwen3-VL-Embedding-2B",
    batch_size: int = 32,
    embedding_cache_path: str | None = None,
    download_workers: int = 16,  # 异步并发数（实测16为最优）
    torch_dtype=torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
):
    """
    构建资产向量数据库
    
    参数:
        mesh_info_path: mesh_info.json 文件路径
        bgids_data_path: bgids_data.json 文件路径
        db_uri: LanceDB 数据库路径
        table_name: 表名
        model_path: Qwen3-VL-Embedding 模型路径
        batch_size: 批处理大小
        embedding_cache_path: embedding 缓存文件路径
        download_workers: 异步下载的最大并发数
        torch_dtype: PyTorch 数据类型
        attn_implementation: 注意力实现方式
    """
    
    if embedding_cache_path is None:
        embedding_cache_path = os.path.join(db_uri, "embeddings.pkl")

    print("=" * 60)
    print("开始构建资产向量数据库")
    print("=" * 60)
    
    # 1. 读取数据文件
    print(f"\n[1/6] 读取数据文件...")
    with open(mesh_info_path, 'r', encoding='utf-8') as f:
        mesh_info = json.load(f)
    print(f"  - mesh_info.json: {len(mesh_info)} 条记录")
    
    with open(bgids_data_path, 'r', encoding='utf-8') as f:
        bgids_data = json.load(f)
    print(f"  - bgids_data.json: {len(bgids_data)} 条记录")
    
    # 2. 初始化模型
    print(f"\n[2/6] 检查模型和缓存...")
    
    # 3. 准备数据
    print(f"\n[3/6] 准备数据...")
    mesh_id_to_input = {}  # 使用字典避免后续 O(n²) 查找
    
    for mesh_id, info in tqdm(mesh_info.items(), desc="  处理中"):
        # 获取基本信息
        category_zh = info.get('category_zh', '')
        label = info.get('label', '')
        caption = info.get('caption', '')
        
        # 构建文本模板
        text = f"this is a {category_zh}, the label is {label}, the caption is {caption}"
        
        # 获取图片 URL
        image_url = None
        if mesh_id in bgids_data:
            image_url = bgids_data[mesh_id].get('pngPreviewImgUrl')
        
        # 只处理有图片的数据
        if image_url:
            mesh_id_to_input[mesh_id] = {
                "text": text,
                "image": image_url
            }
    
    mesh_ids = list(mesh_id_to_input.keys())
    print(f"  - 共准备 {len(mesh_ids)} 条有效数据（有图片 URL）")
    
    # 4. 检查缓存或计算 embeddings
    print(f"\n[4/6] 处理向量计算...")
    embeddings_dict = {}
    
    if os.path.exists(embedding_cache_path):
        print(f"  - 发现缓存文件: {embedding_cache_path}")
        print(f"  - 正在加载缓存...")
        with open(embedding_cache_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        print(f"  - 缓存加载完成，共 {len(embeddings_dict)} 个向量")
        
        # 检查是否所有 mesh_id 都在缓存中
        missing_ids = [mid for mid in mesh_ids if mid not in embeddings_dict]
        if missing_ids:
            print(f"  - 警告: 发现 {len(missing_ids)} 个 mesh_id 不在缓存中，需要重新计算")
        else:
            print(f"  - ✓ 所有数据都在缓存中，跳过向量计算")
    else:
        print(f"  - 未发现缓存文件，需要计算所有向量")
        missing_ids = mesh_ids
    
    # 如果有需要计算的向量
    if missing_ids:
        print(f"\n  开始计算 {len(missing_ids)} 个向量...")
        
        # 初始化模型
        print(f"  - 初始化 Qwen3-VL-Embedding 模型...")
        model = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation
        )
        print("  - 模型加载完成")
        
        # 准备需要计算的输入（使用字典查找，O(1) 复杂度）
        print(f"  - 准备待计算的输入数据...")
        missing_inputs = [mesh_id_to_input[mid] for mid in missing_ids]
        print(f"  - 准备完成，共 {len(missing_inputs)} 条数据")
        
        # 统计成功和失败的数量
        success_count = 0
        failed_count = 0
        
        # 批量计算
        print(f"  - 使用异步方式下载图片，并发数: {download_workers}")
        for i in tqdm(range(0, len(missing_inputs), batch_size), desc="  计算向量"):
            batch_inputs = missing_inputs[i:i + batch_size]
            batch_ids = missing_ids[i:i + batch_size]
            
            try:
                # 预下载这个batch的所有图片（多线程并行）
                batch_with_images = prepare_batch_with_images(batch_inputs, download_workers)
                
                # 计算embeddings
                embeddings = model.process(batch_with_images)
                
                # 将 tensor 转换为 list（先转为 float32 避免 BFloat16 不支持问题）
                if isinstance(embeddings, torch.Tensor):
                    # 如果是 BFloat16，先转换为 Float32
                    if embeddings.dtype == torch.bfloat16:
                        embeddings = embeddings.float()
                    embeddings = embeddings.cpu().numpy().tolist()
                
                # 保存到字典
                for j, mesh_id in enumerate(batch_ids):
                    embeddings_dict[mesh_id] = embeddings[j]
                
                success_count += len(batch_ids)
                    
            except Exception as e:
                print(f"\n  警告: 批次 {i//batch_size} 处理失败: {e}")
                
                # 尝试单个处理，找出哪些成功哪些失败
                for j, single_input in enumerate(batch_inputs):
                    mesh_id = batch_ids[j]
                    try:
                        # 为单个样本下载图片（使用较小的并发数）
                        single_batch = prepare_batch_with_images([single_input], download_workers=4)
                        single_embedding = model.process(single_batch)
                        
                        if isinstance(single_embedding, torch.Tensor):
                            if single_embedding.dtype == torch.bfloat16:
                                single_embedding = single_embedding.float()
                            single_embedding = single_embedding.cpu().numpy().tolist()
                        embeddings_dict[mesh_id] = single_embedding[0]
                        success_count += 1
                    except Exception as single_e:
                        # 单个也失败，记录错误并使用零向量
                        if "vision info" not in str(single_e) and "proxy" not in str(single_e).lower():
                            print(f"    mesh_id {mesh_id} 失败: {single_e}")
                        # 使用零向量占位
                        if len(embeddings_dict) > 0:
                            # 使用已有的 embedding 维度
                            embedding_dim = len(next(iter(embeddings_dict.values())))
                        else:
                            embedding_dim = 512  # 默认维度
                        embeddings_dict[mesh_id] = [0.0] * embedding_dim
                        failed_count += 1
        
        # 保存缓存
        print(f"\n  - 向量计算完成")
        print(f"    成功: {success_count} 个")
        print(f"    失败（使用零向量）: {failed_count} 个")
        print(f"  - 保存向量缓存到: {embedding_cache_path}")
        with open(embedding_cache_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"  - 缓存保存完成，共 {len(embeddings_dict)} 个向量")
    
    # 5. 构建数据库
    print(f"\n[5/6] 构建 LanceDB 数据库...")
    print(f"  - 数据库路径: {db_uri}")
    print(f"  - 表名: {table_name}")
    
    # 准备最终数据
    data = []
    for mesh_id in mesh_ids:
        info = mesh_info[mesh_id]
        
        # 获取 size 信息并转换单位（mm -> m）
        size = [0.0, 0.0, 0.0]  # 默认值
        if mesh_id in bgids_data:
            size_dict = bgids_data[mesh_id].get('size', {})
            if size_dict:
                size = [
                    size_dict.get('x', 0.0) / 1000.0,  # mm -> m
                    size_dict.get('y', 0.0) / 1000.0,
                    size_dict.get('z', 0.0) / 1000.0
                ]
        
        data.append({
            "asset_id": mesh_id,
            "exist": info.get('exist', False),
            "category_id": info.get('category_id', 0),
            "category_zh": info.get('category_zh', ''),
            "label": info.get('label', ''),
            "caption": info.get('caption', ''),
            "size": size,
            "vector": embeddings_dict[mesh_id]
        })
    
    # 连接数据库并创建表
    db = lancedb.connect(db_uri)
    table = db.create_table(table_name, data=data, mode="overwrite")
    
    print(f"  - 数据库创建完成")
    print(f"  - 共插入 {len(data)} 条记录")
    
    print("\n" + "=" * 60)
    print("✓ 资产向量数据库构建完成！")
    print("=" * 60)
    
    return table


def build_hole_table(
    window_info_path: str = "/data-nas/data/experiments/mushui/datasets/manycore/window_info.json",
    door_info_path: str = "/data-nas/data/experiments/mushui/datasets/manycore/door_info.json",
    db_uri: str = "/data-nas/data/experiments/mushui/projects/utils/fast-scene/scenebuilder/manycore"
):
    """
    构建 window 和 door 向量数据库表
    """
    print(f"\n[Hole] 开始构建 window 和 door 表...")
    print(f"  - 数据库路径: {db_uri}")
    
    # 连接数据库
    db = lancedb.connect(db_uri)
    
    configs = [
        {"name": "window", "path": window_info_path},
        {"name": "door", "path": door_info_path}
    ]
    
    for config in configs:
        table_name = config["name"]
        file_path = config["path"]
        
        print(f"  - 正在处理 {table_name} 表, 文件: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"    警告: 文件 {file_path} 不存在，跳过")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        table_data = []
        for mesh_id, info in raw_data.items():
            # 提取信息并构建数据项
            # vector 为 [width, height]
            table_data.append({
                "asset_id": str(mesh_id),
                "exist": info.get("exist", False),
                "category": info.get("category", ""),
                "vector": [float(info.get("width", 0.0)), float(info.get("height", 0.0))]
            })
            
        if table_data:
            db.create_table(table_name, data=table_data, mode="overwrite")
            print(f"    ✓ {table_name} 表创建完成, 共 {len(table_data)} 条记录")
        else:
            print(f"    警告: {table_name} 数据为空，未创建表")


if __name__ == "__main__":
    # # 直接运行此文件时，执行构建
    # table = build_asset_db(batch_size=128, download_workers=12)
    db_uri = "/data-nas/data/experiments/mushui/projects/utils/fast-scene/scenebuilder/manycore"
    # table_name = "furniture"
    # db = lancedb.connect(db_uri)
    # table = db.open_table(table_name)
    # # 测试查询
    # print("\n测试查询...")
    # query_vector = table.to_pandas()['vector'].iloc[0]
    # result = table.search(query_vector).select(["asset_id", "category_zh", "label"]).limit(3).to_polars()
    # print(result)



    build_hole_table(db_uri=db_uri)
    db = lancedb.connect(db_uri)

    table_name = "window"
    table = db.open_table(table_name)
    query_vector = table.to_pandas()['vector'].iloc[0]
    result = table.search(query_vector).where("exist = true").metric("l2").select(["asset_id", "category", "exist", "vector"]).limit(3).to_polars()
    print(result)
    best_asset_id = result["asset_id"].to_list()[0]
    print("Best asset_id:", best_asset_id)
    