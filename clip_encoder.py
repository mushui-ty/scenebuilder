#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP文本编码器模块
用于将文本描述转换为512维特征向量
"""

import os
import torch
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from typing import Union, List
from functools import lru_cache

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class CLIPTextEncoder(nn.Module):
    """CLIP文本编码器"""
    
    def __init__(
        self,
        name=os.path.expandvars("$MY_HOME/.cache/huggingface/hub/clip-vit-base-patch32/"),
        max_length=77,
        device="cpu"
    ):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(name).to(device).eval()

        self.text_emb_dim = self.text_encoder.config.hidden_size
        self.max_length = max_length
        self.device = device

        assert self.max_length == self.tokenizer.model_max_length

    @torch.no_grad()
    def forward(self, prompt: Union[str, List[str]], norm=True):
        """
        编码文本为特征向量
        
        Args:
            prompt: 文本或文本列表
            norm: 是否L2归一化
            
        Returns:
            text_last_hidden_state: (num_prompts, max_length, text_emb_dim)
            text_embeds: (num_prompts, text_emb_dim) - 通常为512维
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        text_encoder_output = self.text_encoder(
            text_input_ids.to(self.device)
        )

        text_last_hidden_state = text_encoder_output.last_hidden_state.float()
        text_embeds = text_encoder_output.text_embeds.float()
        
        if norm:
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return text_last_hidden_state, text_embeds
    
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        简化接口：直接返回文本嵌入
        
        Args:
            text: 文本或文本列表
            
        Returns:
            text_embeds: (num_prompts, 512) 已L2归一化的特征向量
        """
        _, text_embeds = self.forward(text, norm=True)
        return text_embeds


class CachedCLIPEncoder:
    """带缓存的CLIP编码器，避免重复编码相同文本"""
    
    def __init__(self, device="cuda", cache_size=1024):
        self.encoder = CLIPTextEncoder(device=device)
        self.cache_size = cache_size
        self._cache = {}
    
    def encode(self, text: str):
        """
        编码文本，使用LRU缓存
        
        Args:
            text: 文本描述
            
        Returns:
            feature: (512,) numpy array
        """
        if text in self._cache:
            return self._cache[text]
        
        # 编码
        with torch.no_grad():
            text_embeds = self.encoder.encode(text)
            feature = text_embeds[0].cpu().numpy()
        
        # 缓存管理（简单LRU）
        if len(self._cache) >= self.cache_size:
            # 删除最旧的项
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[text] = feature
        return feature
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()


if __name__ == "__main__":
    # 测试
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    encoder = CLIPTextEncoder(device=device)
    
    # 测试单个文本
    text = "white tufted bed"
    _, embeds = encoder(text)
    print(f"文本: '{text}'")
    print(f"特征形状: {embeds.shape}")
    print(f"特征维度: {embeds.shape[-1]}")
    
    # 测试缓存编码器
    cached_encoder = CachedCLIPEncoder(device=device)
    feature = cached_encoder.encode(text)
    print(f"\n缓存编码器特征形状: {feature.shape}")
    
    # 测试缓存
    import time
    start = time.time()
    for _ in range(100):
        cached_encoder.encode(text)
    elapsed = time.time() - start
    print(f"\n缓存命中100次耗时: {elapsed*1000:.2f}ms")

