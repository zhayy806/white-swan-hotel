"""
HuggingFace Inference API Embeddings — 无需本地模型，无需 API Token
====================================================================
使用 HuggingFace 免费 Inference API 生成向量。
同一模型 (shibing624/text2vec-base-chinese)，向量与本地 ONNX 完全一致。

优点：零本地依赖，服务秒启动，适合云部署
缺点：首次请求可能慢（HF 冷启动模型），之后快
"""

import time
import requests
import numpy as np
from typing import List
from langchain_core.embeddings import Embeddings


class APIEmbeddings(Embeddings):
    """
    通过 HuggingFace Inference API 生成 embedding 向量。
    匿名访问（无 token），低频使用免费。
    """

    def __init__(self, model_name: str, api_key: str = "", max_retries: int = 3):
        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = max_retries
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call_api(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text])[0].tolist()

    def _call_api(self, texts: List[str]) -> np.ndarray:
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True},
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    self.api_url,
                    json=payload,
                    headers=self._headers,
                    timeout=90,  # HF 模型冷启动可能需要 30-60 秒
                )

                if r.status_code == 200:
                    embeddings = r.json()
                    # 单文本时返回一维数组，包装成二维
                    if isinstance(embeddings[0], float):
                        embeddings = [embeddings]
                    arr = np.array(embeddings, dtype=np.float32)
                    # L2 归一化
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    return arr / np.maximum(norms, 1e-12)

                elif r.status_code == 503:
                    # 模型正在加载中，等待后重试
                    wait = r.json().get("estimated_time", 30)
                    print(f"   ⏳ HF 模型加载中，等待 {wait:.0f}s (attempt {attempt+1}/{self.max_retries})...")
                    time.sleep(min(wait, 60))
                    last_error = f"503: 模型加载超时"
                    continue

                else:
                    last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                    if attempt < self.max_retries - 1:
                        time.sleep(5)
                        continue

            except requests.Timeout:
                last_error = "请求超时"
                if attempt < self.max_retries - 1:
                    time.sleep(5)
                    continue
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(5)
                    continue

        raise RuntimeError(f"Embedding API 调用失败（{self.max_retries}次重试后）: {last_error}")
