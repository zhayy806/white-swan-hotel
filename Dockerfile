FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖（无需 PyTorch / ONNX 模型）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目文件（不包含 ONNX 模型，用 HF API 替代）
COPY config.py .
COPY scripts/step4_serve.py scripts/onnx_embed.py scripts/api_embed.py scripts/
COPY data/白天鹅宾馆FAQ.md data/
COPY db/ai_data_agent.db db/
COPY chroma_db/ chroma_db/
COPY static/ static/

# 暴露端口
EXPOSE 8000

# 环境变量：使用 HuggingFace API 生成向量（无需本地模型）
ENV EMBEDDING_MODE=api
ENV HF_ENDPOINT=https://hf-mirror.com
ENV LLM_MODEL=deepseek-v4-pro
ENV LLM_BASE_URL=https://api.deepseek.com/v1

CMD ["python3", "scripts/step4_serve.py"]
