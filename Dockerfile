FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖（无 PyTorch，仅 ONNX Runtime ~30MB）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目文件
COPY config.py .
COPY scripts/step4_serve.py scripts/onnx_embed.py scripts/
COPY data/白天鹅宾馆FAQ.md data/
COPY db/ai_data_agent.db db/
COPY chroma_db/ chroma_db/
COPY static/ static/

# ONNX 模型（已导出，无需联网下载）
COPY text2vec_onnx/ text2vec_onnx/

# 暴露端口
EXPOSE 8000

# 环境变量（可在 Render 面板覆盖）
ENV HF_ENDPOINT=https://hf-mirror.com
ENV LLM_MODEL=deepseek-v4-pro
ENV LLM_BASE_URL=https://api.deepseek.com/v1

CMD ["python3", "scripts/step4_serve.py"]
