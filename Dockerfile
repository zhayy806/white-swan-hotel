FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目文件
COPY config.py .
COPY scripts/step4_serve.py scripts/
COPY data/白天鹅宾馆FAQ.md data/
COPY db/ai_data_agent.db db/
COPY chroma_db/ chroma_db/
COPY static/ static/

# 预下载 Embedding 模型（使用国内镜像）
ENV HF_ENDPOINT=https://hf-mirror.com
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('shibing624/text2vec-base-chinese')"

# 暴露端口
EXPOSE 8000

# 启动
ENV HF_ENDPOINT=https://hf-mirror.com
ENV LLM_MODEL=deepseek-v4-pro
ENV LLM_BASE_URL=https://api.deepseek.com/v1
CMD ["python3", "scripts/step4_serve.py"]
